"""Source of truth for the helper block embedded at the end of each models/*.py Function."""

import asyncio
import base64
import glob
import inspect
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from typing import Callable, Iterator, List, Optional, Tuple

import requests

# --- shared helpers: generated from shared/helpers.py by setup/sync_shared.py; edit it there ---

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
ATTACHMENT_TYPES = {None, "file", "image", "video"}

PIPELINE_ERRORS = (
    requests.RequestException,
    subprocess.CalledProcessError,
    OSError,
    RuntimeError,
    ValueError,
    KeyError,
)

# The vision caption is where "the image shows…" leaks into prompts, so it's told to describe the scene.
SCENE_ONLY_SUFFIX = (
    " Describe only what is present in the scene itself. Do not say that this is an image, "
    "photo, picture, frame, video or clip; do not comment on resolution, sharpness or quality; "
    "and do not state what you cannot tell. Write it as a description of a real scene."
)


def extract_user_text(body: dict) -> str:
    """Returns the text of the last message."""
    messages = body.get("messages") or []
    content = messages[-1].get("content") if messages else None
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return " ".join(p for p in parts if p).strip()
    return ""


def extract_images_b64(body: dict) -> Tuple[List[str], str]:
    """Returns base64 images inlined in the last message, plus a note on any that aren't image data."""
    messages = body.get("messages") or []
    content = messages[-1].get("content") if messages else None
    if not isinstance(content, list):
        return [], ""

    images, unusable = [], 0
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "image_url":
            continue
        image_url = block.get("image_url")
        url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url or "")
        header, _, data = url.partition(",")
        if header.startswith("data:") and header.endswith(";base64") and data:
            images.append(data)
        else:
            unusable += 1
    return images, (f"{unusable} inline image(s) arrived as a link rather than image data" if unusable else "")


async def resolve_attachments(files: Optional[list]) -> Tuple[List[str], List[str], str]:
    """Turns __files__ into (video_paths, image_paths, unresolved_reason)."""
    videos: List[str] = []
    images: List[str] = []
    problems = []

    for f in files or []:
        if not isinstance(f, dict) or f.get("type") not in ATTACHMENT_TYPES:
            continue
        inner = f["file"] if isinstance(f.get("file"), dict) else f
        meta = inner.get("meta") if isinstance(inner.get("meta"), dict) else {}
        file_id = f.get("id") or inner.get("id")
        name = inner.get("filename") or meta.get("name") or file_id or "attachment"

        path, content_type = await _resolve_file_by_id(file_id) if file_id else (None, "")
        content_type = content_type or meta.get("content_type") or ""
        # The payload path is client-supplied, so it's only trusted inside the uploads directory.
        if not path and _inside_uploads(inner.get("path")):
            path = inner.get("path")

        ext = os.path.splitext(path or "")[1].lower()
        if not path:
            problems.append(f"{name}: could not locate it on disk")
        elif not os.path.exists(path):
            problems.append(f"{name}: stored path no longer exists")
        elif content_type.startswith("image/") or (not content_type and ext in IMAGE_EXTENSIONS):
            images.append(path)
        elif content_type.startswith("video/") or ext in VIDEO_EXTENSIONS:
            videos.append(path)
        else:
            problems.append(f"{name}: unsupported type ({content_type or 'unknown'})")

    return videos, images, "; ".join(problems)


async def _resolve_file_by_id(file_id: str) -> Tuple[Optional[str], str]:
    """Looks up an upload's path and content type via Open WebUI, falling back to the uploads layout."""
    try:
        from open_webui.models.files import Files

        record = Files.get_file_by_id(file_id)
        # A coroutine in current Open WebUI; un-awaited it silently looks like "no file".
        if inspect.isawaitable(record):
            record = await record
        if record is not None:
            meta = getattr(record, "meta", None)
            content_type = (meta.get("content_type") or "") if isinstance(meta, dict) else ""
            return getattr(record, "path", None), content_type
    except Exception:
        pass

    if not re.fullmatch(r"[\w-]+", file_id):
        return None, ""
    try:
        from open_webui.config import UPLOAD_DIR

        hits = glob.glob(os.path.join(glob.escape(str(UPLOAD_DIR)), f"{file_id}_*"))
        if hits:
            return hits[0], ""
    except Exception:
        pass
    return None, ""


def _inside_uploads(path: Optional[str]) -> bool:
    """Checks that a path points inside Open WebUI's uploads directory."""
    if not path:
        return False
    try:
        from open_webui.config import UPLOAD_DIR
    except Exception:
        return False
    root = os.path.realpath(str(UPLOAD_DIR))
    return os.path.realpath(path).startswith(root + os.sep)


def attachment_problem(video_count: int, image_count: int, unresolved: str) -> str:
    """Returns a warning when the attachments can't be used as sent, else an empty string."""
    if unresolved:
        return (
            f"⚠️ An attachment couldn't be read ({unresolved}). Nothing was generated, "
            "because answering without it would silently ignore it.\n\n"
            "Re-encode it (mp4 for clips, png or jpg for images), or describe it in the message text instead."
        )
    if video_count > 1 or (video_count and image_count):
        return (
            "⚠️ Attach either one clip or images, not both (or several clips) in the same "
            "message: only one source is described per prompt."
        )
    return ""


def caption_attachments(
    v, video: Optional[str], images: List[str], video_instruction: str, image_instruction: str,
    workdir: str, need_last_frame: bool = False,
) -> Tuple[str, Optional[float], Optional[str], Optional[str]]:
    """Describes the clip or images; returns (caption, clip_seconds, first_frame, last_frame)."""
    if video:
        seconds = probe_duration(video, v.FFPROBE_BINARY)
        frames = extract_frames(video, workdir, seconds, v.FRAME_COUNT, v.FFMPEG_BINARY)
        last = extract_last_frame(video, os.path.join(workdir, "frame_last.jpg"), seconds, v.FFMPEG_BINARY)
        if need_last_frame and not last:
            raise RuntimeError("ffmpeg could not extract the clip's final frame")
        encoded = [file_to_b64(p) for p in frames + ([last] if last else [])]
        instruction, first = video_instruction, frames[0]
    elif images:
        seconds, first, last, encoded, instruction = None, None, None, images, image_instruction
    else:
        return "", None, None, None

    caption = ollama_vision(
        v.OLLAMA_BASE_URL,
        v.VISION_MODEL,
        instruction + SCENE_ONLY_SUFFIX,
        cap_images(encoded, v.MAX_VISION_IMAGES),
        v.REQUEST_TIMEOUT_SECONDS,
        v.VISION_NUM_CTX,
    )
    return clean_caption(caption), seconds, first, last


def build_idea(text: str, caption: str, duration: Optional[float]) -> str:
    """Frames the caption as facts about the scene, so the prompt never points back at a reference."""
    parts = []
    if duration is not None:
        parts.append(f"Target duration: {duration:.2f} seconds.")
    if text:
        parts.append("MY INTENT (primary — this is what the prompt must deliver):\n" + text)
    if caption:
        parts.append(
            "SCENE DETAILS observed from a reference (state these as facts about the scene "
            "itself; never refer back to the reference, and never call it an image, photo, "
            "video or attachment):\n" + caption
        )
        if not text:
            parts.append(
                "No separate intent was given: treat the scene details above as the subject "
                "and write the prompt for that scene."
            )
    return "\n\n".join(parts)


def run_tool(cmd: List[str], text: bool = False) -> subprocess.CompletedProcess:
    """Runs ffmpeg/ffprobe, turning a missing binary into a readable error."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=text)
    except FileNotFoundError:
        raise RuntimeError(
            f"{cmd[0]} was not found; install ffmpeg or set the FFMPEG_BINARY / FFPROBE_BINARY valves"
        ) from None


def stderr_tail(exc: subprocess.CalledProcessError) -> str:
    """Returns the last non-empty stderr line of a failed tool run."""
    err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    lines = [line.strip() for line in err.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def probe_duration(video_path: str, ffprobe_bin: str) -> float:
    """Returns the video stream's length, falling back to the container and then the last packet."""
    # The container can outlast the video (longer audio) or report N/A (streamed webm).
    for entries in (
        ["-select_streams", "v:0", "-show_entries", "stream=duration"],
        ["-show_entries", "format=duration"],
        ["-select_streams", "v:0", "-show_entries", "packet=pts_time"],
    ):
        out = run_tool([ffprobe_bin, "-v", "error", *entries, "-of", "csv=p=0", video_path], text=True).stdout
        values = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", out)]
        if values and max(values) > 0:
            return max(values)
    raise RuntimeError("could not determine the clip's duration")


def extract_frame_at(video_path: str, out_path: str, timestamp: float, ffmpeg_bin: str) -> None:
    run_tool([ffmpeg_bin, "-y", "-ss", f"{max(timestamp, 0):.3f}", "-i", video_path,
              "-frames:v", "1", "-q:v", "2", out_path])


def extract_frames(video_path: str, out_dir: str, duration: float, frame_count: int, ffmpeg_bin: str) -> List[str]:
    """Samples frames evenly across the clip, skipping any timestamp ffmpeg can't decode."""
    count = max(1, frame_count)
    paths, last_error = [], ""
    for i in range(count):
        out_path = os.path.join(out_dir, f"frame_{i:02d}.jpg")
        try:
            extract_frame_at(video_path, out_path, duration * i / count, ffmpeg_bin)
        except subprocess.CalledProcessError as exc:
            last_error = stderr_tail(exc)
            continue
        if os.path.exists(out_path):
            paths.append(out_path)
    if not paths:
        raise RuntimeError("ffmpeg could not extract any frames from the clip" + (f": {last_error}" if last_error else ""))
    return paths


def extract_last_frame(video_path: str, out_path: str, duration: float, ffmpeg_bin: str) -> Optional[str]:
    """Writes the clip's true final frame to out_path, or returns None if ffmpeg can't."""
    # Seeking to duration-epsilon overshoots on short/low-fps clips; decoding the tail with -update 1 keeps the last frame.
    try:
        run_tool([ffmpeg_bin, "-y", "-ss", f"{max(duration - 3, 0):.3f}", "-i", video_path,
                  "-update", "1", "-q:v", "2", out_path])
    except subprocess.CalledProcessError:
        return None
    return out_path if os.path.exists(out_path) else None


def save_video_references(first_path: str, last_path: str, policy: str, ref_dir: str) -> List[Tuple[str, str]]:
    """Copies the frames the mode needs into ref_dir; returns (label, path) pairs."""
    stamp = _ref_stamp()
    wanted = {
        "none": [],
        "first": [("Reference frame (Picture 1, t=0)", first_path, "picture1")],
        "last": [("Reference frame (Picture 1, final frame)", last_path, "picture1")],
        "first_last": [
            ("Reference frame (Picture 1, t=0)", first_path, "picture1"),
            ("Reference frame (Picture 2, final frame)", last_path, "picture2"),
        ],
    }[policy]
    return [(label, _copy_ref(src, ref_dir, f"{name}_{stamp}")) for label, src, name in wanted]


def save_image_references(images_b64: List[str], policy: str, ref_dir: str) -> List[Tuple[str, str]]:
    """Writes the reference images the mode needs into ref_dir; returns (label, path) pairs."""
    if policy == "none" or not images_b64:
        return []
    stamp = _ref_stamp()
    if policy in ("first", "last"):
        return [("Reference image (Picture 1)", _write_ref(images_b64[0], ref_dir, f"picture1_{stamp}"))]
    refs = [("Reference image (Picture 1, opening)", _write_ref(images_b64[0], ref_dir, f"picture1_{stamp}"))]
    if len(images_b64) > 1:
        refs.append(("Reference image (Picture 2, ending)", _write_ref(images_b64[-1], ref_dir, f"picture2_{stamp}")))
    return refs


def reference_footer(refs: List[Tuple[str, str]], policy: str) -> str:
    """Lists the saved reference files, or says which ones the user has to supply."""
    if policy == "none":
        return ""
    if not refs:
        needed = {
            "first": "a first-frame",
            "last": "a final-frame",
            "first_last": "an opening- and an ending-frame",
        }[policy]
        return (
            "\n\nℹ️ No reference image was saved (nothing was attached), so supply "
            f"{needed} image to the generator yourself."
        )
    listed = "\n".join(f"{label} saved to: {path}" for label, path in refs)
    if policy == "first_last" and len(refs) < 2:
        listed += "\nℹ️ Only one image was attached, so supply the ending frame (Picture 2) yourself."
    return "\n\n" + listed + "\nSupply these to your MiniMax generation alongside the prompt above."


def _ref_stamp() -> str:
    """Returns a sortable, collision-free suffix shared by one request's reference files."""
    return f"{int(time.time())}_{uuid.uuid4().hex[:6]}"


def _copy_ref(frame_path: str, ref_dir: str, name: str) -> str:
    os.makedirs(ref_dir, exist_ok=True)
    dest = os.path.join(ref_dir, f"{name}.jpg")
    shutil.copyfile(frame_path, dest)
    return dest


def _write_ref(image_b64: str, ref_dir: str, name: str) -> str:
    data = base64.b64decode(image_b64)
    os.makedirs(ref_dir, exist_ok=True)
    dest = os.path.join(ref_dir, name + _image_ext(data))
    with open(dest, "wb") as fh:
        fh.write(data)
    return dest


def _image_ext(data: bytes) -> str:
    """Picks a file extension from the image's magic bytes."""
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith(b"GIF8"):
        return ".gif"
    return ".jpg"


def file_to_b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def cap_images(images: List[str], limit: int) -> List[str]:
    """Subsamples evenly down to the vision model's image budget, keeping both endpoints."""
    if limit <= 0 or len(images) <= limit:
        return images
    if limit == 1:
        return images[:1]
    picked, seen = [], set()
    for i in range(limit):
        idx = round(i * (len(images) - 1) / (limit - 1))
        if idx not in seen:
            seen.add(idx)
            picked.append(images[idx])
    return picked


def ollama_vision(
    base_url: str, model: str, instruction: str, images_b64: List[str], timeout: int, num_ctx: int = 0
) -> str:
    payload = {"model": model, "prompt": instruction, "images": images_b64, "stream": False}
    return _ollama_generate(base_url, payload, timeout, num_ctx)


def ollama_generate(
    base_url: str, model: str, system: str, prompt: str, timeout: int, num_ctx: int = 0
) -> str:
    payload = {"model": model, "system": system, "prompt": prompt, "stream": False}
    return _ollama_generate(base_url, payload, timeout, num_ctx)


def _ollama_generate(base_url: str, payload: dict, timeout: int, num_ctx: int) -> str:
    """Posts to Ollama's generate endpoint, surfacing Ollama's own error text on failure."""
    if num_ctx:
        payload["options"] = {"num_ctx": num_ctx}
    resp = requests.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=timeout)
    if not resp.ok:
        raise RuntimeError(_ollama_error(resp, payload["model"]))
    text = (resp.json().get("response") or "").strip()
    if not text:
        raise RuntimeError(f"{payload['model']} returned an empty response")
    return text


def _ollama_error(resp: requests.Response, model: str) -> str:
    """Formats a failed Ollama response, preferring Ollama's own error text."""
    try:
        detail = resp.json().get("error") or resp.text
    except ValueError:
        detail = resp.text
    return f"Ollama returned {resp.status_code} for {model}: {detail.strip()[:300]}"


def ollama_stream(
    base_url: str, model: str, system: str, prompt: str, timeout: int, num_ctx: int = 0
) -> Iterator[str]:
    """Yields the writer's text as Ollama generates it."""
    payload = {"model": model, "system": system, "prompt": prompt, "stream": True}
    if num_ctx:
        payload["options"] = {"num_ctx": num_ctx}
    got_text = False
    with requests.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=timeout, stream=True) as resp:
        if not resp.ok:
            raise RuntimeError(_ollama_error(resp, model))
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("error"):
                raise RuntimeError(f"Ollama failed mid-response for {model}: {str(data['error'])[:300]}")
            piece = data.get("response") or ""
            got_text = got_text or bool(piece.strip())
            if piece:
                yield piece
            if data.get("done"):
                break
    if not got_text:
        raise RuntimeError(f"{model} returned an empty response")


# Covers the longest edit a transform makes near the end: a leading "Alignment Instruction:" or "the image's".
STREAM_HOLDBACK = 40


def stream_transformed(chunks: Iterator[str], transform: Callable[[str], str]) -> Iterator[str]:
    """Yields transform(full text) as it grows, holding back a tail the transform may still rewrite."""
    raw, sent = "", ""
    for chunk in chunks:
        raw += chunk
        safe = transform(raw)[:-STREAM_HOLDBACK]
        if len(safe) > len(sent) and safe.startswith(sent):
            yield safe[len(sent):]
            sent = safe
    final = transform(raw)
    rest = final[len(os.path.commonprefix([sent, final])):]
    if rest:
        yield rest


async def stream_in_thread(make: Callable[..., Iterator[str]], *args):
    """Drives a blocking generator from a worker thread so Open WebUI's event loop stays free."""
    gen, done = make(*args), object()
    try:
        while True:
            chunk = await asyncio.to_thread(next, gen, done)
            if chunk is done:
                return
            yield chunk
    finally:
        try:
            gen.close()
        except ValueError:
            pass


def pipeline_error(exc: Exception, had_attachment: bool) -> str:
    """Formats a pipeline failure as a user-facing message."""
    if isinstance(exc, requests.ConnectionError):
        detail = "could not reach Ollama. Is it running?"
    elif isinstance(exc, requests.Timeout):
        detail = "Ollama timed out. The model may still be loading: retry, or raise REQUEST_TIMEOUT_SECONDS"
    elif isinstance(exc, subprocess.CalledProcessError):
        detail = f"{os.path.basename(str(exc.cmd[0]))} failed: {stderr_tail(exc) or f'exit status {exc.returncode}'}"
    else:
        detail = str(exc) or type(exc).__name__
    message = f"⚠️ Could not build the prompt: {detail}"
    if had_attachment:
        message += (
            "\n\nFallback: describe the attachment in the message text instead. The typed path "
            "needs no vision model or ffmpeg."
        )
    return message


# Vision models emit medium-talk regularly and an 8B writer echoes it, so it's removed deterministically.
_META_SENTENCE = re.compile(
    r"\b(resolution|low[- ]quality|blurry|pixelated|two[- ]dimensional|flat image|"
    r"no discernible|cannot (?:be )?determine|difficult to (?:describe|discern|tell)|"
    r"appears to be an image|it is unclear|not visible in the (?:image|photo|frame))\b",
    re.IGNORECASE,
)

_MEDIUM = r"(?:image|photo|picture|frame|video|clip|scene)"
_PROVIDED = r"(?:\s+you(?:'ve|\s+have)?\s+provided)?"
# Lead-ins need a verb or "In the …," so noun phrases like "The frame of the bike" survive.
_META_LEADIN = re.compile(
    rf"^\s*(?:in\s+(?:the|this)\s+{_MEDIUM}{_PROVIDED}\s*,?\s*"
    rf"|(?:the|this)\s+{_MEDIUM}{_PROVIDED}\s+(?:appears\s+to\s+|seems\s+to\s+)?"
    rf"(?:shows?|depicts?|features?|contains?|presents?|captures?|displays?|is\s+of)\s+"
    rf"|(?:the|this)\s+(?:image|photo|picture|video){_PROVIDED}\s+(?:appears\s+to\s+be|seems\s+to\s+be|is)\s+)",
    re.IGNORECASE,
)


def clean_caption(caption: str) -> str:
    """Strips medium-talk from the vision caption so the prompt describes a scene, not a file."""
    kept = []
    for raw in re.split(r"(?<=[.!?])\s+", caption or ""):
        sentence = raw.strip()
        if not sentence or _META_SENTENCE.search(sentence):
            continue
        sentence = _META_LEADIN.sub("", sentence, count=1)
        if sentence:
            kept.append(sentence[0].upper() + sentence[1:])
    return " ".join(kept).strip() or (caption or "").strip()


def tidy_output(text: str) -> str:
    """Strips code fences and a stray "Alignment Instruction:" label the writer sometimes adds."""
    out = (text or "").strip()
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z]*\n?", "", out)
        out = re.sub(r"\n?```$", "", out).strip()
    out = re.sub(r"^\s*alignment instruction\s*:\s*", "", out, count=1, flags=re.IGNORECASE)
    return out.strip()
