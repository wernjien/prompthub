"""
title: Krea2
author: PromptHub
version: 0.6.0
license: MIT
description: >
    Writes a Krea 2 image prompt from a typed scene, an attached image, or both together.
    Accepts a typed message, an attachment, or both. Runs in-process inside
    Open WebUI and talks straight to Ollama on localhost — deliberately NOT
    via an Open WebUI model connection, so no raw Ollama model has to be
    exposed in the model picker.
requirements: requests
"""

import base64
import glob
import inspect
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple

import requests
from pydantic import BaseModel

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg"}

SYSTEM_PROMPT = """You are a prompt writer for Krea 2, an image generation model. Your only job is to turn the input into one high-quality image prompt. Krea 2 reads natural descriptive language: never use comma-separated tags, (word:1.3) weighting syntax, or a separate negative-prompt section.

INPUT
The input contains a MY INTENT section, SCENE DETAILS observed from a reference, or both. MY INTENT decides what the image depicts; the scene details supply concrete specifics (subject, setting, lighting, angle, palette, style) for the elements they describe. Merge them into one coherent scene and never contradict MY INTENT: for "same scene but at night", keep the described subject and setting and relight them for night. If only scene details are given, write the prompt that recreates that scene.

The prompt is pasted into a generator that cannot see any reference, so it must stand on its own. Never refer to a reference, image, picture, attachment or upload ("the image shows", "as depicted", "in the provided photo"), and never comment on resolution, blur or image quality. Describe the scene itself. Naming the medium as a style ("a street photograph", "an oil painting") is fine.

OUTPUT RULES
- Output ONLY the final prompt: no preamble, explanation, headings, markdown or surrounding quotes.
- One flowing paragraph of natural descriptive sentences.
- 50-120 words. Go shorter for simple ideas; never pad.
- Write in English, even if the input is in another language.

STRUCTURE (in this order)
1. Subject: who or what, with defining physical details. Always first.
2. Action or pose: what the subject is doing, expression, body language.
3. Setting: location, environment, time of day, background elements.
4. Lighting: always name the source, direction and quality (e.g. "low golden-hour sun raking from the left", "single tungsten lamp casting deep shadows").
5. Camera: shot type, angle, lens and depth of field (e.g. "close-up at eye level, 85mm lens, shallow depth of field"). For non-photographic styles, use composition and framing terms instead.
6. Style and medium: photograph, film stock, oil painting, 3D render, anime, etc.
7. Color palette and mood: concrete colors and the emotional tone.

QUALITY PRINCIPLES
- Be specific about materials and textures (brushed steel, linen with visible weave, skin with pores and freckles); this is where realism comes from.
- For photorealistic requests, aim for an authentic, non-AI look: candid framing, natural skin texture, real-world imperfections, believable backgrounds.
- Tie every attribute to its object so details don't bleed together: "a woman in a red coat holding a blue umbrella", not "woman, red, blue, coat, umbrella".
- Keep the scene focused on 3-5 key elements; if the input is overloaded, keep the ones that matter most to MY INTENT.
- Keep every detail consistent: never pair night with bright sunlight, or minimalist with a crowded scene.

NEVER
- Quality tags or filler: "masterpiece", "best quality", "8k", "ultra HD", "highly detailed", "trending on artstation", "award-winning".
- Negative phrasing ("no people", "without text"). Describe what IS there instead ("an empty street", "a plain unmarked wall").
- Text or lettering in the scene unless MY INTENT asks for it or the scene details include it.

TEXT IN THE SCENE
- When there is text, put the exact words in double quotes and say where and how they appear, e.g. a hand-painted sign reading "OPEN LATE" above the door.

RESPECTING THE USER
- Keep everything MY INTENT specifies (style, colors, composition, subject details); only fill in what it leaves open.
- If a style is named (anime, watercolor, pixel art, etc.), commit to it fully and use that medium's vocabulary instead of camera terms.
- If the idea is vague, make confident, tasteful creative choices; never ask questions.
- If the user asks for variations, output that many prompts as separate paragraphs divided by a blank line, varying lighting, angle or setting while keeping the core subject.

EXAMPLE 1
Input: MY INTENT: old woman at a market
Output: A candid street photograph of an elderly woman laughing at a fruit stall in a Lisbon market, her silver hair tied back and a knitted cardigan over her shoulders. Late afternoon sun filters through a striped canvas awning, casting warm dappled light across her face and the crates of oranges beside her. Shot at eye level with a 35mm lens and shallow depth of field, natural skin texture, subtle film grain, muted warm palette with soft oranges and faded blues.

EXAMPLE 2
Input: MY INTENT: same scene but at night, in the rain
SCENE DETAILS: A young man in a yellow raincoat rides a bicycle along a canal lined with brick houses. Bright midday sun, clear blue sky, wide shot from street level, realistic style.
Output: A young man in a yellow raincoat pedals a bicycle along a narrow canal lined with old brick houses, his hood up and shoulders hunched against the weather. Steady night rain streaks through the glow of iron streetlamps, their warm light rippling across wet cobblestones and the black water beside him. Wide shot from street level with a 28mm lens and deep depth of field, a realistic photograph with glistening reflections and rain-beaded fabric, palette of amber, deep navy and saturated yellow, quiet and melancholy."""

VISION_INSTRUCTION_IMAGE = "Describe this image in detail: main subject and action, setting/background, materials and textures, lighting direction and quality, apparent camera angle or lens characteristics, color palette, overall style, and any visible text quoted exactly."

VISION_INSTRUCTION_VIDEO = "These frames are sampled in order across a short clip. Describe the scene in detail: main subject and action, setting/background, lighting, camera angle or lens characteristics, color palette, and overall style."

# Which reference frames this mode hands to the generator, if any:
# "none" | "first" | "last" | "first_last".
REFERENCE_POLICY = "none"

# Whether the prompt format needs the clip's real duration.
NEEDS_DURATION = False

EMPTY_INPUT_HINT = "Type the scene you want, attach an image, or both — then send."

# Appended to every vision instruction. The caption is the real source of
# "the image shows…" leakage: whatever meta-commentary the vision model
# writes gets echoed by the writer into a prompt that will be pasted
# somewhere with no reference attached.
SCENE_ONLY_SUFFIX = (
    " Describe only what is present in the scene itself. Do not say that this is an image, "
    "photo, picture, frame, video or clip; do not comment on resolution, sharpness or quality; "
    "and do not state what you cannot tell. Write it as a description of a real scene."
)


class Pipe:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        VISION_MODEL: str = os.getenv("PROMPTHUB_VISION_MODEL", "llava:13b")
        TEXT_MODEL: str = os.getenv("PROMPTHUB_TEXT_MODEL", "dolphin3:8b")
        FFMPEG_BINARY: str = os.getenv("PROMPTHUB_FFMPEG_BINARY", "ffmpeg")
        FFPROBE_BINARY: str = os.getenv("PROMPTHUB_FFPROBE_BINARY", "ffprobe")
        # 4 sampled frames + the final frame = 5 images, which is what fits.
        # llava:13b's vision context is 4096 tokens and each image costs
        # ~576, so 9 images (the 8 the original spec asked for, plus the last
        # frame) hard-fails with exceed_context_size_error. Raising num_ctx
        # does NOT lift that image budget — only sending fewer images does.
        FRAME_COUNT: int = int(os.getenv("PROMPTHUB_FRAME_COUNT", "4"))
        MAX_VISION_IMAGES: int = int(os.getenv("PROMPTHUB_MAX_VISION_IMAGES", "5"))
        # Does improve the vision model's reasoning over the frames it can
        # see, even though it doesn't raise the image budget.
        VISION_NUM_CTX: int = int(os.getenv("PROMPTHUB_VISION_NUM_CTX", "8192"))
        # The system prompt plus a long caption can outgrow Ollama's default
        # context, which truncates silently instead of failing.
        TEXT_NUM_CTX: int = int(os.getenv("PROMPTHUB_TEXT_NUM_CTX", "8192"))
        REQUEST_TIMEOUT_SECONDS: int = 300
        REFERENCE_FRAME_DIR: str = os.getenv(
            "PROMPTHUB_REFERENCE_FRAME_DIR", os.path.expanduser("~/PromptHub/output")
        )

    def __init__(self):
        self.id = "krea2"
        self.name = "Krea2"
        self.valves = self.Valves()

    async def pipe(self, body: dict, __files__: Optional[list] = None) -> str:
        v = self.valves
        text = extract_user_text(body)
        images = extract_images_b64(body)
        video, file_images, unresolved = await resolve_attachments(__files__)
        images = images + file_images

        if not text and not images and not video:
            return "\u26a0\ufe0f " + EMPTY_INPUT_HINT

        # Never quietly ignore an attachment: writing a prompt from the text
        # alone would look like it worked while silently dropping the file.
        if unresolved and not video and not images:
            return (
                "\u26a0\ufe0f You attached a file, but it couldn't be read "
                f"({unresolved}). Nothing was generated, because answering from "
                "your text alone would have silently ignored the attachment.\n\n"
                "If it's an unusual format, try re-encoding to mp4 — or describe "
                "it in the message text instead, which needs no attachment at all."
            )

        workdir = tempfile.mkdtemp(prefix="prompthub_")
        try:
            caption = ""
            duration = None
            refs: List[Tuple[str, str]] = []

            if video:
                duration = probe_duration(video, v.FFPROBE_BINARY)
                frames = extract_frames(video, workdir, duration, v.FRAME_COUNT, v.FFMPEG_BINARY)

                # Best-effort: only the keyframe modes actually need the
                # final frame, so a failure here shouldn't sink a T2VA run.
                last_frame = os.path.join(workdir, "frame_last.jpg")
                try:
                    extract_last_frame(video, last_frame, v.FFMPEG_BINARY)
                except subprocess.CalledProcessError:
                    if REFERENCE_POLICY in ("last", "first_last"):
                        raise
                    last_frame = None

                frames_b64 = [file_to_b64(p) for p in frames]
                if last_frame:
                    frames_b64.append(file_to_b64(last_frame))

                caption = ollama_vision(
                    v.OLLAMA_BASE_URL,
                    v.VISION_MODEL,
                    VISION_INSTRUCTION_VIDEO + SCENE_ONLY_SUFFIX,
                    cap_images(frames_b64, v.MAX_VISION_IMAGES),
                    v.REQUEST_TIMEOUT_SECONDS,
                    v.VISION_NUM_CTX,
                )
                refs = save_video_references(
                    frames[0], last_frame or frames[-1], REFERENCE_POLICY, v.REFERENCE_FRAME_DIR
                )
            elif images:
                caption = ollama_vision(
                    v.OLLAMA_BASE_URL,
                    v.VISION_MODEL,
                    VISION_INSTRUCTION_IMAGE + SCENE_ONLY_SUFFIX,
                    cap_images(images, v.MAX_VISION_IMAGES),
                    v.REQUEST_TIMEOUT_SECONDS,
                    v.VISION_NUM_CTX,
                )
                refs = save_image_references(images, REFERENCE_POLICY, v.REFERENCE_FRAME_DIR)

            idea = build_idea(text, clean_caption(caption), duration if NEEDS_DURATION else None)
            final_prompt = tidy_output(ollama_generate(
                v.OLLAMA_BASE_URL, v.TEXT_MODEL, SYSTEM_PROMPT, idea, v.REQUEST_TIMEOUT_SECONDS, v.TEXT_NUM_CTX)
            )
        except (
            requests.RequestException,
            subprocess.CalledProcessError,
            RuntimeError,
            ValueError,
        ) as exc:
            return (
                f"\u26a0\ufe0f Could not build the prompt: {exc}\n\nManual fallback: describe the "
                "image/video yourself in the message text instead — the typed path needs no "
                "vision model or ffmpeg."
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        return final_prompt + reference_footer(refs, REFERENCE_POLICY)


# --- helpers --------------------------------------------------------
# Duplicated (not imported) across the function files on purpose: Open WebUI
# stores each Function's code independently and can't import siblings, so
# each file has to stand alone. Keep this block in sync across all five.


def extract_user_text(body: dict) -> str:
    messages = body.get("messages", [])
    if not messages:
        return ""
    content = messages[-1].get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return " ".join(p for p in parts if p).strip()
    return ""


def extract_images_b64(body: dict) -> List[str]:
    """Open WebUI embeds uploaded images inline in the last user message as
    OpenAI-style content blocks — they never arrive via __files__."""
    messages = body.get("messages", [])
    if not messages:
        return []
    content = messages[-1].get("content")
    if not isinstance(content, list):
        return []

    out = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "image_url":
            continue
        url = block.get("image_url", {}).get("url", "")
        if url.startswith("data:"):
            out.append(url.split(",", 1)[1])
        elif url:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            out.append(base64.b64encode(resp.content).decode())
    return out


async def resolve_attachments(files: Optional[list]) -> Tuple[Optional[str], List[str], str]:
    """Turns __files__ into (video_path, image_b64_list, unresolved_reason).

    __files__ entries are not a fixed shape: the browser sends a fat record,
    the REST API may send as little as {"type": "file", "id": "..."} with no
    path at all. So resolve the id through Open WebUI's own Files model —
    we're in-process, so that's available — and fall back to the uploads
    directory layout. Note Files.get_file_by_id is a coroutine in current
    versions; un-awaited it yields an object with no usable attributes,
    which silently looks like "no file attached".
    """
    if not files:
        return None, [], ""

    video_path = None
    image_b64: List[str] = []
    problems = []

    for f in files:
        if not isinstance(f, dict):
            continue
        inner = f.get("file") if isinstance(f.get("file"), dict) else f
        file_id = f.get("id") or inner.get("id")
        path = inner.get("path")
        meta = inner.get("meta") if isinstance(inner.get("meta"), dict) else {}
        content_type = (meta or {}).get("content_type") or ""

        if not path and file_id:
            path, resolved_type = await _resolve_file_by_id(file_id)
            content_type = content_type or resolved_type

        name = inner.get("filename") or (meta or {}).get("name") or file_id or "attachment"
        if not path:
            problems.append(f"{name}: could not locate it on disk")
            continue
        if not os.path.exists(path):
            problems.append(f"{name}: stored path no longer exists")
            continue

        if content_type.startswith("image/"):
            image_b64.append(file_to_b64(path))
        elif content_type.startswith("video/") or os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
            video_path = video_path or path
        else:
            problems.append(f"{name}: unsupported type ({content_type or 'unknown'})")

    return video_path, image_b64, "; ".join(problems)


async def _resolve_file_by_id(file_id: str) -> Tuple[Optional[str], str]:
    try:
        from open_webui.models.files import Files

        record = Files.get_file_by_id(file_id)
        if inspect.isawaitable(record):
            record = await record
        if record is not None:
            meta = getattr(record, "meta", None) or {}
            content_type = meta.get("content_type") or "" if isinstance(meta, dict) else ""
            return getattr(record, "path", None), content_type
    except Exception:
        pass

    # Layout fallback: <UPLOAD_DIR>/<file id>_<original filename>
    try:
        from open_webui.config import UPLOAD_DIR

        hits = glob.glob(os.path.join(str(UPLOAD_DIR), f"{file_id}_*"))
        if hits:
            return hits[0], ""
    except Exception:
        pass

    return None, ""


def build_idea(text: str, caption: str, duration: Optional[float]) -> str:
    """Frames the vision caption as facts about the scene rather than as
    "an attached image", so the written prompt doesn't inherit references
    to something the downstream generator can't see."""
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


def probe_duration(video_path: str, ffprobe_bin: str) -> float:
    result = subprocess.run(
        [ffprobe_bin, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video_path],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise RuntimeError("could not determine video duration")
    return duration


def extract_frame_at(video_path: str, out_path: str, timestamp: float, ffmpeg_bin: str) -> None:
    subprocess.run(
        [ffmpeg_bin, "-y", "-ss", f"{max(timestamp, 0):.3f}", "-i", video_path, "-frames:v", "1", "-q:v", "2", out_path],
        check=True,
        capture_output=True,
    )


def extract_frames(video_path: str, out_dir: str, duration: float, frame_count: int, ffmpeg_bin: str) -> List[str]:
    paths = []
    for i in range(frame_count):
        out_path = os.path.join(out_dir, f"frame_{i:02d}.jpg")
        extract_frame_at(video_path, out_path, duration * i / frame_count, ffmpeg_bin)
        paths.append(out_path)
    return paths


def extract_last_frame(video_path: str, out_path: str, ffmpeg_bin: str) -> None:
    """Grabs the true final frame.

    Seeking to `duration - epsilon` is unreliable: on a short or low-fps clip
    that timestamp can land past the last frame and ffmpeg exits non-zero
    with no output (a 3.00s @ 15fps clip has its last frame at 2.933s, so
    even -0.05 overshoots). Seeking relative to the end and letting
    `-update 1` overwrite each decoded frame leaves the final one in place,
    and clamps harmlessly to the start on clips shorter than the window.
    """
    subprocess.run(
        [ffmpeg_bin, "-y", "-sseof", "-3", "-i", video_path, "-update", "1", "-q:v", "2", out_path],
        check=True,
        capture_output=True,
    )


def save_video_references(first_path: str, last_path: str, policy: str, ref_dir: str):
    if policy == "none":
        return []
    if policy == "first":
        return [("Reference frame (Picture 1, t=0)", _copy_ref(first_path, ref_dir, "picture1"))]
    if policy == "last":
        return [("Reference frame (Picture 1, final frame)", _copy_ref(last_path, ref_dir, "picture1"))]
    return [
        ("Reference frame (Picture 1, t=0)", _copy_ref(first_path, ref_dir, "picture1")),
        ("Reference frame (Picture 2, final frame)", _copy_ref(last_path, ref_dir, "picture2")),
    ]


def save_image_references(images_b64: List[str], policy: str, ref_dir: str):
    if policy == "none" or not images_b64:
        return []
    if policy in ("first", "last"):
        return [("Reference image (Picture 1)", _write_ref(images_b64[0], ref_dir, "picture1"))]
    refs = [("Reference image (Picture 1, opening)", _write_ref(images_b64[0], ref_dir, "picture1"))]
    if len(images_b64) > 1:
        refs.append(("Reference image (Picture 2, ending)", _write_ref(images_b64[1], ref_dir, "picture2")))
    return refs


def reference_footer(refs, policy: str) -> str:
    if policy == "none":
        return ""
    if not refs:
        needed = {
            "first": "a first-frame",
            "last": "a final-frame",
            "first_last": "an opening- and an ending-frame",
        }[policy]
        return (
            "\n\n\u2139\ufe0f No reference image was saved (nothing was attached), so supply "
            f"{needed} image to the generator yourself."
        )
    listed = "\n".join(f"{label} saved to: {path}" for label, path in refs)
    return "\n\n" + listed + "\nSupply these to your MiniMax generation alongside the prompt above."


def _copy_ref(frame_path: str, ref_dir: str, label: str) -> str:
    os.makedirs(ref_dir, exist_ok=True)
    dest = os.path.join(ref_dir, f"{label}_{int(time.time())}.jpg")
    shutil.copyfile(frame_path, dest)
    return dest


def _write_ref(image_b64: str, ref_dir: str, label: str) -> str:
    os.makedirs(ref_dir, exist_ok=True)
    dest = os.path.join(ref_dir, f"{label}_{int(time.time())}.jpg")
    with open(dest, "wb") as fh:
        fh.write(base64.b64decode(image_b64))
    return dest


def file_to_b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def cap_images(images: List[str], limit: int) -> List[str]:
    """Keeps the image count inside the vision model's budget.

    llava:13b allows ~4096 vision tokens at ~576 per image, so more than
    about five images fails outright with exceed_context_size_error. Rather
    than surface that as an error, subsample evenly and keep the endpoints,
    which are the frames that matter most for first/last references.
    """
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
    if num_ctx:
        payload["options"] = {"num_ctx": num_ctx}
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def ollama_generate(
    base_url: str, model: str, system: str, prompt: str, timeout: int, num_ctx: int = 0
) -> str:
    payload = {"model": model, "system": system, "prompt": prompt, "stream": False}
    if num_ctx:
        payload["options"] = {"num_ctx": num_ctx}
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"].strip()


# Sentences that are about the medium rather than the scene. The vision
# model emits these regularly and an 8B writer happily echoes them, so they
# are removed deterministically instead of being forbidden by instruction.
_META_SENTENCE = re.compile(
    r"\b(resolution|low[- ]quality|blurry|pixelated|two[- ]dimensional|flat image|"
    r"no discernible|cannot (?:be )?determine|difficult to (?:describe|discern|tell)|"
    r"appears to be an image|it is unclear|not visible in the (?:image|photo|frame))\b",
    re.IGNORECASE,
)

_META_LEADIN = re.compile(
    r"^\s*(?:in\s+)?(?:the|this)\s+(?:image|photo|picture|frame|video|clip|scene)\s*"
    r"(?:you(?:'ve|\s+have)?\s+provided\s*)?,?\s*"
    r"(?:appears\s+to\s+|seems\s+to\s+)?"
    r"(?:is\s+|shows|depicts|features|contains|presents|captures|displays)?\s*",
    re.IGNORECASE,
)


def clean_caption(caption: str) -> str:
    """Strips medium-talk out of the vision caption before it reaches the
    writer, so the finished prompt describes a scene rather than a file."""
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
    """Removes wrappers the writer occasionally adds around the prompt."""
    out = (text or "").strip()
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z]*\n?", "", out)
        out = re.sub(r"\n?```$", "", out).strip()
    # e.g. 'Alignment Instruction: For the target video, at 0.00 seconds…'
    out = re.sub(r"^\s*alignment instruction\s*:\s*", "", out, count=1, flags=re.IGNORECASE)
    return out.strip()
