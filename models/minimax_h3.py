"""
title: MiniMax H3
author: PromptHub
version: 0.8.0
license: MIT
description: Writes a MiniMax H3 video prompt from typed text, an attached image or clip, or both, inferring the mode (T2VA / I2VA / FL2VA / L2VA) and saving any reference frames it needs.
requirements: requests
"""

import asyncio
import base64
import glob
import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Callable, Iterator, List, Optional, Tuple

import requests
from pydantic import BaseModel

# Which reference frames each mode hands the generator.
REFERENCE_POLICY = {"t2va": "none", "i2va": "first", "fl2va": "first_last", "l2va": "last"}

MODE_REASON = {
    "t2va": "nothing was attached, so the prompt has to describe everything itself",
    "i2va": "one reference image, treated as the opening frame",
    "fl2va": "two reference images, treated as the opening and ending frames",
    "l2va": "one reference image, treated as the closing frame",
}

EMPTY_INPUT_HINT = "Type what you want, attach an image or clip, or both — then send."

H3_MIN_SECONDS, H3_MAX_SECONDS = 4.0, 15.0

_ENDING_WORDS = re.compile(r"\b(?:ends? on|ending on|finish(?:es)? on|final frame|last frame)\b")
_OPENING_WORDS = re.compile(r"\b(?:starts? from|starting from|first frame|opening frame)\b")
_BOTH_WORDS = re.compile(r"\b(?:first and last|between these|morph(?:s|ed|ing)?|transition from)\b")
# A length the user typed ("[6s]", "10 seconds", "5-second") takes priority over the attached clip's.
_STATED_DURATION = re.compile(
    r"\[\s*\d+(?:\.\d+)?\s*s\s*\]|\b\d+(?:\.\d+)?\s*-?\s*(?:secs?|seconds?)\b", re.IGNORECASE
)
_THE_IMAGE = re.compile(r"\b(the) image('s)?\b", re.IGNORECASE)

# Built from shared parts; format follows MiniMax-AI/MiniMax-H3 skills/h3-prompt-writing/references/base-en.txt.
_INPUT_RULES = """INPUT
The input may contain a MY INTENT section, SCENE DETAILS observed from a reference, or both. MY INTENT decides what the video depicts; the scene details supply concrete specifics for the elements they describe. Merge them into one coherent result and never contradict MY INTENT.

Your output is pasted into a video generator. Never refer to an attachment or upload as something the reader can see: no "the attached image", "the uploaded video", "the provided photo", "as depicted", and no remarks about resolution or image quality. The ONLY permitted reference to supplied imagery is MiniMax's own <Picture 1> / Picture 2 notation, used where this mode allows it. Everything else must read as a direct description of the scene.

DURATION
If the input includes a "Target duration: X.XX seconds" line, use exactly that number. Otherwise use a length the user states (a tag like [6s] or "a 10 second clip"); otherwise pick 4-15 seconds from the idea, 6 if there is no hint. Never ask the user anything, including the duration. Plan the clip as beats of about 2-3 seconds each (a 6-second clip has 2-3 beats, a 10-second clip 3-5) and cover the full duration, nothing past the end. Beats are only for your planning: never write the word "beat" or number the beats in the output."""

_BODY_RULES = """integrated_multimodal_description (the main body)
- Always begin [Shot 1] with the overall visual style, then the initial composition: "[Shot 1] Live-action, cinematic, ..." for anything realistic. Use another style (2D-animated, 3D CG, claymation, watercolor, vintage film) only when the user or the scene details call for it. [Shot 1] has no timestamp.
- Write the beats as clearly ordered actions within the shot, using plain time phrases where they help ("for the first two seconds", "then", "toward the end"). Only a real cut gets a timestamp.
- Add a later shot only when a real cut is needed: new subject, space, state, viewpoint, or time. It starts with a strictly increasing timestamp within the duration, e.g. "[Shot 2] At 00:03.500, the camera cuts to...". If only distance or a slight angle changes, use camera motion instead.
- One core action arc per clip; every beat advances it. Never cram unrelated actions into separate beats.
- Use strong, specific verbs (strides, flinches, drifts, slams, unfurls), never vague ones (moves, goes), and state speed and intensity (slowly, suddenly, gently).
- Add secondary motion for realism: hair and fabric in the wind, dust, water, reflections, background activity.
- Keep physics plausible. Avoid complex hand interactions and crowded multi-character choreography.
- Camera: at most two camera moves per shot, written as a natural action inside the sentence, never a bracketed tag. Motion types: Zoom In / Zoom Out, Push In / Pull Out, Pan Left / Pan Right, Truck Left / Truck Right, Tilt Up / Tilt Down, Pedestal Up / Pedestal Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly / Shake Strongly, POV, Roll Clockwise / Roll Counterclockwise. Add "with small amplitude" / "with large amplitude" and "at slow speed" / "at fast speed" only when meaningful. For a stable shot, say the camera holds a static shot.
- Speakers: every subject who speaks, sings, or voices off-screen gets a stable ID like (S1), (S2); simultaneous speakers get a compound ID like (S1,S2). Put the identifying phrase, ID, action, and delivery (tone, volume, emotion) outside <d>; put ONLY a language tag and the exact spoken words inside <d>, e.g. The old man with a hoarse whisper (S1) says: <d>[English] We're not alone here.</d> For voiceover use exactly "says in an off-screen voiceover" and immediately state that the on-screen character's lips remain closed.
- Dialogue budget: about 2.5 words per second of clip, with pauses; never fill the whole clip with speech. Place each line at the moment it happens. Keep the words in the language the user wants spoken.
- Any banner, sign, subtitle, or on-screen text goes in English double quotes, verbatim, e.g. a sign reading "OPEN".

overall_soundscape: one paragraph, 1-4 sentences: ambience, physical action sounds, and non-verbal human sounds (wind, rain, traffic, footsteps, impacts, breathing, laughter). Tie each sound to a visible action and its moment, and match it in intensity. Never describe sounds with no source in the scene. Never repeat dialogue, singing, or diegetic music here; those stay in the main body. Use exactly "N/A" only if the user explicitly asked for total silence.

non_diegetic_music: 1-3 sentences about audience-only background score: instrumentation, tempo, rhythm, and dynamic changes. No mood words and no explaining its emotional function. Music the characters can hear (radio, phone, singing) is diegetic and belongs in the main body. Use exactly "N/A" if the scene has only natural sound.

QUALITY
- Use concrete, filmable visual and audio detail; never abstract words like "cinematic" or "beautiful" standing alone, and never filler such as "masterpiece", "best quality", "8k", "ultra HD" or "highly detailed".
- Keep every detail consistent: never night with bright sunlight, or a static shot that also pans.
- Never use negative phrasing ("no shaking", "no music"). State the positive version ("the camera holds a static shot", non_diegetic_music: N/A).
- Keep everything the user explicitly specified and fill in only what they left open. If the idea is vague, make confident, tasteful choices. If the user gives no audio direction, choose fitting natural ambience and sound effects, and add dialogue only if the scene clearly calls for it.
- Write every field in English; dialogue, lyrics, and visible text stay in their original language."""


def _output_rule(first: str) -> str:
    return (
        f"OUTPUT\nOutput ONLY {first} — no preamble, no mode or duration label, no explanation, "
        "no markdown, no surrounding quotes. Keep the main body to about 60-150 words for clips "
        "up to 6 seconds and up to about 200 for longer clips; never pad. Output exactly one "
        "prompt. Only if the user explicitly asks for variations, output that many complete "
        "prompts separated by a line containing only \"---\", varying camera, timing, or sound "
        "while keeping the core idea."
    )


def _compose(*parts: str) -> str:
    return "\n\n".join(p.strip() for p in parts)


SYSTEM_PROMPTS = {
    "t2va": _compose(
        """You are a prompt writer for MiniMax H3, a video generation model with native audio, in TEXT-TO-VIDEO-AUDIO (T2VA) mode. There is no reference image: build the complete audiovisual timeline from the input, describing subject appearance, setting, lighting, camera, style, and palette yourself. You may add scene, character, action, and sound detail as long as it stays consistent with the user's intent. Never use <Picture> notation in this mode.""",
        _INPUT_RULES,
        """FORMAT
Output exactly three fields, each starting on its own line, separated by one blank line:

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

Structure the timeline as an opening state, the main action, and how the shot ends, with the subject and main action first.""",
        _BODY_RULES,
        _output_rule("the three fields"),
        """EXAMPLE
Input: MY INTENT: [6s] astronaut walking on mars
Output:
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot frames a lone astronaut in a scuffed white suit standing on a red desert plain under a hazy orange sky, long shadows stretching from the low sun across rippled dunes. He stands still for a moment, then takes a slow, heavy first step forward. The camera tracks alongside him at waist height as he settles into a steady stride, each boot kicking up fine rust-colored dust that drifts sideways in the thin wind. Toward the end, the camera pulls out with large amplitude at slow speed, shrinking him to a small figure against the vast, empty landscape.

overall_soundscape: Slow, steady breathing fills the helmet from the start, joined by the heavy crunch of boots on loose gravel once he begins walking. A thin, hollow wind sweeps across the plain throughout.

non_diegetic_music: A low synthesizer drone at a slow tempo enters softly during the pull-out and gradually swells in volume until the end.""",
    ),
    "i2va": _compose(
        """You are a prompt writer for MiniMax H3, a video generation model with native audio, in IMAGE-TO-VIDEO-AUDIO (I2VA) mode. A reference image is supplied separately to the generator as the literal first frame of the video (<Picture 1>, belonging to [Shot 1], at 0.00 seconds). You cannot see it: treat the scene details as exactly what <Picture 1> shows.""",
        _INPUT_RULES,
        """FORMAT
Output the alignment instruction as the very first line, exactly as written here, then one blank line, then the three fields, each separated by one blank line:

For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

DEVELOP FORWARD FROM THE FIRST FRAME
- The first frame already defines the look. Anchor [Shot 1] in one short sentence that names the style and refers to the subject briefly as shown in <Picture 1> ("the woman shown in <Picture 1> remains beside the window, preserving her appearance and the room's light"). Do not re-describe its appearance in detail.
- Spend the rest on what happens next, beat by beat: what moves, how the camera moves, what changes. Structure: first-frame anchor, action onset, continuous development, result or reaction.
- Never contradict the first frame (identity, clothing, colors, key objects, lighting, setting) unless MY INTENT asks for that change.""",
        _BODY_RULES,
        _output_rule("the alignment instruction line and the three fields"),
        """EXAMPLE
Input: MY INTENT: [6s] she hears something outside
SCENE DETAILS: A woman in a grey cardigan sits in an armchair beside a sunlit window with white curtains, soft morning light, live-action.
Output:
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the woman shown in <Picture 1> remains seated in the armchair beside the sunlit window, preserving her grey cardigan, pose, and the soft morning light. A gust lifts the white curtains inward and stirs loose strands of her hair as dust drifts through the sunbeam. She suddenly turns her head toward the window, her calm expression tightening into alertness. The camera pushes in with small amplitude at slow speed on her face as she leans slightly forward, and the woman with a low, uneasy voice (S1) whispers: <d>[English] Who's there?</d>

overall_soundscape: Soft wind and faint birdsong drift in through the glass as the curtains rustle. A distant dog barks twice outside just before she turns, and the armchair creaks as she leans forward.

non_diegetic_music: N/A""",
    ),
    "fl2va": _compose(
        """You are a prompt writer for MiniMax H3, a video generation model with native audio, in FIRST-AND-LAST-FRAME-TO-VIDEO-AUDIO (FL2VA) mode. Two reference images are supplied separately to the generator: Picture 1 is the literal opening frame (Shot 1, at 0.00 seconds) and Picture 2 is the literal closing frame (at the end of the duration). You cannot see them: treat the scene details as exactly what the opening and ending states show.""",
        _INPUT_RULES,
        """FORMAT
The very first line of your response must be this sentence, with your real duration written to two decimal places in place of 6.00:

How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 6.00-second mark of the target video.

Use "Shot 1" for both pictures unless the user explicitly asked for a cut. Then one blank line, then the three fields, each separated by one blank line:

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

DESCRIBE THE PATH BETWEEN THE FRAMES
- Do not describe Picture 1 and Picture 2 as two static images; refer to their elements briefly ("in the position and framing established by Picture 1"). Describe only the continuous path between them, beat by beat: how the subject moves, how poses change, how objects are handled, how the composition, camera, and lighting evolve.
- Structure: first-frame state, observable intermediate changes, progressively narrowing differences, last-frame state. The final beat lands exactly on the ending state ("settling into the pose and composition established by Picture 2 at the end of the shot").
- Prefer a single shot spanning the whole duration so the model can interpolate continuously; add a cut only if the user asked for one.
- If the two frames differ greatly, describe a believable bridging action or camera move rather than a sudden jump.""",
        _BODY_RULES,
        _output_rule("the alignment instruction line and the three fields, starting from line 1")
        + ' Never write the literal text "S.SS"; always a real number.',
        """EXAMPLE
Input: MY INTENT: [10s] he walks to the pier
SCENE DETAILS: Opening: a man in a long navy coat sits on a wooden bench by the sea. Ending: the same man stands at the edge of a pier, gazing at the horizon. Overcast daylight, live-action.
Output:
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 10.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the man in the long navy coat begins seated on the bench in the position and framing established by Picture 1, under flat overcast light. He rises slowly and pauses to look out at the grey water as the sea breeze stirs his coat, then walks steadily along the weathered boardwalk toward the pier while gulls glide overhead. The camera tracks behind him at shoulder height, slowing as he nears the end. He takes a final step to the edge and stands still, settling into the pose, spacing, and composition established by Picture 2 at the end of the shot.

overall_soundscape: Waves lap softly against the wooden pilings while gulls cry in the distance. Planks creak under his footsteps as he walks, fading once he stops at the edge, and the breeze rustles his coat throughout.

non_diegetic_music: N/A""",
    ),
    "l2va": _compose(
        """You are a prompt writer for MiniMax H3, a video generation model with native audio, in LAST-FRAME-TO-VIDEO-AUDIO (L2VA) mode. One reference image is supplied separately to the generator as the literal FINAL frame of the video (<Picture 1>, belonging to the last shot, at the end of the duration); it does NOT belong to Shot 1 unless the video is a single shot. You cannot see it: treat the scene details as exactly what the ending state shows.""",
        _INPUT_RULES,
        """FORMAT
Decide how many shots the video has (1 unless a cut is clearly needed). The very first line of your response must be this sentence, with the real final shot number in place of 1 and your real duration to two decimal places in place of 6.00:

How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 6.00-second mark of the target video.

Then one blank line, then the three fields, each separated by one blank line:

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

INFER THE OPENING, CONVERGE ON THE LAST FRAME
- The clip must END exactly on <Picture 1>. Infer a plausible earlier state for the first beat that differs from it (a different pose, position, distance, or framing) and stays consistent with its subject and setting and with MY INTENT.
- Describe the action that converges on the final composition beat by beat. Structure: plausible preceding state, explicit action and transition path, gradual convergence in the final shot, landing exactly on the ending ("...and settles into the exact pose, framing, and lighting established by <Picture 1>").""",
        _BODY_RULES,
        _output_rule("the alignment instruction line and the three fields, starting from line 1")
        + ' Never write the literal text "S.SS" or "[Shot N]"; always the real number and shot label.',
        """EXAMPLE
Input: MY INTENT: [6s] the final sprint
SCENE DETAILS: A female runner in a blue singlet bursts through a finish-line tape with arms raised on a red stadium track, bright afternoon sun, crowd in the stands behind, live-action.
Output:
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 6.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a low shot frames the runner in the blue singlet charging down the final straight of the red track under bright afternoon sun, face tense and arms pumping hard, a rival just behind her shoulder. The camera tracks backward in front of her as she finds a last burst of speed and pulls clear, sweat flying from her brow. She breaks through the tape and throws her arms into the air, and the runner with a breathless, triumphant voice (S1) shouts: <d>[English] Yes!</d> The motion settles into the exact pose, framing, and lighting established by <Picture 1>.

overall_soundscape: Rapid footsteps pound the track as her breathing grows ragged. The crowd's roar builds steadily and peaks in a thunderous cheer as she crosses the line.

non_diegetic_music: A driving percussion rhythm at a fast tempo that builds in volume and cuts out the instant she breaks the tape.""",
    ),
}

VISION_INSTRUCTION_IMAGE = {
    "t2va": """Describe this image in detail — visual style, subject appearance, setting, lighting, composition. It will be used to write a video prompt with no reference image, so be thorough about appearance.""",
    "i2va": """This image is the literal first frame of a video that hasn't been made yet. Describe it precisely — visual style, subject appearance, clothing, colours, key objects, setting, lighting, composition — so a prompt can develop forward from it consistently.""",
    "fl2va": """These images are the opening and ending frames of a video that hasn't been made yet (the first image is the opening, the last is the ending). Describe the motion path that would connect them — how the subject moves, how poses change, how the composition and lighting evolve — rather than describing each as a static image.""",
    "l2va": """This image is the literal FINAL frame of a video that hasn't been made yet. Describe it precisely — subject appearance, setting, lighting, composition, and the exact final state of everything in it — so a prompt can converge onto it.""",
}

VISION_INSTRUCTION_VIDEO = {
    "t2va": """These frames are sampled in order across a short video clip. Describe the full scene in detail — visual style, subject appearance, setting, lighting — and describe how the subject and/or camera move across the frames. This will be used to generate a new video with no reference image, so be thorough about appearance.""",
    "i2va": """These frames are sampled in order across a short video clip, starting at its very first frame. Describe the full scene in detail — visual style, subject appearance, setting, lighting — and how the subject and/or camera move across the frames. The first frame will be supplied separately as a fixed reference image, so describe it precisely and keep the rest consistent with it.""",
    "fl2va": """These frames are sampled in order across a short video clip, from its first frame to its last. Describe ONLY the continuous motion path between the first and last frame — how the subject moves, how poses change, how the composition and lighting evolve. The first and last frames will be supplied separately as fixed reference images, so focus on the transition between them.""",
    "l2va": """These frames are sampled in order across a short video clip, ending at its very last frame. Describe the full trajectory that leads up to that last frame — subject appearance, setting, lighting, and how the subject and/or camera move across the frames. The last frame will be supplied separately as a fixed reference image, so describe it precisely and make sure the path converges onto it.""",
}


def choose_mode(text: str, images: List[str], video: Optional[str]) -> Tuple[str, str]:
    """Picks the H3 mode: an explicit mention wins, otherwise it follows what was attached."""
    low = (text or "").lower()

    for mode in ("fl2va", "i2va", "l2va", "t2va"):
        if re.search(rf"\b{mode}\b", low):
            return mode, f"you named {mode.upper()} in the message"

    wants_end = bool(_ENDING_WORDS.search(low))
    wants_start = bool(_OPENING_WORDS.search(low))
    wants_both = bool(_BOTH_WORDS.search(low))

    if len(images) >= 2:
        return "fl2va", MODE_REASON["fl2va"]
    if len(images) == 1:
        if wants_end:
            return "l2va", "one reference image, and your wording points at the ending"
        return "i2va", MODE_REASON["i2va"]
    if video:
        if wants_both:
            return "fl2va", "a clip, and your wording asks for a first-to-last transition"
        if wants_end:
            return "l2va", "a clip, and your wording points at the ending"
        if wants_start:
            return "i2va", "a clip, and your wording points at the opening"
        return "t2va", "a clip used purely as reference material, so the prompt describes it standalone"
    return "t2va", MODE_REASON["t2va"]


def clip_target(text: str, clip_seconds: Optional[float]) -> Tuple[Optional[float], str]:
    """Returns the target duration to send (None leaves it to the writer) and a note if it was clamped."""
    if clip_seconds is None or _STATED_DURATION.search(text or ""):
        return None, ""
    target = min(max(clip_seconds, H3_MIN_SECONDS), H3_MAX_SECONDS)
    if target == clip_seconds:
        return target, ""
    return target, (
        f"The clip runs {clip_seconds:.1f}s, outside H3's {H3_MIN_SECONDS:g}-{H3_MAX_SECONDS:g}s range, "
        f"so the prompt targets {target:.2f}s."
    )


def frame_not_image(prompt: str) -> str:
    """Says "frame" rather than "image" so a video prompt doesn't read as describing the reference picture."""
    return _THE_IMAGE.sub(lambda m: f"{m.group(1)} frame{m.group(2) or ''}", prompt)


class Pipe:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        VISION_MODEL: str = os.getenv("PROMPTHUB_VISION_MODEL", "llava:13b")
        TEXT_MODEL: str = os.getenv("PROMPTHUB_TEXT_MODEL", "dolphin3:8b")
        FFMPEG_BINARY: str = os.getenv("PROMPTHUB_FFMPEG_BINARY", "ffmpeg")
        FFPROBE_BINARY: str = os.getenv("PROMPTHUB_FFPROBE_BINARY", "ffprobe")
        # llava:13b fails past ~5 images (~576 tokens each of a 4096 budget); the final frame is one of them.
        FRAME_COUNT: int = int(os.getenv("PROMPTHUB_FRAME_COUNT", "4"))
        MAX_VISION_IMAGES: int = int(os.getenv("PROMPTHUB_MAX_VISION_IMAGES", "5"))
        # Ollama clamps this to the model's own limit (4096 for llava:13b).
        VISION_NUM_CTX: int = int(os.getenv("PROMPTHUB_VISION_NUM_CTX", "8192"))
        # Ollama silently truncates prompts that overflow its default context.
        TEXT_NUM_CTX: int = int(os.getenv("PROMPTHUB_TEXT_NUM_CTX", "8192"))
        REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("PROMPTHUB_REQUEST_TIMEOUT_SECONDS", "300"))
        REFERENCE_FRAME_DIR: str = os.getenv(
            "PROMPTHUB_REFERENCE_FRAME_DIR", os.path.expanduser("~/PromptHub/output")
        )

    def __init__(self):
        self.id = "minimax_h3"
        self.name = "MiniMax H3"
        self.valves = self.Valves()

    async def pipe(self, body: dict, __files__: Optional[list] = None, __task__: Optional[str] = None):
        # Title/tag/follow-up jobs are routed here too; "" makes Open WebUI fall back without an LLM run.
        if __task__:
            return ""
        videos, image_paths, unresolved = await resolve_attachments(__files__)
        return stream_in_thread(self._run, body, videos, image_paths, unresolved)

    def _run(self, body: dict, videos: List[str], image_paths: List[str], unresolved: str) -> Iterator[str]:
        v = self.valves
        text = extract_user_text(body)
        images, inline_problem = extract_images_b64(body)
        problem = attachment_problem(
            len(videos), len(images) + len(image_paths), "; ".join(p for p in (unresolved, inline_problem) if p)
        )
        if problem:
            yield problem
            return
        if not text and not images and not image_paths and not videos:
            yield "⚠️ " + EMPTY_INPUT_HINT
            return

        video = videos[0] if videos else None
        workdir = tempfile.mkdtemp(prefix="prompthub_")
        streamed = False
        try:
            images += [file_to_b64(p) for p in image_paths]
            mode, reason = choose_mode(text, images, video)
            policy = REFERENCE_POLICY[mode]
            caption, clip_seconds, first_frame, last_frame = caption_attachments(
                v, video, images, VISION_INSTRUCTION_VIDEO[mode], VISION_INSTRUCTION_IMAGE[mode], workdir,
                need_last_frame=policy in ("last", "first_last"),
            )
            duration, duration_note = clip_target(text, clip_seconds)
            idea = build_idea(text, caption, duration)
            chunks = ollama_stream(
                v.OLLAMA_BASE_URL, v.TEXT_MODEL, SYSTEM_PROMPTS[mode], idea, v.REQUEST_TIMEOUT_SECONDS,
                v.TEXT_NUM_CTX,
            )
            for piece in stream_transformed(chunks, lambda t: frame_not_image(tidy_output(t))):
                streamed = True
                yield piece
            if video:
                refs = save_video_references(first_frame, last_frame or first_frame, policy, v.REFERENCE_FRAME_DIR)
            else:
                refs = save_image_references(images, policy, v.REFERENCE_FRAME_DIR)
        except PIPELINE_ERRORS as exc:
            yield ("\n\n" if streamed else "") + pipeline_error(exc, bool(video or images))
            return
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        others = " / ".join(m.upper() for m in ("t2va", "i2va", "fl2va", "l2va") if m != mode)
        note = f"\n\n---\nWrote a **{mode.upper()}** prompt — {reason}.\n"
        if duration_note:
            note += duration_note + "\n"
        note += f"To force a different one, just say so in the message ({others})."
        yield reference_footer(refs, policy) + note


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
