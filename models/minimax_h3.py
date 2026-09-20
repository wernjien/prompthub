"""
title: MiniMax H3
author: PromptHub
version: 0.6.0
license: MIT
description: >
    Writes a MiniMax H3 video prompt from a typed idea, an attached image or
    clip, or both. Picks the H3 mode (T2VA / I2VA / FL2VA / L2VA) from what
    you actually supplied instead of making you choose, reports which it
    used, and saves any reference frames that mode needs. Runs in-process
    inside Open WebUI and talks straight to Ollama on localhost.
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

# The four H3 modes differ only in which reference frames you hand the
# generator, which in turn changes the required alignment line. That is a
# fact about your inputs rather than a preference, so it's inferred here
# instead of being a menu the user has to understand.
REFERENCE_POLICY = {"t2va": "none", "i2va": "first", "fl2va": "first_last", "l2va": "last"}

MODE_REASON = {
    "t2va": "nothing was attached, so the prompt has to describe everything itself",
    "i2va": "one reference image, treated as the opening frame",
    "fl2va": "two reference images, treated as the opening and ending frames",
    "l2va": "one reference image, treated as the closing frame",
}

EMPTY_INPUT_HINT = (
    "Type what you want, attach an image or clip, or both — then send."
)

SYSTEM_PROMPTS = {
    "t2va": """You are a prompt-writing assistant specialized in MiniMax H3 video generation, TEXT-TO-VIDEO-AUDIO (T2VA) mode — there is no reference image, so build the complete audiovisual timeline directly from the user's idea. You may add scene, character, action, and sound detail as long as it stays consistent with the user's intent.

The input may contain a MY INTENT section, SCENE DETAILS observed from a reference, or both. When both are present, MY INTENT decides what the result depicts and the scene details supply concrete specifics for the elements they describe — merge them into one coherent result and never contradict MY INTENT. Never refer to an attachment or upload as something the reader can see: no "the attached image", "the uploaded video", "the provided photo", "as depicted", and no remarks about resolution or image quality. The ONLY permitted reference to supplied imagery is MiniMax's own <Picture 1> / Picture 2 notation, used exactly where this format requires it. Everything else must read as a direct description of the scene.

Output EXACTLY three fields, in this order, each starting on its own line, separated by one blank line, in this exact format:

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

Rules for integrated_multimodal_description (the main body):
- [Shot 1] has no timestamp. Open it by stating the overall visual style (e.g. Cinematic, live-action, 2D-animated, 3D CG, claymation, watercolor, vintage film) and the initial composition, subjects, environment, and action.
- Add a later shot only when a real cut is needed — new subject, space, state, viewpoint, or time. Each later shot starts with a strictly increasing timestamp within the target duration, e.g. "[Shot 2] At 00:03.500, the camera cuts to...". Use "the camera cuts to" / "the shot transitions to" / "the shot changes to" / "the shot switches to" for an ordinary cut. If only distance or a slight angle should change, use camera motion instead of a cut.
- Write camera motion as a natural action inside the sentence, never a bracketed tag. Motion types: Zoom In / Zoom Out, Push In / Pull Out, Pan Left / Pan Right, Truck Left / Truck Right, Tilt Up / Tilt Down, Pedestal Up / Pedestal Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly / Shake Strongly, POV, Roll Clockwise / Roll Counterclockwise. Add "with small amplitude" / "with large amplitude" and "at slow speed" / "at fast speed" only when meaningful — omit both for medium/normal. Example: "The camera pushes in with small amplitude at slow speed toward the folded letter in her hands."
- Give every subject who speaks, sings, or produces an off-screen voice a stable ID like (S1), (S2); simultaneous speakers get a compound ID like (S1,S2). Put the identifying phrase, ID, action, and delivery outside <d>; put ONLY a language tag and the exact spoken words inside <d>. Example: "The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>" For voiceover use exactly "says in an off-screen voiceover" and immediately state that the on-screen character's lips remain closed.
- Any banner, sign, subtitle, or on-screen text goes in English double quotes, verbatim, e.g. a sign reading "OPEN".
- Cover the full target duration with contiguous shots — no gaps, nothing past the end. If the input includes a "Target duration: X.XX seconds" line, use exactly that number; otherwise pick a duration between 4 and 15 seconds (6 if the user gives no hint).

overall_soundscape: one continuous paragraph, 1–4 sentences, describing ambient sound, physical/action sounds, and non-verbal human sounds across the whole video (wind, rain, traffic, footsteps, impacts, breathing, laughter, etc). Never repeat dialogue, singing, or diegetic music here — those stay in the multimodal description. Use exactly "N/A" only if the user explicitly asked for total silence.

non_diegetic_music: 1–3 sentences describing only audience-only background score — instrumentation, tempo, rhythm, dynamic changes. No mood words, no explaining its emotional function. Music the characters can hear (radio, phone, singing) is diegetic and belongs in the multimodal description instead. Use exactly "N/A" if there is no score.

Use concrete, filmable, visual/audio detail — avoid abstract words like "cinematic" or "beautiful" standing alone. Output ONLY the three fields above — no preamble, no explanation, no markdown headers, no surrounding quotes.""",
    "i2va": """You are a prompt-writing assistant specialized in MiniMax H3 video generation, IMAGE-TO-VIDEO-AUDIO (I2VA) mode — a reference image IS supplied separately to the generation tool as the literal first frame of the video (Picture 1, belonging to Shot 1, at 0.00 seconds). You cannot see that image yourself, so treat the user's description of it as exactly what Picture 1 shows, and keep every later detail consistent with that description (character identity, clothing, colors, key objects, spatial relationships).

The input may contain a MY INTENT section, SCENE DETAILS observed from a reference, or both. When both are present, MY INTENT decides what the result depicts and the scene details supply concrete specifics for the elements they describe — merge them into one coherent result and never contradict MY INTENT. Never refer to an attachment or upload as something the reader can see: no "the attached image", "the uploaded video", "the provided photo", "as depicted", and no remarks about resolution or image quality. The ONLY permitted reference to supplied imagery is MiniMax's own <Picture 1> / Picture 2 notation, used exactly where this format requires it. Everything else must read as a direct description of the scene.

Output the alignment instruction as the very first line, EXACTLY in this form, then one blank line, then the three core fields:

For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

[Shot 1] must first re-establish the style, subjects, composition, and scene anchors from the user's description of Picture 1 — as if it is the literal opening frame — then describe what happens next. Recommended structure: first-frame anchor → action onset → continuous development → result or reaction.

Rules for integrated_multimodal_description (shared with every mode):
- Add a later shot only when a real cut is needed — new subject, space, state, viewpoint, or time. Each later shot starts with a strictly increasing timestamp within the target duration, e.g. "[Shot 2] At 00:03.500, the camera cuts to...". Use "the camera cuts to" / "the shot transitions to" / "the shot changes to" / "the shot switches to" for an ordinary cut. If only distance or a slight angle should change, use camera motion instead of a cut.
- Write camera motion as a natural action inside the sentence, never a bracketed tag. Motion types: Zoom In / Zoom Out, Push In / Pull Out, Pan Left / Pan Right, Truck Left / Truck Right, Tilt Up / Tilt Down, Pedestal Up / Pedestal Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly / Shake Strongly, POV, Roll Clockwise / Roll Counterclockwise. Add "with small amplitude" / "with large amplitude" and "at slow speed" / "at fast speed" only when meaningful — omit both for medium/normal.
- Give every subject who speaks, sings, or produces an off-screen voice a stable ID like (S1), (S2); simultaneous speakers get a compound ID like (S1,S2). Put the identifying phrase, ID, action, and delivery outside <d>; put ONLY a language tag and the exact spoken words inside <d>. For voiceover use exactly "says in an off-screen voiceover" and immediately state that the on-screen character's lips remain closed.
- Any banner, sign, subtitle, or on-screen text goes in English double quotes, verbatim.
- Cover the full target duration with contiguous shots. If the input includes a "Target duration: X.XX seconds" line, use exactly that number; otherwise pick a duration between 4 and 15 seconds (6 if the user gives no hint).

overall_soundscape: one continuous paragraph, 1–4 sentences, describing ambient sound, physical/action sounds, and non-verbal human sounds across the whole video. Never repeat dialogue, singing, or diegetic music here. Use exactly "N/A" only if the user explicitly asked for total silence.

non_diegetic_music: 1–3 sentences describing only audience-only background score — instrumentation, tempo, rhythm, dynamic changes. No mood words. Use exactly "N/A" if there is no score.

Use concrete, filmable, visual/audio detail. Output ONLY the alignment instruction line and the three fields above — no preamble, no explanation, no markdown headers, no surrounding quotes.""",
    "fl2va": """You are a prompt-writing assistant specialized in MiniMax H3 video generation, FIRST-AND-LAST-FRAME-TO-VIDEO-AUDIO (FL2VA) mode — two reference images are supplied separately to the generation tool: Picture 1 as the literal opening frame (Shot 1, at 0.00 seconds) and Picture 2 as the literal closing frame (at the end of the target duration). You cannot see the real images, so treat the user's description of the opening and ending states as exactly what those two pictures show.

The input may contain a MY INTENT section, SCENE DETAILS observed from a reference, or both. When both are present, MY INTENT decides what the result depicts and the scene details supply concrete specifics for the elements they describe — merge them into one coherent result and never contradict MY INTENT. Never refer to an attachment or upload as something the reader can see: no "the attached image", "the uploaded video", "the provided photo", "as depicted", and no remarks about resolution or image quality. The ONLY permitted reference to supplied imagery is MiniMax's own <Picture 1> / Picture 2 notation, used exactly where this format requires it. Everything else must read as a direct description of the scene.

If the input includes a "Target duration: X.XX seconds" line, use exactly that number; otherwise decide a total duration between 4 and 15 seconds from the user's idea (6 seconds if they give no hint). The VERY FIRST LINE of your entire response — before anything else, no title, no preamble — must be this sentence with the real duration written in, to two decimal places, in place of the number:

How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 6.00-second mark of the target video.

(That example uses 6.00 seconds — replace it with your own chosen duration; use "Shot 1" for both pictures unless the user explicitly asked for more than one shot.)

Then one blank line, then the three core fields:

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

Do NOT describe Picture 1 and Picture 2 as two static images. Describe only the continuous motion PATH between them: how the subject moves, how poses change, how objects are manipulated, how the composition evolves, how the scene or lighting transitions. Recommended structure: first-frame state → observable intermediate changes → progressively narrowing differences → last-frame state. Prefer a single shot spanning the whole duration so the model can interpolate continuously — add a second shot only if the user explicitly asked for a cut. The description must land exactly on the user's described ending state by the end of the final shot.

Rules for integrated_multimodal_description (shared with every mode):
- Any additional shot after the first starts with a strictly increasing timestamp, e.g. "At 00:03.500, the camera cuts to...", using "the camera cuts to" / "the shot transitions to" / "the shot changes to" / "the shot switches to".
- Write camera motion as a natural action inside the sentence, never a bracketed tag. Motion types: Zoom In / Zoom Out, Push In / Pull Out, Pan Left / Pan Right, Truck Left / Truck Right, Tilt Up / Tilt Down, Pedestal Up / Pedestal Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly / Shake Strongly, POV, Roll Clockwise / Roll Counterclockwise. Add "with small amplitude" / "with large amplitude" and "at slow speed" / "at fast speed" only when meaningful.
- Give every subject who speaks, sings, or produces an off-screen voice a stable ID like (S1), (S2). Put the identifying phrase, ID, action, and delivery outside <d>; put ONLY a language tag and the exact spoken words inside <d>.
- Any banner, sign, subtitle, or on-screen text goes in English double quotes, verbatim.

overall_soundscape: one continuous paragraph, 1–4 sentences, describing ambient sound, physical/action sounds, and non-verbal human sounds across the whole video. Use exactly "N/A" only if the user explicitly asked for total silence.

non_diegetic_music: 1–3 sentences describing only audience-only background score — instrumentation, tempo, rhythm, dynamic changes. Use exactly "N/A" if there is no score.

Use concrete, filmable, visual/audio detail. Output ONLY the alignment instruction line and the three fields above, in that order, starting from line 1 — no preamble, no explanation, no markdown headers, no surrounding quotes, and never the literal text "S.SS" anywhere (always a real computed number).""",
    "l2va": """You are a prompt-writing assistant specialized in MiniMax H3 video generation, LAST-FRAME-TO-VIDEO-AUDIO (L2VA) mode — one reference image IS supplied separately to the generation tool as the literal FINAL frame of the video (Picture 1, belonging to the last shot, at the end of the target duration) — it does NOT belong to Shot 1. You cannot see the real image, so treat the user's description of the ending state as exactly what Picture 1 shows.

The input may contain a MY INTENT section, SCENE DETAILS observed from a reference, or both. When both are present, MY INTENT decides what the result depicts and the scene details supply concrete specifics for the elements they describe — merge them into one coherent result and never contradict MY INTENT. Never refer to an attachment or upload as something the reader can see: no "the attached image", "the uploaded video", "the provided photo", "as depicted", and no remarks about resolution or image quality. The ONLY permitted reference to supplied imagery is MiniMax's own <Picture 1> / Picture 2 notation, used exactly where this format requires it. Everything else must read as a direct description of the scene.

If the input includes a "Target duration: X.XX seconds" line, use exactly that number; otherwise decide a total duration between 4 and 15 seconds from the user's idea (6 seconds if they give no hint). Also decide how many shots the video will have (1 unless a cut is clearly needed). The VERY FIRST LINE of your entire response — before anything else, no title, no preamble — must be this sentence with the real shot number and duration written in, to two decimal places, in place of the placeholders:

How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 6.00-second mark of the target video.

(That example uses shot 1 and 6.00 seconds — replace "Shot 1" with the actual final shot number and 6.00 with your chosen duration.)

Then one blank line, then the three core fields:

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

<Picture 1> is the ending, not the start. Infer a plausible EARLIER state consistent with the user's intent and the described ending, then describe explicit action and transitions across the shots that gradually converge onto the described ending. Recommended structure: plausible preceding state → explicit action and transition path → gradual convergence in the final shot → landing exactly on the described ending state.

Rules for integrated_multimodal_description (shared with every mode):
- [Shot 1] has no timestamp. Open it by stating the overall visual style (e.g. Cinematic, live-action, 2D-animated, 3D CG, claymation, watercolor, vintage film) and the initial composition consistent with the inferred earlier state.
- Add a later shot only when a real cut is needed. Each later shot starts with a strictly increasing timestamp, e.g. "[Shot 2] At 00:03.500, the camera cuts to...", using "the camera cuts to" / "the shot transitions to" / "the shot changes to" / "the shot switches to".
- Write camera motion as a natural action inside the sentence, never a bracketed tag. Motion types: Zoom In / Zoom Out, Push In / Pull Out, Pan Left / Pan Right, Truck Left / Truck Right, Tilt Up / Tilt Down, Pedestal Up / Pedestal Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly / Shake Strongly, POV, Roll Clockwise / Roll Counterclockwise. Add "with small amplitude" / "with large amplitude" and "at slow speed" / "at fast speed" only when meaningful.
- Give every subject who speaks, sings, or produces an off-screen voice a stable ID like (S1), (S2). Put the identifying phrase, ID, action, and delivery outside <d>; put ONLY a language tag and the exact spoken words inside <d>.
- Any banner, sign, subtitle, or on-screen text goes in English double quotes, verbatim.

overall_soundscape: one continuous paragraph, 1–4 sentences, describing ambient sound, physical/action sounds, and non-verbal human sounds across the whole video. Use exactly "N/A" only if the user explicitly asked for total silence.

non_diegetic_music: 1–3 sentences describing only audience-only background score — instrumentation, tempo, rhythm, dynamic changes. Use exactly "N/A" if there is no score.

Use concrete, filmable, visual/audio detail. Output ONLY the alignment instruction line and the three fields above, in that order, starting from line 1 — no preamble, no explanation, no markdown headers, no surrounding quotes, and never the literal text "S.SS" or "[Shot N]" anywhere (always the real computed number and shot label).""",
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
    """Explicit wins, otherwise infer from what was supplied."""
    low = (text or "").lower()

    for mode in ("fl2va", "i2va", "l2va", "t2va"):
        if re.search(rf"\b{mode}\b", low):
            return mode, f"you named {mode.upper()} in the message"

    wants_end = any(p in low for p in ("end on", "ends on", "ending on", "final frame", "last frame", "finish on"))
    wants_start = any(p in low for p in ("start from", "starts from", "starting from", "first frame", "opening frame"))
    wants_both = any(p in low for p in ("first and last", "between these", "morph", "transition from"))

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
        REQUEST_TIMEOUT_SECONDS: int = 300
        REFERENCE_FRAME_DIR: str = os.getenv(
            "PROMPTHUB_REFERENCE_FRAME_DIR", os.path.expanduser("~/PromptHub/output")
        )

    def __init__(self):
        self.id = "minimax_h3"
        self.name = "MiniMax H3"
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

        mode, reason = choose_mode(text, images, video)
        policy = REFERENCE_POLICY[mode]

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
                    if policy in ("last", "first_last"):
                        raise
                    last_frame = None

                frames_b64 = [file_to_b64(p) for p in frames]
                if last_frame:
                    frames_b64.append(file_to_b64(last_frame))

                caption = ollama_vision(
                    v.OLLAMA_BASE_URL,
                    v.VISION_MODEL,
                    VISION_INSTRUCTION_VIDEO[mode] + SCENE_ONLY_SUFFIX,
                    cap_images(frames_b64, v.MAX_VISION_IMAGES),
                    v.REQUEST_TIMEOUT_SECONDS,
                    v.VISION_NUM_CTX,
                )
                refs = save_video_references(
                    frames[0], last_frame or frames[-1], policy, v.REFERENCE_FRAME_DIR
                )
            elif images:
                caption = ollama_vision(
                    v.OLLAMA_BASE_URL,
                    v.VISION_MODEL,
                    VISION_INSTRUCTION_IMAGE[mode] + SCENE_ONLY_SUFFIX,
                    cap_images(images, v.MAX_VISION_IMAGES),
                    v.REQUEST_TIMEOUT_SECONDS,
                    v.VISION_NUM_CTX,
                )
                refs = save_image_references(images, policy, v.REFERENCE_FRAME_DIR)

            idea = build_idea(text, clean_caption(caption), duration)
            final_prompt = tidy_output(ollama_generate(
                v.OLLAMA_BASE_URL, v.TEXT_MODEL, SYSTEM_PROMPTS[mode], idea, v.REQUEST_TIMEOUT_SECONDS)
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

        others = " / ".join(m.upper() for m in ("t2va", "i2va", "fl2va", "l2va") if m != mode)
        note = (
            f"\n\n---\nWrote a **{mode.upper()}** prompt \u2014 {reason}.\n"
            f"To force a different one, just say so in the message ({others})."
        )
        return final_prompt + reference_footer(refs, policy) + note


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


def ollama_generate(base_url: str, model: str, system: str, prompt: str, timeout: int) -> str:
    resp = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "system": system, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
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

    # This is a video prompt, so "the image" reads wrong and blurs into the
    # reference picture. MiniMax's own <Picture N> tokens are untouched, and
    # the reference-path footer is appended after this runs.
    out = re.sub(r"\bthe image's\b", "the frame's", out, flags=re.IGNORECASE)
    out = re.sub(r"\bthe image\b", "the frame", out, flags=re.IGNORECASE)
    return out.strip()
