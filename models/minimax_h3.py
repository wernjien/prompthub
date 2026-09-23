"""
title: MiniMax H3
author: PromptHub
version: 0.7.0
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

# The four modes share one format and differ only in their header, the
# alignment line, how the timeline relates to the reference frames, and the
# worked example. Assembling them from shared parts keeps the common rules
# from drifting apart between modes. Format follows MiniMax's official
# guide (MiniMax-AI/MiniMax-H3, skills/h3-prompt-writing/references/base-en.txt).
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
        # The system prompt plus a long caption can outgrow Ollama's default
        # context, which truncates silently instead of failing.
        TEXT_NUM_CTX: int = int(os.getenv("PROMPTHUB_TEXT_NUM_CTX", "8192"))
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
                v.OLLAMA_BASE_URL, v.TEXT_MODEL, SYSTEM_PROMPTS[mode], idea, v.REQUEST_TIMEOUT_SECONDS,
                v.TEXT_NUM_CTX)
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

    # This is a video prompt, so "the image" reads wrong and blurs into the
    # reference picture. MiniMax's own <Picture N> tokens are untouched, and
    # the reference-path footer is appended after this runs.
    out = re.sub(r"\bthe image's\b", "the frame's", out, flags=re.IGNORECASE)
    out = re.sub(r"\bthe image\b", "the frame", out, flags=re.IGNORECASE)
    return out.strip()
