"""Rule checks for writer output, mirroring the rules stated in each Function's system prompt."""

import re
from typing import List, Optional

KREA_WORDS = (50, 120)
# The MiniMax length rules say "about", so the upper bound gets some slack.
MINIMAX_MIN_WORDS, MINIMAX_WORD_SLACK = 40, 1.2
MINIMAX_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
DIALOGUE_WORDS_PER_SECOND = 2.5

I2VA_ALIGNMENT = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
FL2VA_ALIGNMENT = re.compile(
    r"^How the reference pictures align with the target video — Picture 1 \(from Shot 1\) aligns with the "
    r"0\.00-second mark of the target video; Picture 2 \(from Shot (\d+)\) aligns with the "
    r"(\d+\.\d{2})-second mark of the target video\.$"
)
L2VA_ALIGNMENT = re.compile(
    r"^How the reference pictures align with the target video — <Picture 1> \(from \[Shot (\d+)\]\) "
    r"aligns with the (\d+\.\d{2})-second mark of the target video\.$"
)

FILLER = re.compile(
    r"\b(?:masterpiece|best quality|8k|4k|ultra[- ]?hd|highly detailed|trending on artstation|award[- ]winning)\b",
    re.IGNORECASE,
)
LEAK = re.compile(
    r"\b(?:attached|uploaded|provided|reference) (?:image|photo|picture|video|clip|frame)s?\b"
    r"|\bas depicted\b|\b(?:the|this) (?:image|photo|picture) (?:shows|depicts)\b|\bresolution\b",
    re.IGNORECASE,
)
NEGATIVE = re.compile(r"\b(?:no|without)\s+(?!longer\b|sooner\b|doubt\b|one\b)[a-z]+", re.IGNORECASE)
WEIGHTING = re.compile(r"\([^()]*:\s*\d+(?:\.\d+)?\)")
MARKDOWN = re.compile(r"\*\*|```|^\s*#|^\s*[-*]\s", re.MULTILINE)
PREAMBLE = re.compile(r"^\s*(?:here(?:'s| is)|sure\b|certainly\b|prompt\s*:|output\s*:)", re.IGNORECASE)
DIALOGUE = re.compile(r"<d>(.*?)</d>", re.DOTALL)
STATED_SECONDS = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*s\s*\]")


def stated_duration(text: str) -> Optional[float]:
    """Returns a duration written as a [6s] tag in the input, if any."""
    m = STATED_SECONDS.search(text or "")
    return float(m.group(1)) if m else None


def word_count(text: str) -> int:
    return len(re.findall(r"[\w'’-]+", text))


def sentence_count(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def _shared_problems(text: str) -> List[str]:
    """Checks the rules both prompts share, ignoring spoken dialogue."""
    prose = DIALOGUE.sub("", text)
    problems = []
    for label, pattern in (
        ("filler tag", FILLER), ("refers to the reference", LEAK), ("negative phrasing", NEGATIVE),
    ):
        hits = sorted({m.group(0) for m in pattern.finditer(prose)})
        if hits:
            problems.append(f"{label}: {', '.join(repr(h) for h in hits)}")
    if MARKDOWN.search(text):
        problems.append("contains markdown")
    if PREAMBLE.search(text):
        problems.append("starts with a preamble")
    return problems


def check_krea2(output: str, variations: int = 1) -> List[str]:
    """Returns the Krea 2 rules the output breaks; empty means it passed."""
    text = (output or "").strip()
    if not text:
        return ["empty output"]
    problems = _shared_problems(text)
    if text[0] in "\"“" and text[-1] in "\"”":
        problems.append("wrapped in quotes")
    if WEIGHTING.search(text):
        problems.append("uses (word:1.3) weighting syntax")
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) != variations:
        problems.append(f"{len(paragraphs)} paragraphs, expected {variations}")
    low, high = KREA_WORDS
    for i, paragraph in enumerate(paragraphs, 1):
        words = word_count(paragraph)
        if not low <= words <= high:
            problems.append(f"paragraph {i} has {words} words, expected {low}-{high}")
    return problems


def _split_fields(text: str) -> tuple:
    """Returns (text before the first field, {field: value}, problems)."""
    pattern = re.compile(rf"^({'|'.join(MINIMAX_FIELDS)}):[ \t]*", re.MULTILINE)
    matches = list(pattern.finditer(text))
    names = [m.group(1) for m in matches]
    problems = []
    if names != list(MINIMAX_FIELDS):
        problems.append(f"fields are {names or 'missing'}, expected {list(MINIMAX_FIELDS)} once each in order")
    fields = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fields.setdefault(m.group(1), text[m.end():end].strip())
    head = text[: matches[0].start()].strip() if matches else text
    return head, fields, problems


def _alignment_problems(head: str, mode: str, duration: Optional[float]) -> List[str]:
    if mode == "t2va":
        return [f"unexpected text before the fields: {head[:80]!r}"] if head else []
    if mode == "i2va":
        return [] if head == I2VA_ALIGNMENT else [f"alignment line should be exactly {I2VA_ALIGNMENT!r}, got {head[:120]!r}"]
    m = (FL2VA_ALIGNMENT if mode == "fl2va" else L2VA_ALIGNMENT).match(head)
    if not m:
        return [f"alignment line doesn't match the {mode.upper()} template: {head[:160]!r}"]
    seconds = float(m.group(2))
    if duration is not None and abs(seconds - duration) > 0.005:
        return [f"alignment line says {seconds:.2f}s, expected {duration:.2f}s"]
    if duration is None and not 4 <= seconds <= 15:
        return [f"alignment line says {seconds:.2f}s, outside 4-15s"]
    return []


def _body_problems(body: str, mode: str, duration: Optional[float]) -> List[str]:
    problems = []
    if not body.startswith("[Shot 1]"):
        problems.append("main body doesn't start with [Shot 1]")
    if re.match(r"\[Shot 1\]\s*At \d", body):
        problems.append("[Shot 1] has a timestamp")
    shots = [int(n) for n in re.findall(r"\[Shot (\d+)\]", body)]
    if shots != list(range(1, len(shots) + 1)):
        problems.append(f"shot numbers out of order: {shots}")
    stamps = [int(mm) * 60 + float(ss) for mm, ss in re.findall(r"\[Shot \d+\]\s*At (\d{2}):(\d{2}(?:\.\d+)?)", body)]
    if stamps != sorted(set(stamps)):
        problems.append(f"shot timestamps not strictly increasing: {stamps}")
    if duration is not None and any(t >= duration for t in stamps):
        problems.append(f"shot timestamp past the {duration:.2f}s duration")
    if re.search(r"\bbeats?\b", body, re.IGNORECASE):
        problems.append("says 'beat' in the output")
    if "S.SS" in body or "[Shot N]" in body:
        problems.append("left a template placeholder in")

    limit = 150 if duration is None or duration <= 6 else 200
    words = word_count(DIALOGUE.sub("", body))
    if not MINIMAX_MIN_WORDS <= words <= limit * MINIMAX_WORD_SLACK:
        problems.append(f"main body has {words} words, expected about {MINIMAX_MIN_WORDS}-{limit}")

    if mode == "t2va" and "Picture" in body:
        problems.append("T2VA must not use <Picture> notation")
    if mode in ("i2va", "l2va") and "<Picture 1>" not in body:
        problems.append(f"{mode.upper()} body never refers to <Picture 1>")
    if mode == "fl2va" and "Picture 2" not in body:
        problems.append("FL2VA body never lands on Picture 2")

    if body.count("<d>") != body.count("</d>"):
        problems.append("unbalanced <d> tags")
    spoken = 0
    for m in DIALOGUE.finditer(body):
        line = m.group(1).strip()
        spoken += word_count(re.sub(r"^\[[^\]]+\]", "", line))
        if not re.match(r"\[[A-Za-z][\w -]*\]", line):
            problems.append(f"dialogue lacks a [Language] tag: {line[:40]!r}")
        if not re.search(r"\(S\d+(?:,\s*S\d+)*\)", body[max(0, m.start() - 160): m.start()]):
            problems.append(f"dialogue has no (S1)-style speaker ID before it: {line[:40]!r}")
    budget = DIALOGUE_WORDS_PER_SECOND * (duration or 15)
    if spoken > budget:
        problems.append(f"{spoken} spoken words, over the {budget:.0f}-word budget")
    return problems


def check_minimax(output: str, mode: str, duration: Optional[float] = None) -> List[str]:
    """Returns the MiniMax H3 rules the output breaks for this mode; empty means it passed."""
    text = (output or "").strip()
    if not text:
        return ["empty output"]
    head, fields, problems = _split_fields(text)
    alignment = _alignment_problems(head, mode, duration)
    # The required alignment line itself says "reference pictures".
    problems += _shared_problems(text[len(head):].strip() if mode != "t2va" and not alignment else text)
    problems += alignment
    if "integrated_multimodal_description" in fields:
        problems += _body_problems(fields["integrated_multimodal_description"], mode, duration)

    soundscape = fields.get("overall_soundscape", "")
    if soundscape and soundscape != "N/A":
        if not 1 <= sentence_count(soundscape) <= 4:
            problems.append(f"overall_soundscape has {sentence_count(soundscape)} sentences, expected 1-4")
        if "<d>" in soundscape:
            problems.append("dialogue in overall_soundscape")
    music = fields.get("non_diegetic_music", "")
    if music and music != "N/A":
        if not 1 <= sentence_count(music) <= 3:
            problems.append(f"non_diegetic_music has {sentence_count(music)} sentences, expected 1-3 or N/A")
        if "\n" in music:
            problems.append("text after non_diegetic_music")
    return problems
