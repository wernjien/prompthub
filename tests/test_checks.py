import re

import pytest
from checks import check_krea2, check_minimax, stated_duration


def _krea_examples(krea2):
    return [re.split(r"\n\nEXAMPLE", part)[0].strip() for part in krea2.SYSTEM_PROMPT.split("\nOutput: ")[1:]]


def _minimax_example(minimax, mode):
    prompt = minimax.SYSTEM_PROMPTS[mode]
    example = prompt.split("\nEXAMPLE\n", 1)[1]
    given, output = example.split("\nOutput:\n", 1)
    return output.strip(), stated_duration(given)


def test_krea_prompt_examples_pass(krea2):
    examples = _krea_examples(krea2)
    assert len(examples) == 2
    for example in examples:
        assert check_krea2(example) == []


@pytest.mark.parametrize("mode", ["t2va", "i2va", "fl2va", "l2va"])
def test_minimax_prompt_examples_pass(minimax, mode):
    output, duration = _minimax_example(minimax, mode)
    assert check_minimax(output, mode, duration) == []


GOOD_KREA = " ".join(["A weathered lighthouse keeper stands on a windswept cliff at dawn."] * 6)


@pytest.mark.parametrize("bad, expected", [
    ("", "empty output"),
    ("Here is your prompt: " + GOOD_KREA, "preamble"),
    ("**Prompt** " + GOOD_KREA, "markdown"),
    (GOOD_KREA + " masterpiece, 8k.", "filler tag"),
    (GOOD_KREA + " The image shows a gull.", "refers to the reference"),
    (GOOD_KREA + " A street with no people.", "negative phrasing"),
    (GOOD_KREA + " (sunset:1.3).", "weighting"),
    (GOOD_KREA + "\n\n" + GOOD_KREA, "2 paragraphs"),
    ("A lighthouse on a cliff.", "words, expected 50-120"),
    ('"' + GOOD_KREA + '"', "wrapped in quotes"),
])
def test_krea_flags_rule_breaks(bad, expected):
    assert any(expected in p for p in check_krea2(bad)), check_krea2(bad)


def test_minimax_flags_missing_fields(minimax):
    output, _ = _minimax_example(minimax, "t2va")
    broken = output.replace("overall_soundscape:", "soundscape:")
    assert any("fields are" in p for p in check_minimax(broken, "t2va", 6.0))


def test_minimax_flags_wrong_alignment_duration(minimax):
    output, _ = _minimax_example(minimax, "fl2va")
    problems = check_minimax(output, "fl2va", 8.0)
    assert any("expected 8.00s" in p for p in problems)


def test_minimax_flags_picture_in_t2va(minimax):
    output, _ = _minimax_example(minimax, "t2va")
    broken = output.replace("[Shot 1] Live-action", "[Shot 1] Live-action, matching <Picture 1>")
    assert any("<Picture> notation" in p for p in check_minimax(broken, "t2va", 6.0))


def test_minimax_flags_dialogue_rules(minimax):
    output, _ = _minimax_example(minimax, "i2va")
    no_tag = output.replace("<d>[English] Who's there?</d>", "<d>Who's there?</d>")
    assert any("[Language] tag" in p for p in check_minimax(no_tag, "i2va", 6.0))
    no_id = output.replace(" (S1)", "")
    assert any("speaker ID" in p for p in check_minimax(no_id, "i2va", 6.0))


def test_minimax_flags_beats_and_timestamps(minimax):
    output, _ = _minimax_example(minimax, "t2va")
    broken = output.replace("[Shot 1] Live-action", "[Shot 1] At 00:01.000, beat one, Live-action")
    problems = check_minimax(broken, "t2va", 6.0)
    assert any("'beat'" in p for p in problems) and any("has a timestamp" in p for p in problems)


def test_minimax_flags_preamble_text(minimax):
    output, _ = _minimax_example(minimax, "t2va")
    assert any("before the fields" in p for p in check_minimax("T2VA prompt\n\n" + output, "t2va", 6.0))


def test_stated_duration():
    assert stated_duration("[6s] a fox") == 6.0
    assert stated_duration("[ 7.5 s ]") == 7.5
    assert stated_duration("a fox") is None
