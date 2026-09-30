import pytest

IMG = "aGVsbG8="


@pytest.mark.parametrize(
    "text, images, video, mode",
    [
        ("a fox running", [], None, "t2va"),
        ("a fox running", [IMG], None, "i2va"),
        ("a fox running", [IMG, IMG], None, "fl2va"),
        ("ends on this frame", [IMG], None, "l2va"),
        ("anything", [], "clip.mp4", "t2va"),
        ("starts from the first frame", [], "clip.mp4", "i2va"),
        ("ends on the last frame", [], "clip.mp4", "l2va"),
        ("morph between these", [], "clip.mp4", "fl2va"),
        ("make it L2VA", [IMG, IMG], None, "l2va"),
        ("use T2VA please", [IMG], None, "t2va"),
    ],
)
def test_choose_mode(minimax, text, images, video, mode):
    assert minimax.choose_mode(text, images, video)[0] == mode


def test_explicit_mode_wins_over_attachments(minimax):
    mode, reason = minimax.choose_mode("fl2va", [IMG], None)
    assert mode == "fl2va" and "FL2VA" in reason


def test_every_mode_has_policy_and_prompt(minimax):
    for mode in ("t2va", "i2va", "fl2va", "l2va"):
        assert mode in minimax.REFERENCE_POLICY
        assert mode in minimax.SYSTEM_PROMPTS
        assert mode in minimax.VISION_INSTRUCTION_IMAGE
        assert mode in minimax.VISION_INSTRUCTION_VIDEO
