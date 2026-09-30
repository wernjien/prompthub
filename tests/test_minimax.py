def test_clip_target_in_range(minimax):
    assert minimax.clip_target("", 8.0) == (8.0, "")


def test_clip_target_clamped_with_note(minimax):
    target, note = minimax.clip_target("", 30.0)
    assert target == minimax.H3_MAX_SECONDS and "30.0s" in note
    target, _ = minimax.clip_target("", 1.0)
    assert target == minimax.H3_MIN_SECONDS


def test_clip_target_none_without_clip(minimax):
    assert minimax.clip_target("", None) == (None, "")


def test_clip_target_respects_stated_duration(minimax):
    assert minimax.clip_target("make it 10 seconds", 30.0) == (None, "")


def test_frame_not_image(minimax):
    assert minimax.frame_not_image("Keep the image sharp.") == "Keep the frame sharp."
    assert minimax.frame_not_image("the image's edge") == "the frame's edge"
    assert minimax.frame_not_image("an imagery of") == "an imagery of"
