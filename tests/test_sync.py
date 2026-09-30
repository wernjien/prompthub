import sync_shared


def test_functions_match_shared_helpers():
    assert sync_shared.drifted() == []
