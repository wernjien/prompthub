import subprocess

import pytest
import requests


def test_tidy_output_strips_fences_and_label(helpers):
    assert helpers.tidy_output("```text\nhello\n```") == "hello"
    assert helpers.tidy_output("Alignment Instruction: hello") == "hello"
    assert helpers.tidy_output("  plain  ") == "plain"
    assert helpers.tidy_output(None) == ""


def test_clean_caption_drops_meta_sentences(helpers):
    out = helpers.clean_caption("The image shows a red car. The resolution is low. It has chrome trim.")
    assert out == "A red car. It has chrome trim."


def test_clean_caption_keeps_noun_phrases(helpers):
    assert helpers.clean_caption("The frame of the bike is steel.") == "The frame of the bike is steel."


def test_clean_caption_falls_back_to_original(helpers):
    text = "The resolution is poor."
    assert helpers.clean_caption(text) == text
    assert helpers.clean_caption("") == ""


def test_build_idea_text_only(helpers):
    idea = helpers.build_idea("a fox", "", None)
    assert idea.startswith("MY INTENT") and "SCENE DETAILS" not in idea


def test_build_idea_caption_only_and_duration(helpers):
    idea = helpers.build_idea("", "A fox in snow.", 6.0)
    assert "Target duration: 6.00 seconds." in idea
    assert "SCENE DETAILS" in idea and "No separate intent" in idea


def test_build_idea_both(helpers):
    idea = helpers.build_idea("at night", "A fox.", None)
    assert "MY INTENT" in idea and "SCENE DETAILS" in idea and "No separate intent" not in idea


def test_attachment_problem(helpers):
    assert helpers.attachment_problem(0, 0, "") == ""
    assert helpers.attachment_problem(1, 0, "") == ""
    assert helpers.attachment_problem(0, 3, "") == ""
    assert "couldn't be read" in helpers.attachment_problem(0, 0, "a.png: missing")
    assert "either one clip" in helpers.attachment_problem(2, 0, "")
    assert "either one clip" in helpers.attachment_problem(1, 1, "")


def test_extract_user_text(helpers):
    assert helpers.extract_user_text({}) == ""
    assert helpers.extract_user_text({"messages": [{"content": "  hi  "}]}) == "hi"
    body = {"messages": [{"content": [{"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}]}]}
    assert helpers.extract_user_text(body) == "a b"


def test_extract_images_b64(helpers):
    body = {"messages": [{"content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "text", "text": "x"},
    ]}]}
    images, note = helpers.extract_images_b64(body)
    assert images == ["QUJD"] and "1 inline image" in note
    assert helpers.extract_images_b64({"messages": [{"content": "text"}]}) == ([], "")


@pytest.mark.parametrize("count, limit, expected", [
    (3, 5, [0, 1, 2]),
    (10, 0, list(range(10))),
    (10, 1, [0]),
    (10, 2, [0, 9]),
    (9, 3, [0, 4, 8]),
])
def test_cap_images(helpers, count, limit, expected):
    items = [str(i) for i in range(count)]
    assert helpers.cap_images(items, limit) == [str(i) for i in expected]


def test_image_ext(helpers):
    assert helpers._image_ext(b"\x89PNG....") == ".png"
    assert helpers._image_ext(b"RIFF\0\0\0\0WEBPxx") == ".webp"
    assert helpers._image_ext(b"GIF89a") == ".gif"
    assert helpers._image_ext(b"\xff\xd8\xff") == ".jpg"


def test_reference_footer(helpers):
    assert helpers.reference_footer([], "none") == ""
    assert "supply a first-frame" in helpers.reference_footer([], "first")
    refs = [("Reference image (Picture 1, opening)", "/tmp/p1.png")]
    footer = helpers.reference_footer(refs, "first_last")
    assert "/tmp/p1.png" in footer and "ending frame" in footer


def test_save_image_references(helpers, tmp_path):
    assert helpers.save_image_references(["QUJD"], "none", str(tmp_path)) == []
    refs = helpers.save_image_references(["QUJD", "REVG"], "first_last", str(tmp_path))
    assert [label for label, _ in refs] == [
        "Reference image (Picture 1, opening)", "Reference image (Picture 2, ending)"
    ]
    assert all((tmp_path / p.split("/")[-1]).exists() for _, p in refs)


def test_save_video_references(helpers, tmp_path):
    src = tmp_path / "f.jpg"
    src.write_bytes(b"x")
    out = tmp_path / "out"
    refs = helpers.save_video_references(str(src), str(src), "first_last", str(out))
    assert len(refs) == 2 and all(p.endswith(".jpg") for _, p in refs)
    assert helpers.save_video_references(str(src), str(src), "none", str(out)) == []


def test_stderr_tail(helpers):
    exc = subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"first\n\nlast line\n")
    assert helpers.stderr_tail(exc) == "last line"
    assert helpers.stderr_tail(subprocess.CalledProcessError(1, ["x"])) == ""


def test_pipeline_error_messages(helpers):
    assert "Is it running" in helpers.pipeline_error(requests.ConnectionError(), False)
    assert "timed out" in helpers.pipeline_error(requests.Timeout(), False)
    exc = subprocess.CalledProcessError(2, ["/usr/bin/ffmpeg"], stderr=b"bad codec")
    msg = helpers.pipeline_error(exc, True)
    assert "ffmpeg failed: bad codec" in msg and "Fallback" in msg
    assert "Fallback" not in helpers.pipeline_error(RuntimeError("boom"), False)


class _Resp:
    def __init__(self, ok=True, status=200, data=None, text=""):
        self.ok, self.status_code, self._data, self.text = ok, status, data, text

    def json(self):
        if self._data is None:
            raise ValueError
        return self._data


def test_ollama_generate_success_and_payload(helpers, monkeypatch):
    seen = {}

    def fake_post(url, json, timeout):
        seen.update(url=url, json=json, timeout=timeout)
        return _Resp(data={"response": "  done  "})

    monkeypatch.setattr(helpers.requests, "post", fake_post)
    assert helpers.ollama_generate("http://h:1/", "m", "sys", "p", 5, 4096) == "done"
    assert seen["url"] == "http://h:1/api/generate"
    assert seen["json"]["options"] == {"num_ctx": 4096} and seen["json"]["system"] == "sys"


def test_ollama_generate_errors(helpers, monkeypatch):
    monkeypatch.setattr(helpers.requests, "post", lambda *a, **k: _Resp(False, 404, {"error": "no model"}))
    with pytest.raises(RuntimeError, match="404 for m: no model"):
        helpers.ollama_generate("http://h", "m", "s", "p", 5)
    monkeypatch.setattr(helpers.requests, "post", lambda *a, **k: _Resp(data={"response": " "}))
    with pytest.raises(RuntimeError, match="empty response"):
        helpers.ollama_generate("http://h", "m", "s", "p", 5)


def test_ollama_vision_sends_images_without_ctx_when_zero(helpers, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        helpers.requests, "post", lambda url, json, timeout: seen.update(json=json) or _Resp(data={"response": "ok"})
    )
    helpers.ollama_vision("http://h", "v", "look", ["QUJD"], 5)
    assert seen["json"]["images"] == ["QUJD"] and "options" not in seen["json"]


class _StreamResp(_Resp):
    def __init__(self, lines, ok=True, status=200, data=None):
        super().__init__(ok, status, data)
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ollama_stream_yields_pieces(helpers, monkeypatch):
    lines = [b'{"response": "A "}', b"", b'{"response": "fox"}', b'{"done": true}', b'{"response": "late"}']
    seen = {}
    monkeypatch.setattr(helpers.requests, "post", lambda url, **k: seen.update(k) or _StreamResp(lines))
    assert list(helpers.ollama_stream("http://h", "m", "s", "p", 5, 2048)) == ["A ", "fox"]
    assert seen["stream"] is True and seen["json"]["stream"] is True and seen["json"]["options"] == {"num_ctx": 2048}


def test_ollama_stream_errors(helpers, monkeypatch):
    monkeypatch.setattr(helpers.requests, "post", lambda *a, **k: _StreamResp([], False, 404, {"error": "no model"}))
    with pytest.raises(RuntimeError, match="404 for m: no model"):
        list(helpers.ollama_stream("http://h", "m", "s", "p", 5))
    monkeypatch.setattr(helpers.requests, "post", lambda *a, **k: _StreamResp([b'{"error": "oom"}']))
    with pytest.raises(RuntimeError, match="mid-response for m: oom"):
        list(helpers.ollama_stream("http://h", "m", "s", "p", 5))
    monkeypatch.setattr(helpers.requests, "post", lambda *a, **k: _StreamResp([b'{"response": " "}', b'{"done": true}']))
    with pytest.raises(RuntimeError, match="empty response"):
        list(helpers.ollama_stream("http://h", "m", "s", "p", 5))


@pytest.mark.parametrize("text", [
    "```text\n" + "A long prompt about a fox in the snow. " * 5 + "\n```",
    "Alignment Instruction: " + "For the target video, the image's edge glows. " * 4,
    "short",
    "x" * 200,
])
@pytest.mark.parametrize("size", [1, 3, 17])
def test_stream_transformed_matches_whole_text_transform(helpers, text, size):
    transform = helpers.tidy_output
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    pieces = list(helpers.stream_transformed(iter(chunks), transform))
    assert "".join(pieces) == transform(text)
    if len(text) > 100:
        assert len(pieces) > 1


def test_stream_in_thread_yields_and_closes(helpers):
    closed = []

    def gen(n):
        try:
            yield from (str(i) for i in range(n))
        finally:
            closed.append(True)

    async def collect():
        return [c async for c in helpers.stream_in_thread(gen, 3)]

    import asyncio
    assert asyncio.run(collect()) == ["0", "1", "2"] and closed == [True]
