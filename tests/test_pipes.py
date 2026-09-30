import asyncio
import inspect

import pytest


@pytest.fixture(params=["krea2", "minimax"])
def pipe_module(request):
    return request.getfixturevalue(request.param)


def run(pipe_module, body, files=None, task=None):
    """Calls the pipe like Open WebUI does and joins whatever it streams."""
    async def go():
        res = await pipe_module.Pipe().pipe(body, files, task)
        if isinstance(res, str):
            return res
        assert inspect.isasyncgen(res)
        return "".join([chunk async for chunk in res])
    return asyncio.run(go())


def fake_stream(*pieces):
    def stream(base, model, system, prompt, timeout, num_ctx=0):
        fake_stream.calls.append((model, prompt))
        yield from pieces
    fake_stream.calls = []
    return stream


def test_task_requests_skip_generation(pipe_module):
    assert run(pipe_module, {"messages": []}, None, "title_generation") == ""


def test_empty_input_returns_hint(pipe_module):
    out = run(pipe_module, {"messages": [{"content": "  "}]}, [])
    assert out.startswith("⚠️") and pipe_module.EMPTY_INPUT_HINT in out


def test_text_only_streams_writer_and_tidies(pipe_module, monkeypatch):
    monkeypatch.setattr(pipe_module, "ollama_stream", fake_stream("```\nA fox ", "in snow.", "\n```"))
    out = run(pipe_module, {"messages": [{"content": "a fox"}]}, [])
    assert out.startswith("A fox in snow.") and "```" not in out
    assert fake_stream.calls == [("dolphin3:8b", "MY INTENT (primary — this is what the prompt must deliver):\na fox")]


def test_ollama_down_gives_readable_error(pipe_module, monkeypatch):
    def boom(*a, **k):
        raise pipe_module.requests.ConnectionError()
        yield

    monkeypatch.setattr(pipe_module, "ollama_stream", boom)
    out = run(pipe_module, {"messages": [{"content": "a fox"}]}, [])
    assert out.startswith("⚠️ Could not build the prompt") and "Is it running" in out


def test_error_mid_stream_keeps_partial_text(pipe_module, monkeypatch):
    def partial(*a, **k):
        yield "A fox in the snow at dusk, " * 3
        raise pipe_module.requests.Timeout()

    monkeypatch.setattr(pipe_module, "ollama_stream", partial)
    out = run(pipe_module, {"messages": [{"content": "a fox"}]}, [])
    assert out.startswith("A fox in the snow") and "\n\n⚠️ Could not build the prompt" in out


def test_minimax_appends_mode_note_and_rewrites_image(minimax, monkeypatch):
    monkeypatch.setattr(minimax, "ollama_stream", fake_stream("Keep the ", "ima", "ge sharp."))
    out = run(minimax, {"messages": [{"content": "a fox"}]}, [])
    assert out.startswith("Keep the frame sharp.")
    assert "Wrote a **T2VA** prompt" in out and "I2VA / FL2VA / L2VA" in out
