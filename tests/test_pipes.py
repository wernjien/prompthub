import asyncio

import pytest


@pytest.fixture(params=["krea2", "minimax"])
def pipe_module(request):
    return request.getfixturevalue(request.param)


def test_task_requests_skip_generation(pipe_module):
    pipe = pipe_module.Pipe()
    assert asyncio.run(pipe.pipe({"messages": []}, None, "title_generation")) == ""


def test_empty_input_returns_hint(pipe_module):
    out = asyncio.run(pipe_module.Pipe().pipe({"messages": [{"content": "  "}]}, []))
    assert out.startswith("⚠️") and pipe_module.EMPTY_INPUT_HINT in out


def test_text_only_runs_writer_and_tidies(pipe_module, monkeypatch):
    calls = []

    def fake_generate(base, model, system, prompt, timeout, num_ctx=0):
        calls.append((model, prompt))
        return "```\nA fox in snow.\n```"

    monkeypatch.setattr(pipe_module, "ollama_generate", fake_generate)
    out = asyncio.run(pipe_module.Pipe().pipe({"messages": [{"content": "a fox"}]}, []))
    assert out.startswith("A fox in snow.")
    assert calls == [("dolphin3:8b", "MY INTENT (primary — this is what the prompt must deliver):\na fox")]


def test_ollama_down_gives_readable_error(pipe_module, monkeypatch):
    def boom(*a, **k):
        raise pipe_module.requests.ConnectionError()

    monkeypatch.setattr(pipe_module, "ollama_generate", boom)
    out = asyncio.run(pipe_module.Pipe().pipe({"messages": [{"content": "a fox"}]}, []))
    assert "Could not build the prompt" in out and "Is it running" in out


def test_minimax_appends_mode_note(minimax, monkeypatch):
    monkeypatch.setattr(minimax, "ollama_generate", lambda *a, **k: "Prompt.")
    out = asyncio.run(minimax.Pipe().pipe({"messages": [{"content": "a fox"}]}, []))
    assert "Wrote a **T2VA** prompt" in out and "I2VA / FL2VA / L2VA" in out
