"""Tests for the vLLM client (llms/vllm_client.py) and the backend switch
(llms/llm_client.py). The vLLM server is replaced by an httpx.MockTransport,
so these run anywhere -- no GPU, no vLLM install, no network.
"""
from __future__ import annotations

import json

import httpx
import pytest

from nbx.core import config
from nbx.llms import llm_client, vllm_client

SERVED = ["mistral:7b", "mistralai/Mistral-7B-Instruct-v0.3"]


class _Requests(list):
    """Requests the fake server received, plus its knobs in .state."""


@pytest.fixture(autouse=True)
def _fresh_vllm_client(monkeypatch):
    """vllm_client pools one HTTP client per event loop; start each test clean."""
    monkeypatch.setattr(vllm_client, "_client_cache", None)


@pytest.fixture
def vllm(monkeypatch):
    """Routes vllm_client's HTTP calls to a fake vLLM server. Returns the
    list of requests it received, for assertions on the payload."""
    seen = _Requests()
    state = {"chat_status": 200, "content": '{"offers": []}'}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": m} for m in SERVED]})
        if request.url.path == "/v1/chat/completions":
            if state["chat_status"] != 200:
                return httpx.Response(state["chat_status"], json={"error": "boom"})
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": state["content"]}}]})
        return httpx.Response(404)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(vllm_client.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(config, "VLLM_URL", "http://vllm.test")
    monkeypatch.setattr(config, "LOCAL_HOSTS", config.LOCAL_HOSTS | {"vllm.test"})   # the fake server counts as local
    monkeypatch.setattr(config, "VLLM_API_KEY", "")
    monkeypatch.setattr(vllm_client, "_served_cache", (0.0, []))
    seen.state = state
    return seen


async def test_lists_served_model_names(vllm):
    assert await vllm_client.is_available()
    assert await vllm_client.list_installed_models() == SERVED


async def test_generate_sends_openai_payload_and_returns_ollama_shape(vllm):
    vllm.state["content"] = '{"offers": [{"offer_name": "X"}]}'
    result = await vllm_client.generate("mistral:7b", "hello", system="sys", force_json=True)

    assert result["error"] is None
    assert json.loads(result["content"]) == {"offers": [{"offer_name": "X"}]}
    assert set(result) == {"content", "duration_ms", "model", "error"}

    body = json.loads(vllm[-1].content)
    assert body["model"] == "mistral:7b"
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == config.VLLM_MAX_TOKENS


async def test_generate_without_json_mode_omits_response_format(vllm):
    await vllm_client.generate("mistral:7b", "hello")
    assert "response_format" not in json.loads(vllm[-1].content)


async def test_api_key_sent_as_bearer(vllm, monkeypatch):
    monkeypatch.setattr(config, "VLLM_API_KEY", "secret")
    await vllm_client.generate("mistral:7b", "hello")
    assert vllm[-1].headers["Authorization"] == "Bearer secret"


async def test_unknown_model_404_becomes_error_not_exception(vllm):
    vllm.state["chat_status"] = 404
    result = await vllm_client.generate("gemma2:9b", "hello")
    assert result["content"] == ""
    assert "not served by vLLM" in result["error"]


async def test_unreachable_server_becomes_error(monkeypatch):
    monkeypatch.setattr(config, "VLLM_URL", "http://127.0.0.1:9")  # nothing listens on the discard port
    assert not await vllm_client.is_available()
    result = await vllm_client.generate("mistral:7b", "hello", timeout=2.0)
    assert result["error"]


async def test_resolve_model_falls_back_to_the_served_model(vllm):
    assert await vllm_client.resolve_model("mistralai/Mistral-7B-Instruct-v0.3") == "mistralai/Mistral-7B-Instruct-v0.3"
    assert await vllm_client.resolve_model("gemma2:9b") == "mistral:7b"


def test_model_is_installed_is_exact_match_on_vllm():
    assert vllm_client.model_is_installed(SERVED, "mistral:7b")
    assert not vllm_client.model_is_installed(SERVED, "mistral:latest")


async def test_llm_client_dispatches_on_config(vllm, monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "vllm")
    assert llm_client.backend_url() == "http://vllm.test"
    assert await llm_client.list_installed_models() == SERVED
    assert llm_client.model_is_installed(SERVED, "mistral:7b")

    monkeypatch.setattr(config, "LLM_BACKEND", "offline")
    assert llm_client.backend_url() == "in-process"


async def test_resolve_or_none_gives_the_rule_based_path_offline_or_down(vllm, monkeypatch):
    assert await llm_client.resolve_or_none("mistral:7b") is None           # offline backend
    monkeypatch.setattr(config, "LLM_BACKEND", "vllm")
    assert await llm_client.resolve_or_none("mistral:7b") == "mistral:7b"

    async def down():
        return False

    monkeypatch.setattr(vllm_client, "is_available", down)
    assert await llm_client.resolve_or_none("mistral:7b") is None


@pytest.mark.parametrize("backend, attr, url", [("ollama", "OLLAMA_URL", "http://127.0.0.1:11434"),
                                                 ("llamacpp", "LLAMACPP_URL", "http://127.0.0.1:8080"),
                                                 ("lmstudio", "LMSTUDIO_URL", "http://127.0.0.1:1234")])
async def test_every_local_server_backend_uses_the_openai_compatible_client(vllm, monkeypatch, backend, attr, url):
    monkeypatch.setattr(config, "LLM_BACKEND", backend)
    monkeypatch.setattr(config, attr, "http://vllm.test")
    assert llm_client.backend_name() == backend and not llm_client.is_offline()
    assert llm_client.backend_url() == "http://vllm.test"
    out = await llm_client.generate("qwen2.5:3b", "hello", force_json=True)
    assert out["error"] is None
    body = json.loads(vllm[-1].content)
    assert ("response_format" in body) == config.LLM_JSON_MODE[backend]       # LM Studio: no json_object
    assert config.llm_url(backend) == "http://vllm.test"
    monkeypatch.setattr(config, attr, url)
    assert config.is_local_url(config.llm_url(backend))


async def test_local_only_refuses_a_remote_llm_server(monkeypatch):
    for backend, attr in (("vllm", "VLLM_URL"), ("ollama", "OLLAMA_URL")):
        monkeypatch.setattr(config, "LLM_BACKEND", backend)
        monkeypatch.setattr(config, attr, "https://llm.example.com")
        monkeypatch.setattr(config, "LOCAL_ONLY", True)
        assert not await vllm_client.is_available() and await vllm_client.list_installed_models() == []
        out = await vllm_client.generate("m", "a prompt that must not leave")
        assert out["error"].startswith("FSDR_LOCAL_ONLY")
        assert await llm_client.resolve_or_none(None) is None                  # agents take their rule-based path
        assert config.local_violations() and "not this machine" in config.local_violations()[0]


def test_local_hosts_and_cloud_apis():
    assert config.is_local_url("http://127.0.0.1:8001") and config.is_local_url("http://localhost:11434")
    assert config.is_local_url("http://[::1]:8080") and not config.is_local_url("https://api.anthropic.com")
    assert not config.is_local_url("https://api.openai.com/v1")
    assert config.LOCAL_ONLY                                                   # the default


def test_tracing_to_the_cloud_is_refused_in_local_only_mode(monkeypatch):
    from nbx.core import tracing

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)                    # default = the LangSmith cloud
    assert not tracing.enabled()
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://127.0.0.1:1984")
    assert tracing.enabled()


def test_telemetry_is_off_and_no_cloud_llm_sdk_is_used():
    import os
    from pathlib import Path

    for k, v in {"GRADIO_ANALYTICS_ENABLED": "False", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                 "VLLM_NO_USAGE_STATS": "1"}.items():
        assert os.environ.get(k) == v, k
    root = Path(__file__).resolve().parents[1]
    code = "\n".join(p.read_text(encoding="utf-8") for p in (root / "nbx").rglob("*.py")) + (root / "run.py").read_text(encoding="utf-8")
    for needle in ("import anthropic", "import openai", "from openai", "from anthropic", "api.anthropic.com",
                   "api.openai.com", "GoogleFont", "fonts.googleapis"):
        assert needle not in code, needle
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "anthropic" not in pyproject and "openai" not in pyproject


def test_invalid_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "tgi")
    with pytest.raises(ValueError, match="LLM_BACKEND"):
        llm_client.backend_name()


async def test_reuses_one_pooled_client_per_event_loop(vllm):
    await vllm_client.generate("mistral:7b", "a")
    first = vllm_client._client_cache[1]
    await vllm_client.generate("mistral:7b", "b")
    await vllm_client.list_installed_models()
    assert vllm_client._client_cache[1] is first


def test_ui_page_and_api_docs_load_nothing_from_outside():
    import re

    from fastapi.testclient import TestClient

    from nbx.api.main import app
    from nbx.ui.gradio_app import strip_external

    page = TestClient(app).get("/docs")
    assert page.status_code == 200 and not re.findall(r"(?:src|href)=\"https?://", page.text)
    html = (b'<link rel="preconnect" href="https://fonts.googleapis.com" />\n<link rel="preconnect"\n'
            b' href="https://fonts.gstatic.com" crossorigin="anonymous" />\n<script\n src="https://cdnjs.cloudflare.com/ajax/'
            b'libs/iframe-resizer/4.3.1/iframeResizer.contentWindow.min.js" async></script><div>ok</div>')
    assert strip_external(html).strip() == b"<div>ok</div>"
