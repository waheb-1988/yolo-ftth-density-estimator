"""Tests for the vLLM client (llms/vllm_client.py) and the backend switch
(llms/llm_client.py). The vLLM server is replaced by an httpx.MockTransport,
so these run anywhere -- no GPU, no vLLM install, no network.
"""
from __future__ import annotations

import json

import httpx
import pytest

from nbx.core import config
from nbx.llms import llm_client, ollama_client, vllm_client

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

    called = {}

    async def fake_ollama_list():
        called["ollama"] = True
        return ["mistral:latest"]

    monkeypatch.setattr(ollama_client, "list_installed_models", fake_ollama_list)
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    assert await llm_client.list_installed_models() == ["mistral:latest"]
    assert called == {"ollama": True}
    # Ollama keeps its looser tag matching.
    assert llm_client.model_is_installed(["mistral:latest"], "mistral:7b")


def test_ollama_backend_is_refused_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_ENABLED", False)
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    with pytest.raises(ValueError, match="OLLAMA_ENABLED"):
        llm_client.backend_name()


def test_ollama_embedder_is_refused_when_disabled(monkeypatch):
    from nbx.llms import embeddings
    monkeypatch.setattr(config, "OLLAMA_ENABLED", False)
    with pytest.raises(ValueError, match="OLLAMA_ENABLED"):
        embeddings.embed_texts(["x"], "qwen3-embedding:0.6b")


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
