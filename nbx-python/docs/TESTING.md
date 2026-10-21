# Testing

The suite needs **Python packages only**: no LLM server, no network and no
running app. `tests/conftest.py` forces `LLM_BACKEND=offline` for every
test, so your `.env` cannot change the results.

## Run

```bash
python run.py test                 # everything (about 20-30 s)
python run.py test -v              # one line per test
python run.py test -k agentic      # tests whose name matches "agentic"
python run.py test tests/test_api_smoke.py
python run.py test -x              # stop at the first failure
```

`python -m pytest` works the same way from the project root. Settings are in
`pyproject.toml` (`asyncio_mode = auto`, `testpaths = tests`).

## What is covered

| File | Covers |
|---|---|
| `test_api_smoke.py` | end to end over HTTP (FastAPI `TestClient`): every read endpoint answers; `/recommend`, `/nba/*`, `/agentic/decide`, `/agentic/stream` and `/copilot/chat` return complete decisions on the offline backend |
| `test_offline_backend.py` | the offline backend: dispatch, health, the council in fallback mode with **zero** LLM calls, NBO / NBA / copilot fallbacks |
| `test_agentic.py` | the council controller, with a scripted fake LLM: required evidence, loop guard, unknown offer codes, critic revision, auto-fix, currency and template guardrails, mid-run LLM failure |
| `test_languages.py` | nationality → language mapping, native-language SMS templates, unsafe-text fallback |
| `test_competitors.py` | parsing and normalizing competitor pages, like-for-like comparison |
| `test_ingestion_pipeline.py` | upload validation, atomic writes, per-step status, retry from the failed step |
| `test_llm_client.py` | backend dispatch, Ollama / vLLM clients (with a fake HTTP server) |
| `test_gradio_helpers.py` | UI helpers: MSISDN parsing, batch CSV export |

## Writing tests

- Async tests are plain `async def test_...()`; no decorator is needed.
- To simulate a **real** LLM, set `config.LLM_BACKEND` to `"ollama"` and
  monkeypatch `llm_client.is_available`, `resolve_model` and `generate`.
  See the `fake_llm` fixture in `test_agentic.py`.
- Tests read the shipped `data/`. Don't call `/seed` or `/upload` against it
  in a test; point `config` paths at `tmp_path` instead, as
  `test_ingestion_pipeline.py` does.

## Manual check of the running app

With `python run.py start` running, in another terminal:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/agentic/decide -H "Content-Type: application/json" -d '{"msisdn": 1, "scope": "full"}'
curl "http://localhost:8000/competitors/search?q=unlimited%20data&top_k=2"
```

Then open http://localhost:7860 and run both recommender tabs on customer `15`.
