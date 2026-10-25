# API reference

Base URL: `http://localhost:8000`. The interactive docs, where you can try
every call from the browser, are at **http://localhost:8000/docs**, and the
OpenAPI schema is at `/openapi.json`.

Examples use bash `curl`. On Windows PowerShell, use `curl.exe` (the plain
`curl` is an alias), or `Invoke-RestMethod`:

```powershell
Invoke-RestMethod http://localhost:8000/agentic/decide -Method Post -ContentType "application/json" -Body '{"msisdn": 15, "scope": "full"}'
```

---

## Decisions

### `POST /agentic/decide`: agent council (main endpoint)

```bash
curl -X POST http://localhost:8000/agentic/decide \
  -H "Content-Type: application/json" \
  -d '{"msisdn": 15, "scope": "full"}'
```

| Field | Default | Meaning |
|---|---|---|
| `msisdn` | required | customer id |
| `scope` | `"full"` | `"nba"` = action only; `"nbo"` / `"full"` = action, then offers for it |
| `llm` | `DEFAULT_LLM` | model name (ignored on the offline backend) |
| `use_last_nba` | `false` | offers only: reuse this customer's last action decision (made in the last 30 min) |

Response: `{"decision": {...}, "trace": [...]}`. Key `decision` fields:
`mode` (`agentic` / `fallback` / `partial_fallback`), `action`, `channel`,
`send_window`, `confidence`, `nba_rationale`, `offers[]` (code, name, price,
`p_accept`, `expected_value_omr`, `why`), `critic`, `message` (`sms_en`,
`sms_ar`, `sms_native`, `agent_script`), `explanation`, `stats`.

### `GET /agentic/stream?msisdn=15&scope=full`: same, streamed live

Server-Sent Events, one `data: {json}` line per event (`run_start`,
`agent_start`, `thought`, `tool_call`, `tool_result`, `policy`, `guardrails`,
`revision`, `auto_fix`, `agent_final`, `run_end`, or `error`).

```bash
curl -N "http://localhost:8000/agentic/stream?msisdn=15&scope=nba"
```

### `GET /agentic/recent`

The most recent council decisions (short form).

### `POST /recommend`: offer pipeline (non-agentic)

```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" \
  -d '{"msisdn": 15, "retrieval_mode": "graph"}'
```

Optional: `llm`, `retrieval_mode` (`graph` | `vector`), `vectorstore`
(`chroma`, `faiss`, `qdrant`, `milvus`, `lancedb`), `embedding_model`.

### `POST /nba/action`: bandit action (+ LLM confirmation)

```bash
curl -X POST http://localhost:8000/nba/action -H "Content-Type: application/json" -d '{"msisdn": 15}'
```

### `POST /nba/decide`: offer pipeline wrapped in the NBA/rules envelope

Same body as `/recommend`.

### `POST /copilot/chat`: questions about the portfolio

```bash
curl -X POST http://localhost:8000/copilot/chat -H "Content-Type: application/json" \
  -d '{"question": "Which customers are most at risk?"}'
```

---

## Customers and portfolio

| Endpoint | Returns |
|---|---|
| `GET /customers` | every customer with risk level, segment and policy action |
| `GET /customer/{msisdn}` | profile + purchase history (404 if unknown) |
| `GET /customers/{msisdn}/360` | full 360° view (scores, trends, revenue) |
| `GET /portfolio/summary` | portfolio KPIs + recent decisions |
| `GET /offers` | the offer catalog |

## Models and graph

| Endpoint | Returns |
|---|---|
| `GET /nba/bandit/stats` | bandit training stats |
| `GET /nba/bandit/context/{msisdn}` | the bandit's input features for one customer |
| `GET /cf/stats`, `GET /cf/score/{msisdn}` | collaborative-filtering stats / scores |
| `GET /graph/stats` | node and edge counts |
| `GET /graph/cohort-offers/{msisdn}` | what similar customers bought |
| `GET /graph/similar-customers/{msisdn}` | nearest customers in the graph |
| `GET /rules` | active business rules and action policy |

## Competitors

| Endpoint | Returns |
|---|---|
| `GET /competitors/offers?operator=COMPETITOR_A&segment=mass` | competitor plans (both filters optional) |
| `GET /competitors/compare/{program_rule}` | like-for-like competitor plans for one of our offers |
| `GET /competitors/search?q=unlimited data&top_k=5` | semantic search (needs `run.py setup`) |
| `POST /competitors/refresh` | re-scrape the public pages (internet, ~30 s) |

## System

| Endpoint | Returns |
|---|---|
| `GET /health` | `status`, `llm_backend`, vector-store status, loaded models |
| `GET /models` | LLM roster and which are installed |
| `GET /embeddings` | embedding roster and availability |
| `GET /vectorstores` | vector-store roster, availability, document counts |
| `GET /logs` | the last 20 API calls |
| `POST /vectorstore/switch` | `{"store": "faiss"}`: set the active store |
| `POST /vectorstore/compare` | `{"query": "...", "top_k": 5}`: query every store |
| `POST /benchmark` | run `/recommend` on every installed LLM (needs Ollama/vLLM) |

## Data (these overwrite `data/`)

| Endpoint | Does |
|---|---|
| `POST /seed` | regenerate the synthetic dataset and re-index |
| `POST /seed/embedding` | `{"embedding_model": "..."}`: index another embedding model |
| `POST /upload` | multipart `customers`, `offers`, `purchases`, `segments` files → `{"job_id"}` |
| `POST /upload/quick` | re-ingest the files already in `data/` |
| `GET /upload/status/{job_id}` | per-step progress |
| `POST /upload/retry/{job_id}` | resume from the failed step |
| `POST /nba/upload`, `/nba/upload/quick`, `GET /nba/upload/status/{id}`, `POST /nba/upload/retry/{id}` | same, for the 7 NBA score files |

```bash
curl -X POST http://localhost:8000/upload \
  -F customers=@customers.json -F offers=@offers.json \
  -F purchases=@purchases.json -F segments=@segments.json
curl http://localhost:8000/upload/status/<job_id>
```

Every response carries an `X-Process-Time-Ms` header.
