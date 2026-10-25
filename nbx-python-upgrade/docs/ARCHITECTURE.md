# Architecture

## The two processes

```
 Browser ──▶ Gradio UI (nbx/ui, :7860) ──HTTP──▶ FastAPI (nbx/api/main.py, :8000)
                                                    │
                     ┌──────────────────────────────┼──────────────────────────────┐
                     ▼                              ▼                              ▼
          agents/agentic (council)        agents/nbo_agent, nba_agent      ingestion pipelines
                     │                              │
        ┌────────────┼─────────────┬────────────────┼──────────────┐
        ▼            ▼             ▼                ▼              ▼
  core/nba_bandit  core/collab_  core/knowledge_  vectorstores/   core/competitors
  (Vowpal Wabbit)  filter (BPR)  graph (networkx) (5 backends)    (snapshot + Chroma)
                                                    │
                                       llms/embeddings (MiniLM)
        llms/llm_client ──▶ offline_client | ollama_client | vllm_client
```

All state lives in process memory and in `data/`. There is no database
server. The Gradio UI holds no logic of its own: every button calls an API
endpoint, so anything you can do in the UI you can also do over HTTP.

## How a decision is made

`POST /agentic/decide` (and the two recommender tabs) runs the **agent council**
in `nbx/agents/agentic/council.py`. NBA always runs first; the offer step
builds on its result.

```
 1. NBA strategist   ── tools: risk scores, usage trend, revenue trend, bandit
                        pick, contact policy, market overview
                     ──▶ one action + channel + send window + confidence
 2. NBO specialist   ── tools: catalog search, CF scores, cohort graph,
    (scope nbo/full)    eligibility, expected value, competitor benchmark
                     ──▶ up to 4 offers that serve the action
 3. Critic           ── checks the decision against guardrails
                        (e.g. no push action on an at-risk customer,
                        offer codes must exist) ──▶ approve / revise / auto-fix
 4. Messenger        ── SMS (EN / AR / customer's language) + agent script
 5. Explanation      ── computed facts per signal and per offer, plus a short narrative
```

Each agent is a **ReAct loop** (`agents/agentic/react.py`): the LLM replies
with one JSON object, either a tool call or a final answer. The controller
enforces the rules, not the model:

- **required evidence**: an agent cannot finalize before calling the tools it needs;
- **loop guard**: repeated identical calls are refused;
- **validation**: unknown actions or offer codes are rejected;
- **step budget**: the agent stops after a fixed number of steps.

### Deterministic fallback (the offline backend)

Every agent has a deterministic policy that uses **the same tools**
(`council.py`, "Deterministic policies"). It runs when:

- `LLM_BACKEND=offline` (the default here). The council skips the LLM
  entirely and reports `mode: "fallback"`, with `llm_calls: 0`;
- the LLM server is unreachable;
- an LLM call fails mid-run. The remaining agents fall back, and the run
  reports `mode: "partial_fallback"`.

So the guardrails, numbers, offers and explanations are always produced. The
LLM only adds its own reasoning and wording on top.

## The non-agentic pipelines

These are kept for comparison and for the plain endpoints:

- **`POST /recommend`**: `agents/nbo_agent.py`:
  profile + need detection → retrieval (`graph` by default, or `vector`) + CF
  candidates → LLM picks 4 offers → `agents/reranker.py` blends LLM, retrieval,
  graph, CF and price-fit scores → `agents/rules_engine.py`.
- **`POST /nba/action`**: `agents/nba_agent.py`:
  the Vowpal Wabbit bandit proposes an action; the LLM may confirm it or
  override it. On the offline backend, the bandit's pick is used.
- **`POST /nba/decide`**: NBO wrapped in the NBA envelope (`agents/final_agent.py`).

## Models (trained at first use, in memory)

| Model | File | Library | Trained on |
|---|---|---|---|
| Contextual bandit | `core/nba_bandit.py` | `vowpalwabbit` (`--cb_explore_adf`) | NBA score + usage files, `nba_targeting.json` |
| Collaborative filtering | `core/collab_filter.py` | `implicit` (BPR) | `purchases.json` |
| Cohort graph | `core/knowledge_graph.py` | `networkx` | customers, purchases, segments |
| Embeddings | `llms/embeddings.py` | `sentence-transformers` (MiniLM, 384-d) | offers, segments, purchase patterns |

The first request after start-up trains the bandit and the CF model (a second
or two). Later requests reuse them.

## Retrieval modes

`DEFAULT_RETRIEVAL_MODE=graph` (default): "what fits this need" comes from real
peer purchases and segment affinity in the cohort graph. No embeddings are
needed, which matches production telco data (relational, no free text).
`vector` uses an embedded vector store instead: Chroma, FAISS, Qdrant, Milvus
Lite or LanceDB. Each (store, embedding model) pair has its own collection,
so switching the model never corrupts an index.

## Adding business rules

`agents/rules_engine.py` holds `RULES`, a list of `Rule(name, fn)` where
`fn(recommendations, customer, context) -> recommendations`. Add a rule there.
It applies to `/recommend` and `/nba/decide`, and it shows in the Business
Rules tab.

## Package layout and imports

Everything is under the `nbx` package and imported absolutely
(`from nbx.core import config`). Run commands from the project root, either
through `run.py` (which sets `PYTHONPATH`) or with `python -m ...`, so that
`nbx` is importable.
