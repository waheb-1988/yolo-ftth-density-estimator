"""FastAPI backend for the NBO Agentic RAG system."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from nbx.core import collab_filter, competitors, config, insights, knowledge_graph, nba_bandit, readiness
from nbx.core.models import (
    AgenticRequest,
    CopilotRequest,
    BenchmarkRequest,
    BenchmarkResponse,
    CohortOfferHit,
    Customer,
    EmbeddingModelInfo,
    GraphStatsResponse,
    HealthResponse,
    IndexEmbeddingRequest,
    IndexEmbeddingResponse,
    ModelInfo,
    NBAActionRequest,
    NBAActionResponse,
    NBADecideRequest,
    NBADecideResponse,
    Offer,
    RecommendRequest,
    RecommendResponse,
    SeedResponse,
    SimilarCustomer,
    UploadJobStatus,
    UploadStartResponse,
    VectorStoreCompareRequest,
    VectorStoreInfo,
    VectorStoreSwitchRequest,
)
from nbx.llms import llm_client, ollama_client, ranker
from nbx.agents import final_agent, ingestion_pipeline, nba_agent, nba_ingestion_pipeline, rules_engine
from nbx.agents.agentic import copilot, council
from nbx.agents.nbo_agent import run_pipeline
from nbx.agents.final_agent import decide_next_best_action
from nbx.agents.retriever import HybridRetriever
from nbx.vectorstores import get_store, try_get_store

app = FastAPI(title="NBO Agentic RAG API", version="1.0.0")

from nbx.api import fsdr as fsdr_api  # noqa: E402  (FSDR routes, /fsdr/*)

app.include_router(fsdr_api.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {"current_store": config.DEFAULT_VECTORSTORE}
_api_log: list[dict] = []
_MAX_LOG = 20


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)
    _api_log.append(
        {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    while len(_api_log) > _MAX_LOG:
        _api_log.pop(0)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response


def _load_json(path):
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return await readiness.compute_health()


@app.get("/models", response_model=list[ModelInfo])
async def get_models():
    installed = await llm_client.list_installed_models()
    return [
        ModelInfo(**m, available=llm_client.model_is_installed(installed, m["name"]))
        for m in config.OLLAMA_MODELS
    ]


@app.get("/embeddings", response_model=list[EmbeddingModelInfo])
async def get_embeddings():
    installed = await ollama_client.list_installed_models() if config.OLLAMA_ENABLED else []
    out = []
    for e in config.EMBEDDING_MODELS:
        if e["backend"] == "sentence-transformers":
            available = True  # downloaded on first use from the HF cache, not gated on Ollama
        else:
            available = ollama_client._model_is_installed(installed, e["model_name"])
        out.append(
            EmbeddingModelInfo(
                key=e["key"],
                display=e["display"],
                backend=e["backend"],
                dimension=e["dimension"],
                strength=e["strength"],
                recommended=e["recommended"],
                available=available,
            )
        )
    return out


@app.get("/graph/stats", response_model=GraphStatsResponse)
async def graph_stats():
    return knowledge_graph.get_stats()


@app.get("/graph/cohort-offers/{msisdn}", response_model=list[CohortOfferHit])
async def graph_cohort_offers(msisdn: int):
    customers = _load_json(config.CUSTOMERS_FILE)
    customer = next((c for c in customers if c["msisdn"] == msisdn), None)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"No customer found for msisdn={msisdn}")
    return knowledge_graph.cohort_offers(customer, top_k=10)


@app.get("/graph/similar-customers/{msisdn}", response_model=list[SimilarCustomer])
async def graph_similar_customers(msisdn: int):
    return knowledge_graph.similar_customers(msisdn, top_k=10)


@app.get("/cf/stats")
async def cf_stats():
    """Collaborative-filtering model health: item universe (monthly,
    non-voice offers), how many customers have a trained embedding, how many
    90-day interactions the model was trained on. See core/collab_filter.py."""
    return collab_filter.get_stats()


@app.get("/cf/score/{msisdn}")
async def cf_score(msisdn: int):
    customers = _load_json(config.CUSTOMERS_FILE)
    customer = next((c for c in customers if c["msisdn"] == msisdn), None)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"No customer found for msisdn={msisdn}")
    return collab_filter.score_customer(customer, top_k=10)


@app.get("/nba/bandit/stats")
async def nba_bandit_stats():
    """Contextual bandit health: action distribution explored during
    training, cumulative avg_reward (fraction of exploratory picks that
    matched the customer's historical action), how many labeled takers it
    trained on. See core/nba_bandit.py."""
    return nba_bandit.get_stats()


@app.get("/nba/bandit/context/{msisdn}")
async def nba_bandit_context(msisdn: int):
    row = nba_bandit.get_score_row(msisdn)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No NBA score simulated for msisdn={msisdn}")
    return row


@app.get("/vectorstores", response_model=list[VectorStoreInfo])
async def get_vectorstores():
    out = []
    for cfg in config.VECTOR_STORES:
        store = try_get_store(cfg["key"])
        if store is None:
            out.append(
                VectorStoreInfo(
                    key=cfg["key"], display=cfg["display"], status="unavailable",
                    document_count=0, index_size_bytes=0,
                )
            )
            continue
        stats = store.get_stats()
        out.append(
            VectorStoreInfo(
                key=cfg["key"],
                display=cfg["display"],
                status=stats.get("status", "ok"),
                document_count=stats.get("count", 0),
                index_size_bytes=stats.get("size_bytes", 0),
            )
        )
    return out


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(body: RecommendRequest):
    try:
        result = await run_pipeline(
            msisdn=body.msisdn,
            llm=body.llm,
            vectorstore=body.vectorstore or _state["current_store"],
            embedding_model=body.embedding_model,
            retrieval_mode=body.retrieval_mode,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
    return result


@app.post("/nba/decide", response_model=NBADecideResponse)
async def nba_decide(body: NBADecideRequest):
    """The FINAL decision: agents/final_agent.py composes agents/nba_agent.py
    (bandit + LLM -> action) and agents/nbo_agent.py (retrieval/CF/graph +
    LLM -> offers) sequentially -- NBA's action steers NBO's need signal
    when it names a category -- then applies agents/rules_engine.py. For NBA
    alone (no NBO, no rules), see POST /nba/action."""
    try:
        result = await decide_next_best_action(
            msisdn=body.msisdn,
            llm=body.llm,
            vectorstore=body.vectorstore or _state["current_store"],
            embedding_model=body.embedding_model,
            retrieval_mode=body.retrieval_mode,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NBA pipeline failed: {e}")
    return result


@app.post("/nba/action", response_model=NBAActionResponse)
async def nba_action(body: NBAActionRequest):
    """NBA alone: the contextual bandit (core/nba_bandit.py) plus its LLM
    confirm-or-override step (agents/nba_agent.py) -- no NBO call, no
    business rules. Useful for testing/inspecting NBA's own decision in
    isolation from the combined POST /nba/decide."""
    try:
        result = await nba_agent.decide_action(msisdn=body.msisdn, llm=body.llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NBA action failed: {e}")
    return result


@app.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark(body: BenchmarkRequest):
    try:
        result = await ranker.run_benchmark(
            msisdn=body.msisdn, vectorstore=body.vectorstore or _state["current_store"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {e}")
    return result


@app.post("/seed", response_model=SeedResponse)
async def seed():
    from nbx.scripts.seed_data import seed_all

    try:
        result = seed_all()
        knowledge_graph.get_graph(force_rebuild=True)
        collab_filter.get_model(force_rebuild=True)
        nba_bandit.get_bandit(force_rebuild=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Seeding failed: {e}")
    return result


@app.post("/upload", response_model=UploadStartResponse)
async def upload(
    background_tasks: BackgroundTasks,
    customers: UploadFile = File(...),
    offers: UploadFile = File(...),
    purchases: UploadFile = File(...),
    segments: UploadFile = File(...),
):
    """Starts the step-by-step ingestion pipeline (agents/ingestion_pipeline.py)
    on 4 uploaded JSON files, replacing whatever data is currently seeded.
    Runs in the background; poll GET /upload/status/{job_id} for progress.
    Does not touch /seed or scripts/seed_data.py's standalone behavior."""
    files = {
        "customers": await customers.read(),
        "offers": await offers.read(),
        "purchases": await purchases.read(),
        "segments": await segments.read(),
    }
    job_id = ingestion_pipeline.start_job(files)
    background_tasks.add_task(ingestion_pipeline.run_job, job_id, files)
    return {"job_id": job_id}


@app.get("/upload/status/{job_id}", response_model=UploadJobStatus)
async def upload_status(job_id: str):
    job = ingestion_pipeline.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job found for job_id={job_id}")
    return {
        "job_id": job.job_id,
        "overall_status": job.overall_status.value,
        "steps": [
            {
                "name": s.name,
                "display": ingestion_pipeline.STEP_DISPLAY[s.name],
                "status": s.status.value,
                "message": s.message,
                "count": s.count,
                "error": s.error,
            }
            for s in job.steps
        ],
    }


@app.post("/upload/retry/{job_id}", response_model=UploadStartResponse)
async def upload_retry(
    job_id: str,
    background_tasks: BackgroundTasks,
    customers: UploadFile | None = File(None),
    offers: UploadFile | None = File(None),
    purchases: UploadFile | None = File(None),
    segments: UploadFile | None = File(None),
):
    """Re-runs a failed job from its first failed step onward, not from the
    beginning. Only pass files back in if the failed step was "Upload &
    validate" (nothing was written to disk yet, so there's nothing to
    resume from) -- every later step just re-runs against the data already
    on disk from a prior successful "Parse & normalize" step."""
    job = ingestion_pipeline.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job found for job_id={job_id}")

    failed_step = ingestion_pipeline.first_failed_step(job)
    if failed_step is None:
        raise HTTPException(status_code=400, detail="Job has no failed step to retry.")

    files = None
    if any([customers, offers, purchases, segments]):
        files = {
            "customers": await customers.read() if customers else b"",
            "offers": await offers.read() if offers else b"",
            "purchases": await purchases.read() if purchases else b"",
            "segments": await segments.read() if segments else b"",
        }
    if failed_step == ingestion_pipeline.STEP_NAMES[0] and not files:
        raise HTTPException(
            status_code=400,
            detail="Retrying the Upload & validate step requires re-uploading all 4 files.",
        )

    background_tasks.add_task(ingestion_pipeline.retry_job, job_id, files)
    return {"job_id": job_id}


@app.post("/upload/quick", response_model=UploadStartResponse)
async def upload_quick(background_tasks: BackgroundTasks):
    """Same pipeline as POST /upload, but reads the 4 files straight from
    disk (config.CUSTOMERS_FILE etc.) instead of a browser upload -- for
    re-validating/re-indexing data that's already on this machine without
    dragging files into the picker. 404s with exactly which file is missing
    if POST /seed hasn't been run yet."""
    files = {
        "customers": config.CUSTOMERS_FILE.read_bytes() if config.CUSTOMERS_FILE.exists() else b"",
        "offers": config.OFFERS_FILE.read_bytes() if config.OFFERS_FILE.exists() else b"",
        "purchases": config.PURCHASES_FILE.read_bytes() if config.PURCHASES_FILE.exists() else b"",
        "segments": config.SEGMENTS_FILE.read_bytes() if config.SEGMENTS_FILE.exists() else b"",
    }
    missing = [k for k, v in files.items() if not v]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Missing on-disk file(s): {', '.join(f + '.json' for f in missing)}. Run POST /seed first.",
        )
    job_id = ingestion_pipeline.start_job(files)
    background_tasks.add_task(ingestion_pipeline.run_job, job_id, files)
    return {"job_id": job_id}


@app.post("/nba/upload", response_model=UploadStartResponse)
async def nba_upload(
    background_tasks: BackgroundTasks,
    churn: UploadFile = File(...),
    reactivation: UploadFile = File(...),
    appetency: UploadFile = File(...),
    fragility: UploadFile = File(...),
    revenue_decrease: UploadFile = File(...),
    activity_usage: UploadFile = File(...),
    targeting: UploadFile = File(...),
):
    """Starts the NBA-side ingestion pipeline (agents/nba_ingestion_pipeline.py)
    on 7 uploaded JSON files -- split to mirror the production script's own
    per-Dataiku-dataset boundaries, see core.config.NBA_SOURCE_FILE_ATTRS --
    replacing whatever NBA data is currently on disk and retraining the
    contextual bandit. Runs in the background; poll
    GET /nba/upload/status/{job_id} for progress. Does not touch POST /seed's
    synthetic generation."""
    files = {
        "churn": await churn.read(),
        "reactivation": await reactivation.read(),
        "appetency": await appetency.read(),
        "fragility": await fragility.read(),
        "revenue_decrease": await revenue_decrease.read(),
        "activity_usage": await activity_usage.read(),
        "targeting": await targeting.read(),
    }
    job_id = nba_ingestion_pipeline.start_job()
    background_tasks.add_task(nba_ingestion_pipeline.run_job, job_id, files)
    return {"job_id": job_id}


@app.get("/nba/upload/status/{job_id}", response_model=UploadJobStatus)
async def nba_upload_status(job_id: str):
    job = nba_ingestion_pipeline.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job found for job_id={job_id}")
    return {
        "job_id": job.job_id,
        "overall_status": job.overall_status.value,
        "steps": [
            {
                "name": s.name,
                "display": nba_ingestion_pipeline.STEP_DISPLAY[s.name],
                "status": s.status.value,
                "message": s.message,
                "count": s.count,
                "error": s.error,
            }
            for s in job.steps
        ],
    }


@app.post("/nba/upload/retry/{job_id}", response_model=UploadStartResponse)
async def nba_upload_retry(
    job_id: str,
    background_tasks: BackgroundTasks,
    churn: UploadFile | None = File(None),
    reactivation: UploadFile | None = File(None),
    appetency: UploadFile | None = File(None),
    fragility: UploadFile | None = File(None),
    revenue_decrease: UploadFile | None = File(None),
    activity_usage: UploadFile | None = File(None),
    targeting: UploadFile | None = File(None),
):
    """Re-runs a failed NBA upload job from its first failed step onward.
    Only pass files back in if the failed step was "Upload & validate" --
    every later step just re-runs against the files already on disk from a
    prior successful "Parse & normalize" step."""
    job = nba_ingestion_pipeline.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job found for job_id={job_id}")

    failed_step = nba_ingestion_pipeline.first_failed_step(job)
    if failed_step is None:
        raise HTTPException(status_code=400, detail="Job has no failed step to retry.")

    uploads = {"churn": churn, "reactivation": reactivation, "appetency": appetency, "fragility": fragility,
               "revenue_decrease": revenue_decrease, "activity_usage": activity_usage, "targeting": targeting}
    files = None
    if any(uploads.values()):
        files = {k: (await v.read() if v else b"") for k, v in uploads.items()}
    if failed_step == nba_ingestion_pipeline.STEP_NAMES[0] and not files:
        raise HTTPException(
            status_code=400,
            detail="Retrying the Upload & validate step requires re-uploading all 7 files.",
        )

    background_tasks.add_task(nba_ingestion_pipeline.retry_job, job_id, files)
    return {"job_id": job_id}


@app.post("/nba/upload/quick", response_model=UploadStartResponse)
async def nba_upload_quick(background_tasks: BackgroundTasks):
    """Same pipeline as POST /nba/upload, but reads the 7 NBA source files
    straight from disk (core.config.NBA_SOURCE_FILE_ATTRS) instead of a
    browser upload -- for re-validating/retraining against data that's
    already on this machine without dragging 7 files into pickers."""
    files, missing = {}, []
    for key, attr in config.NBA_SOURCE_FILE_ATTRS.items():
        path = getattr(config, attr)
        if path.exists():
            files[key] = path.read_bytes()
        else:
            missing.append(key)
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Missing on-disk file(s): {', '.join(f + '.json' for f in missing)}. Run POST /seed first.",
        )
    job_id = nba_ingestion_pipeline.start_job()
    background_tasks.add_task(nba_ingestion_pipeline.run_job, job_id, files)
    return {"job_id": job_id}


@app.get("/rules")
async def get_rules():
    """Business rules for both agents, read live from the running code, not
    a static description: NBO's agents/rules_engine.py (voice exclusion,
    affordability priority) and NBA's policy (core.config's segment->action
    map, propensity table, exclusions) plus agents/final_agent.py's
    category-steering map between them."""
    return {
        "nbo_rules": [{"rule_name": r.name, "description": r.description} for r in rules_engine.RULES],
        "affordability_band_omr": config.AFFORDABILITY_BAND_OMR,
        "nba_policy": {
            "actions": config.NBA_ACTIONS,
            "segment_action_map": config.NBA_SEGMENT_ACTION_MAP,
            "action_propensity": config.NBA_ACTION_PROPENSITY,
            "excluded_segments": config.NBA_EXCLUDED_SEGMENTS,
            "non_push_actions": sorted(config.NBA_NON_PUSH_ACTIONS),
            "reward_window_days": config.NBA_REWARD_WINDOW_DAYS,
            "vw_args": config.NBA_VW_ARGS,
        },
        "action_need_override": final_agent.ACTION_NEED_OVERRIDE,
    }


@app.post("/seed/embedding", response_model=IndexEmbeddingResponse)
async def seed_embedding(body: IndexEmbeddingRequest):
    """Indexes the current data into all 5 stores under an additional
    embedding model, without regenerating data or touching other models'
    collections. Use this to bring a Qwen3-Embedding variant online after
    pulling it in Ollama."""
    if body.embedding_model not in config.EMBEDDING_MODEL_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown embedding model '{body.embedding_model}'")
    from nbx.scripts.seed_data import index_embedding_model

    try:
        indexed = index_embedding_model(body.embedding_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    return {"embedding_model": body.embedding_model, "indexed": indexed}


@app.get("/customer/{msisdn}")
async def get_customer(msisdn: int):
    customers = _load_json(config.CUSTOMERS_FILE)
    customer = next((c for c in customers if c["msisdn"] == msisdn), None)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"No customer found for msisdn={msisdn}")

    purchases = _load_json(config.PURCHASES_FILE)
    history = [p for p in purchases if p["msisdn"] == msisdn]
    history.sort(key=lambda p: p["purchase_date"], reverse=True)

    return {"customer": customer, "purchase_history": history}


@app.get("/offers", response_model=list[Offer])
async def get_offers(
    category: str | None = None,
    max_price: float | None = Query(default=None),
    is_data: bool | None = None,
    is_voice: bool | None = None,
):
    offers = _load_json(config.OFFERS_FILE)
    if category:
        offers = [o for o in offers if o["category"].lower() == category.lower()]
    if max_price is not None:
        offers = [o for o in offers if o["price_omr"] <= max_price]
    if is_data is not None:
        offers = [o for o in offers if o["is_data"] == is_data]
    if is_voice is not None:
        offers = [o for o in offers if o["is_voice"] == is_voice]
    return offers


@app.post("/vectorstore/switch")
async def switch_vectorstore(body: VectorStoreSwitchRequest):
    if body.store not in config.VECTOR_STORE_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown store '{body.store}'")
    store = try_get_store(body.store)
    if store is None:
        raise HTTPException(status_code=503, detail=f"Store '{body.store}' failed to initialize")
    _state["current_store"] = body.store
    return {"store": body.store, "status": "switched", "stats": store.get_stats()}


@app.post("/vectorstore/compare")
async def compare_vectorstores(body: VectorStoreCompareRequest):
    rows = []
    for cfg in config.VECTOR_STORES:
        key = cfg["key"]
        store = try_get_store(key)
        if store is None:
            rows.append({"store": key, "results": [], "time_ms": 0, "top_result": None, "status": "unavailable"})
            continue
        start = time.time()
        try:
            results = HybridRetriever(store).retrieve([body.query], top_k=body.top_k)
        except Exception as e:
            rows.append({"store": key, "results": [], "time_ms": 0, "top_result": None, "status": f"error: {e}"})
            continue
        elapsed_ms = int((time.time() - start) * 1000)
        rows.append(
            {
                "store": key,
                "results": results,
                "time_ms": elapsed_ms,
                "top_result": results[0]["id"] if results else None,
                "status": "ok",
            }
        )
    return {"query": body.query, "top_k": body.top_k, "rows": rows}


@app.get("/logs")
async def get_logs():
    return {"logs": list(reversed(_api_log))}


# ---------------------------------------------------------------------------
# Agentic NBA/NBO council, copilot, and portfolio read-models (UI)
# ---------------------------------------------------------------------------
_AGENTIC_SCOPES = ("nba", "nbo", "full")


def _check_scope(scope: str) -> None:
    if scope not in _AGENTIC_SCOPES:
        raise HTTPException(status_code=400, detail=f"scope must be one of {_AGENTIC_SCOPES}")


@app.post("/agentic/decide")
async def agentic_decide(body: AgenticRequest):
    _check_scope(body.scope)
    try:
        return await council.run_council(body.msisdn, body.scope, body.llm, use_last_nba=body.use_last_nba)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/agentic/stream")
async def agentic_stream(msisdn: int, scope: str = "full", llm: str | None = None, use_last_nba: bool = False):
    """Server-Sent Events: one `data: {json}` line per council event
    (run_start, agent_start, thought, tool_call, tool_result, policy,
    guardrails, revision, auto_fix, agent_final, run_end | error)."""
    _check_scope(scope)
    if insights.customer_360(msisdn) is None:
        raise HTTPException(status_code=404, detail=f"No customer found for msisdn={msisdn}")
    queue: asyncio.Queue = asyncio.Queue()

    async def _run():
        try:
            await council.run_council(msisdn, scope, llm, emit=queue.put_nowait, use_last_nba=use_last_nba)
        except Exception as e:  # surfaced to the UI instead of a silently dead stream
            queue.put_nowait({"type": "error", "detail": f"{type(e).__name__}: {e}"})
        finally:
            queue.put_nowait(None)

    async def _events():
        task = asyncio.create_task(_run())
        try:
            while (ev := await queue.get()) is not None:
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        finally:
            task.cancel()

    return StreamingResponse(_events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/agentic/recent")
async def agentic_recent():
    return list(council.RECENT_RUNS)


@app.post("/copilot/chat")
async def copilot_chat(body: CopilotRequest):
    return await copilot.ask(body.question, body.llm, body.history)


@app.get("/customers")
async def customers():
    return insights.customers_list()


@app.get("/customers/{msisdn}/360")
async def customer_360(msisdn: int):
    c = insights.customer_360(msisdn)
    if c is None:
        raise HTTPException(status_code=404, detail=f"No customer found for msisdn={msisdn}")
    return c


@app.get("/portfolio/summary")
async def portfolio_summary():
    return {**insights.portfolio_summary(), "recent_decisions": list(council.RECENT_RUNS)[:8]}


# ---------------------------------------------------------------------------
# Competitor intelligence (COMPETITOR_A / COMPETITOR_B public price pages)
# ---------------------------------------------------------------------------
@app.get("/competitors/offers")
async def competitor_offers(operator: str | None = None, segment: str | None = None):
    rows = competitors.load_offers()
    if operator:
        rows = [r for r in rows if r["operator"].lower() == operator.lower()]
    if segment:
        rows = [r for r in rows if r.get("segment") == segment]
    return rows


@app.get("/competitors/search")
async def competitor_search(q: str, top_k: int = 5, operator: str | None = None):
    return competitors.search(q, top_k=top_k, operator=operator)


@app.get("/competitors/compare/{program_rule}")
async def competitor_compare(program_rule: str):
    offers = {o["program_rule"]: o for o in _load_json(config.OFFERS_FILE)}
    if program_rule not in offers:
        raise HTTPException(status_code=404, detail=f"Unknown offer {program_rule}")
    return competitors.compare_offer(offers[program_rule], per_operator=2)


@app.post("/competitors/refresh")
async def competitor_refresh():
    """Re-scrapes the public pages (about 25 s: one request every 1.5 s) and
    rebuilds the competitor collection. Runs in a worker thread."""
    return await asyncio.to_thread(competitors.refresh)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("nbx.api.main:app", host="0.0.0.0", port=config.FASTAPI_PORT, reload=True)
