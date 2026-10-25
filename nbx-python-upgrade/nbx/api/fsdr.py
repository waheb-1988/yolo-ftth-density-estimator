"""FSDR API routes (mounted by nbx/api/main.py under /fsdr).

Same style as main.py: thin handlers over the FSDR modules, HTTPException with
a `detail` message on bad input, long steps run in a worker thread so the
event loop (and the SSE stream) stays responsive.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nbx.fsdr import config, registry, schemas
from nbx.fsdr.calendar import anchor_date

router = APIRouter(prefix="/fsdr", tags=["fsdr"])

_workspaces: dict[tuple[str, float], object] = {}


def _run_dir(run: str) -> Path:
    try:
        date.fromisoformat(run)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"run must be a date YYYY-MM-DD, got '{run}'")
    folder = config.FSDR_DATA_DIR / run
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"No FSDR run folder {run}. Generate or upload data first.")
    return folder


def _workspace(run: str):
    """One loaded FsdrWorkspace per run, rebuilt when the scoring output changes."""
    from nbx.fsdr.agents.tools import FsdrWorkspace

    folder = _run_dir(run)
    scored = folder / "outputs" / "results_scoring_nba.json"
    if not scored.exists():
        raise HTTPException(status_code=409, detail="No scoring output for this run: train and score first.")
    key = (run, scored.stat().st_mtime)
    if key not in _workspaces:
        for k in [k for k in _workspaces if k[0] == run]:
            _workspaces.pop(k)
        _workspaces[key] = FsdrWorkspace(folder)
    return _workspaces[key]


def _fail(e: Exception):
    """Pipeline exceptions -> HTTP errors with the original message."""
    from nbx.fsdr.bandit import IncompatibleModel
    from nbx.fsdr.pipelines.score import StaleModel
    from nbx.fsdr.pipelines.train import DataBlocked

    if isinstance(e, HTTPException):
        raise e
    if isinstance(e, (DataBlocked, StaleModel, IncompatibleModel)):
        raise HTTPException(status_code=409, detail=str(e))
    if isinstance(e, (FileNotFoundError, LookupError)):
        raise HTTPException(status_code=404, detail=str(e))
    if isinstance(e, ValueError):
        raise HTTPException(status_code=400, detail=str(e))
    raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Runs and data
# ---------------------------------------------------------------------------
@router.get("/runs")
async def list_runs():
    out = []
    if config.FSDR_DATA_DIR.exists():
        for p in sorted((p for p in config.FSDR_DATA_DIR.glob("20??-??-??") if p.is_dir()), reverse=True):
            m = p / config.MANIFEST_FILE
            manifest = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}
            v = p / config.VALIDATION_JSON
            outputs = sorted(f.stem for f in (p / "outputs").glob("*.json")) if (p / "outputs").exists() else []
            out.append({"run": p.name, "anchor_date": anchor_date(date.fromisoformat(p.name)).isoformat(),
                        "source": manifest.get("source", "export"), "n": (manifest.get("generator") or {}).get("n"),
                        "scenarios": (manifest.get("generator") or {}).get("scenarios"),
                        "validation": json.loads(v.read_text(encoding="utf-8"))["status"] if v.exists() else None,
                        "trained": "nba_training_report" in outputs, "scored": "results_scoring_nba" in outputs,
                        "arms_recommended": [a for a in ("cstr", "free", "llm") if f"recos_monthly_{a}" in outputs]})
    return out


class GenerateRequest(BaseModel):
    date: str | None = None
    n: int = Field(config.GENERATOR_DEFAULT_N, ge=50, le=config.GENERATOR_MAX_N)
    scenarios: list[str] = Field(default_factory=lambda: ["baseline"])
    seed: int = config.GENERATOR_SEED
    force: bool = False


@router.post("/generate")
async def generate(body: GenerateRequest):
    from nbx.fsdr.synth import generator, scenarios

    run_date = date.fromisoformat(body.date) if body.date else date.today()
    out = config.FSDR_DATA_DIR / run_date.isoformat()
    m = out / config.MANIFEST_FILE
    if out.exists() and any(out.iterdir()) and not body.force and not (
            m.exists() and json.loads(m.read_text(encoding="utf-8")).get("source") == "generator"):
        raise HTTPException(status_code=409, detail=f"{out.name} holds data not produced by the generator; set force to overwrite")
    try:
        sc = scenarios.parse(body.scenarios)
        run = await asyncio.to_thread(generator.generate, body.n, run_date, sc, body.seed, out)
    except Exception as e:  # noqa: BLE001
        _fail(e)
    return {"run": run_date.isoformat(), "anchor_date": run.anchor_date.isoformat(), "row_counts": run.manifest["row_counts"],
            "scenarios": sc, "seconds": run.manifest["generation_seconds"]}


@router.post("/validate/{run}")
async def validate_run(run: str):
    from nbx.fsdr.validation import validate_folder

    rep = await asyncio.to_thread(validate_folder, _run_dir(run))
    return {**rep.to_dict(), "markdown": rep.to_markdown()}


@router.get("/tables/{run}")
async def list_tables(run: str):
    folder = _run_dir(run)
    return [{"table": n, "group": s.group, "present": any((folder / f"{n}.{ext}").exists() for ext in ("json", "jsonl"))}
            for n, s in schemas.TABLES.items()]


@router.get("/tables/{run}/{table}")
async def preview_table(run: str, table: str, limit: int = Query(20, ge=1, le=500)):
    from nbx.fsdr.datasource import JsonSource

    src = JsonSource(_run_dir(run))
    if not src.has(table):
        raise HTTPException(status_code=404, detail=f"No table {table} in run {run}")
    df = src.read(table)
    return {"table": table, "rows": len(df), "columns": list(df.columns), "records": df.head(limit).to_dict("records")}


# ---------------------------------------------------------------------------
# Training, scoring, models
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    mode: str = config.TRAIN_MODE
    reward: str = config.REWARD_DEFINITION
    passes: int = Field(config.TRAIN_PASSES, ge=config.TRAIN_PASSES_RANGE[0], le=config.TRAIN_PASSES_RANGE[1])
    epsilon: float = config.VW_EPSILON
    promote: str = "auto"


@router.post("/train/{run}")
async def train(run: str, body: TrainRequest):
    from nbx.fsdr.pipelines.train import run_training

    try:
        return await asyncio.to_thread(run_training, _run_dir(run), None, body.mode, body.reward, body.passes, body.epsilon,
                                       config.HOLDOUT_SHARE, config.GENERATOR_SEED, body.promote)
    except Exception as e:  # noqa: BLE001
        _fail(e)


class ScoreRequest(BaseModel):
    population: str = "target"
    policy: str = config.SCORING_POLICY
    allow_stale: bool = False


@router.post("/score/{run}")
async def score(run: str, body: ScoreRequest):
    from nbx.fsdr.pipelines.score import run_scoring

    try:
        return await asyncio.to_thread(run_scoring, _run_dir(run), None, body.population, body.policy, None, None, None,
                                       body.allow_stale)
    except Exception as e:  # noqa: BLE001
        _fail(e)


@router.get("/models")
async def models():
    return registry.models()


@router.post("/models/{version}/promote")
async def promote(version: str, force: bool = False):
    try:
        return registry.promote(version, force=force, reason="promoted from the API")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/models/rollback")
async def rollback():
    try:
        return registry.rollback()
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ---------------------------------------------------------------------------
# Offers and the agentic recommender
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    arm: str
    mapping: str | None = None
    llm: str | None = None
    limit: int | None = None
    restart: bool = False


@router.post("/recommend/{run}")
async def recommend(run: str, body: RecommendRequest):
    folder = _run_dir(run)
    try:
        if body.arm == "llm":
            from nbx.fsdr.agents.recommender_agent import run_batch

            return await run_batch(folder, llm=body.llm, limit=body.limit, resume=not body.restart, ws=_workspace(run))
        if body.arm not in ("cstr", "free"):
            raise HTTPException(status_code=400, detail="arm must be cstr, free or llm")
        from nbx.fsdr.pipelines.recommend import run_recommend

        return await asyncio.to_thread(run_recommend, folder, body.arm, body.mapping)
    except Exception as e:  # noqa: BLE001
        _fail(e)


@router.get("/recos/{run}/{arm}")
async def recos(run: str, arm: str, limit: int = Query(200, ge=1, le=5000)):
    from nbx.fsdr.pipelines import read_table

    folder = _run_dir(run)
    rows = []
    for name in ("monthly", "weekly"):
        try:
            df = read_table(folder, f"recos_{name}_{arm}")
            rows += df.where(df.notna(), None).to_dict("records")
        except FileNotFoundError:
            pass
    if not rows:
        raise HTTPException(status_code=404, detail=f"No recommendations for arm {arm} in run {run}")
    rows.sort(key=lambda r: (str(r.get("ACCESS_METHOD_VAL")), -(r.get("VALIDITY") or 0)))
    return {"arm": arm, "rows": len(rows), "records": rows[:limit]}


@router.get("/customers/{run}")
async def customers(run: str, arm: str = "llm", limit: int = Query(300, ge=1, le=5000)):
    ws = _workspace(run)
    out = []
    for v in ws.lines(arm)[:limit]:
        ctx = ws.context(v, arm)
        out.append({"msisdn": v, "action": ctx.scored["ACTION"], "churn": round(ctx.churn(), 3), "segment": ctx.segment()})
    return out


class BatchRequest(BaseModel):
    msisdns: list[str] | None = None        # None = the first `limit` customers of the arm
    arm: str = "llm"
    llm: str | None = None
    limit: int = Field(config.BATCH_SIZE, ge=1, le=config.BATCH_MAX)


@router.post("/batch/{run}")
async def batch(run: str, body: BatchRequest):
    """One batch: action -> top 4 offers -> top 2 roaming -> international offers (if the customer calls abroad)."""
    from nbx.fsdr.pipelines.batch import flat_row, recommend_batch

    ws = _workspace(run)
    if body.arm not in ("llm", "cstr", "free"):
        raise HTTPException(status_code=400, detail="arm must be llm, cstr or free")
    msisdns = body.msisdns if body.msisdns else ws.lines(body.arm)[: body.limit]
    try:
        rows = await recommend_batch(ws, [str(m) for m in msisdns], body.arm, body.llm)
    except Exception as e:  # noqa: BLE001
        _fail(e)
    return {"run": run, "arm": body.arm, "count": len(rows), "rows": rows, "table": [flat_row(r) for r in rows]}


class DecideRequest(BaseModel):
    msisdn: str
    llm: str | None = None


@router.post("/decide/{run}")
async def decide(run: str, body: DecideRequest):
    from nbx.fsdr.agents.recommender_agent import decide as run_decide

    try:
        return await run_decide(_workspace(run), body.msisdn, "llm", body.llm)
    except Exception as e:  # noqa: BLE001
        _fail(e)


@router.get("/stream/{run}")
async def stream(run: str, msisdn: str, llm: str | None = None):
    """Server-Sent Events, one `data: {json}` per agent event (same pattern as /agentic/stream)."""
    from nbx.fsdr.agents.recommender_agent import decide as run_decide

    ws = _workspace(run)
    try:
        ws.context(msisdn)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    queue: asyncio.Queue = asyncio.Queue()

    async def _run():
        try:
            out = await run_decide(ws, msisdn, "llm", llm, emit=queue.put_nowait)
            queue.put_nowait({"type": "decision", "decision": out["decision"]})
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


# ---------------------------------------------------------------------------
# Real JSON upload, campaign log, feedback, weekly report, simulation
# ---------------------------------------------------------------------------
@router.post("/upload/{run}")
async def upload(run: str, files: list[UploadFile] = File(...), force: bool = False):
    """Real exports: one <table>.json / .jsonl per table into data/fsdr/<run>/, then the validation report."""
    from nbx.fsdr.validation import validate_folder

    try:
        date.fromisoformat(run)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"run must be a date YYYY-MM-DD, got '{run}'")
    folder = config.FSDR_DATA_DIR / run
    m = folder / config.MANIFEST_FILE
    if folder.exists() and m.exists() and json.loads(m.read_text(encoding="utf-8")).get("source") == "generator" and not force:
        raise HTTPException(status_code=409, detail=f"{run} holds generated data; set force to replace it with real files")
    names = [Path(f.filename or "").name for f in files]
    wrong = [n for n in names if n.rsplit(".", 1)[-1].lower() not in ("json", "jsonl")]
    if wrong:                                            # checked before anything is written: no half-made run
        raise HTTPException(status_code=400, detail=f"only .json or .jsonl files: {', '.join(wrong)}")
    folder.mkdir(parents=True, exist_ok=True)
    saved, unknown = [], []
    for f, name in zip(files, names):
        stem = name.rsplit(".", 1)[0]
        (folder / name).write_bytes(await f.read())
        (saved if schemas.TABLE_ALIASES.get(stem, stem) in schemas.TABLES else unknown).append(name)
    if m.exists() and json.loads(m.read_text(encoding="utf-8")).get("source") == "generator":
        m.unlink()                                      # the generated manifest no longer describes the folder
    rep = await asyncio.to_thread(validate_folder, folder)
    return {"run": run, "saved": saved, "unknown_files": unknown, "validation": {**rep.to_dict(), "markdown": rep.to_markdown()}}


@router.get("/campaign-log")
async def get_campaign_log(start: str | None = None, end: str | None = None, arm: str | None = None,
                           limit: int = Query(500, ge=1, le=50_000)):
    """Append-only campaign log (newest first): what was sent to whom, with the logged propensity."""
    from nbx.fsdr import campaign_log

    root = campaign_log.root_of()
    try:
        log = campaign_log.read_log(root, date.fromisoformat(start) if start else None, date.fromisoformat(end) if end else None, arm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    rows = json.loads(log.iloc[::-1].head(limit).to_json(orient="records")) if len(log) else []
    return {"rows": rows, "total": int(len(log)), "per_arm": log["arm"].value_counts().to_dict() if len(log) else {}}


@router.post("/feedback/{run}")
async def feedback(run: str):
    from nbx.fsdr.pipelines.feedback import run_feedback

    try:
        return await asyncio.to_thread(run_feedback, _run_dir(run))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:  # noqa: BLE001
        _fail(e)


@router.get("/report/{week}")
async def report(week: str):
    """Weekly report of the campaign run on `week`: take rate and revenue per arm, uplift vs control."""
    from nbx.fsdr import campaign_log
    from nbx.fsdr.pipelines.feedback import weekly_report

    try:
        d = date.fromisoformat(week)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"week must be a date YYYY-MM-DD, got '{week}'")
    rep = weekly_report(campaign_log.root_of(), d)
    if not rep["logged_lines"]:
        raise HTTPException(status_code=404, detail=f"No campaign logged in the week of {week}.")
    return rep


@router.post("/agents/{kind}/{run}")
async def ops_agent(kind: str, run: str, llm: str | None = None):
    """Data / training / scoring / monitor agent (LLM-driven, rule-based without an LLM; guards in code)."""
    from nbx.fsdr.agents.ops_agents import OPS_AGENTS, run_ops_agent

    if kind not in OPS_AGENTS:
        raise HTTPException(status_code=400, detail=f"agent must be one of {OPS_AGENTS}")
    try:
        return await run_ops_agent(kind, _run_dir(run), llm=llm)
    except Exception as e:  # noqa: BLE001
        _fail(e)


class SimulateRequest(BaseModel):
    weeks: int = Field(3, ge=1, le=12)
    n: int = Field(2000, ge=100, le=config.GENERATOR_MAX_N)
    start: str | None = None
    scenarios: str = "baseline"
    arms: list[str] = Field(default_factory=lambda: ["free", "cstr", "llm"])
    seed: int = config.GENERATOR_SEED


@router.post("/simulate")
async def simulate(body: SimulateRequest):
    """Closed-loop simulation in data/fsdr/simulation/ (its own runs, models and campaign log)."""
    from nbx.fsdr.pipelines.simulate import run_simulation

    try:
        start = date.fromisoformat(body.start) if body.start else date.today()
        return await asyncio.to_thread(run_simulation, config.FSDR_DATA_DIR / "simulation", start, body.weeks, body.n,
                                       body.scenarios, body.arms, body.seed)
    except Exception as e:  # noqa: BLE001
        _fail(e)
