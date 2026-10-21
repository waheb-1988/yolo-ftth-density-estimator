"""Step-by-step data-upload ingestion pipeline: Upload & validate -> Parse &
normalize -> Vector indexing -> Knowledge graph build -> Readiness check.

Each step is its own plain function, independently callable and testable.
This module only orchestrates and reports progress -- it does not duplicate
business logic: step 3 calls scripts/seed_data.py's existing per-store
indexing loop, step 4 calls core/knowledge_graph.py's existing graph builder,
step 5 calls core/readiness.py's existing health check (the same one GET
/health uses). Nothing here regenerates synthetic data or touches
POST /seed / scripts/seed_data.py's standalone CLI behavior.

Progress is reported via polling (see main.py's GET /upload/status/{job_id})
rather than a streaming connection (SSE): every step here is a bounded local
operation (no long think-time to hide latency for), and a job's in-memory
JobState is naturally resumable across polls/reconnects, which a dropped SSE
stream is not. This matches the rest of the app's existing in-process,
single-worker state model (main.py's _state / _api_log, the graph's
_graph_cache) -- no new infrastructure (no DB, no task queue).

A note on step 3 (vector indexing) and step 5 (readiness): an individual
vector store failing to index, or Ollama being unreachable, does NOT fail
the step or the job -- both are reported as "done" with the real per-store /
per-service state visible in the message. This mirrors how GET /health and
GET /vectorstores already treat store/Ollama unavailability as "degraded",
not "down", elsewhere in this codebase; the ingestion pipeline doesn't invent
a stricter policy than the rest of the app already has for the exact same
failure modes.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from pydantic import ValidationError

from nbx.core import config, knowledge_graph, readiness
from nbx.core.models import Customer, Offer, Purchase, Segment
from nbx.scripts.seed_data import index_embedding_model


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


STEP_NAMES = [
    "upload_validate",
    "parse_normalize",
    "vector_indexing",
    "knowledge_graph",
    "readiness_check",
]

STEP_DISPLAY = {
    "upload_validate": "Upload & validate",
    "parse_normalize": "Parse & normalize",
    "vector_indexing": "Vector indexing",
    "knowledge_graph": "Knowledge graph build",
    "readiness_check": "Readiness check",
}

REQUIRED_FILES = ["customers", "offers", "purchases", "segments"]
MODEL_BY_FILE = {
    "customers": Customer,
    "offers": Offer,
    "purchases": Purchase,
    "segments": Segment,
}
_FINAL_PATH_BY_FILE = {
    "customers": "CUSTOMERS_FILE",
    "offers": "OFFERS_FILE",
    "purchases": "PURCHASES_FILE",
    "segments": "SEGMENTS_FILE",
}


# ---------------------------------------------------------------------------
# Step 1: Upload & validate
# ---------------------------------------------------------------------------
class UploadValidationError(Exception):
    """Raised with the *complete* list of problems found across all 4 files
    (not just the first one hit), per the "fail fast with a clear message
    listing exactly what's missing or malformed" requirement."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_upload(files: dict[str, bytes]) -> dict[str, list[dict]]:
    """Validates the 4 uploaded JSON files against core.models' pydantic
    schemas. Returns {file_key: [validated record dict, ...]} on success.

    "Malformed" here means: invalid JSON, a top-level value that isn't a
    JSON array, or a row that fails its pydantic model (missing/wrong-typed
    required field). Referential-integrity checks (e.g. every purchase's
    msisdn exists in customers) and business-rule validation (price >= 0,
    msisdn uniqueness) are NOT performed -- nothing in this codebase
    specifies them today. TODO: add if/when those rules are supplied.
    """
    errors: list[str] = []
    missing = [f for f in REQUIRED_FILES if not files.get(f)]
    if missing:
        errors.append(f"Missing required file(s): {', '.join(f + '.json' for f in missing)}")

    parsed: dict[str, list[dict]] = {}
    for key in REQUIRED_FILES:
        raw = files.get(key)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"{key}.json: invalid JSON ({e})")
            continue
        if not isinstance(data, list):
            errors.append(f"{key}.json: expected a JSON array of records, got {type(data).__name__}")
            continue

        model = MODEL_BY_FILE[key]
        rows: list[dict] = []
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                errors.append(f"{key}.json row {i}: expected an object, got {type(row).__name__}")
                continue
            try:
                rows.append(model(**row).model_dump())
            except ValidationError as e:
                for err in e.errors():
                    field_path = ".".join(str(x) for x in err["loc"])
                    errors.append(f"{key}.json row {i}: {err['msg']} (field: {field_path})")
        parsed[key] = rows

    if errors:
        raise UploadValidationError(errors)
    return parsed


# ---------------------------------------------------------------------------
# Step 2: Parse & normalize (write validated data to disk, atomically)
# ---------------------------------------------------------------------------
def write_files(parsed: dict[str, list[dict]]) -> dict[str, int]:
    """Writes the already-validated records to config.DATA_DIR under the
    same filenames scripts/seed_data.py's simulator uses, so every
    downstream step (indexing, graph build) sees no difference between
    uploaded and simulated data. Writes to temp files first and only
    replaces the real files once every file has serialized successfully, so
    a mid-write failure never leaves a half-updated dataset on disk."""
    tmp_paths: dict[str, Path] = {}
    try:
        for key, rows in parsed.items():
            final_path = getattr(config, _FINAL_PATH_BY_FILE[key])
            tmp_path = final_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            tmp_paths[key] = tmp_path

        for key, tmp_path in tmp_paths.items():
            final_path = getattr(config, _FINAL_PATH_BY_FILE[key])
            tmp_path.replace(final_path)
    finally:
        for tmp_path in tmp_paths.values():
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    return {key: len(rows) for key, rows in parsed.items()}


# ---------------------------------------------------------------------------
# Step 3: Vector indexing
# ---------------------------------------------------------------------------
def index_vectors(progress_cb: Optional[Callable[[str, int, Optional[str]], None]] = None) -> dict[str, int]:
    """Indexes the (now on-disk) data into all 5 stores under the default
    embedding model. Thin wrapper -- all indexing logic lives in
    scripts/seed_data.py::index_embedding_model, unchanged for its existing
    callers (seed_all, POST /seed/embedding, the CLI)."""
    return index_embedding_model(progress_cb=progress_cb)


# ---------------------------------------------------------------------------
# Step 4: Knowledge graph build
# ---------------------------------------------------------------------------
def build_graph_step() -> dict:
    graph = knowledge_graph.get_graph(force_rebuild=True)
    return knowledge_graph.get_stats(graph)


# ---------------------------------------------------------------------------
# Step 5: Readiness check
# ---------------------------------------------------------------------------
async def readiness_step() -> dict:
    return await readiness.compute_health()


# ---------------------------------------------------------------------------
# Job state + orchestration
# ---------------------------------------------------------------------------
@dataclass
class StepState:
    name: str
    status: StepStatus = StepStatus.PENDING
    message: str = ""
    count: Optional[int] = None
    error: Optional[str] = None


@dataclass
class JobState:
    job_id: str
    steps: list[StepState] = field(default_factory=lambda: [StepState(name=n) for n in STEP_NAMES])
    overall_status: StepStatus = StepStatus.PENDING

    def step(self, name: str) -> StepState:
        return next(s for s in self.steps if s.name == name)


_jobs: dict[str, JobState] = {}


def _set_running(step: StepState) -> None:
    step.status, step.message, step.error = StepStatus.RUNNING, "", None


def _set_done(step: StepState, message: str, count: Optional[int] = None) -> None:
    step.status, step.message, step.count = StepStatus.DONE, message, count


def _set_failed(step: StepState, error: str) -> None:
    step.status, step.error = StepStatus.FAILED, error


async def _execute(job: JobState, files: Optional[dict[str, bytes]], start_from: str) -> None:
    start_idx = STEP_NAMES.index(start_from)
    job.overall_status = StepStatus.RUNNING
    parsed: Optional[dict[str, list[dict]]] = None

    # --- Step 1: Upload & validate -------------------------------------
    step = job.step("upload_validate")
    if start_idx <= 0:
        _set_running(step)
        try:
            parsed = validate_upload(files or {})
        except UploadValidationError as e:
            _set_failed(step, "; ".join(e.errors))
            job.overall_status = StepStatus.FAILED
            return
        except Exception as e:
            _set_failed(step, str(e))
            job.overall_status = StepStatus.FAILED
            return
        counts = {k: len(v) for k, v in parsed.items()}
        _set_done(step, ", ".join(f"{v} {k}" for k, v in counts.items()), sum(counts.values()))

    # --- Step 2: Parse & normalize --------------------------------------
    step = job.step("parse_normalize")
    if start_idx <= 1:
        _set_running(step)
        if parsed is None:
            _set_failed(step, "No validated data available -- step 1 must complete first.")
            job.overall_status = StepStatus.FAILED
            return
        try:
            written = write_files(parsed)
        except Exception as e:
            _set_failed(step, str(e))
            job.overall_status = StepStatus.FAILED
            return
        _set_done(step, ", ".join(f"{v} {k}" for k, v in written.items()), sum(written.values()))

    # --- Step 3: Vector indexing ----------------------------------------
    step = job.step("vector_indexing")
    if start_idx <= 2:
        _set_running(step)
        per_store: dict[str, dict] = {}

        def _cb(store_key: str, count: int, error: Optional[str]) -> None:
            per_store[store_key] = {"count": count, "error": error}
            ok = sum(1 for v in per_store.values() if v["error"] is None)
            step.message = f"{ok}/{len(config.VECTOR_STORES)} stores indexed so far ({', '.join(per_store.keys())})"

        try:
            indexed = index_vectors(progress_cb=_cb)
        except Exception as e:
            _set_failed(step, str(e))
            job.overall_status = StepStatus.FAILED
            return
        failed_stores = [k for k, v in per_store.items() if v["error"]]
        ok_count = len(indexed) - len(failed_stores)
        message = f"{ok_count}/{len(indexed)} stores indexed"
        if failed_stores:
            message += f"; failed: {', '.join(failed_stores)}"
        _set_done(step, message, sum(indexed.values()))

    # --- Step 4: Knowledge graph build ----------------------------------
    step = job.step("knowledge_graph")
    if start_idx <= 3:
        _set_running(step)
        try:
            stats = build_graph_step()
        except Exception as e:
            _set_failed(step, str(e))
            job.overall_status = StepStatus.FAILED
            return
        _set_done(step, f"{stats['nodes']} nodes, {stats['edges']} edges", stats["nodes"])

    # --- Step 5: Readiness check -----------------------------------------
    step = job.step("readiness_check")
    if start_idx <= 4:
        _set_running(step)
        try:
            health = await readiness_step()
        except Exception as e:
            _set_failed(step, str(e))
            job.overall_status = StepStatus.FAILED
            return
        backend = health.get("llm_backend", "LLM backend")
        message = f"{backend} reachable" if health["ollama_connected"] else (
            f"{backend} not reachable -- LLM-dependent tabs (Recommend, NBA Decision, Benchmark) "
            "will stay disabled until it's started; everything else is fully usable."
        )
        _set_done(step, message)

    job.overall_status = StepStatus.DONE


def start_job(files: dict[str, bytes]) -> str:
    """Registers a new job and returns its id. Call run_job(job_id, files)
    (typically via FastAPI BackgroundTasks) to actually execute it -- kept
    separate so the endpoint can return the job_id immediately."""
    job = JobState(job_id=str(uuid.uuid4()))
    _jobs[job.job_id] = job
    return job.job_id


async def run_job(job_id: str, files: dict[str, bytes]) -> None:
    job = _jobs[job_id]
    await _execute(job, files, start_from=STEP_NAMES[0])


def get_job(job_id: str) -> Optional[JobState]:
    return _jobs.get(job_id)


def first_failed_step(job: JobState) -> Optional[str]:
    failed = next((s for s in job.steps if s.status == StepStatus.FAILED), None)
    return failed.name if failed else None


async def retry_job(job_id: str, files: Optional[dict[str, bytes]]) -> None:
    """Resumes a job from its first failed step onward -- NOT from the
    beginning. Steps 1-2 (validate/write) are only re-run if that's where it
    failed (files are required in that case, since nothing was validated or
    written yet); steps >=3 just re-run against the data already on disk
    from a prior successful step 2, so no re-upload is needed to retry a
    step 3/4/5 failure."""
    job = _jobs.get(job_id)
    if job is None:
        raise KeyError(f"No job found for job_id={job_id}")

    start_from = first_failed_step(job) or STEP_NAMES[0]
    if start_from == STEP_NAMES[0] and not files:
        raise ValueError("Retrying the Upload & validate step requires re-uploading all 4 files.")

    idx = STEP_NAMES.index(start_from)
    for s in job.steps[idx:]:
        s.status, s.message, s.error, s.count = StepStatus.PENDING, "", None, None

    await _execute(job, files, start_from=start_from)
