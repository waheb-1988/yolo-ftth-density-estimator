"""Step-by-step NBA data-upload pipeline: Upload & validate -> Parse &
normalize -> Bandit training -> Readiness check.

The NBA-side equivalent of agents/ingestion_pipeline.py's "Upload & Setup"
tab for NBO -- same shape (reuses its StepStatus/StepState/JobState, same
polling-not-streaming rationale, same atomic-write and collect-every-error
policies), but for 7 files instead of 4: the NBA source datasets split to
mirror the production script's own per-Dataiku-dataset boundaries (see
core.config.NBA_SOURCE_FILE_ATTRS for exactly which production dataset each
one stands in for), not one merged nba_scores.json. Replaces whatever NBA
data is currently on disk and retrains core/nba_bandit.py's contextual
bandit -- doesn't touch core/data_simulator.py's synthetic generation or
POST /seed's standalone behavior.

Referential integrity against customers.json (does this msisdn actually
exist?) and cross-file integrity (does every file cover the same msisdns?)
are NOT checked -- nothing in this codebase specifies either rule, same
policy the NBO pipeline states for its own uploads. A msisdn missing from
one of the 6 non-targeting files simply gets a 0.0 default for those
features at merge time (core/nba_bandit.py::_load_scores()).
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import ValidationError

from nbx.core import config, nba_bandit, readiness
from nbx.core.models import (
    NBAActivityUsage,
    NBAAppetencyScore,
    NBAChurnScore,
    NBAFragilityScore,
    NBARevDecreaseScore,
    NBATargeting,
    NBAReactivationScore,
)
from nbx.agents.ingestion_pipeline import JobState, StepState, StepStatus

STEP_NAMES = ["upload_validate", "parse_normalize", "bandit_training", "readiness_check"]
STEP_DISPLAY = {
    "upload_validate": "Upload & validate",
    "parse_normalize": "Parse & normalize",
    "bandit_training": "Bandit training",
    "readiness_check": "Readiness check",
}

REQUIRED_FILES = ["churn", "reactivation", "appetency", "fragility", "revenue_decrease", "activity_usage", "targeting"]
MODEL_BY_FILE = {
    "churn": NBAChurnScore,
    "reactivation": NBAReactivationScore,
    "appetency": NBAAppetencyScore,
    "fragility": NBAFragilityScore,
    "revenue_decrease": NBARevDecreaseScore,
    "activity_usage": NBAActivityUsage,
    "targeting": NBATargeting,
}


class UploadValidationError(Exception):
    """Raised with the *complete* list of problems across every file/row,
    not just the first -- same policy as agents/ingestion_pipeline.py."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


# ---------------------------------------------------------------------------
# Step 1: Upload & validate
# ---------------------------------------------------------------------------
def validate_upload(files: dict[str, bytes]) -> dict[str, list[dict]]:
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
# Step 2: Parse & normalize
# ---------------------------------------------------------------------------
def write_files(parsed: dict[str, list[dict]]) -> dict[str, int]:
    """Same temp-file-then-atomic-replace pattern as
    agents/ingestion_pipeline.py::write_files -- a mid-write failure never
    leaves a half-updated set of NBA files on disk."""
    tmp_paths = {}
    try:
        for key, rows in parsed.items():
            final_path = getattr(config, config.NBA_SOURCE_FILE_ATTRS[key])
            tmp_path = final_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            tmp_paths[key] = tmp_path

        for key, tmp_path in tmp_paths.items():
            final_path = getattr(config, config.NBA_SOURCE_FILE_ATTRS[key])
            tmp_path.replace(final_path)
    finally:
        for tmp_path in tmp_paths.values():
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    return {key: len(rows) for key, rows in parsed.items()}


# ---------------------------------------------------------------------------
# Step 3: Bandit training
# ---------------------------------------------------------------------------
def train_step() -> dict:
    nba_bandit.get_bandit(force_rebuild=True)
    return nba_bandit.get_stats()


# ---------------------------------------------------------------------------
# Step 4: Readiness check
# ---------------------------------------------------------------------------
async def readiness_step() -> dict:
    return await readiness.compute_health()


# ---------------------------------------------------------------------------
# Job state + orchestration -- own _jobs registry, but reuses
# agents.ingestion_pipeline's StepStatus/StepState/JobState dataclasses
# (they carry no NBO-specific fields).
# ---------------------------------------------------------------------------
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
        excluded = sum(1 for r in parsed["targeting"] if r["excluded"])
        _set_done(step, f"{', '.join(f'{v} {k}' for k, v in counts.items())} ({excluded} excluded / Nursering)", sum(counts.values()))

    # --- Step 2: Parse & normalize ---------------------------------------
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

    # --- Step 3: Bandit training ------------------------------------------
    step = job.step("bandit_training")
    if start_idx <= 2:
        _set_running(step)
        try:
            stats = train_step()
        except Exception as e:
            _set_failed(step, str(e))
            job.overall_status = StepStatus.FAILED
            return
        _set_done(
            step,
            f"{stats['n_takers']} labeled takers, avg_reward={stats['avg_reward']} "
            f"over {stats['iterations']} iterations",
            stats["n_takers"],
        )

    # --- Step 4: Readiness check -------------------------------------------
    step = job.step("readiness_check")
    if start_idx <= 3:
        _set_running(step)
        try:
            health = await readiness_step()
        except Exception as e:
            _set_failed(step, str(e))
            job.overall_status = StepStatus.FAILED
            return
        backend = health.get("llm_backend", "LLM backend")
        message = (
            f"{backend} reachable" if health["ollama_connected"] else
            f"{backend} not reachable -- NBO's LLM step (inside NBA Decision) stays degraded until it's "
            "started; the bandit's own action decision doesn't depend on it."
        )
        _set_done(step, message)

    job.overall_status = StepStatus.DONE


def start_job() -> str:
    """Registers a new job and returns its id. Call run_job(job_id, files)
    (typically via FastAPI BackgroundTasks) to execute it."""
    import uuid

    job = JobState(job_id=str(uuid.uuid4()), steps=[StepState(name=n) for n in STEP_NAMES])
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
    """Resumes from the first failed step, not from the beginning -- same
    policy as agents/ingestion_pipeline.py::retry_job. Only pass files back
    in if the failed step was "Upload & validate" (nothing was written yet);
    later steps re-run against the files already on disk from a prior
    successful "Parse & normalize" step."""
    job = _jobs.get(job_id)
    if job is None:
        raise KeyError(f"No job found for job_id={job_id}")

    start_from = first_failed_step(job) or STEP_NAMES[0]
    if start_from == STEP_NAMES[0] and not files:
        raise ValueError("Retrying the Upload & validate step requires re-uploading all 7 files.")

    idx = STEP_NAMES.index(start_from)
    for s in job.steps[idx:]:
        s.status, s.message, s.error, s.count = StepStatus.PENDING, "", None, None

    await _execute(job, files, start_from=start_from)
