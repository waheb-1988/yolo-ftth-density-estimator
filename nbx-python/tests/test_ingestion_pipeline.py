# -*- coding: utf-8 -*-
"""Tests for the step-by-step ingestion pipeline (agents/ingestion_pipeline.py):
sequencing (a failed step stops the pipeline before the next one runs) and
retry (resuming from the failed step, not restarting from the top).

Steps 3-5 (vector indexing / graph build / readiness) are monkeypatched with
call-tracking stubs for the sequencing/retry tests -- they don't need a real
vector-store/Ollama environment to prove the *orchestration* is correct.
Steps 1-2 (validate/write) are tested against their real logic separately,
since those are pure and fast (JSON/pydantic validation, file I/O).
"""
from __future__ import annotations

import json

import pytest

from nbx.agents import ingestion_pipeline as ip
from nbx.core import config


# ---------------------------------------------------------------------------
# Fixtures: minimal valid records for each of the 4 required files
# ---------------------------------------------------------------------------
def _valid_customer_row(msisdn=1):
    return {
        "msisdn": msisdn, "tenure_days": 100, "arpu": 5.0, "dv_segment": "new smart",
        "revenue_segment": "Mid Value", "rfm_score": 555, "usage_type": "BALANCED",
        "data_usage_tier": "LOW_DATA", "activity_level": "MEDIUM_ACTIVITY",
        "is_intl_caller": False, "is_roamer": False, "days_since_last_purchase": 10,
        "days_since_last_network_activity": 5, "avg_offer_price": 5.0,
        "preferred_recharge_channel": "APP", "data_volume_mb_p1": 1000.0,
        "voice_traffic_min_p1": 50.0, "active_days_count_p1": 20,
        "favorite_category": "NBO", "last_product_name": "PS_1D_1R",
        "nationality": "OMN", "nwk_type": "4G",
    }


def _valid_offer_row():
    return {
        "program_rule": "PS_1D_1R", "offer_name": "PS_1D_1R", "category": "NBO",
        "offer_type": "Prepaid Add-on", "price_omr": 1.0, "data_gb": "1GB",
        "voice_mins": "0", "validity_days": 1, "is_data": True, "is_voice": False,
        "is_intl": False, "is_roaming": False, "discount_pct": 0.0, "description": "x",
    }


def _valid_purchase_row():
    return {
        "transaction_id": "TXN-1", "msisdn": 1, "purchase_date": "2026-01-01T00:00:00",
        "offer_name": "PS_1D_1R", "program_rule": "PS_1D_1R", "category": "NBO",
        "price_omr": 1.0, "purchase_count": 1,
    }


def _valid_segment_row():
    return {"segment": "new smart", "typical_behavior": "test behavior"}


def _files(**overrides) -> dict[str, bytes]:
    base = {
        "customers": json.dumps([_valid_customer_row()]).encode(),
        "offers": json.dumps([_valid_offer_row()]).encode(),
        "purchases": json.dumps([_valid_purchase_row()]).encode(),
        "segments": json.dumps([_valid_segment_row()]).encode(),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Step 1: validate_upload -- real logic
# ---------------------------------------------------------------------------
class TestValidateUpload:
    def test_valid_files_parse_cleanly(self):
        parsed = ip.validate_upload(_files())
        assert len(parsed["customers"]) == 1
        assert len(parsed["offers"]) == 1
        assert len(parsed["purchases"]) == 1
        assert len(parsed["segments"]) == 1

    def test_missing_file_is_reported(self):
        files = _files()
        del files["segments"]
        with pytest.raises(ip.UploadValidationError) as exc_info:
            ip.validate_upload(files)
        assert any("segments" in e for e in exc_info.value.errors)

    def test_invalid_json_is_reported(self):
        files = _files(offers=b"{not valid json")
        with pytest.raises(ip.UploadValidationError) as exc_info:
            ip.validate_upload(files)
        assert any("offers.json" in e and "invalid JSON" in e for e in exc_info.value.errors)

    def test_all_malformed_rows_are_collected_not_just_the_first(self):
        bad_customers = [
            {**_valid_customer_row(1), "arpu": "not-a-number"},
            {**_valid_customer_row(2), "tenure_days": "also-not-a-number"},
        ]
        files = _files(customers=json.dumps(bad_customers).encode())
        with pytest.raises(ip.UploadValidationError) as exc_info:
            ip.validate_upload(files)
        assert any("row 0" in e for e in exc_info.value.errors)
        assert any("row 1" in e for e in exc_info.value.errors)


# ---------------------------------------------------------------------------
# Step 2: write_files -- real logic, isolated to a tmp_path data dir
# ---------------------------------------------------------------------------
class TestWriteFiles:
    def test_writes_all_four_files_and_returns_counts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CUSTOMERS_FILE", tmp_path / "customers.json")
        monkeypatch.setattr(config, "OFFERS_FILE", tmp_path / "offers.json")
        monkeypatch.setattr(config, "PURCHASES_FILE", tmp_path / "purchases.json")
        monkeypatch.setattr(config, "SEGMENTS_FILE", tmp_path / "segments.json")

        parsed = {
            "customers": [_valid_customer_row()],
            "offers": [_valid_offer_row(), _valid_offer_row()],
            "purchases": [_valid_purchase_row()],
            "segments": [_valid_segment_row()],
        }
        counts = ip.write_files(parsed)
        assert counts == {"customers": 1, "offers": 2, "purchases": 1, "segments": 1}
        assert json.loads((tmp_path / "offers.json").read_text())[1]["program_rule"] == "PS_1D_1R"
        assert not list(tmp_path.glob("*.tmp"))  # no leftover temp files


# ---------------------------------------------------------------------------
# Sequencing + retry -- monkeypatch steps with call-tracking stubs
# ---------------------------------------------------------------------------
@pytest.fixture
def call_log():
    return []


def _patch_all_steps(monkeypatch, call_log, fail_at: str | None = None):
    """Replaces every step function with a stub that records its name into
    call_log (and raises if it matches fail_at), so tests can assert exactly
    which steps ran and in what order."""

    def _make(name, result):
        def _sync(*args, **kwargs):
            call_log.append(name)
            if name == fail_at:
                raise RuntimeError(f"{name} failed (test-injected)")
            return result

        async def _async(*args, **kwargs):
            return _sync(*args, **kwargs)

        return _async if name == "readiness_check" else _sync

    monkeypatch.setattr(
        ip, "validate_upload",
        _make("upload_validate", {"customers": [_valid_customer_row()], "offers": [], "purchases": [], "segments": []}),
    )
    monkeypatch.setattr(ip, "write_files", _make("parse_normalize", {"customers": 1, "offers": 0, "purchases": 0, "segments": 0}))
    monkeypatch.setattr(ip, "index_vectors", _make("vector_indexing", {"chroma": 3, "faiss": 3, "qdrant": 3, "milvus": 3, "lancedb": 3}))
    monkeypatch.setattr(ip, "build_graph_step", _make("knowledge_graph", {"nodes": 10, "edges": 5}))
    monkeypatch.setattr(ip, "readiness_step", _make("readiness_check", {"ollama_connected": True}))


class TestSequencing:
    async def test_all_steps_run_in_order_on_success(self, monkeypatch, call_log):
        _patch_all_steps(monkeypatch, call_log)
        job_id = ip.start_job(_files())
        await ip.run_job(job_id, _files())

        assert call_log == ["upload_validate", "parse_normalize", "vector_indexing", "knowledge_graph", "readiness_check"]
        job = ip.get_job(job_id)
        assert job.overall_status == ip.StepStatus.DONE
        assert all(s.status == ip.StepStatus.DONE for s in job.steps)

    async def test_failure_at_step3_stops_the_pipeline_before_step4(self, monkeypatch, call_log):
        _patch_all_steps(monkeypatch, call_log, fail_at="vector_indexing")
        job_id = ip.start_job(_files())
        await ip.run_job(job_id, _files())

        assert call_log == ["upload_validate", "parse_normalize", "vector_indexing"]
        job = ip.get_job(job_id)
        assert job.overall_status == ip.StepStatus.FAILED
        assert job.step("vector_indexing").status == ip.StepStatus.FAILED
        assert job.step("knowledge_graph").status == ip.StepStatus.PENDING
        assert job.step("readiness_check").status == ip.StepStatus.PENDING

    async def test_failure_at_step1_never_calls_step2(self, monkeypatch, call_log):
        _patch_all_steps(monkeypatch, call_log, fail_at="upload_validate")
        job_id = ip.start_job(_files())
        await ip.run_job(job_id, _files())

        assert call_log == ["upload_validate"]
        job = ip.get_job(job_id)
        assert job.overall_status == ip.StepStatus.FAILED
        assert job.step("parse_normalize").status == ip.StepStatus.PENDING


class TestRetry:
    async def test_retry_resumes_from_the_failed_step_not_from_the_top(self, monkeypatch, call_log):
        _patch_all_steps(monkeypatch, call_log, fail_at="vector_indexing")
        job_id = ip.start_job(_files())
        await ip.run_job(job_id, _files())
        assert call_log == ["upload_validate", "parse_normalize", "vector_indexing"]

        call_log.clear()
        _patch_all_steps(monkeypatch, call_log)  # now everything succeeds
        await ip.retry_job(job_id, files=None)

        # steps 1-2 must NOT re-run on retry -- only the failed step onward
        assert call_log == ["vector_indexing", "knowledge_graph", "readiness_check"]
        job = ip.get_job(job_id)
        assert job.overall_status == ip.StepStatus.DONE

    async def test_retry_from_step1_failure_requires_files(self, monkeypatch, call_log):
        _patch_all_steps(monkeypatch, call_log, fail_at="upload_validate")
        job_id = ip.start_job(_files())
        await ip.run_job(job_id, _files())
        assert ip.get_job(job_id).overall_status == ip.StepStatus.FAILED

        with pytest.raises(ValueError, match="requires re-uploading"):
            await ip.retry_job(job_id, files=None)

    async def test_retry_unknown_job_raises(self):
        with pytest.raises(KeyError):
            await ip.retry_job("does-not-exist", files=None)
