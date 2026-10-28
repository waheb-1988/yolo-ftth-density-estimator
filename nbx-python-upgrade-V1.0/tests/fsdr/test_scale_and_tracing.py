"""1.7 M-line readiness (vectorised helpers give the same results as the per-value code), the weekly runner,
and LangSmith tracing (with a fake recorder: nothing is sent)."""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nbx.fsdr import config, schemas

MIXED = pd.Series([91234567, "91234567", 91234567.0, " 91234567 ", "91234567.0", None, np.nan, 1001.5, "", "abc", True,
                   np.int64(7), "007", ".0"], dtype=object)


def test_vectorised_key_normalisation_equals_the_per_value_rules():
    from nbx.fsdr.features import _norm_val_per_value, norm_val
    from nbx.fsdr.validation import _norm_key, norm_keys

    assert norm_val(MIXED).tolist() == _norm_val_per_value(MIXED).tolist()
    assert norm_keys(MIXED).tolist() == MIXED.map(_norm_key).tolist()
    for s in (pd.Series([1, 2, 3]), pd.Series([1.0, np.nan, 2.5]), pd.Series(["a ", None, "1.0"], dtype=object), pd.Series([], dtype=object)):
        assert norm_val(s).tolist() == _norm_val_per_value(s).tolist()
        assert norm_keys(s).tolist() == s.map(_norm_key).tolist()


@pytest.mark.parametrize("raw_type", ["string", "int", "bigint", "double", "date", "timestamp"])
def test_vectorised_type_check_equals_value_matches(raw_type):
    s = pd.Series([1, 2.5, "x", "2026-10-21", "2026-10-21T10:00:00", "2026-13-99T", True, np.int64(3), None, "21/10/2026"], dtype=object)
    expected = sum(not schemas.value_matches(raw_type, v) for v in s if v is not None)
    assert schemas.type_errors(s, raw_type)[0] == expected
    for col in (pd.Series(["a", "b"], dtype=object), pd.Series([1, 2], dtype=object), pd.Series([1.5, None], dtype=object)):
        nn = col[col.notna()]
        assert schemas.type_errors(col, raw_type)[0] == sum(not schemas.value_matches(raw_type, v) for v in nn)


def test_write_table_writes_the_same_json_as_the_per_value_version(tmp_path):
    from nbx.fsdr.pipelines import _clean, write_table

    df = pd.DataFrame({"A": [1, 2], "B": [0.1234567891, np.nan], "C": ["x", None], "D": [True, False],
                       "E": pd.array([1, None], dtype="Int64"), "F": [date(2026, 10, 21), None], "G": [{"k": 1}, [1, 2]]})
    p = write_table(df, tmp_path, "t")
    old = [{k.lower(): _clean(v) for k, v in r.items()} for r in df.to_dict("records")]
    assert p.read_text(encoding="utf-8") == json.dumps(old, ensure_ascii=False)


def test_typed_source_keeps_text_and_types_numbers(small_folder):
    from nbx.fsdr.datasource import JsonSource

    raw, typed = JsonSource(small_folder), JsonSource(small_folder, compact=True)
    r, t = raw.read("activities_m1_score"), typed.read("activities_m1_score")
    assert r["ACCESS_METHOD_VAL"].tolist() == t["ACCESS_METHOD_VAL"].tolist()          # text column untouched
    num = [c for c in t.columns if t[c].dtype.kind in "if"]
    assert num and all(r[c].dtype == object for c in num)
    assert t.memory_usage(deep=True).sum() < r.memory_usage(deep=True).sum()


def test_value_policy_gives_the_same_offers_one_line_at_a_time_or_in_a_batch(workspace):
    eng = workspace.engine
    vals = workspace.lines("cstr")[:40]
    action = "UPSELL"
    batch = eng.select_value_batch(vals, action, "cstr", chunk=7)
    for v in vals:
        one = eng.select_value(v, action, "cstr")
        assert [p.as_dict() for p in one] == [p.as_dict() for p in batch[v]]


def test_weekly_runner_over_two_weeks(tmp_path, monkeypatch):
    from nbx.fsdr.agents.recommender_agent import run_batch  # noqa: F401  (import check)
    from nbx.fsdr.pipelines.weekly import run_weekly
    from nbx.fsdr.synth import generator
    from nbx.fsdr.synth.world import World
    from nbx.fsdr import campaign_log

    monkeypatch.setattr(config, "FSDR_STATE_DIR", tmp_path / "state")
    d0 = date(2026, 10, 21)
    world = World.create(600, seed=5, as_of=d0 + timedelta(days=7))
    f0 = generator.write_run(generator.generate_run(world, d0), tmp_path / d0.isoformat())
    rep0 = run_weekly(f0, llm_limit=5)
    assert rep0["status"] == "done"
    assert rep0["steps"]["feedback"]["status"] == "skipped" and rep0["steps"]["report"]["status"] == "skipped"
    assert rep0["steps"]["train"]["label_source"] == "targeting" and rep0["steps"]["offers"]["cstr"]["offers"] > 0
    d1 = d0 + timedelta(days=7)
    prev = campaign_log.read_log(campaign_log.root_for(f0), d0, d0 + timedelta(days=6))
    f1 = generator.write_run(generator.generate_run(world, d1, prev), tmp_path / d1.isoformat())
    rep1 = run_weekly(f1, arms=("cstr",))
    assert rep1["steps"]["feedback"]["status"] == "done" and rep1["steps"]["train"]["label_source"] == "campaign_log"
    assert rep1["steps"]["report"]["status"] == "done" and "cg" in rep1["steps"]["report"]["arms"]
    assert json.loads((f1 / "outputs" / "weekly_run_report.json").read_text())["status"] == "done"


def test_weekly_runner_stops_on_blocked_data(small_folder, tmp_path):
    import shutil

    from nbx.fsdr.pipelines.train import DataBlocked
    from nbx.fsdr.pipelines.weekly import run_weekly

    f = tmp_path / small_folder.name
    shutil.copytree(small_folder, f, ignore=shutil.ignore_patterns("outputs"))
    (f / "base_prepaid.json").unlink()                               # a required table missing -> blocked
    with pytest.raises(DataBlocked):
        run_weekly(f)
    rep = json.loads((f / "outputs" / "weekly_run_report.json").read_text())
    assert rep["status"] == "failed at validate" and rep["steps"]["validate"]["status"] == "failed"


class FakeTrace:
    """Stands in for langsmith.run_helpers.trace: records the runs, sends nothing."""
    runs: list = []

    def __init__(self, name, run_type="chain", inputs=None, **kw):
        self.rec = {"name": name, "run_type": run_type, "inputs": inputs, "outputs": None, "error": None, **kw}
        FakeTrace.runs.append(self.rec)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def end(self, outputs=None, error=None, **kw):
        self.rec["outputs"], self.rec["error"] = outputs, error


@pytest.fixture
def fake_langsmith(monkeypatch):
    import langsmith.run_helpers

    FakeTrace.runs = []
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key-not-sent")
    monkeypatch.setattr(langsmith.run_helpers, "trace", FakeTrace)
    return FakeTrace.runs


def test_tracing_is_off_without_key(monkeypatch):
    from nbx.core import tracing

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert not tracing.enabled()
    with tracing.span("x") as sp:
        sp.end(outputs={"a": 1})                                    # no-op


async def test_a_decision_is_traced_with_nested_agent_and_tool_runs_and_no_msisdn(workspace, fake_langsmith):
    from nbx.fsdr.agents.recommender_agent import decide

    v = workspace.lines("llm")[0]
    d = (await decide(workspace, v))["decision"]
    names = [r["name"] for r in fake_langsmith]
    assert names[0] == "fsdr_decision" and any(n.startswith("tool:") for n in names)
    top = fake_langsmith[0]
    assert top["outputs"]["action"] == d["action"] and top["inputs"]["customer_ref"].startswith("C-")
    assert v not in json.dumps([r["inputs"] for r in fake_langsmith], default=str)


async def test_llm_calls_are_traced_as_llm_runs(fake_langsmith, monkeypatch):
    from nbx.core import config as nbx_config
    from nbx.llms import llm_client, vllm_client

    async def fake_generate(**kw):
        return {"content": '{"ok": true}', "duration_ms": 3, "model": kw["model"], "error": None}

    monkeypatch.setattr(nbx_config, "LLM_BACKEND", "vllm")
    monkeypatch.setattr(vllm_client, "generate", fake_generate)
    out = await llm_client.generate("qwen2.5:3b", "hello", system="sys", force_json=True)
    run = fake_langsmith[-1]
    assert out["content"] == '{"ok": true}' and run["run_type"] == "llm" and run["name"] == "llm:qwen2.5:3b"
    assert run["inputs"]["messages"][1]["content"] == "hello" and run["outputs"]["content"] == '{"ok": true}'


def test_tracing_errors_never_break_a_run(monkeypatch):
    import langsmith.run_helpers

    from nbx.core import tracing

    def broken(*a, **k):
        raise RuntimeError("langsmith down")

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setattr(langsmith.run_helpers, "trace", broken)
    with tracing.span("x") as sp:
        sp.end(outputs={"a": 1})
    assert True
