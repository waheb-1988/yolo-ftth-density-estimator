"""V3.0: target upload folders, code 5 (F-01 … F-19) golden cases, the final step (push / per target / details)."""
from __future__ import annotations

import json
import shutil
from datetime import date

import pandas as pd
import pytest

from nbx.fsdr import config, csvstd, targets
from nbx.fsdr.contract import data_agent
from nbx.fsdr.final_output import build_input_flytxt, f05_anchors
from nbx.fsdr.pipelines import scoring_groups as sg
from nbx.fsdr.pipelines import training

from .conftest import RUN_DATE

VAL = "ACCESS_METHOD_VAL"
D = date(2026, 10, 22)


class Src:
    def __init__(self, tables):
        self.t = tables

    def has(self, n):
        return n in self.t

    def read(self, n):
        return self.t[n].copy()


def _cat(rows):
    return pd.DataFrame([{"btl_name": n, "type": t, "generosity": "L2", "gb": "1", "min": "0", "price": str(p),
                          "Price data": str(p), "Price voice": "0", "service": s, "validity": str(v)}
                         for n, t, s, p, v in rows])


CAT = [("M68", "Combo Bundle", "COMBO", 6.8, 30), ("M34", "Data Bundle", "DATA", 3.4, 30),
       ("W12", "Data Bundle", "DATA", 1.2, 7), ("W15", "Combo Bundle", "COMBO", 1.5, 7),
       ("DUP34", "Data Bundle", "DATA", 3.4, 30), ("FAV", "Data Bundle", "DATA", 2.0, 30),
       ("CM_3R_8GB_240MIN", "Combo Bundle", "COMBO", 3.0, 30), ("CM_4R_10GB_300MIN", "Combo Bundle", "COMBO", 4.0, 30),
       ("CM_5R_14GB_360MIN", "Combo Bundle", "COMBO", 5.0, 30)]


def _src(subs_m: dict, bought=(), conflicts=()):
    vals = list(subs_m)
    st = pd.DataFrame({"ACCESS_METHOD_VAL": vals, **{f"Total_Sub_Amt_{p}": [subs_m[v][i] for v in vals]
                                                     for i, p in enumerate(["M1", "M2", "M3"])}})
    for w in ("w1", "w2", "w3", "w4"):
        st[f"Total_Sub_Amt_{w}"] = 0.0
    use = pd.DataFrame({"ACCESS_METHOD_VAL": vals, **{f"PG_Total_Amount_{p}": 0.0 for p in
                                                       ["m1", "m2", "m3", "w1", "w2", "w3", "w4"]}})
    cat = _cat(CAT)
    scripts = pd.DataFrame({"Sl No.": range(len(cat)), "destination_offer_name": cat["btl_name"],
                            "English Script": "EN " + cat["btl_name"], "Arabic Script": "عرض " + cat["btl_name"]})
    conf = pd.DataFrame([{"Offer_A": a, "Offer_B": b, "Offer_Type_A": t} for a, b, t in conflicts],
                        columns=["Offer_A", "Offer_B", "Offer_Type_A"])
    sb = pd.DataFrame([{"access_method_val": v, "PRODUCTNAME": n, "sub_flag": 1, "nb_sub": 1} for v, n in bought],
                      columns=["access_method_val", "PRODUCTNAME", "sub_flag", "nb_sub"])
    return Src({"btl_catalog": cat, "subscriptions_tg": st, "usage_data_voice_payg_tg": use, "sub_btl_3months": sb,
                "BTL_script_english_arabic_ok": scripts, "conflicting_offers_h": conf})


def _recos(rows):
    return pd.DataFrame(rows, columns=["access_method_val", "btl_name", "price"])


def _scored(vals, action="MAINTAIN"):
    return pd.DataFrame({"access_method_val": vals, "action": action, "proba": 1.0, "tg_flag": "constraint"})


# ---------------------------------------------------------------------------
# Code 5 golden cases (master §23)
# ---------------------------------------------------------------------------
def test_f05_anchor_zero_periods_are_missing():
    src = _src({"91000000": (0.0, 4.0, 6.0)})
    assert f05_anchors(src).loc["91000000", "F05_MONTHLY"] == pytest.approx(6.0)        # min(5 + 1.414, 6)


def test_slots_from_monthly_and_weekly_price_rank_and_the_38_columns():
    src = _src({"91000000": (6, 6, 6)})
    m = _recos([("91000000", "M34", 3.4), ("91000000", "M68", 6.8)])
    w = _recos([("91000000", "W12", 1.2)])
    fly, slots = build_input_flytxt(m, w, _scored(["91000000"], "UPSELL"), src, D)
    r = fly.iloc[0]
    assert list(fly.columns) == config.INPUT_FLYTXT_COLUMNS and len(fly.columns) == 38
    assert (r["destination_offer_name_1"], r["destination_offer_name_2"], r["destination_offer_name_3"]) == ("M68", "M34", "W12")
    assert r["opportunity_name_1"] == "nbo_LFM_M68" and r["opportunity_name_4"] == "" and r["offer_script_en_4"] == ""
    assert (r["date"], r["start_date"], r["end_date"], r["flag_splio"]) == ("2026-10-22", "2026-10-23", "", 1)
    assert (r["segment"], r["segment_btl_taker_upsell"], r["segment_btl_taker"]) == ("UPSELL", "BTL_upsell", "No_BTL_Taker")
    assert r["offer_script_ar_1"] == "عرض M68"


@pytest.mark.parametrize("monthly,expected", [(2.5, "CM_3R_8GB_240MIN"), (3.5, "CM_4R_10GB_300MIN"), (5.2, "")])
def test_forced_bundle_in_slot_6_by_monthly_anchor(monthly, expected):
    src = _src({"91000000": (monthly, monthly, monthly)})
    fly, _ = build_input_flytxt(_recos([("91000000", "M68", 6.8)]), _recos([]), _scored(["91000000"]), src, D)
    assert fly.iloc[0]["destination_offer_name_6"] == expected


def test_no_forced_bundle_when_a_popular_offer_is_already_held():
    src = _src({"91000000": (2.5, 2.5, 2.5)}, bought=[("91000000", "CM_4R_10GB_300MIN")])
    fly, _ = build_input_flytxt(_recos([("91000000", "M68", 6.8)]), _recos([]), _scored(["91000000"]), src, D)
    r = fly.iloc[0]
    assert r["destination_offer_name_5"] == "CM_4R_10GB_300MIN" and r["destination_offer_name_6"] == ""
    assert r["segment_btl_taker"] == "BTL_Taker"


def test_f09_same_type_and_price_cleared_and_f12_conflict_removes_offer_a():
    src = _src({"91000000": (9, 9, 9)}, bought=[("91000000", "DUP34")],
               conflicts=[("W12", "M68", "DATA"), ("W15", "M68", "VOICE")])
    m = _recos([("91000000", "M68", 6.8)])
    w = _recos([("91000000", "W12", 1.2), ("91000000", "W15", 1.5)])
    fly, slots = build_input_flytxt(m, w, _scored(["91000000"]), src, D)
    r, s = fly.iloc[0], slots.iloc[0]
    assert r["destination_offer_name_3"] == "W15"                          # VOICE conflict row ignored
    assert r["destination_offer_name_4"] == "" and s["REMOVED_4"] == "F-12: W12"   # (W12, M68) -> Offer_A W12 removed
    m2 = _recos([("91000000", "M68", 6.8), ("91000000", "M34", 3.4)])
    fly2, slots2 = build_input_flytxt(m2, _recos([]), _scored(["91000000"]), src, D)
    assert fly2.iloc[0]["destination_offer_name_5"] == "" and slots2.iloc[0]["REMOVED_5"] == "F-09: DUP34"
    assert fly2.iloc[0]["offer_script_en_5"] == "" and fly2.iloc[0]["opportunity_name_5"] == ""   # issue (a) fixed


def test_f04_customer_without_slots_1_to_4_is_dropped():
    src = _src({"91000000": (6, 6, 6), "91000001": (6, 6, 6)})
    fly, slots = build_input_flytxt(_recos([("91000000", "M68", 6.8)]), _recos([]),
                                    _scored(["91000000", "91000001"]), src, D)
    assert list(fly["msisdn"]) == ["91000000"]
    assert slots.set_index(VAL).loc["91000001", "DROP_RULE"] == "F-04"


# ---------------------------------------------------------------------------
# Target lists and the final step, end to end
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def run_v3(small_folder, tmp_path_factory):
    root = tmp_path_factory.mktemp("v3 final")
    folder = root / "input" / RUN_DATE.isoformat()
    shutil.copytree(small_folder, folder)
    runs, cache = root / "runs", root / "cache"
    assert data_agent.run_t0(folder, runs_root=runs, cache_root=cache)["status"] == "done"
    training.run_features(RUN_DATE, runs_root=runs, cache_root=cache, workers=1)
    gen = sg.generate_targets(RUN_DATE, runs_root=runs, cache_root=cache)
    p1 = sg.run_p1_panel(RUN_DATE, runs_root=runs, cache_root=cache)
    p6 = sg.run_p6_offers(RUN_DATE, runs_root=runs, cache_root=cache)
    fin = sg.run_p7_final(RUN_DATE, runs_root=runs, cache_root=cache)
    return {"folder": folder, "runs": runs, "cache": cache, "gen": gen, "p1": p1, "p6": p6, "final": fin,
            "dir": runs / RUN_DATE.isoformat() / "final"}


def test_dummy_targets_are_written_in_the_upload_folders_and_used_by_the_panel(run_v3):
    tdir = targets.targets_dir(run_v3["folder"])
    for rel in config.TARGET_FILES.values():
        assert (tdir / f"{rel}.csv.gz").exists() and (tdir / f"{rel}.schema.json").exists()
    assert run_v3["p1"]["mode"] == "uploaded" and run_v3["p1"]["rejected"] == 0
    assert run_v3["p1"]["sizes"] == {k: int(v) for k, v in run_v3["gen"]["sizes"].items()}
    assert not [i for i in run_v3["p1"]["target_issues"] if i["level"] != "info"]


def test_final_writes_push_per_target_and_details(run_v3):
    d, fin = run_v3["dir"], run_v3["final"]
    assert fin["push_ready"] and all(c["status"] == "PASS" for c in fin["output_checks"])
    push = csvstd.read_csv(d / "push" / f"input_flytxt{csvstd.SUFFIX}")
    assert list(push.columns) == config.INPUT_FLYTXT_COLUMNS and push["msisdn"].is_unique
    parts = pd.concat([csvstd.read_csv(d / t / f"input_flytxt_{k}{csvstd.SUFFIX}")
                       for k, t in (("t1c", "target1"), ("t1f", "target1"), ("t2c", "target2"), ("t2f", "target2"))])
    assert sorted(parts["msisdn"]) == sorted(push["msisdn"])
    t1 = csvstd.read_csv(d / "target1" / f"input_flytxt_target1{csvstd.SUFFIX}")
    t1_parts = pd.concat([csvstd.read_csv(d / "target1" / f"input_flytxt_{k}{csvstd.SUFFIX}") for k in ("t1c", "t1f")])
    assert set(t1["msisdn"]) == set(t1_parts["msisdn"])
    panel = pd.read_parquet(run_v3["runs"] / RUN_DATE.isoformat() / "score" / "P1_panel" / "panel.parquet")
    det = pd.read_parquet(d / "details_all.parquet")
    assert len(det) == len(panel) and det[VAL].is_unique
    for col in ("ACTION_ORIGINAL", "ACTION_AGENT", "ACTION_SENT", "B26_AVG_REV_MONTHLY", "F05_MONTHLY", "SLOT_1",
                "REASON_1", "IN_INPUT_FLYTXT", "DROP_RULE", "PROBA_CHURN", "reco_m1_selected_by"):
        assert col in det.columns, col
    assert set(det.loc[det["IN_INPUT_FLYTXT"], VAL]) == set(push["msisdn"])
    ctrl = det[det["GROUP_KEY"] == "control"]
    assert (ctrl["DROP_RULE"] == "control (H-01)").all() and not set(ctrl[VAL]) & set(push["msisdn"])
    man = json.loads((d / csvstd.OUTPUTS_MANIFEST).read_text(encoding="utf-8"))
    assert any(f["file"] == "input_flytxt.csv.gz" for f in man["files"])


def test_push_is_blocked_when_an_output_check_fails(run_v3, monkeypatch):
    monkeypatch.setattr(sg, "output_checks", lambda *a: [{"id": "A-20", "check": "broken on purpose", "status": "FAIL"}])
    fin = sg.run_p7_final(RUN_DATE, runs_root=run_v3["runs"], cache_root=run_v3["cache"], force=True)
    d = run_v3["dir"]
    assert not fin["push_ready"] and (d / "push" / "BLOCKED.txt").exists()
    assert not (d / "push" / f"input_flytxt{csvstd.SUFFIX}").exists()        # the earlier PASS file is removed
    assert "broken on purpose" in (d / "push" / "BLOCKED.txt").read_text(encoding="utf-8")


def test_target_checks_block_a_line_in_two_groups_and_drop_bad_lines(run_v3):
    seg = sg.base_segments(RUN_DATE, run_v3["runs"])
    good = seg[seg["EXCLUDED"] == 0][VAL].iloc[:3].tolist()
    nurs = seg[seg["EXCLUDED"] == 1][VAL].iloc[:1].tolist()
    t = pd.DataFrame({VAL: [good[0], good[0], good[1], "123", "99999999", *nurs],
                      "GROUP_KEY": ["t1c", "t2c", "t1f", "t1f", "t2f", "t2f"]})
    kept, rejected, issues = targets.check_targets(t, seg)
    checks = {i["check"] for i in issues}
    assert "line_in_two_groups" in checks and any(i["level"] == "blocking" for i in issues)
    reasons = set(rejected["REJECT_REASON"])
    assert {"not an 8-digit MSISDN", "not in the eligible base (D-07)"} <= reasons
    if nurs:
        assert "Nursering segment (B-02)" in reasons
