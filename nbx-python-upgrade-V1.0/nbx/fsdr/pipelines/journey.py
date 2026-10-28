"""Step-by-step NBA / NBO journey (UI tab "FSDR Journey", API /fsdr/journey/*).

    1 ingest        copy every contract table of a source folder (the dummy data) into data/fsdr/<D>/, byte for
                    byte (SHA-256 checked), validate, check the joins and recompute the key calculations, then
                    train and score on that data
    2 action        Best Action agent: the action, the bandit evidence, the guards and the LLM's own explanation
    3 offers        Offer agent: the top 4 offers (2 monthly + 2 weekly) with P(accept), value and the LLM rationale
    4 addons        top 2 roaming packs + international packs (if the customer calls abroad)
    5 competitors   Competitor agent: each offer against the closest competitor plans, verdict, LLM summary, swaps
    6 output        everything about the line (every column of every table row that concerns it), every step's
                    result and LLM answer -> outputs/journey/<msisdn>_output.json (+ a CSV row per line)

Each step works for one MSISDN or a list (batch). The state of each line is saved after each step in
outputs/journey/<msisdn>.json, so a step continues from the previous one; a missing previous step is run first.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import shutil
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import config, schemas
from nbx.fsdr.calendar import MONTHS, WEEKS
from nbx.fsdr.datasource import JsonSource
from nbx.fsdr.features import norm_val, upper
from nbx.fsdr.pipelines import outputs_dir

JOURNEY_DIR = "journey"
STEPS = ["action", "offers", "addons", "competitors", "output"]
DUMMY_ROOT = config.FSDR_DATA_DIR.parent / "fsdr_dummy"          # data/fsdr_dummy/<D>/ : the dummy data to ingest


# ---------------------------------------------------------------------------
# 1. Ingest: copy, prove it is the same data, validate, check joins and calculations
# ---------------------------------------------------------------------------
def sources(root: Path = DUMMY_ROOT) -> list[dict]:
    """Dummy data folders available for ingestion: data/fsdr_dummy/<D>/."""
    out = []
    for p in sorted(Path(root).glob("20??-??-??")) if Path(root).is_dir() else []:
        m = p / config.MANIFEST_FILE
        man = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}
        out.append({"source": str(p), "run": p.name, "files": sum(1 for f in p.glob("*.json*") if not f.name.startswith("_")),
                    "n": (man.get("generator") or {}).get("n"), "seed": (man.get("generator") or {}).get("seed")})
    return out


def last_ingest(root: Path | None = None) -> dict | None:
    """The most recent step-1 upload (its ingestion report), so a reloaded page continues the journey."""
    root = Path(root or config.FSDR_DATA_DIR)
    reps = [p for p in root.glob(f"20??-??-??/outputs/{JOURNEY_DIR}/ingestion_report.json")] if root.is_dir() else []
    for p in sorted(reps, key=lambda p: p.stat().st_mtime, reverse=True):
        rep = json.loads(p.read_text(encoding="utf-8"))
        if rep.get("usable") and rep.get("training"):
            return rep
    return None


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _table_file(folder: Path, table: str) -> Path | None:
    names = [table] + [a for a, t in schemas.TABLE_ALIASES.items() if t == table]
    for n in names:
        for ext in (".json", ".jsonl"):
            if (folder / f"{n}{ext}").exists():
                return folder / f"{n}{ext}"
    return None


def copy_tables(source: Path, run: str | None = None, data_root: Path | None = None) -> tuple[Path, list[dict]]:
    """Copies the source's table files (and its manifest) into <data root>/<D>/; derived outputs of an older
    ingestion of the same date are removed (they were computed on other data)."""
    source = Path(source)
    if not source.is_dir():
        raise FileNotFoundError(f"no source folder {source}")
    man = source / config.MANIFEST_FILE
    run = run or (json.loads(man.read_text(encoding="utf-8")).get("run_date") if man.exists() else None) or source.name
    date.fromisoformat(run)
    dest = Path(data_root or config.FSDR_DATA_DIR) / run
    dest.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dest / "outputs", ignore_errors=True)
    for f in dest.glob("*.json*"):                               # a previous ingestion's files
        f.unlink()
    files = []
    for table in schemas.TABLES:
        src_file = _table_file(source, table)
        if src_file is None:
            files.append({"table": table, "file": None, "copied": False, "identical": None})
            continue
        shutil.copy2(src_file, dest / src_file.name)
        files.append({"table": table, "file": src_file.name, "copied": True, "bytes": src_file.stat().st_size,
                      "sha256": _sha(src_file)[:16], "identical": _sha(src_file) == _sha(dest / src_file.name)})
    if man.exists():
        shutil.copy2(man, dest / config.MANIFEST_FILE)
    return dest, files


def _share(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def table_status(files: list[dict], rep) -> list[dict]:
    """One row per contract table: green (ok), yellow (warning), red (blocking / missing required), grey (optional,
    not provided)."""
    by_table: dict[str, list] = {}
    for i in rep.issues:
        by_table.setdefault(i.table, []).append(i)
    rows = []
    for f in files:
        s = schemas.TABLES[f["table"]]
        tr = rep.tables.get(f["table"])
        issues = by_table.get(f["table"], [])
        levels = {i.level for i in issues}
        if not f["copied"]:
            status = "grey" if not s.used_by_model and s.family != "model_score" else "red" if "blocking" in levels else "yellow"
        else:
            status = "red" if "blocking" in levels else "yellow" if "warning" in levels else "green"
        rows.append({"table": f["table"], "group": s.group, "used_by": _used_by(s), "status": status,
                     "rows": tr.rows if tr else 0, "fields": len(s.fields),
                     "missing_fields": ", ".join(tr.missing_fields) if tr and tr.missing_fields else "",
                     "identical_copy": f.get("identical"), "sha256": f.get("sha256", ""),
                     "issues": "; ".join(i.message for i in issues if i.level != "info")[:300]})
    return rows


def _used_by(s) -> str:
    if s.family == "model_score":
        return "NBA features (scores)"
    if s.name == config.TARGET_SAMPLE_TABLE:
        return "arms / scoring population"
    return {"base": "eligibility, features", "usage": "NBA features (trends)", "pricing": "ARPU anchor (offers)",
            "campaign": "training labels, rewards, ranker", "sample": "arms"}.get(s.group, s.group) \
        if s.used_by_model else "not used by the model"


def merge_checks(src: JsonSource) -> list[dict]:
    """The joins the NBA agent relies on, from the scoring population's point of view."""
    out = []

    def add(name, why, left, right, green_at=0.99, yellow_at=0.70):
        left = {v for v in left if v is not None}
        m = len(left & right)
        share = _share(m, len(left))
        out.append({"join": name, "why": why, "left_keys": len(left), "matched": m, "match_rate": share,
                    "status": "green" if share >= green_at else "yellow" if share >= yellow_at else "red"})

    base = upper(src.read("base_prepaid"))
    sample = src.read(config.TARGET_SAMPLE_TABLE)
    scored = set(norm_val(sample.loc[sample["group"].astype(str).str.lower() != "cg", "access_method_val"]).dropna())
    add("test sample -> base_prepaid (ACCESS_METHOD_VAL)", "a scored line must be an active prepaid line", scored,
        set(base["ACCESS_METHOD_VAL"].dropna()))
    acct = upper(src.read("consumer_aact_account"))
    add("base_prepaid -> consumer_aact_account (ACCOUNT_ID)", "account -> party (BR-11)", set(base["ACCOUNT_ID"].dropna()),
        set(acct["ACCOUNT_ID"].dropna()))
    party = upper(src.read("party_scoring"))
    add("consumer_aact_account -> party_scoring (PARTY_ID)", "nationality group (BR-15)", set(acct["PARTY_ID"].dropna()),
        set(party["PARTY_ID"].dropna()))
    for m in config.MODEL_SCORE_REGISTRY:
        if src.has(m.table):
            ids = set(norm_val(src.read(m.table)["access_method_val"]).dropna())
            limited = m.tenure_range is not None
            add(f"test sample -> {m.table}", f"{m.feature}" + (f" (tenure {m.tenure_range[0]}-{m.tenure_range[1]} d only)" if limited else ""),
                scored, ids, green_at=0.0 if limited else config.MIN_SCORE_COVERAGE, yellow_at=0.0 if limited else 0.3)
    for p in WEEKS + MONTHS:
        t = schemas.activity_table_name(p, "D")
        if src.has(t):
            add(f"test sample -> {t}", f"usage {p} (missing = 0, BR-48)", scored, set(upper(src.read(t))["ACCESS_METHOD_VAL"].dropna()),
                green_at=0.5, yellow_at=0.2)
    for t, why in (("subs_amt_score", "subscription revenue (BR-49)"), ("subscriptions_tg", "ARPU anchor: subscriptions (BR-68)"),
                   ("usage_data_voice_payg_tg", "ARPU anchor: PAYG (BR-68)")):
        if src.has(t):
            add(f"test sample -> {t}", why, scored, set(upper(src.read(t))["ACCESS_METHOD_VAL"].dropna()), green_at=0.5, yellow_at=0.2)
    am = upper(src.read("access_method_id_auto"))
    tg = upper(src.read("targeting_dataiku_auto"))
    add("targeting_dataiku_auto -> access_method_id_auto (ACCESS_METHOD_ID)", "last campaign -> MSISDN (training labels)",
        set(tg["ACCESS_METHOD_ID"].dropna()), set(am["ACCESS_METHOD_ID"].dropna()))
    cat = set(src.read("btl_catalog")["btl_name"].astype(str).str.strip())
    for t, why in (("subs_btl_auto", "rewards D-7..D-1 (BR-23/40)"), ("sub_btl_3months", "ranker history (BR-24/83)")):
        add(f"{t}.PRODUCTNAME -> btl_catalog", why + "; non-BTL products are expected", set(src.read(t)["PRODUCTNAME"].astype(str).str.strip()),
            cat, green_at=0.5, yellow_at=0.2)
    return out


def calculation_checks(src: JsonSource, ws) -> list[dict]:
    """Recomputes key values from the raw tables, independently of the feature code, and compares them with what
    the engine uses for the scored lines."""
    feat = ws.features
    vals = list(ws.scored.index)
    out = []

    def add(name, rule, expected: pd.Series, actual: pd.Series, tol=1e-6):
        e, a = expected.reindex(vals), actual.reindex(vals)
        ok = e.notna() & a.notna()
        bad = int((~np.isclose(e[ok].astype(float), a[ok].astype(float), atol=tol, rtol=1e-6)).sum())
        out.append({"check": name, "rule": rule, "lines_checked": int(ok.sum()), "mismatches": bad,
                    "status": "green" if bad == 0 and ok.sum() else "red" if bad else "yellow",
                    "example": _example(e, a, ok)})

    def raw_sum(table, col):
        if not src.has(table):
            return pd.Series(dtype=float)
        df = upper(src.read(table))
        return pd.to_numeric(df.get(col), errors="coerce").fillna(0).groupby(df["ACCESS_METHOD_VAL"]).sum() \
            if col in df.columns else pd.Series(dtype=float)

    for p in ("W1", "M1", "M3"):
        sub = raw_sum("subs_amt_score", f"SUB_AMT_{p}")
        pg = raw_sum(schemas.activity_table_name(p, "D"), f"PG_DATA_AMOUNT_{p}")
        exp = sub.reindex(vals).fillna(0) + pg.reindex(vals).fillna(0)
        add(f"REV_{p} = SUB_AMT_{p} + PG_DATA_AMOUNT_{p}", "BR-49 (missing usage = 0, BR-48)", exp, feat[f"REV_{p}"])
    w1, w2 = feat["REV_W1"].reindex(vals), feat["REV_W2"].reindex(vals)
    exp = pd.Series(np.where(w2 == 0, 0.0, np.clip((w1 - w2) / w2.where(w2 != 0, 1) * 100, *config.PCT_CLIP)), index=vals)
    add("VAR_REV_W1_W2 = (W1 - W2) / W2 x 100", "BR-50..53 (0 when W2 = 0, clipped to [-100, 500])", exp, feat["VAR_REV_W1_W2"])
    churn = [m.table for m in config.MODEL_SCORE_REGISTRY if m.feature == "PROBA_CHURN" and src.has(m.table)]
    frames = [pd.DataFrame({"V": norm_val(src.read(t)["access_method_val"]), "P": pd.to_numeric(src.read(t)["proba_1"], errors="coerce")})
              for t in churn]
    raw = pd.concat(frames).dropna().groupby("V")["P"].max() if frames else pd.Series(dtype=float)
    add(f"PROBA_CHURN = max over the {len(churn)} churn files", "BR-46 (rows of one feature merged by max)", raw, feat["PROBA_CHURN"])
    rev = {}
    for p in MONTHS:
        rev[p] = raw_sum("subscriptions_tg", f"TOTAL_SUB_AMT_{p}").reindex(vals).fillna(0) + \
            raw_sum("usage_data_voice_payg_tg", f"PG_TOTAL_AMOUNT_{p}").reindex(vals).fillna(0)
    m = pd.DataFrame(rev)
    exp = m.mean(axis=1) + m.std(axis=1, ddof=1).fillna(0)
    anchors = ws.engine.anchors["AVG_REV_MONTHLY"].copy()
    anchors.index = anchors.index.astype(str)
    if config.ANCHOR_MAX_WEEKLY_GROWTH is not None and "ANCHOR_CAPPED" in ws.engine.anchors.columns:
        capped = set(ws.engine.anchors.index[ws.engine.anchors["ANCHOR_CAPPED"]].astype(str))
        exp = exp[[v not in capped for v in exp.index]]
    add("monthly ARPU anchor = mean + std(M1..M3) of subscriptions + PAYG", "BR-68..71 (ddof = 1)", exp, anchors, tol=1e-4)
    nat = feat["NATIONALITY"].reindex(vals)
    grp = pd.Series(np.where(nat.isin(config.NATIONALITY_KEEP), nat, config.NATIONALITY_OTHERS), index=vals)
    same = (grp == feat["NATIONALITY_GROUP"].reindex(vals)).astype(float)
    add("NATIONALITY_GROUP from party_scoring.NATIONALITY", f"BR-15 ({', '.join(config.NATIONALITY_KEEP)}, else others)",
        pd.Series(1.0, index=vals), same)
    return out


def _example(e: pd.Series, a: pd.Series, ok: pd.Series) -> str:
    idx = ok[ok].index
    if not len(idx):
        return ""
    bad = [i for i in idx if not np.isclose(float(e[i]), float(a[i]), atol=1e-4, rtol=1e-6)]
    i = bad[0] if bad else idx[0]
    return f"{i}: recomputed {float(e[i]):.4f}, engine {float(a[i]):.4f}"


def ingest(source: Path, run: str | None = None, prepare: bool = True) -> dict:
    """Step 1: copy -> validate -> joins -> train + score -> recomputed calculations."""
    from nbx.fsdr.agents.tools import FsdrWorkspace
    from nbx.fsdr.pipelines.score import run_scoring
    from nbx.fsdr.pipelines.train import run_training
    from nbx.fsdr.validation import validate_folder

    t0 = time.time()
    dest, files = copy_tables(source, run)
    rep = validate_folder(dest)
    src = JsonSource(dest)
    out = {"run": dest.name, "source": str(source), "destination": str(dest), "validation_status": rep.status,
           "usable": rep.status != "blocked", "tables": table_status(files, rep), "joins": merge_checks(src),
           "same_data": all(f["identical"] for f in files if f["copied"]),
           "tables_copied": sum(f["copied"] for f in files), "tables_in_contract": len(files)}
    if prepare and out["usable"]:
        tr = run_training(dest)
        sc = run_scoring(dest)
        out["training"] = {"model": tr["version"], "promoted": tr["promoted"], "rule": tr["promotion_rule"],
                           "labelled_rows": tr["labels"]["labelled_rows"], "features": tr["features"]["model_features"],
                           "label_source": tr["labels"]["label_source"]}
        out["scoring"] = {"model": sc["model_version"], "scored_lines": sc["population_report"]["scored_lines"],
                          "groups": sc["population_report"]["sample_groups"], "max_psi": sc["drift"]["max_psi"]}
        ws = FsdrWorkspace(dest)
        out["calculations"] = calculation_checks(src, ws)
        out["lines"] = {arm: ws.lines(arm)[:5] for arm in ("llm", "cstr", "free")}
    out["seconds"] = round(time.time() - t0, 1)
    (outputs_dir(dest) / JOURNEY_DIR).mkdir(parents=True, exist_ok=True)
    (outputs_dir(dest) / JOURNEY_DIR / "ingestion_report.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 2-6. Steps for one line (state saved between steps)
# ---------------------------------------------------------------------------
def _state_path(ws, v: str) -> Path:
    p = outputs_dir(ws.data_dir) / JOURNEY_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{v}.json"


def load_state(ws, v: str) -> dict:
    p = _state_path(ws, v)
    st = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if st.get("model_version") != ws.model_version:            # re-scored since: start again
        st = {}
    return st or {"msisdn": v, "run": ws.data_dir.name, "model_version": ws.model_version, "steps": {}, "decision_state": {}}


def save_state(ws, v: str, st: dict) -> None:
    st["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _state_path(ws, v).write_text(json.dumps(st, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _tool(events: list[dict], tool: str):
    return next((e["result"] for e in reversed(events) if e.get("type") == "tool_result" and e.get("tool") == tool), None)


def _action_explanation(run, contact: dict, seg: dict | None) -> str:
    """Plain-words reason for the action, from the evidence (used as is offline, beside the LLM's words online)."""
    pmf = run.ctx.pmf()
    top = run.ctx.bandit_top()
    parts = [f"The bandit (model {run.ws.model_version}) gives {top} the highest probability ({pmf.get(top, 0):.2f})"]
    llm_action = run.justification.get("llm_action")
    if run.policy_source == "fallback":
        parts.append(f"no LLM reachable: the action sent is {run.action}, sampled from that distribution "
                     f"(p = {pmf.get(run.action, 0):.2f}, production BR-57: sampling keeps the model learning)")
    elif run.policy_source == "bandit":
        parts.append(f"the LLM chose {llm_action}, but this decision fell in the {config.EPSILON_LLM:.0%} exploration share "
                     f"(FSDR_EPSILON_LLM): the code sent a bandit sample instead, {run.action} (logged propensity "
                     f"{float(run.propensity):.3f}), so the llm arm's logs stay usable for learning")
    else:
        parts.append(f"the LLM chose {run.action}" + (f", overriding the bandit's {top}" if run.action != top else ", as the bandit")
                     + f" (logged propensity {float(run.propensity):.3f})")
    if seg and seg.get("segment"):
        parts.append(f"the legacy segment policy maps {seg.get('segment')} to {seg.get('segment_action')}")
    parts.append(contact.get("verdict", ""))
    if run.findings:
        parts.append("guards applied: " + "; ".join(run.findings))
    return ". ".join(p.rstrip(".")[:1].upper() + p.rstrip(".")[1:] for p in parts if p) + "."


async def _run_for(ws, v: str, st: dict, llm: str | None):
    from nbx.fsdr.agents.recommender_agent import DecisionRun

    run = await DecisionRun(ws, v).open(llm)
    if st.get("decision_state"):
        run.restore(st["decision_state"])
    return run


async def step_action(ws, v: str, llm: str | None = None) -> dict:
    from nbx.fsdr.agents.tools import FSDR_TOOLS

    st = load_state(ws, v)
    run = await _run_for(ws, v, {}, llm)                         # a new action always starts a new decision
    await run.action_stage()
    contact = _tool(run.events, "contact_policy") or FSDR_TOOLS["contact_policy"].fn(run.ctx, {})
    seg = FSDR_TOOLS["segment_policy"].fn(run.ctx, {})
    hist = FSDR_TOOLS["campaign_history"].fn(run.ctx, {})
    j = run.justification
    res = {"action": run.action, "policy_source": run.policy_source, "propensity": round(float(run.propensity), 4),
           "bandit_top_action": run.ctx.bandit_top(), "bandit_pmf": {a: round(p, 4) for a, p in run.ctx.pmf().items()},
           "mode": run.mode, "llm_model": run.model,
           "llm_answer": {"thoughts": j.get("llm_words", []), "llm_action": j.get("llm_action"),
                          "signals_used": j.get("signals_used", []), "override_reason": j.get("override_reason", ""),
                          "confidence": j.get("confidence"), "expected_acceptance": j.get("expected_acceptance")},
           "explanation": _action_explanation(run, contact, seg), "guards": run.findings,
           "contact_policy": contact, "segment_policy": seg, "campaign_history": hist,
           "scores": {k: round(float(run.ctx.row.get(k, np.nan)), 4) for k in config.proba_features()},
           "trace": run.events}
    st = {**st, "steps": {"action": res}, "decision_state": run.state()}     # later steps are re-done after a new action
    save_state(ws, v, st)
    return res


async def _ensure(ws, v: str, step: str, llm: str | None) -> dict:
    st = load_state(ws, v)
    order = STEPS[: STEPS.index(step)]
    for s in order:
        if s not in st["steps"]:
            await STEP_FUNCS[s](ws, v, llm)
            st = load_state(ws, v)
    return st


async def step_offers(ws, v: str, llm: str | None = None) -> dict:
    st = await _ensure(ws, v, "offers", llm)
    run = await _run_for(ws, v, st, llm)
    run.events = []
    await run.offer_stage()
    acc = ws.engine.acceptance
    cat = ws.catalog
    picks = run.picks
    offers = []
    for o in run.offer_details(run.offers):
        n = o["offer"]
        row = cat.loc[n]
        p = picks.get(n)
        p_acc = p.p_accept if p is not None and p.p_accept is not None else (
            float(acc.p_accept(v, run.action, cat.loc[[n], ["SERVICE", "GENEROSITY", "VALIDITY", "PRICE"]])[0]) if acc else None)
        offers.append({**o, "gb": float(row.get("GB") or 0), "minutes": int(row.get("MIN") or 0),
                       "p_accept": round(p_acc, 4) if p_acc is not None else None,
                       "expected_value_omr": round(p_acc * o["price"] * config.OFFER_MARGIN, 4) if p_acc is not None else None,
                       "rank_in_family": p.rank if p is not None else None,
                       "selected_by": {1: "price band pass 1", 2: "price band pass 2", 3: "price band pass 3", 4: "value (P x price)",
                                       5: "exploration"}.get(o["pass"], "LLM choice"),
                       "sms_en": ws.scripts.get(n, ("", ""))[0]})
    res = {"action": run.action, "offers": offers, "rule_based_offers": list(picks), "offer_policy": "value" if ws.engine.uses_value else "price_band",
           "llm_answer": {"thoughts": [e["text"] for e in run.events if e.get("type") == "thought"], "rationale": run.offer_rationale,
                          "override_reason": next((o["reason"] for o in run.overrides if o["type"] == "offers"), "")},
           "guards": run.findings, "mode": run.mode, "llm_model": run.model,
           "explanation": _offer_explanation(run, offers), "trace": run.events}
    st["steps"] = {k: st["steps"][k] for k in ("action",)} | {"offers": res}
    st["decision_state"] = run.state()
    save_state(ws, v, st)
    return res


def _offer_explanation(run, offers: list[dict]) -> str:
    if run.action == config.NO_CONTACT:
        return "No contact this week: no offer."
    if not offers:
        return ("No offer: the line's ARPU anchor is above every price of the action's families (BR-82), or the family "
                "has no offer for this validity.")
    fams = ", ".join(f"{s} {g}" for s, g in __import__("nbx.fsdr.offer_rules", fromlist=["families"]).families(run.action, "llm"))
    return (f"Action {run.action} allows the families {fams} (BR-66). Up to 2 offers per validity, 4 in total (BR-82), no "
            f"conflicting pair. Within each family the offer with the highest P(accept) x price above the line's ARPU anchor "
            f"is chosen" + (" (value policy)." if run.ws.engine.uses_value else " (production price bands)."))


async def step_addons(ws, v: str, llm: str | None = None) -> dict:
    st = await _ensure(ws, v, "addons", llm)
    main = [o["offer"] for o in st["steps"]["offers"]["offers"]]
    if st["steps"]["action"]["action"] == config.NO_CONTACT:
        res = {"roaming": [], "international": [], "international_reason": "no contact this week", "roaming_basis": None}
    else:
        res = ws.addons.recommend(v, main)
    res["rules"] = (f"top {config.ADDON_TOP_ROAMING} roaming packs ranked on the line's roaming history (popularity + price fit "
                    f"without history); international packs only with >= {config.INTL_MIN_CALLS_M1} international calls in M1; "
                    "never an add-on that conflicts with a main offer")
    st["steps"] = {k: st["steps"][k] for k in ("action", "offers")} | {"addons": res}
    save_state(ws, v, st)
    return res


async def step_competitors(ws, v: str, llm: str | None = None) -> dict:
    st = await _ensure(ws, v, "competitors", llm)
    run = await _run_for(ws, v, st, llm)
    run.events = []
    await run.competitor_stage()
    c = run.competitor
    res = {"verdict": c["verdict"], "summary": c["summary"], "comparisons": c["comparisons"], "swaps_applied": c["swaps_applied"],
           "final_offers": run.offers, "llm_answer": {"thoughts": c.get("llm_words", []), "summary": c["summary"]},
           "mode": run.mode, "llm_model": run.model, "guards": run.findings, "trace": run.events}
    st["steps"] = {k: st["steps"][k] for k in ("action", "offers", "addons")} | {"competitors": res}
    st["decision_state"] = run.state()
    save_state(ws, v, st)
    return res


def _row(df: pd.DataFrame | None, v: str) -> dict:
    if df is None or v not in df.index:
        return {}
    r = df.loc[v]
    r = r.iloc[-1] if isinstance(r, pd.DataFrame) else r
    return {k: (None if (isinstance(x, float) and np.isnan(x)) else x.item() if hasattr(x, "item") else x) for k, x in r.to_dict().items()}


def customer_context(ws, v: str) -> dict:
    """Every column of every table row that concerns the line (raw tables) + what the engine computed for it."""
    src = ws.src
    raw: dict[str, list[dict]] = {}
    aid = ws.base.at[v, "ACCESS_METHOD_ID"] if v in ws.base.index else None
    acc = ws.base.at[v, "ACCOUNT_ID"] if v in ws.base.index else None
    party = ws.base.at[v, "PARTY_ID"] if v in ws.base.index else None
    for table in schemas.TABLES:
        if not src.has(table):
            continue
        df = src.read(table)
        cols = {c.upper(): c for c in df.columns}
        mask = pd.Series(False, index=df.index)
        for key, val in (("ACCESS_METHOD_VAL", v), ("MSISDN", v), ("ACCESS_METHOD_ID", aid), ("ACCOUNT_ID", acc), ("PARTY_ID", party)):
            if key in cols and val is not None and not pd.isna(val):
                col = df[cols[key]]
                mask |= norm_val(col).astype(str) == str(int(val) if isinstance(val, (int, np.integer, float)) else val)
        rows = df[mask]
        if len(rows):
            raw[table] = json.loads(rows.head(20).to_json(orient="records", date_format="iso"))
    anchors = ws.engine.anchors.copy()
    anchors.index = anchors.index.astype(str)
    return {"msisdn": v, "arm": _row(ws.scored, v).get("TG_FLAG"), "raw_tables": raw,
            "engine": {"features": _row(ws.features, v), "scoring": _row(ws.scored, v), "anchors": _row(anchors, v),
                       "eligible_base": _row(ws.base, v), "consent_blocked": v in ws.engine.consent_blocked}}


async def step_output(ws, v: str, llm: str | None = None) -> dict:
    st = await _ensure(ws, v, "output", llm)
    run = await _run_for(ws, v, st, llm)
    run.events = []
    decision = run.finish()
    steps = {k: {kk: vv for kk, vv in s.items() if kk != "trace"} for k, s in st["steps"].items()}
    out = {"msisdn": v, "run": ws.data_dir.name, "generated_at": datetime.now().isoformat(timespec="seconds"),
           "model_version": ws.model_version, "llm_backend": decision["mode"], "llm_model": decision["model"],
           "decision": decision, "steps": steps, "context": customer_context(ws, v),
           "traces": {k: s.get("trace", []) for k, s in st["steps"].items()}}
    path = outputs_dir(ws.data_dir) / JOURNEY_DIR / f"{v}_output.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    st["steps"]["output"] = {"file": str(path)}
    save_state(ws, v, st)
    return {"file": str(path), "summary": flat(out), "output": out}


STEP_FUNCS = {"action": step_action, "offers": step_offers, "addons": step_addons, "competitors": step_competitors,
              "output": step_output}


# ---------------------------------------------------------------------------
# Batch: the same step for a list of lines
# ---------------------------------------------------------------------------
def parse_msisdns(text: str) -> list[str]:
    """MSISDNs from a pasted list or an uploaded .csv / .txt: the 'msisdn' / 'access_method_val' column if there is
    a header, else the first column; duplicates removed, order kept."""
    lines = [l for l in text.replace("\r", "\n").split("\n") if l.strip()]
    if not lines:
        return []
    rows = list(csv.reader(lines))
    head = [h.strip().lower() for h in rows[0]]
    col = next((i for i, h in enumerate(head) if h in ("msisdn", "access_method_val")), None)
    body = rows[1:] if col is not None or not head[0].replace(".", "").isdigit() else rows
    col = col or 0
    out: list[str] = []
    for r in body:
        if len(r) > col:
            s = str(r[col]).strip().split(".")[0]
            if s.isdigit() and s not in out:
                out.append(s)
    return out


async def run_step(ws, step: str, msisdns: list[str], llm: str | None = None, concurrency: int | None = None) -> dict:
    if step not in STEP_FUNCS:
        raise ValueError(f"step must be one of {STEPS}")
    if len(msisdns) > config.BATCH_MAX:
        raise ValueError(f"at most {config.BATCH_MAX} lines per call")
    sem = asyncio.Semaphore(max(1, concurrency or config.AGENT_CONCURRENCY))

    async def one(v: str) -> dict:
        v = str(v).strip()
        if v not in ws.scored.index:
            return {"msisdn": v, "error": "not in the scored sample of this run (control group or unknown line)"}
        async with sem:
            from nbx.core import tracing
            from nbx.core.privacy import pseudonym

            with tracing.span(f"fsdr_journey:{step}", "chain", inputs={"customer_ref": pseudonym(v), "run": ws.data_dir.name}) as sp:
                try:
                    res = {"msisdn": v, **await STEP_FUNCS[step](ws, v, llm)}
                except Exception as e:  # noqa: BLE001  (one bad line must not stop the batch)
                    res = {"msisdn": v, "error": f"{type(e).__name__}: {e}"}
                sp.end(outputs={k: x for k, x in res.items() if k in ("action", "verdict", "explanation", "file", "error")},
                       error=res.get("error"))
                return res

    t0 = time.time()
    results = list(await asyncio.gather(*(one(v) for v in msisdns)))
    files = {}
    if step == "output" or len(msisdns) > 1:
        files = write_batch(ws, step, results)
    return {"step": step, "count": len(results), "errors": sum(1 for r in results if r.get("error")),
            "seconds": round(time.time() - t0, 1), "results": results, "files": files}


def flat(out: dict) -> dict:
    """One wide row per line: the decision, every step, every engine column of the line."""
    d, s, ctx = out["decision"], out["steps"], out["context"]["engine"]
    act, offers, add, comp = s.get("action", {}), s.get("offers", {}), s.get("addons", {}), s.get("competitors", {})
    row = {"msisdn": out["msisdn"], "run": out["run"], "arm": out["context"].get("arm"), "model_version": out["model_version"],
           "llm_backend": out["llm_backend"], "llm_model": out["llm_model"], "action": d["action"],
           "policy_source": d["policy_source"], "propensity": d["propensity"], "bandit_top_action": d["bandit_top_action"],
           "action_explanation": act.get("explanation"), "action_llm_answer": " | ".join(act.get("llm_answer", {}).get("thoughts") or []),
           "action_override_reason": act.get("llm_answer", {}).get("override_reason"),
           "offer_llm_rationale": offers.get("llm_answer", {}).get("rationale"),
           "competitor_verdict": comp.get("verdict"), "competitor_summary": comp.get("summary"),
           "international_reason": add.get("international_reason"), "critic": d["critic"]["verdict"]}
    for i, o in enumerate(d["offers"][:4]):
        po = next((x for x in offers.get("offers", []) if x["offer"] == o["offer"]), {})
        row.update({f"offer_{i + 1}": o["offer"], f"offer_{i + 1}_price": o["price"], f"offer_{i + 1}_validity": o["validity"],
                    f"offer_{i + 1}_p_accept": po.get("p_accept")})
    for i, o in enumerate((add.get("roaming") or [])[:2]):
        row.update({f"roaming_{i + 1}": o["offer"], f"roaming_{i + 1}_price": o["price"]})
    for i, o in enumerate((add.get("international") or [])[:2]):
        row.update({f"international_{i + 1}": o["offer"], f"international_{i + 1}_price": o["price"]})
    for prefix, block in (("feat", ctx.get("features", {})), ("score", ctx.get("scoring", {})), ("anchor", ctx.get("anchors", {})),
                          ("base", ctx.get("eligible_base", {}))):
        for k, val in block.items():
            row[f"{prefix}_{k}"] = val
    return row


def write_batch(ws, step: str, results: list[dict]) -> dict:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = outputs_dir(ws.data_dir) / JOURNEY_DIR
    jpath = folder / f"batch_{step}_{stamp}.json"
    slim = [{k: v for k, v in r.items() if k not in ("trace", "output")} for r in results]
    jpath.write_text(json.dumps(slim, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    rows = [r["summary"] if step == "output" and "summary" in r else _step_row(step, r) for r in results]
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    cpath = folder / f"batch_{step}_{stamp}.csv"
    with cpath.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return {"json": str(jpath), "csv": str(cpath)}


def _step_row(step: str, r: dict) -> dict:
    if r.get("error"):
        return {"msisdn": r["msisdn"], "error": r["error"]}
    if step == "action":
        return {"msisdn": r["msisdn"], "action": r["action"], "propensity": r["propensity"], "bandit_top_action": r["bandit_top_action"],
                "policy_source": r["policy_source"], "mode": r["mode"], "explanation": r["explanation"],
                "llm_answer": " | ".join(r["llm_answer"]["thoughts"]), "override_reason": r["llm_answer"]["override_reason"]}
    if step == "offers":
        row = {"msisdn": r["msisdn"], "action": r["action"], "rationale": r["llm_answer"]["rationale"]}
        for i, o in enumerate(r["offers"][:4]):
            row.update({f"offer_{i + 1}": o["offer"], f"offer_{i + 1}_price": o["price"], f"offer_{i + 1}_p_accept": o["p_accept"]})
        return row
    if step == "addons":
        return {"msisdn": r["msisdn"], "roaming": ", ".join(o["offer"] for o in r.get("roaming", [])),
                "international": ", ".join(o["offer"] for o in r.get("international", [])), "international_reason": r.get("international_reason")}
    if step == "competitors":
        return {"msisdn": r["msisdn"], "verdict": r["verdict"], "summary": r["summary"],
                "swaps": "; ".join(f"{s['out']} -> {s['in']}" for s in r.get("swaps_applied", []))}
    return {"msisdn": r["msisdn"]}
