"""Tools of the FSDR agents (action, offer, competitor). Evidence, not orders.

Same Tool dataclass and same call path as the NBX council tools
(nbx/agents/agentic/tools.py); the FSDR agents pass FSDR_TOOLS as the
registry of the shared ReAct loop. Every tool reads the run loaded once in an
FsdrWorkspace (the JSON folder + the scoring output) and returns compact,
rounded facts that fit a small local model's context.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nbx.agents.agentic.tools import Tool, ToolError
from nbx.core import vw_format
from nbx.fsdr import campaign_log, config, registry
from nbx.fsdr.calendar import MONTHS, WEEKS
from nbx.fsdr.datasource import JsonSource
from nbx.fsdr.features import build_feature_frame, eligible_base, norm_id, norm_val
from nbx.fsdr.labels import access_method_map, load_fulfilments
from nbx.fsdr.offer_rules import anchor_value, check_offers, customer_eligibility, families, k_margin
from nbx.fsdr.pipelines import read_table
from nbx.fsdr.pipelines.recommend import OfferEngine
from nbx.fsdr.pipelines.score import target_population
from nbx.fsdr.pipelines.train import load_and_validate

SERVICE_TO_CATEGORY = {"DATA": "Data", "COMBO": "Bundle", "VOICE": "Voice"}
_LABEL_RE = re.compile("|".join(re.escape(k) for k in config.COMPETITOR_LABELS) or r"$^")


def label(text: Any) -> Any:
    """Real competitor names -> configurable labels (COMPETITOR_A / COMPETITOR_B), also inside plan names."""
    if not isinstance(text, str):
        return text
    return _LABEL_RE.sub(lambda m: config.COMPETITOR_LABELS[m.group(0)], text)


def _r(x, nd: int = 3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, nd)


# ---------------------------------------------------------------------------
# Workspace: one run loaded once, shared by every customer and agent
# ---------------------------------------------------------------------------
class FsdrWorkspace:
    def __init__(self, data_dir: str | Path, models_root: Path | None = None, engine: OfferEngine | None = None):
        self.data_dir = Path(data_dir)
        self.src, self.run_date, _ = load_and_validate(self.data_dir)
        try:
            self.scored = read_table(self.data_dir, "results_scoring_nba")
        except FileNotFoundError as e:
            raise FileNotFoundError("no scoring output: run fsdr-score first") from e
        self.scored["ACCESS_METHOD_VAL"] = self.scored["ACCESS_METHOD_VAL"].astype(str)
        self.scored = self.scored.set_index("ACCESS_METHOD_VAL", drop=False)
        self.model_version = str(self.scored["MODEL_VERSION"].iat[0]) if len(self.scored) else None
        pop, _ = target_population(self.src)
        # same seed as the scoring pipeline's default, so missing-score fills match the scored features
        self.log_root = campaign_log.root_for(self.data_dir)
        feat, _ = build_feature_frame(self.src, pop, "D", self.run_date.toordinal(), log_root=self.log_root,
                                      context_date=self.run_date)
        self.features = feat.set_index("ACCESS_METHOD_VAL", drop=False)
        base, _ = eligible_base(self.src)
        self.base = base.set_index("ACCESS_METHOD_VAL", drop=False)
        self.engine = engine or OfferEngine(self.src, log_root=self.log_root, run_date=self.run_date)
        self.catalog = self.engine.catalog.set_index("BTL_NAME", drop=False)
        self._load_campaign()
        self._load_scripts()
        self._competitors = None
        self._addons = None

    @property
    def cohort_graph(self):
        """core/knowledge_graph over THIS run's data: lines -> device segment (DEVICEBEHAVIORSEGMENTATION.dv_segment),
        lines -> BTL bought in the last 270 days (sub_btl_3months), BTL -> service (built on first use)."""
        if getattr(self, "_cohort_graph", None) is None:
            from nbx.core import knowledge_graph

            dv = self.src.read("DEVICEBEHAVIORSEGMENTATION.dv_segment") if self.src.has("DEVICEBEHAVIORSEGMENTATION.dv_segment") else None
            seg_by_id = dict(zip(norm_id(dv["ACCESS_METHOD_ID"]), dv["DV_SEGMENT"].astype(str))) if dv is not None else {}
            lines = self.base[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID"]].drop_duplicates("ACCESS_METHOD_VAL")
            customers = [{"msisdn": v, "dv_segment": seg_by_id.get(i, "UNKNOWN"), "revenue_segment": "-", "usage_type": "-"}
                         for v, i in zip(lines["ACCESS_METHOD_VAL"], lines["ACCESS_METHOD_ID"])]
            cat = self.catalog
            offers = [{"program_rule": n, "category": s, "price_omr": float(p), "offer_name": n}
                      for n, s, p in zip(cat["BTL_NAME"], cat["SERVICE"], cat["PRICE"])]
            service = dict(zip(cat["BTL_NAME"], cat["SERVICE"]))
            purchases = [{"msisdn": v, "program_rule": b, "category": service.get(b, ""), "purchase_count": float(w)}
                         for (v, b), w in self.history.items()]
            segments = [{"segment": s, "typical_behavior": ""} for s in sorted(set(seg_by_id.values()) | {"UNKNOWN"})]
            self._cohort_graph = knowledge_graph.build_graph_from(customers, offers, purchases, segments)
        return self._cohort_graph

    @property
    def addons(self):
        """Roaming / international add-on engine (built on first use)."""
        if self._addons is None:
            from nbx.fsdr.addons import AddonEngine

            self._addons = AddonEngine(self.src, self.engine)
        return self._addons

    def _load_campaign(self) -> None:
        raw = self.src.read("targeting_dataiku_auto")
        offer_cols = [c for c in raw.columns if str(c).lower().startswith("destination_offer_name_")]
        tg = pd.DataFrame({"ACCESS_METHOD_ID": norm_id(raw["access_method_id"]), "DATE1": raw["date1"], "SEGMENT": raw["segment"],
                           "OFFERS": [[o for o in r if o] for r in raw[offer_cols].itertuples(index=False)]})
        tg = tg.merge(access_method_map(self.src), on="ACCESS_METHOD_ID", how="left").dropna(subset=["ACCESS_METHOD_VAL"])
        self.targeting = tg.sort_values("DATE1").drop_duplicates("ACCESS_METHOD_VAL", keep="last").set_index("ACCESS_METHOD_VAL")
        self.fulfilments = load_fulfilments(self.src, self.run_date)
        hist = self.engine.interactions
        self.history = hist.groupby(["ACCESS_METHOD_VAL", "BTL_NAME"])["WEIGHT"].sum()

    def _load_scripts(self) -> None:
        s = self.src.read("BTL_script_english_arabic_ok") if self.src.has("BTL_script_english_arabic_ok") else pd.DataFrame()
        self.scripts = {r["destination_offer_name"]: (r["English Script"], r["Arabic Script"]) for r in s.to_dict("records")} if len(s) else {}

    def competitor_offers(self) -> list[dict]:
        if self._competitors is None:
            from nbx.core import competitors

            self._competitors = competitors.load_offers()
        return self._competitors

    def lines(self, arm: str | None = None) -> list[str]:
        flag = config.GROUP_TO_FLAG.get(arm) if arm else None
        s = self.scored if flag is None else self.scored[self.scored["TG_FLAG"] == flag]
        return s["ACCESS_METHOD_VAL"].tolist()

    def context(self, msisdn: str, arm: str = "llm") -> "FsdrContext":
        val = str(msisdn)
        if val not in self.scored.index:
            raise LookupError(f"msisdn {val} is not in the scored population of {self.data_dir.name}")
        return FsdrContext(ws=self, msisdn=val, arm=arm)


@dataclass
class FsdrContext:
    """Per-customer run state shared by the three agents (same role as NBX's RunContext)."""

    ws: FsdrWorkspace
    msisdn: str
    arm: str = "llm"
    action: str | None = None
    offers: list[str] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def row(self) -> dict:
        return self.ws.features.loc[self.msisdn].to_dict() if self.msisdn in self.ws.features.index else {}

    @property
    def scored(self) -> dict:
        return self.ws.scored.loc[self.msisdn].to_dict()

    def pmf(self) -> dict[str, float]:
        s = self.scored
        return {a: float(s.get(f"P_{vw_format.action_token(a)}", 0.0)) for a in config.ACTIONS}

    def bandit_top(self) -> str:
        p = self.pmf()
        return max(p, key=p.get)

    def churn(self) -> float:
        return float(self.row.get("PROBA_CHURN", 0.0))

    def segment(self) -> str | None:
        t = self.ws.targeting
        return t.at[self.msisdn, "SEGMENT"] if self.msisdn in t.index else None

    def eligibility_issues(self) -> list[str]:
        base = self.ws.base.loc[self.msisdn].to_dict() if self.msisdn in self.ws.base.index else None
        return customer_eligibility(base, self.segment())

    def anchors(self) -> dict:
        a = self.ws.engine.anchors
        if self.msisdn not in a.index:
            return {}
        r = a.loc[self.msisdn]
        return {p: {"total": _r(r[f"AVG_REV_{p}"]), "data": _r(r[f"AVG_REV_DATA_{p}"]), "voice": _r(r[f"AVG_REV_VOICE_{p}"])}
                for p in ("MONTHLY", "WEEKLY")}

    def deterministic_offers(self, action: str | None = None) -> list:
        """The constraint-arm answer for this customer (3-pass selection) -- also the offline fallback."""
        action = action or self.action
        key = ("picks", action)
        if key not in self.cache:
            self.cache[key] = self.ws.engine.select(self.msisdn, action, "cstr")
        return self.cache[key]


def _offer_list(args: dict) -> list[str]:
    raw = args.get("offers", args.get("offer"))
    if isinstance(raw, str):
        raw = [s.strip() for s in raw.split(",") if s.strip()]
    if not isinstance(raw, list) or not raw:
        raise ToolError('args must include "offers": a list of BTL offer codes, e.g. ["BTL_DATA_L2_30D_14.4GB"]')
    return [str(x).strip() for x in raw][:6]


def _offer_brief(ctx: FsdrContext, name: str) -> dict:
    o = ctx.ws.catalog.loc[name]
    return {"offer": name, "service": o["SERVICE"], "generosity": o["GENEROSITY"], "validity": int(o["VALIDITY"]),
            "price": _r(o["PRICE"]), "gb": _r(o.get("GB")), "minutes": _r(o.get("MIN"), 0)}


# ---------------------------------------------------------------------------
# Action-agent tools
# ---------------------------------------------------------------------------
def customer_360(ctx: FsdrContext, args: dict) -> dict:
    f = ctx.row
    if not f:
        raise ToolError("no features for this customer")
    voice_m1, data_m1 = float(f.get("LOCAL_VOICE_MIN_M1", 0)), float(f.get("DATA_VOLUME_MB_M1", 0))
    usage = "inactive" if voice_m1 == 0 and data_m1 == 0 else "voice_only" if data_m1 == 0 else "data_only" if voice_m1 == 0 else "dual"
    base = ctx.ws.base.loc[ctx.msisdn].to_dict() if ctx.msisdn in ctx.ws.base.index else {}
    return {
        "msisdn": ctx.msisdn, "plan": base.get("CURRENT_PLAN"), "line_since": base.get("ACCESS_METHOD_SUBS_START_DT"),
        "nationality_group": f.get("NATIONALITY_GROUP"), "usage_type": usage, "eligibility_issues": ctx.eligibility_issues(),
        "scores": {k: _r(f.get(k)) for k in config.proba_features()},
        "revenue_omr": {p: _r(f.get(f"REV_{p}"), 2) for p in WEEKS + MONTHS},
        "trends_pct": {"revenue_W1_vs_W2": _r(f.get("VAR_REV_W1_W2"), 1), "revenue_M1_vs_M2": _r(f.get("VAR_REV_M1_M2"), 1),
                       "voice_W1_vs_W2": _r(f.get("VAR_VOICE_W1_W2"), 1), "data_W1_vs_W2": _r(f.get("VAR_DATA_W1_W2"), 1),
                       "data_M1_vs_M2": _r(f.get("VAR_DATA_M1_M2"), 1)},
        "usage_M1": {"local_voice_min": _r(voice_m1, 1), "data_mb": _r(data_m1, 1)},
    }


def bandit_pmf(ctx: FsdrContext, args: dict) -> dict:
    s = ctx.scored
    pmf = ctx.pmf()
    top = ctx.bandit_top()
    return {"pmf": {a: _r(p) for a, p in sorted(pmf.items(), key=lambda kv: -kv[1])}, "top_action": top,
            "top_probability": _r(pmf[top]), "sampled_action": s.get("ACTION"), "sampled_probability": _r(s.get("PROBA")),
            "model_version": s.get("MODEL_VERSION"), "note": "VW cb_explore_adf, epsilon-greedy; pmf in the 7-action order"}


def segment_policy(ctx: FsdrContext, args: dict) -> dict:
    seg = ctx.segment()
    from nbx.fsdr.labels import segment_action

    excluded = bool(seg) and seg.strip().lower() in config.EXCLUDED_SEGMENTS
    return {"segment": seg, "segment_action": segment_action(seg), "excluded_segment": excluded,
            "note": "BR-39 segment -> action map (what the previous campaign would do)"}


def campaign_history(ctx: FsdrContext, args: dict) -> dict:
    t = ctx.ws.targeting
    last = t.loc[ctx.msisdn] if ctx.msisdn in t.index else None
    ful = ctx.ws.fulfilments
    took = ful.loc[ful["ACCESS_METHOD_VAL"] == ctx.msisdn, "PRODUCTNAME"].tolist()
    offered = list(last["OFFERS"]) if last is not None else []
    hist = ctx.ws.history.loc[ctx.msisdn] if ctx.msisdn in ctx.ws.history.index.get_level_values(0) else pd.Series(dtype=float)
    return {"last_targeting": {"date": last["DATE1"], "segment": last["SEGMENT"], "offers_sent": offered} if last is not None else None,
            "btl_taken_last_7_days": took, "took_an_offered_btl": bool(set(took) & set(offered)),
            "top_btl_last_270_days": [{"offer": k, "times": int(v)} for k, v in hist.sort_values(ascending=False).head(5).items()]}


def cohort_offers(ctx: FsdrContext, args: dict) -> dict:
    """Peers' purchases from core/knowledge_graph.cohort_offers on this run's graph."""
    from nbx.core import knowledge_graph

    g = ctx.ws.cohort_graph
    node = f"customer:{ctx.msisdn}"
    seg = next((v for _, v, d in g.out_edges(node, data=True) if d.get("kind") == "IN_SEGMENT"), None) if node in g else None
    if seg is None:
        return {"dv_segment": None, "offers": [], "note": "line not in the cohort graph"}
    dv = seg.split(":", 1)[1]
    service = str(args.get("service") or "").upper()
    rows = knowledge_graph.cohort_offers({"msisdn": ctx.msisdn, "dv_segment": dv}, top_k=50, graph=g)
    cat = ctx.ws.catalog
    out = [{"offer": r["program_rule"], "service": cat.at[r["program_rule"], "SERVICE"], "price": float(cat.at[r["program_rule"], "PRICE"]),
            "validity": int(cat.at[r["program_rule"], "VALIDITY"]), "graph_score": r["graph_score"], "peers": r["peer_count"]}
           for r in rows if r["program_rule"] in cat.index]
    if service:
        out = [o for o in out if o["service"] == service]
    return {"dv_segment": dv, "offers": out[: max(1, min(int(args.get("top_k", 5)), 10))]}


def contact_policy(ctx: FsdrContext, args: dict) -> dict:
    t = ctx.ws.targeting
    days = None
    if ctx.msisdn in t.index:
        try:
            days = (ctx.ws.run_date - date.fromisoformat(str(t.at[ctx.msisdn, "DATE1"]))).days
        except ValueError:
            days = None
    issues = ctx.eligibility_issues()
    blocked_gap = days is not None and days < config.CONTACT_MIN_GAP_DAYS
    contacts_28d = int(float(ctx.row.get("NB_CONTACTS_LAST_4W", 0) or 0))          # campaign log, D-28..D-1
    churn = ctx.churn()
    blocked = sorted(config.UPSELL_FAMILY) if churn >= config.CHURN_BLOCK_THRESHOLD else []
    reasons = list(issues)
    if blocked_gap:
        reasons.append(f"last contact {days} day(s) ago (minimum gap {config.CONTACT_MIN_GAP_DAYS})")
    if contacts_28d >= config.CONTACT_CAP_28D:
        reasons.append(f"{contacts_28d} contacts in the last {config.CONTACT_CAP_DAYS} days (cap {config.CONTACT_CAP_28D})")
    can = not reasons
    # Plain-words conclusion: small LLMs misread the threshold rule as "this customer is high risk".
    if not can:
        verdict = "Contact NOT allowed this week: " + "; ".join(reasons) + "."
    elif blocked:
        verdict = (f"Contact allowed. churn_score {_r(churn)} >= {config.CHURN_BLOCK_THRESHOLD}: upsell-family actions "
                   "are blocked; RETAIN or MAINTAIN fit.")
    else:
        verdict = (f"Contact allowed and no action is blocked: churn_score {_r(churn)} is below the "
                   f"{config.CHURN_BLOCK_THRESHOLD} threshold (low churn risk).")
    return {"verdict": verdict, "can_contact": can, "blocked_reasons": reasons, "eligibility_issues": issues,
            "days_since_last_contact": days, "min_gap_days": config.CONTACT_MIN_GAP_DAYS,
            "contacts_last_28d": contacts_28d, "contact_cap_28d": config.CONTACT_CAP_28D,
            "churn_score": _r(churn), "blocked_actions": blocked,
            "rule": f"upsell-family actions are blocked when PROBA_CHURN >= {config.CHURN_BLOCK_THRESHOLD}"}


# ---------------------------------------------------------------------------
# Offer-agent tools
# ---------------------------------------------------------------------------
def _action_arg(ctx: FsdrContext, args: dict) -> str:
    a = str(args.get("action") or ctx.action or "").upper().strip()
    if a not in config.ACTIONS:
        raise ToolError(f"action must be one of {config.ACTIONS}")
    return a


def price_band_candidates(ctx: FsdrContext, args: dict) -> dict:
    action = _action_arg(ctx, args)
    picks = ctx.deterministic_offers(action)
    return {"action": action, "rule": "3-pass price band around the ARPU anchor (BR-78..82), generosity " + ctx.ws.engine.mapping,
            "offers": [{"offer": p.btl_name, "service": p.service, "generosity": p.generosity, "validity": p.validity,
                        "price": p.price, "pass": p.pass_no, "anchor": p.anchor, "ranker_score": _r(p.score)} for p in picks]}


def ranker_top_offers(ctx: FsdrContext, args: dict) -> dict:
    action = _action_arg(ctx, args)
    n = max(1, min(int(args.get("n", 5)), 10))
    out = {}
    for validity in (30, 7):
        c = ctx.ws.engine.candidates([ctx.msisdn], action, "cstr", validity).head(n)
        out[f"{validity}d"] = [{"offer": r.BTL_NAME, "service": r.SERVICE, "generosity": r.GENEROSITY, "price": _r(r.PRICE),
                                "score": _r(r.SCORE)} for r in c.itertuples()]
    return {"action": action, "top_offers": out, "note": "BPR ranker on the 270-day BTL history, within the action's families"}


def arpu_anchors(ctx: FsdrContext, args: dict) -> dict:
    action = str(args.get("action") or ctx.action or "UPSELL").upper()
    k = k_margin(action, "cstr") if action in config.ACTIONS else config.PRICE_BAND_K["UPSELL"]
    bands = []
    if action in config.ACTIONS:
        for svc, gen in families(action, "cstr", ctx.ws.engine.mapping):
            for validity in (30, 7):
                a = anchor_value(ctx.ws.engine.anchors, ctx.msisdn, svc, validity)
                bands.append({"service": svc, "generosity": gen, "validity": validity, "soft_band": [_r(a), _r(a + k)]})
    return {"anchors_omr": ctx.anchors(), "k_margin": k, "soft_price_bands": bands,
            "rule": "anchor = mean + std of revenue periods; data 90 % / voice 10 % (BR-71..77). Bands are guidance."}


def search_catalog(ctx: FsdrContext, args: dict) -> dict:
    cat = ctx.ws.catalog
    m = pd.Series(True, index=cat.index)
    for key, col in (("service", "SERVICE"), ("generosity", "GENEROSITY")):
        if args.get(key):
            m &= cat[col] == str(args[key]).upper()
    if args.get("validity"):
        m &= cat["VALIDITY"] == int(args["validity"])
    if args.get("max_price") is not None:
        m &= cat["PRICE"] <= float(args["max_price"])
    if args.get("min_price") is not None:
        m &= cat["PRICE"] >= float(args["min_price"])
    hits = cat[m].sort_values("PRICE")
    return {"count": int(len(hits)), "offers": [_offer_brief(ctx, n) for n in hits.index[:10]]}


def check_offers_tool(ctx: FsdrContext, args: dict) -> dict:
    names = _offer_list(args)
    violations = check_offers(names, ctx.ws.engine.catalog, ctx.ws.engine.conflicts) + ctx.eligibility_issues()
    notes = []
    if ctx.action in config.ACTIONS:
        k = k_margin(ctx.action, "cstr")
        allowed = set(families(ctx.action, "cstr", ctx.ws.engine.mapping))
        for n in names:
            if n not in ctx.ws.catalog.index:
                continue
            o = ctx.ws.catalog.loc[n]
            a = anchor_value(ctx.ws.engine.anchors, ctx.msisdn, o["SERVICE"], int(o["VALIDITY"]))
            if (o["SERVICE"], o["GENEROSITY"]) not in allowed:
                notes.append(f"{n}: outside the action's families/generosity (soft rule, needs a reason)")
            elif not a < o["PRICE"] < a + k:
                notes.append(f"{n}: price {o['PRICE']} outside the soft band {a:.2f}..{a + k:.2f} (soft rule)")
    return {"ok": not violations, "hard_violations": violations, "soft_notes": notes}


def expected_value(ctx: FsdrContext, args: dict) -> dict:
    """Heuristic, NOT a trained uplift model: p = sigmoid(-2.2 + 1.4 x ranker score - 0.8 x relative gap
    above the anchor); value = p x price. Lets the agent compare offers on one scale."""
    out = []
    for n in _offer_list(args):
        if n not in ctx.ws.catalog.index:
            out.append({"offer": n, "error": "not in the catalogue"})
            continue
        o = ctx.ws.catalog.loc[n]
        a = anchor_value(ctx.ws.engine.anchors, ctx.msisdn, o["SERVICE"], int(o["VALIDITY"]))
        c = ctx.ws.engine.candidates([ctx.msisdn], ctx.action or "UPSELL", "cstr", int(o["VALIDITY"])) if ctx.action in config.ACTIONS else pd.DataFrame()
        score = float(c.loc[c["BTL_NAME"] == n, "SCORE"].max()) if len(c) and (c["BTL_NAME"] == n).any() else 0.0
        gap = max(0.0, float(o["PRICE"]) - a) / max(a, 0.5)
        p = 1 / (1 + math.exp(-(-2.2 + 1.4 * score - 0.8 * gap)))
        out.append({"offer": n, "p_accept": _r(p), "expected_value": _r(p * float(o["PRICE"]), 2)})
    return {"ranked": sorted(out, key=lambda r: -(r.get("expected_value") or 0)), "model": "heuristic (see tool docs)"}


# ---------------------------------------------------------------------------
# Competitor-agent tools
# ---------------------------------------------------------------------------
def _as_nbx_offer(o: pd.Series) -> dict:
    return {"program_rule": o["BTL_NAME"], "price_omr": float(o["PRICE"]), "validity_days": int(o["VALIDITY"]),
            "data_gb": float(o["GB"]) if o.get("GB") else None, "category": SERVICE_TO_CATEGORY.get(o["SERVICE"], "Bundle")}


def competitor_compare(ctx: FsdrContext, args: dict) -> dict:
    from nbx.core import competitors

    comp = ctx.ws.competitor_offers()
    if not comp:
        return {"available": False, "note": "no competitor snapshot (run: python run.py setup)"}
    results = []
    for n in _offer_list(args)[:4]:
        if n not in ctx.ws.catalog.index:
            results.append({"offer": n, "error": "not in the catalogue"})
            continue
        cmp = competitors.compare_offer(_as_nbx_offer(ctx.ws.catalog.loc[n]), comp, per_operator=1)
        closest = [{"operator": label(c["operator"]), "plan": label(c["plan"]), "monthly_equiv": c["monthly_equiv_omr"],
                    "data_gb": c.get("data_gb"), "monthly_delta": c["monthly_delta_omr"], "verdict": c["verdict"]}
                   for c in cmp.get("closest", [])]
        worst = max((c["monthly_delta"] for c in closest), default=0.0)
        results.append({"offer": n, "our_monthly_equiv": cmp.get("our_monthly_equiv_omr"), "position": label(cmp.get("position")),
                        "closest": closest, "at_risk": worst > config.COMPETITOR_UNCOMPETITIVE_DELTA})
    ctx.cache["competition"] = results
    return {"results": results, "labels": "competitor names are shown as configurable labels"}


def competitor_market(ctx: FsdrContext, args: dict) -> dict:
    comp = ctx.ws.competitor_offers()
    spend = (ctx.anchors().get("MONTHLY") or {}).get("total") or 0.0
    plans = [c for c in comp if c.get("segment", "mass") == "mass" and c.get("monthly_equiv_omr")
             and abs(c["monthly_equiv_omr"] - spend) <= max(3.0, spend * 0.3)]
    plans.sort(key=lambda c: (c.get("data_gb") or 0) / max(c["monthly_equiv_omr"], 0.01), reverse=True)
    return {"customer_monthly_anchor": _r(spend, 2),
            "competitor_plans_at_this_spend": [{"operator": label(c["operator"]), "plan": label(c["plan"]),
                                                "monthly_equiv": c["monthly_equiv_omr"], "data_gb": c.get("data_gb")} for c in plans[:5]]}


FSDR_TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("customer_360", "Features, revenue W1..M3, trends, the model scores, usage type, eligibility.", "{}", customer_360),
    Tool("bandit_pmf", "Contextual-bandit probability of each of the 7 actions, top action, sampled action.", "{}", bandit_pmf),
    Tool("segment_policy", "Customer segment and the BR-39 segment -> action of the previous policy.", "{}", segment_policy),
    Tool("campaign_history", "Last week's offers, what the customer took, top BTLs of the last 270 days.", "{}", campaign_history),
    Tool("cohort_offers", "What lines of the same device segment bought in the last 270 days (cohort graph).",
         '{"top_k"?: int, "service"?: str}', cohort_offers),
    Tool("contact_policy", "Can we contact this week, eligibility issues, actions blocked by churn risk.", "{}", contact_policy),
    Tool("price_band_candidates", "The rule-based offers for an action (3-pass ARPU price band).", '{"action"?: str}', price_band_candidates),
    Tool("ranker_top_offers", "Top offers of the BPR ranker for an action, per validity.", '{"action"?: str, "n"?: int}', ranker_top_offers),
    Tool("arpu_anchors", "ARPU anchors (total/data/voice, monthly/weekly) and the soft price bands.", '{"action"?: str}', arpu_anchors),
    Tool("search_catalog", "Any catalogue offer by service / validity / generosity / price.",
         '{"service"?: "DATA|VOICE|COMBO", "validity"?: 7|30, "generosity"?: "L0|L1|L2", "max_price"?: number, "min_price"?: number}', search_catalog),
    Tool("check_offers", "Hard rules (catalogue, price, <=2 per validity, <=4, conflicts, eligibility) + soft-band notes.",
         '{"offers": [codes]}', check_offers_tool),
    Tool("expected_value", "Heuristic acceptance probability and value per offer.", '{"offers": [codes]}', expected_value),
    Tool("competitor_compare", "Closest competitor plan per offer (same validity band), monthly price delta, verdict.",
         '{"offers": [codes]}', competitor_compare),
    Tool("competitor_market", "Competitor plans at this customer's monthly spend.", "{}", competitor_market),
]}
