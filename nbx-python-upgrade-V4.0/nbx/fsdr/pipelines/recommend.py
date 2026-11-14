"""Offer recommendation for the free and constraint arms -- port of production Codes 3 and 4.

    cstr  for each NBA action x validity: ranker trained on the 270-day history of the action's
          offer families (BR-66, all_L2 | normal), top-100 candidates per customer, 3-pass price
          bands around the ARPU anchor (BR-68..82), conflicts never together
    free  same, NBA action not used (BR-60): L2 DATA + COMBO only (BR-67)

V1.0 (FSDR_OFFER_POLICY=value, review #10 / #14): inside the same families, each family's offer is the one with
the highest expected revenue P(accept) x price (nbx/fsdr/acceptance.py; pass 4, or 5 for an exploration pick).
Until the acceptance model has enough data the 3 price-band passes are used. Lines without consent
(consent_contact, review #6) get no offer in any arm.

Reads outputs/results_scoring_nba.json (run fsdr-score first) and writes
outputs/recos_monthly_<arm>.json and recos_weekly_<arm>.json (one row per offer,
with the anchor and the pass that selected it).
The llm arm is produced by the recommender agent (nbx/fsdr/agents/recommender_agent.py).
"""
from __future__ import annotations

from nbx.core import tracing

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import campaign_log, config
from nbx.fsdr.acceptance import AcceptanceModel
from nbx.fsdr.datasource import DataSource
from nbx.fsdr.features import consent_blocked, eligible_base
from nbx.fsdr.offer_ranker import load_interactions, make_ranker, top_candidates
from nbx.fsdr.offer_rules import (Pick, anchor_column, anchor_value, arpu_anchors, cap_anchor_growth, catalog_version,
                                  conflicts_with, families, family_items, k_margin, load_catalog, load_conflicts,
                                  select_offers)
from nbx.fsdr.pipelines import read_table, write_report, write_table
from nbx.fsdr.pipelines.train import load_and_validate

ARM_FLAG = {"free": "free", "cstr": "with_constraint", "llm": "llm"}


class OfferEngine:
    """Everything needed to select offers for any customer: catalogue, conflicts, anchors,
    interactions and one trained ranker per (arm, action, validity), built lazily and cached.
    The recommender agent uses the same engine, so its fallback equals the cstr arm."""

    def __init__(self, src: DataSource, mapping: str | None = None, ranker: str | None = None,
                 log_root: Path | None = None, run_date=None, policy: str | None = None):
        self.src = src
        self.mapping = mapping or config.GENEROSITY_MAPPING
        self.ranker_name = ranker or config.RANKER
        self.catalog = load_catalog(src)
        self.version = catalog_version(self.catalog)
        self.conflicts = load_conflicts(src)
        self.anchors = arpu_anchors(src)
        if config.ANCHOR_MAX_WEEKLY_GROWTH is not None and log_root is not None and run_date is not None:   # M4
            self.anchors = cap_anchor_growth(self.anchors, campaign_log.previous_anchors(log_root, run_date),
                                             config.ANCHOR_MAX_WEEKLY_GROWTH)
        self.interactions = load_interactions(src, self.catalog)
        self._rankers: dict = {}
        self.consent_blocked = consent_blocked(src)
        self.policy = (policy or config.OFFER_POLICY).lower()
        if self.policy not in ("value", "price_band"):
            raise ValueError("offer policy must be 'value' or 'price_band'")
        self.acceptance, self.acceptance_report = None, {"trained": False, "reason": "price_band policy"}
        if self.policy == "value":
            from datetime import date
            when = run_date or (date.fromisoformat(src.manifest()["run_date"]) if src.manifest().get("run_date") else date.today())
            self.run_date = when
            self.acceptance, self.acceptance_report = AcceptanceModel.train(src, self.catalog, self.anchors,
                                                                            self.interactions, log_root, when)
        self._seed = (run_date.toordinal() if run_date else 0)

    @property
    def uses_value(self) -> bool:
        """True when offers are chosen on expected value (value policy with a trained acceptance model)."""
        return self.acceptance is not None

    def items(self, action: str | None, arm: str, validity: int) -> pd.DataFrame:
        """Offers the ranker may propose: the family's catalogue offers of this validity (BR-66/67), restricted
        to those bought in the filtered history unless config.RANKER_ITEMS == "catalogue" (BR-83/85)."""
        items = family_items(self.catalog, families(action, arm, self.mapping), validity)
        if config.RANKER_ITEMS != "catalogue":
            items = items[items["BTL_NAME"].isin(set(self.interactions["BTL_NAME"]))]
        return items

    def ranker(self, action: str | None, arm: str, validity: int):
        key = ("free" if arm == "free" else action, validity)
        if key not in self._rankers:
            items = self.items(action, arm, validity)
            self._rankers[key] = (make_ranker(self.ranker_name).fit(self.interactions, items["BTL_NAME"].tolist()), items)
        return self._rankers[key]

    def candidates(self, vals: list[str], action: str | None, arm: str, validity: int) -> pd.DataFrame:
        ranker, items = self.ranker(action, arm, validity)
        return top_candidates(ranker, vals, items)

    def select(self, val: str, action: str | None, arm: str, cands_by_validity: dict[int, pd.DataFrame] | None = None) -> list:
        """Offers for one customer across both validities (<= 2 each, <= 4 total), conflicts excluded."""
        if str(val) in self.consent_blocked:
            return []
        if self.uses_value:
            return self.select_value(val, action, arm)
        fams = families(action, arm, self.mapping)
        k = k_margin(action, arm)
        chosen = []
        for validity in (30, 7):                                   # monthly first, then weekly (production order)
            cands = (cands_by_validity or {}).get(validity)
            if cands is None:
                cands = self.candidates([val], action, arm, validity)
            picks = select_offers(cands[cands["ACCESS_METHOD_VAL"] == val], self.anchors, val, fams, k, validity,
                                  self.conflicts, already=[p.btl_name for p in chosen])
            chosen += picks
        return chosen[: config.MAX_OFFERS_TOTAL]

    def select_value(self, val: str, action: str | None, arm: str) -> list:
        """Value policy for one line (the agents' tools): select_value_batch of a single line."""
        return self.select_value_batch([str(val)], action, arm)[str(val)]

    def select_value_batch(self, vals: list[str], action: str | None, arm: str, chunk: int = 20_000) -> dict[str, list]:
        """Value policy: per validity and family, the catalogue offer with the highest P(accept) x price (the whole
        family, so offers never sold yet compete too); with probability OFFER_EXPLORATION a random family offer.
        P(accept) of a chunk of lines comes from one matrix (review #15); the per-line rules and random draws are
        the same for one line or a million."""
        vals = [str(v) for v in vals]
        fams = families(action, arm, self.mapping)
        items = pd.concat([family_items(self.catalog, [f], v) for v in (30, 7) for f in fams], ignore_index=True)
        items = items[items["PRICE"] > 0].drop_duplicates("BTL_NAME").reset_index(drop=True)
        if items.empty:
            return {v: [] for v in vals}
        names = items["BTL_NAME"].to_numpy()
        prices = items["PRICE"].to_numpy(float)
        pools = {(validity, svc, gen): np.flatnonzero(((items["VALIDITY"] == validity) & (items["SERVICE"] == svc)
                                                       & (items["GENEROSITY"] == gen)).to_numpy())
                 for validity in (30, 7) for svc, gen in fams}
        anchors = self.anchors.copy()
        anchors.index = anchors.index.astype(str)
        anc = {(svc, validity): anchors[anchor_column(svc, validity)].reindex(vals).fillna(0.0).to_numpy(float)
               for validity in (30, 7) for svc, _ in fams}
        floor = config.OFFER_PRICE_FLOOR == "anchor"
        out: dict[str, list] = {}
        for start in range(0, len(vals), chunk):
            part = vals[start:start + chunk]
            P = self.acceptance.p_accept_many(part, action, items)
            EV = P * prices * config.OFFER_MARGIN
            for i, val in enumerate(part):
                k = start + i
                rng = np.random.default_rng([self._seed, int(val) if val.isdigit() else abs(hash(val))])
                chosen, taken = [], []
                for validity in (30, 7):                           # monthly first, then weekly (production order)
                    for svc, gen in fams:
                        a = anc[(svc, validity)][k]
                        keep = [j for j in pools[(validity, svc, gen)]
                                if names[j] not in taken and not conflicts_with(names[j], taken, self.conflicts)
                                and (not floor or prices[j] > a)]           # production's floor (BR-78..80, BR-82)
                        if not keep:
                            continue
                        keep = np.asarray(keep)
                        explore = rng.random() < config.OFFER_EXPLORATION
                        ranked = keep[np.lexsort((prices[keep], -EV[i, keep]))]   # EV desc, then price asc (stable)
                        r = keep[int(rng.integers(len(keep)))] if explore else ranked[0]
                        rank = int(np.flatnonzero(ranked == r)[0]) + 1
                        chosen.append(Pick(names[r], svc, gen, int(validity), float(prices[r]), round(float(EV[i, r]), 6), rank,
                                           5 if explore else 4, round(float(a), 6), anchor_column(svc, validity),
                                           round(float(P[i, r]), 6)))
                        taken.append(names[r])
                out[val] = chosen[: config.MAX_OFFERS_TOTAL]
        return out


def recommend(src: DataSource, scored: pd.DataFrame, arm: str, engine: OfferEngine | None = None) -> tuple[pd.DataFrame, dict]:
    if arm not in ("free", "cstr"):
        raise ValueError("arm must be 'free' or 'cstr' (the llm arm runs through the recommender agent)")
    engine = engine or OfferEngine(src)
    base, _ = eligible_base(src)
    ok_vals = set(base["ACCESS_METHOD_VAL"])
    pop = scored[scored["TG_FLAG"] == ARM_FLAG[arm]].copy()
    pop["ACCESS_METHOD_VAL"] = pop["ACCESS_METHOD_VAL"].astype(str)
    not_eligible = int((~pop["ACCESS_METHOD_VAL"].isin(ok_vals)).sum())
    pop = pop[pop["ACCESS_METHOD_VAL"].isin(ok_vals)]
    groups = [(None, pop)] if arm == "free" else list(pop.groupby("ACTION"))
    rows = []
    for action, g in groups:
        vals = g["ACCESS_METHOD_VAL"].tolist()
        scored_action = dict(zip(g["ACCESS_METHOD_VAL"], g["ACTION"]))       # one lookup per line, not a scan of the arm
        if engine.uses_value:
            batch = engine.select_value_batch([v for v in vals if v not in engine.consent_blocked], action, arm)
            picks_of = lambda val: batch.get(val, [])                          # noqa: E731
        else:
            cands = {v: engine.candidates(vals, action, arm, v) for v in (30, 7)}
            by_val = {v: dict(tuple(c.groupby("ACCESS_METHOD_VAL", sort=False))) for v, c in cands.items()}
            empty = {v: c.iloc[0:0] for v, c in cands.items()}
            picks_of = lambda val: engine.select(val, action, arm, {v: by_val[v].get(val, empty[v]) for v in (30, 7)})  # noqa: E731
        for val in vals:
            for p in picks_of(val):
                rows.append({"ACCESS_METHOD_VAL": val, "ARM": arm, "ACTION": action if arm == "cstr" else scored_action[val],
                             **{k.upper(): v for k, v in p.as_dict().items()}})
    recos = pd.DataFrame(rows)
    n_customers = pop["ACCESS_METHOD_VAL"].nunique()
    covered = recos["ACCESS_METHOD_VAL"].nunique() if len(recos) else 0
    report = {
        "arm": arm, "mapping": engine.mapping, "ranker": engine.ranker_name, "catalog_version": engine.version,
        "offer_policy": "value" if engine.uses_value else "price_band", "acceptance_model": engine.acceptance_report,
        "consent_blocked": int(pop["ACCESS_METHOD_VAL"].isin(engine.consent_blocked).sum()),
        "customers": int(n_customers), "not_eligible_dropped": not_eligible, "with_offers": int(covered),
        "without_offer": int(n_customers - covered), "offers": int(len(recos)),
        "offers_per_pass": recos["PASS_NO"].value_counts().sort_index().to_dict() if len(recos) else {},
        "offers_per_validity": recos["VALIDITY"].value_counts().to_dict() if len(recos) else {},
        "mean_price": round(float(recos["PRICE"].mean()), 3) if len(recos) else None,
    }
    return recos, report


@tracing.traced_pipeline("fsdr_recommend")
def run_recommend(data_dir: str | Path, arm: str, mapping: str | None = None, ranker: str | None = None) -> dict:
    t0 = datetime.now()
    src, run_date, _ = load_and_validate(data_dir)
    try:
        scored = read_table(data_dir, "results_scoring_nba")
    except FileNotFoundError as e:
        raise FileNotFoundError("no scoring output: run fsdr-score first") from e
    engine = OfferEngine(src, mapping, ranker, log_root=campaign_log.root_for(data_dir), run_date=run_date)
    recos, report = recommend(src, scored, arm, engine)
    for validity, name in ((30, "monthly"), (7, "weekly")):
        part = recos[recos["VALIDITY"] == validity] if len(recos) else recos
        write_table(part, data_dir, f"recos_{name}_{arm}")
    logged = campaign_log.append_log(campaign_log.root_for(data_dir),
                                     campaign_log.rows_from_recos(recos, scored, arm, run_date, engine.anchors))   # M1
    report.update({"run_date": run_date.isoformat(), "campaign_log_rows": logged,
                   "seconds": round((datetime.now() - t0).total_seconds(), 1)})
    write_report(report, data_dir, f"nba_recommend_report_{arm}")
    return report
