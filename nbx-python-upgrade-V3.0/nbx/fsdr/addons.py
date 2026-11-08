"""Add-on offers for the batch flow: top roaming and international packs beside the main offers.

These families are not in the BR catalogue (BR-66/67 cover DATA / VOICE / COMBO), so the rules are the app's:
  - catalogue: offers whose service is in config.ROAMING_SERVICES / INTERNATIONAL_SERVICES;
  - ranking: a BPR ranker per family on the 270-day BTL history (BR-24/83), popularity for customers
    without history (BR-86), ties broken by the lower price;
  - roaming: the top ADDON_TOP_ROAMING packs (the data contract has no roaming-usage column, so the
    ranking relies on past roaming purchases);
  - international: only "if available" -- the customer made at least INTL_MIN_CALLS_M1 international
    calls in M1 (Call_Count_M1 - Local_Call_Count_M1: local calls exclude international ones);
  - hard rules: price > 0, never an add-on that conflicts with a main offer or another add-on
    (conflicting_offers_h). BR-82's limit of 4 applies to the main offers, not to these add-ons.
"""
from __future__ import annotations

import pandas as pd

from nbx.fsdr import config
from nbx.fsdr.datasource import DataSource
from nbx.fsdr.features import upper
from nbx.fsdr.offer_ranker import make_ranker
from nbx.fsdr.offer_rules import anchor_value, conflicts_with


def international_calls(src: DataSource) -> pd.Series:
    """International calls in M1 (D-anchored), per line."""
    table = "activities_m1_score"
    if not src.has(table):
        return pd.Series(dtype=float)
    df = upper(src.read(table))
    calls = pd.to_numeric(df.get("CALL_COUNT_M1"), errors="coerce").fillna(0)
    local = pd.to_numeric(df.get("LOCAL_CALL_COUNT_M1"), errors="coerce").fillna(0)
    return (calls - local).clip(lower=0).groupby(df["ACCESS_METHOD_VAL"]).sum()


class AddonEngine:
    def __init__(self, src: DataSource, engine):
        self.engine = engine
        cat = engine.catalog
        self.families = {"roaming": cat[cat["SERVICE"].isin(config.ROAMING_SERVICES)],
                         "international": cat[cat["SERVICE"].isin(config.INTERNATIONAL_SERVICES)]}
        self.rankers = {k: make_ranker().fit(engine.interactions, items["BTL_NAME"].tolist())
                        for k, items in self.families.items() if len(items)}
        self.intl_calls = international_calls(src)
        bought = engine.interactions[engine.interactions["BTL_NAME"].isin(set(self.families["roaming"]["BTL_NAME"]))]
        self.roaming_buyers = set(bought["ACCESS_METHOD_VAL"])

    def _top(self, family: str, val: str, n: int, taken: list[str]) -> list[dict]:
        items = self.families[family].set_index("BTL_NAME")
        ranker = self.rankers.get(family)
        if ranker is None:
            return []
        scores = pd.Series(ranker.score([val])[0], index=ranker.items)
        cand = pd.DataFrame({"BTL_NAME": scores.index, "SCORE": scores.to_numpy()}).merge(
            items[["PRICE", "VALIDITY", "SERVICE", "GENEROSITY", "GB", "MIN"]], left_on="BTL_NAME", right_index=True)
        if val not in getattr(ranker, "user_index", {}):
            # cold start: popularity alone gives everyone the same packs -> blend it with the price fit
            # against the customer's own total ARPU anchor (weekly anchor for 7-day packs, BR-71/77)
            anchor = cand["VALIDITY"].map(lambda v: anchor_value(self.engine.anchors, val, "COMBO", int(v)))
            fit = 1.0 / (1.0 + (cand["PRICE"] - anchor).abs() / anchor.clip(lower=1.0))
            cand["SCORE"] = 0.5 * cand["SCORE"] + fit
        cand = cand[cand["PRICE"] > 0].sort_values(["SCORE", "PRICE"], ascending=[False, True])
        out: list[dict] = []
        for r in cand.itertuples():
            if conflicts_with(r.BTL_NAME, taken + [o["offer"] for o in out], self.engine.conflicts):
                continue
            out.append({"offer": r.BTL_NAME, "service": r.SERVICE, "generosity": r.GENEROSITY, "validity": int(r.VALIDITY),
                        "price": float(r.PRICE), "gb": float(r.GB or 0), "minutes": int(r.MIN or 0), "score": round(float(r.SCORE), 4)})
            if len(out) == n:
                break
        return out

    def recommend(self, val: str, main_offers: list[str]) -> dict:
        val = str(val)
        roaming = self._top("roaming", val, config.ADDON_TOP_ROAMING, main_offers)
        calls = float(self.intl_calls.get(val, 0.0))
        if not len(self.families["international"]):
            intl, why = [], "no international offer in the catalogue"
        elif calls < config.INTL_MIN_CALLS_M1:
            intl, why = [], f"{calls:.0f} international call(s) in M1 (< {config.INTL_MIN_CALLS_M1})"
        else:
            intl = self._top("international", val, config.ADDON_TOP_INTERNATIONAL, main_offers + [o["offer"] for o in roaming])
            why = f"{calls:.0f} international calls in M1"
        return {"roaming": roaming, "roaming_basis": "past roaming purchases" if val in self.roaming_buyers else "popularity",
                "international": intl, "international_reason": why, "international_calls_m1": calls}
