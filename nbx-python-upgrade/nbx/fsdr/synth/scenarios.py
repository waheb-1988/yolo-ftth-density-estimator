"""Use-case scenarios for the dummy data (combinable, e.g. "churn_wave,data_growth").

The behavioural scenarios change the hidden world (synth/world.py reads
World.scenarios): drifts ramp up over the last 4 weeks before the world's
as_of date, so a single generated date already shows them in the W1..W4 trends,
and the weekly simulation keeps them going. `data_quality_issues` instead
damages the generated tables after they are built, so validation has real
problems to catch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nbx.fsdr import config


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    expected_signal: str


SCENARIOS: dict[str, Scenario] = {s.name: s for s in [
    Scenario("baseline", "Normal weekly behaviour.", "-"),
    Scenario("churn_wave", "Churn risk rises over 4 weeks for lines in North Batinah (generate --weeks 4).",
             "PROBA_CHURN and REV_DECR up and usage down in those regions; RETAIN pays off there."),
    Scenario("data_growth", "Voice-only customers with data appetite gradually start using data.",
             "DATA_VOLUME_MB W1 > W4 for them, PROBA_APP_DATA up; UPSELL / CROSS SELL DATA grow."),
    Scenario("price_sensitive", "Price sensitivity x2.5: acceptance drops fast above the ARPU anchor.",
             "Take rate falls steeply with (price - anchor)."),
    Scenario("new_offers", "3 new BTL offers (name ends with _NEW) in the catalogue, with no purchase history (cold start).",
             "Offers in btl_catalog absent from sub_btl_3months; ranker must fall back."),
    Scenario("campaign_fatigue", "Acceptance falls with the number of contacts in the last 4 weeks.",
             "Take rate decreases with NB_CONTACTS_LAST_4W."),
    Scenario("data_quality_issues", "Duplicate MSISDNs, null keys, stacked churn duplicates, keys as strings.",
             "Validation reports each issue."),
]}


def parse(value: str | list[str] | None) -> list[str]:
    """'churn_wave,data_growth' -> ['churn_wave', 'data_growth'] (validated, order kept, duplicates dropped)."""
    if value is None:
        return ["baseline"]
    items = value.split(",") if isinstance(value, str) else list(value)
    out = []
    for s in (x.strip() for x in items):
        if not s:
            continue
        if s not in config.SCENARIOS:
            raise ValueError(f"Unknown scenario '{s}'. Valid: {', '.join(config.SCENARIOS)}")
        if s not in out:
            out.append(s)
    return out or ["baseline"]


# ---------------------------------------------------------------------------
# data_quality_issues: applied to the raw (object) frames just before writing
# ---------------------------------------------------------------------------
QUALITY_SHARE = 0.03          # share of rows damaged per issue (above the 1 % blocking threshold)


def _rows(rng: np.random.Generator, df: pd.DataFrame, share: float = QUALITY_SHARE) -> np.ndarray:
    k = max(1, int(len(df) * share))
    return rng.choice(len(df), size=min(k, len(df)), replace=False)


def inject_quality_issues(tables: dict[str, pd.DataFrame], seed: int) -> list[dict]:
    """Damages the tables in place; returns what was injected (written to the manifest)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 99]))
    log: list[dict] = []

    # 1. duplicate MSISDNs in base_prepaid (same line, another ACCESS_METHOD_ID)
    base = tables["base_prepaid"]
    dup = base.iloc[_rows(rng, base)].copy()
    dup["ACCESS_METHOD_ID"] = [int(v) + 900_000_000 for v in dup["ACCESS_METHOD_ID"]]
    tables["base_prepaid"] = pd.concat([base, dup], ignore_index=True)
    log.append({"issue": "duplicate_msisdn", "table": "base_prepaid", "rows": len(dup)})

    # 2. null keys
    for table, col in [("base_prepaid", "PARTY_ID"), ("activities_w1_auto", "ACCESS_METHOD_ID"),
                       ("base_prepaid", "ACCESS_METHOD_VAL")]:
        df = tables[table]
        idx = _rows(rng, df)
        df.loc[df.index[idx], col] = None
        log.append({"issue": "null_key", "table": table, "column": col, "rows": len(idx)})

    # 3. stacked duplicates in the global churn model (same line scored twice, different probabilities)
    churn_t = config.MODEL_SCORE_REGISTRY[0].table
    ch = tables[churn_t]
    stacked = ch.iloc[_rows(rng, ch, 0.05)].copy()
    stacked["proba_1"] = [round(float(rng.uniform(0, 1)), 6) for _ in range(len(stacked))]
    stacked["proba_0"] = [round(1 - p, 6) for p in stacked["proba_1"]]
    tables[churn_t] = pd.concat([ch, stacked], ignore_index=True)
    log.append({"issue": "stacked_duplicates", "table": churn_t, "rows": len(stacked)})

    # 4. integer keys delivered as strings
    for table, col in [("subs_amt_auto", "ACCESS_METHOD_ID"), (churn_t, "access_method_val")]:
        df = tables[table]
        idx = _rows(rng, df)
        df.loc[df.index[idx], col] = [None if v is None else str(v) for v in df.loc[df.index[idx], col]]
        log.append({"issue": "key_as_string", "table": table, "column": col, "rows": len(idx)})
    return log
