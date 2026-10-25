"""NBA feature engineering -- port of docs/fsdr_reference/nba_lib.py, without Dataiku.

One function, build_features(), is used for BOTH training (T-anchored *_auto
tables) and scoring (D-anchored *_score tables), so the two can never drift.

Fixes kept from nba_lib (README_NBA_pipelines.md):
  - one key type everywhere: ACCESS_METHOD_VAL is a canonical 8-digit string,
    ids are Int64 (the original merged strings with integers);
  - missing model score -> uniform(0, 0.02) drawn PER ROW (BR-47), seeded;
  - trends go from the older period to the recent one, W1 = most recent week,
    % change clipped to [-100, 500] (BR-50..53).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from nbx.fsdr import config, schemas
from nbx.fsdr.calendar import PERIODS
from nbx.fsdr.datasource import DataSource

KEY_ID_COLS = ["ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]


# ---------------------------------------------------------------------------
# Feature names (derived from the registry / config, never hard-coded)
# ---------------------------------------------------------------------------
def trend_feature_names() -> list[str]:
    names = []
    for k, pairs in config.TREND_PAIRS.items():
        for recent, older in pairs:
            names += [f"VAR_{k}_{recent}_{older}", f"LOG_{k}_{recent}_{older}", f"{k}_{older}_ZERO"]
    return names


def numeric_features(registry: list[config.ModelScore] | None = None, extra: list[str] | tuple = ()) -> list[str]:
    """PROBA_* (one per registry feature) + trends (+ extra): the 48 numeric features of production."""
    return config.proba_features(registry) + trend_feature_names() + list(extra)


def model_features(registry: list[config.ModelScore] | None = None) -> list[str]:
    """What the bandit reads: the production features + the last-campaign features (M3) when enabled."""
    return numeric_features(registry, config.LAST_CAMPAIGN_NUMERIC if config.LAST_CAMPAIGN_FEATURES else ())


# ---------------------------------------------------------------------------
# Key normalisation
# ---------------------------------------------------------------------------
def norm_val(s: pd.Series) -> pd.Series:
    """MSISDN -> canonical string: 91234567, 91234567.0 and '91234567' all become '91234567'."""
    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (int, np.integer)):
            out = str(int(v))
        elif isinstance(v, (float, np.floating)):
            out = str(int(v)) if float(v).is_integer() else str(v)
        else:
            out = str(v).strip()
            if out.endswith(".0") and out[:-2].isdigit():
                out = out[:-2]
        return out or None
    return s.map(one).astype(object)


def norm_id(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


def upper(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case column names (production tables mix cases) and normalise the join keys."""
    out = df.copy()
    out.columns = [str(c).upper() for c in out.columns]
    if "ACCESS_METHOD_VAL" in out.columns:
        out["ACCESS_METHOD_VAL"] = norm_val(out["ACCESS_METHOD_VAL"])
    for c in KEY_ID_COLS:
        if c in out.columns:
            out[c] = norm_id(out[c])
    return out


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64") if c in df.columns else 0.0
    return df


# ---------------------------------------------------------------------------
# Loaders (each returns one row per ACCESS_METHOD_VAL)
# ---------------------------------------------------------------------------
def load_party(src: DataSource, table: str) -> pd.DataFrame:
    df = upper(src.read(table))
    return df[["PARTY_ID", "NATIONALITY"]].dropna(subset=["PARTY_ID"]).drop_duplicates("PARTY_ID")


def load_model_scores(src: DataSource, registry: list[config.ModelScore] | None = None) -> tuple[pd.DataFrame, dict]:
    """One row per line with one column per registry feature. Rows of the same
    feature (e.g. the 3 churn files) and stacked duplicates are merged by max."""
    registry = registry or config.MODEL_SCORE_REGISTRY
    per_feature: dict[str, list[pd.DataFrame]] = {}
    coverage: dict[str, int] = {}
    for m in registry:
        if not src.has(m.table):
            coverage[m.table] = 0
            continue
        df = upper(src.read(m.table))
        df = df[["ACCESS_METHOD_VAL", "PROBA_1"]].dropna(subset=["ACCESS_METHOD_VAL"])
        df["PROBA_1"] = pd.to_numeric(df["PROBA_1"], errors="coerce")
        coverage[m.table] = int(df["ACCESS_METHOD_VAL"].nunique())
        per_feature.setdefault(m.feature, []).append(df.rename(columns={"PROBA_1": m.feature}))
    out: pd.DataFrame | None = None
    for feature, frames in per_feature.items():
        agg = "max" if all(m.combine == "max" for m in registry if m.feature == feature) else "mean"
        f = pd.concat(frames, ignore_index=True).groupby("ACCESS_METHOD_VAL", as_index=False)[feature].agg(agg)
        out = f if out is None else out.merge(f, on="ACCESS_METHOD_VAL", how="outer")
    if out is None:
        out = pd.DataFrame(columns=["ACCESS_METHOD_VAL"])
    return out, coverage


def load_usage(src: DataSource, anchor: str) -> pd.DataFrame:
    """anchor 'T' -> activities_*_auto (training), 'D' -> activities_*_score (scoring)."""
    out: pd.DataFrame | None = None
    for p in PERIODS:
        table = schemas.activity_table_name(p, anchor)
        if not src.has(table):
            continue
        df = upper(src.read(table))
        cols = [f"{u}_{p}" for u in config.USAGE_COLS if f"{u}_{p}" in df.columns]
        df = _numeric(df[["ACCESS_METHOD_VAL"] + cols].dropna(subset=["ACCESS_METHOD_VAL"]).copy(), cols)
        df = df.groupby("ACCESS_METHOD_VAL", as_index=False)[cols].sum()
        out = df if out is None else out.merge(df, on="ACCESS_METHOD_VAL", how="outer")
    return out if out is not None else pd.DataFrame(columns=["ACCESS_METHOD_VAL"])


def load_subs(src: DataSource, anchor: str) -> pd.DataFrame:
    table = "subs_amt_auto" if anchor == "T" else "subs_amt_score"
    if not src.has(table):
        return pd.DataFrame(columns=["ACCESS_METHOD_VAL"])
    df = upper(src.read(table))
    cols = [f"SUB_AMT_{p}" for p in PERIODS if f"SUB_AMT_{p}" in df.columns]
    df = _numeric(df[["ACCESS_METHOD_VAL"] + cols].dropna(subset=["ACCESS_METHOD_VAL"]).copy(), cols)
    return df.groupby("ACCESS_METHOD_VAL", as_index=False)[cols].sum()


# ---------------------------------------------------------------------------
# Eligible base (BR-02, BR-08, BR-10)
# ---------------------------------------------------------------------------
def eligible_base(src: DataSource) -> tuple[pd.DataFrame, dict]:
    df = upper(src.read("base_prepaid"))
    n0 = len(df)
    df = df.dropna(subset=["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID"])
    null_keys = n0 - len(df)
    val = df["ACCESS_METHOD_VAL"].astype(str)
    ok_msisdn = val.str.fullmatch(r"\d{8}") & (val > config.MSISDN_MIN_EXCLUSIVE)        # BR-02
    excluded_plan = df["CURRENT_PLAN"].isin(config.EXCLUDED_PLANS)                       # BR-10
    df = df[ok_msisdn & ~excluded_plan]
    before = len(df)
    df = df.sort_values("ACCESS_METHOD_SUBS_START_DT").drop_duplicates("ACCESS_METHOD_VAL", keep="last")   # BR-08
    report = {"base_rows": n0, "eligible": len(df), "null_keys": null_keys, "excluded_plan": int(excluded_plan.sum()),
              "bad_msisdn": int((~ok_msisdn).sum()), "duplicates_removed": before - len(df)}
    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# Feature engineering (identical for training and scoring)
# ---------------------------------------------------------------------------
def build_features(pop: pd.DataFrame, party: pd.DataFrame, scores: pd.DataFrame, usage: pd.DataFrame, subs: pd.DataFrame,
                   seed: int = config.GENERATOR_SEED, registry: list[config.ModelScore] | None = None) -> pd.DataFrame:
    """pop: one row per line with at least ACCESS_METHOD_VAL and PARTY_ID."""
    proba_cols = config.proba_features(registry)
    df = pop.drop(columns=[c for c in ["NATIONALITY"] if c in pop.columns])
    df = df.dropna(subset=["ACCESS_METHOD_VAL"]).drop_duplicates("ACCESS_METHOD_VAL").copy()
    df = (df.merge(party, on="PARTY_ID", how="left")
            .merge(scores, on="ACCESS_METHOD_VAL", how="left")
            .merge(usage, on="ACCESS_METHOD_VAL", how="left")
            .merge(subs, on="ACCESS_METHOD_VAL", how="left"))

    # BR-47: missing model scores -> small random value, drawn PER ROW
    rng = np.random.default_rng(seed)
    lo, hi = config.MISSING_SCORE_RANGE
    for c in proba_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        miss = df[c].isna()
        df.loc[miss, c] = rng.uniform(lo, hi, int(miss.sum()))

    # BR-48 / BR-49: missing usage = 0; NBA revenue = subscriptions + PAYG data
    for p in PERIODS:
        for c in [f"SUB_AMT_{p}", f"PG_DATA_AMOUNT_{p}", f"LOCAL_VOICE_MIN_{p}", f"DATA_VOLUME_MB_{p}"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float64") if c in df.columns else 0.0
        df[f"REV_{p}"] = df[f"SUB_AMT_{p}"] + df[f"PG_DATA_AMOUNT_{p}"]

    # BR-50..53: % change, log change, "older period was zero" flag
    for k, pairs in config.TREND_PAIRS.items():
        for recent, older in pairs:
            cur = df[config.TREND_SOURCE[k].format(p=recent)]
            prev = df[config.TREND_SOURCE[k].format(p=older)]
            zero = prev == 0
            df[f"{k}_{older}_ZERO"] = zero.astype(int)
            pct = np.where(zero, 0.0, (cur - prev) / prev.where(~zero, 1.0) * 100.0)
            df[f"VAR_{k}_{recent}_{older}"] = np.clip(pct, *config.PCT_CLIP)
            df[f"LOG_{k}_{recent}_{older}"] = np.log1p(cur.clip(lower=0)) - np.log1p(prev.clip(lower=0))

    # BR-15 nationality group
    df["NATIONALITY_GROUP"] = np.where(df["NATIONALITY"].isin(config.NATIONALITY_KEEP), df["NATIONALITY"], config.NATIONALITY_OTHERS)
    return df.reset_index(drop=True)


def build_feature_frame(src: DataSource, pop: pd.DataFrame, anchor: str, seed: int = config.GENERATOR_SEED,
                        registry: list[config.ModelScore] | None = None, log_root=None,
                        context_date=None) -> tuple[pd.DataFrame, dict]:
    """Loads every input for `anchor` ('T' training, 'D' scoring) and builds the features of `pop`.
    log_root / context_date: campaign log and the decision date of the last-campaign features (M3);
    without a log every line gets the "never contacted" values."""
    party = load_party(src, "party_auto" if anchor == "T" else "party_scoring")
    scores, coverage = load_model_scores(src, registry)
    feat = build_features(pop, party, scores, load_usage(src, anchor), load_subs(src, anchor), seed, registry)
    if config.LAST_CAMPAIGN_FEATURES:
        from nbx.fsdr import campaign_log
        run_date = src.manifest().get("run_date")
        when = context_date or (date.fromisoformat(run_date) if run_date else date.today())
        lc = campaign_log.last_campaign_features(log_root, feat["ACCESS_METHOD_VAL"], when)
        feat = feat.drop(columns=[c for c in lc.columns if c != "ACCESS_METHOD_VAL" and c in feat.columns])
        feat = feat.merge(lc, on="ACCESS_METHOD_VAL", how="left")
    pop_vals = set(pop["ACCESS_METHOD_VAL"].dropna())
    scored = set(scores["ACCESS_METHOD_VAL"].dropna()) if len(scores) else set()
    report = {"lines": len(feat), "anchor": anchor, "numeric_features": len(numeric_features(registry)), "model_features": len(model_features(registry)),
              "score_coverage": round(len(pop_vals & scored) / len(pop_vals), 4) if pop_vals else 0.0,
              "model_rows": coverage}
    return feat, report
