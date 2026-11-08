"""T2_features (master prompt §6, B-01 … B-12): the features of the full base, partition by partition.

    df, stats = partition_features(CacheSource(D), part=3, run_date=D, anchor="T")   # raw: scores may be NaN
    medians = score_medians(paths)                                                   # B-05, on the whole base
    impute(df, medians)

Every line of one cache partition is in the same partition of every table (contract/cache.py), so a partition is
built with joins inside it only, and 8 partitions run in parallel on the P16 (plan §15). Two passes, because B-05
imputes a missing score with the median of the nationality group over the WHOLE base:
  pass 1  per partition: population, joins, revenue, trends, profile        -> base_features/part-NN.parquet
  between the score medians per nationality group and the ARPU tertiles are computed from those files
  pass 2  per partition: impute the missing scores, keep the *_MISSING flags -> the same file

What changes from V1.0's features.py (the rules of code 1 are the same):
  B-04  the 3 churn tables are stacked in registry order and the FIRST value per line is kept (V1.0: max)
  B-05  missing score -> flag + nationality-group median (V1.0: uniform(0, 0.02) draw, documented as replaced)
        A missing score never triggers "at risk": the S-rules read the flags (segmentation/rules.py)
  §8    AGE and TENURE_DAYS come from customer_profile when present, else party AGE and run date - activation
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import config, schemas
from nbx.fsdr.calendar import PERIODS
from nbx.fsdr.features import add_trends, eligible_frame, norm_id, norm_val, upper

USAGE_V2 = ["LOCAL_VOICE_MIN", "DATA_VOLUME_MB", "PG_DATA_AMOUNT", "PG_TOTAL_AMOUNT", "ACTIVE_DAYS_COUNT"]
KEYS = ["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]
_REFERENCE: dict = {}                              # reference tables, read once per worker process


def anchor_of(pipeline: str) -> str:
    """B-01: week 0 trains on the T-anchored *_auto tables, every later week on the D-anchored *_score tables."""
    return "T" if pipeline == "train_week0" else "D"


def party_table(anchor: str, cache) -> str:
    first, second = ("party_auto", "party_scoring") if anchor == "T" else ("party_scoring", "party_auto")
    return first if cache.has(first) else second


def _part(cache, table: str, part: int) -> pd.DataFrame | None:
    return upper(cache.read(table, part)) if cache.has(table) else None


def _reference(cache, table: str) -> pd.DataFrame | None:
    key = (str(cache.dir), table)
    if key not in _REFERENCE:
        _REFERENCE[key] = upper(cache.read(table)) if cache.has(table) else None
    return _REFERENCE[key]


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def model_scores(cache, part: int, registry: list[config.ModelScore] | None = None) -> pd.DataFrame:
    """One row per line, one column per score feature (NaN = not scored). B-04: churn = first row of the stack."""
    per_feature: dict[str, list[pd.DataFrame]] = {}
    for m in registry or config.MODEL_SCORE_REGISTRY:
        df = _part(cache, m.table, part)
        if df is None or "PROBA_1" not in df.columns:
            continue
        d = pd.DataFrame({"ACCESS_METHOD_VAL": df["ACCESS_METHOD_VAL"],
                          m.feature: pd.to_numeric(df["PROBA_1"], errors="coerce")}).dropna()
        per_feature.setdefault(m.feature, []).append(d)
    out: pd.DataFrame | None = None
    for feature, frames in per_feature.items():
        st = pd.concat(frames, ignore_index=True)
        if feature == config.CHURN_FEATURE:
            st = st.drop_duplicates("ACCESS_METHOD_VAL", keep="first")
        else:
            st = st.groupby("ACCESS_METHOD_VAL", as_index=False)[feature].max()
        out = st if out is None else out.merge(st, on="ACCESS_METHOD_VAL", how="outer")
    return out if out is not None else pd.DataFrame(columns=["ACCESS_METHOD_VAL"])


def usage(cache, part: int, anchor: str) -> pd.DataFrame:
    out: pd.DataFrame | None = None
    for p in PERIODS:
        df = _part(cache, schemas.activity_table_name(p, anchor), part)
        if df is None:
            continue
        cols = [f"{u}_{p}" for u in USAGE_V2 if f"{u}_{p}" in df.columns]
        d = _num(df[["ACCESS_METHOD_VAL"] + cols].dropna(subset=["ACCESS_METHOD_VAL"]).copy(), cols)
        d = d.groupby("ACCESS_METHOD_VAL", as_index=False)[cols].sum()
        out = d if out is None else out.merge(d, on="ACCESS_METHOD_VAL", how="outer")
    return out if out is not None else pd.DataFrame(columns=["ACCESS_METHOD_VAL"])


def subscriptions(cache, part: int, anchor: str) -> pd.DataFrame:
    df = _part(cache, "subs_amt_auto" if anchor == "T" else "subs_amt_score", part)
    if df is None:
        return pd.DataFrame(columns=["ACCESS_METHOD_VAL"])
    cols = [f"SUB_AMT_{p}" for p in PERIODS if f"SUB_AMT_{p}" in df.columns]
    d = _num(df[["ACCESS_METHOD_VAL"] + cols].dropna(subset=["ACCESS_METHOD_VAL"]).copy(), cols)
    return d.groupby("ACCESS_METHOD_VAL", as_index=False)[cols].sum()


def profile(cache, part: int) -> pd.DataFrame:
    df = _part(cache, config.PROFILE_TABLE, part)
    if df is None or "MSISDN" not in df.columns:
        return pd.DataFrame(columns=["ACCESS_METHOD_VAL", "AGE_PROFILE", "TENURE_PROFILE"])
    d = pd.DataFrame({"ACCESS_METHOD_VAL": norm_val(df["MSISDN"]),
                      "AGE_PROFILE": pd.to_numeric(df.get("AGE"), errors="coerce"),
                      "TENURE_PROFILE": pd.to_numeric(df.get("TENURE"), errors="coerce")})
    return d.dropna(subset=["ACCESS_METHOD_VAL"]).drop_duplicates("ACCESS_METHOD_VAL", keep="last")


def region(cache, part: int) -> pd.DataFrame:
    df = _part(cache, "location_auto", part)
    if df is None or "REGION" not in df.columns:
        return pd.DataFrame(columns=["ACCESS_METHOD_ID", "REGION"])
    d = df.assign(_Y=pd.to_numeric(df.get("CALLING_YEAR"), errors="coerce"),
                  _M=pd.to_numeric(df.get("CALLING_MONTH"), errors="coerce"))
    d = d.dropna(subset=["ACCESS_METHOD_ID"]).sort_values(["_Y", "_M"]).drop_duplicates("ACCESS_METHOD_ID", keep="last")
    return d[["ACCESS_METHOD_ID", "REGION"]]


def device(cache, part: int) -> pd.DataFrame:
    df = _part(cache, "DEVICEBEHAVIORSEGMENTATION.dv_segment", part)
    if df is None or "DV_SEGMENT" not in df.columns:
        return pd.DataFrame(columns=["ACCESS_METHOD_ID", "DV_SEGMENT"])
    return df[["ACCESS_METHOD_ID", "DV_SEGMENT"]].dropna(subset=["ACCESS_METHOD_ID"]).drop_duplicates("ACCESS_METHOD_ID", keep="last")


def partition_features(cache, part: int, run_date: date, anchor: str,
                       registry: list[config.ModelScore] | None = None) -> tuple[pd.DataFrame, dict]:
    """Pass 1 of one partition: B-01 population and left joins in code-1 order, B-05 (usage), B-06, B-07, B-08,
    profile. Scores stay NaN where missing (pass 2 imputes them)."""
    proba = config.proba_features(registry)
    base = _part(cache, "base_prepaid", part)
    if base is None:
        raise FileNotFoundError("base_prepaid is not in the cache")
    base, elig = eligible_frame(base)
    df = base[[c for c in KEYS + ["ACCESS_METHOD_SUBS_START_DT", "CURRENT_PLAN"] if c in base.columns]].copy()

    accounts = _reference(cache, "consumer_aact_account")             # B-01: party through the account if missing
    if accounts is not None and {"ACCOUNT_ID", "PARTY_ID"} <= set(accounts.columns) and df["PARTY_ID"].isna().any():
        acc = accounts.dropna(subset=["ACCOUNT_ID"]).drop_duplicates("ACCOUNT_ID").set_index("ACCOUNT_ID")["PARTY_ID"]
        df["PARTY_ID"] = df["PARTY_ID"].fillna(df["ACCOUNT_ID"].map(acc))
    party = _reference(cache, party_table(anchor, cache))
    if party is not None:
        pty = party[[c for c in ("PARTY_ID", "NATIONALITY", "AGE") if c in party.columns]]
        pty = pty.dropna(subset=["PARTY_ID"]).drop_duplicates("PARTY_ID").rename(columns={"AGE": "AGE_PARTY"})
        df = df.merge(pty, on="PARTY_ID", how="left")
    for c in ("NATIONALITY", "AGE_PARTY"):
        if c not in df.columns:
            df[c] = np.nan

    df = (df.merge(model_scores(cache, part, registry), on="ACCESS_METHOD_VAL", how="left")
            .merge(usage(cache, part, anchor), on="ACCESS_METHOD_VAL", how="left")
            .merge(subscriptions(cache, part, anchor), on="ACCESS_METHOD_VAL", how="left")
            .merge(profile(cache, part), on="ACCESS_METHOD_VAL", how="left")
            .merge(region(cache, part), on="ACCESS_METHOD_ID", how="left")
            .merge(device(cache, part), on="ACCESS_METHOD_ID", how="left"))

    cols: dict[str, pd.Series] = {}
    for f in proba:                                                    # B-05 flags (imputed in pass 2)
        s = pd.to_numeric(df[f], errors="coerce").astype("float64") if f in df.columns else pd.Series(np.nan, index=df.index)
        cols[f] = s
        cols[f"{f}{config.MISSING_FLAG_SUFFIX}"] = s.isna().astype("int8")
    for p in PERIODS:                                                  # B-05 missing usage / spend = 0
        for c in [f"SUB_AMT_{p}"] + [f"{u}_{p}" for u in USAGE_V2]:
            cols[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float64") if c in df.columns \
                else pd.Series(0.0, index=df.index)
        cols[f"REV_{p}"] = cols[f"SUB_AMT_{p}"] + cols[f"PG_DATA_AMOUNT_{p}"]       # B-06
    df = pd.concat([df.drop(columns=[c for c in cols if c in df.columns]), pd.DataFrame(cols, index=df.index)], axis=1)
    trends = add_trends(pd.DataFrame(index=df.index).assign(**{c: df[c] for c in df.columns
                                                                  if c.startswith(("REV_", "LOCAL_VOICE_MIN_", "DATA_VOLUME_MB_"))}))
    df = pd.concat([df, trends.drop(columns=[c for c in trends.columns if c in df.columns])], axis=1)   # B-07

    nat = df["NATIONALITY"].astype("string").str.strip()
    df["NATIONALITY_GROUP"] = np.where(nat.isin(config.NATIONALITY_KEEP), nat, config.NATIONALITY_OTHERS)   # B-08
    df["ARPU_M1"] = df["SUB_AMT_M1"] + df["PG_TOTAL_AMOUNT_M1"]          # §7 REV_M1 of the S-rules (and value band)

    age_p = pd.to_numeric(df["AGE_PROFILE"], errors="coerce")           # §8 sources
    df["AGE"] = age_p.fillna(pd.to_numeric(df["AGE_PARTY"], errors="coerce"))
    df["AGE_SOURCE"] = np.where(age_p.notna(), "profile", np.where(df["AGE"].notna(), "party", "missing"))
    start = pd.to_datetime(df["ACCESS_METHOD_SUBS_START_DT"], errors="coerce")
    by_activation = (pd.Timestamp(run_date) - start).dt.days.astype("float64")
    ten_p = pd.to_numeric(df["TENURE_PROFILE"], errors="coerce")
    df["TENURE_DAYS"] = ten_p.fillna(by_activation)
    df["TENURE_SOURCE"] = np.where(ten_p.notna(), "profile", np.where(by_activation.notna(), "activation", "missing"))
    df = df.drop(columns=["AGE_PROFILE", "TENURE_PROFILE"])

    stats = {"part": part, **elig, "lines": len(df),
             "missing_scores": {f: int(df[f"{f}{config.MISSING_FLAG_SUFFIX}"].sum()) for f in proba},
             "age_source": df["AGE_SOURCE"].value_counts().to_dict(),
             "tenure_source": df["TENURE_SOURCE"].value_counts().to_dict(),
             "no_nationality": int(nat.isna().sum()), "no_region": int(df["REGION"].isna().sum()),
             "no_device": int(df["DV_SEGMENT"].isna().sum())}
    return df.reset_index(drop=True), stats


def score_medians(paths: list[Path], registry: list[config.ModelScore] | None = None) -> dict:
    """B-05 medians of the observed scores per nationality group, over every partition (+ "_global")."""
    proba = config.proba_features(registry)
    frames = [pd.read_parquet(p, columns=["NATIONALITY_GROUP"] + proba) for p in paths]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["NATIONALITY_GROUP"] + proba)
    med = {g: {f: (None if pd.isna(v) else round(float(v), 6)) for f, v in row.items()}
           for g, row in df.groupby("NATIONALITY_GROUP")[proba].median().iterrows()}
    med["_global"] = {f: (None if pd.isna(df[f].median()) else round(float(df[f].median()), 6)) for f in proba}
    return med


def arpu_cutoffs(paths: list[Path]) -> list[float]:
    """Tertile cut points of ARPU_M1 among the paying lines (value band, context only: §8)."""
    arpu = pd.concat([pd.read_parquet(p, columns=["ARPU_M1"])["ARPU_M1"] for p in paths], ignore_index=True)
    paying = arpu[arpu > 0]
    return [round(float(q), 4) for q in paying.quantile([1 / 3, 2 / 3])] if len(paying) else [0.0, 0.0]


def impute(df: pd.DataFrame, medians: dict, registry: list[config.ModelScore] | None = None) -> pd.DataFrame:
    """Pass 2 (B-05): every missing score = the median of its nationality group (else global, else 0). In place."""
    glob = medians.get("_global", {})
    for f in config.proba_features(registry):
        miss = df[f].isna()
        if miss.any():
            by_group = {g: m.get(f) for g, m in medians.items() if g != "_global"}
            fill = df.loc[miss, "NATIONALITY_GROUP"].map(by_group).astype("float64")
            df.loc[miss, f] = fill.fillna(glob.get(f) if glob.get(f) is not None else 0.0).fillna(0.0)
    return df


def column_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per numeric column: count, nulls, sum, min, max (summed over partitions into feature_summary.csv)."""
    num = df.select_dtypes("number")
    return pd.DataFrame({"count": num.count(), "nulls": num.isna().sum(), "sum": num.sum(), "min": num.min(),
                         "max": num.max()})
