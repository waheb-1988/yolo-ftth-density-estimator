# -*- coding: utf-8 -*-
"""
FSDR - Next Best Action (NBA) shared library
============================================
Put this file in the Dataiku project library:  lib/python/nba_lib.py
It is imported by:
    01_nba_train.py   (training + scoring of the full base)
    02_nba_score.py   (scoring only, at campaign start, with the saved model)

Everything that must be IDENTICAL between training and scoring lives here
(feature engineering, VW text format, action list), so the two pipelines
can never drift apart.
"""
import json
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import dataiku
import vowpalwabbit

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
VW_ARGS = "--cb_explore_adf -q UA --quiet --epsilon 0.2"
MODEL_FOLDER = "nba_model"          # Dataiku managed folder (create it once in the Flow)
MODEL_FILE = "nba_cb_adf.vw"
META_FILE = "nba_meta.json"
SEED = 42

# Fixed order of actions. NEVER shuffle it: the model's probability vector
# is returned in this order.
ACTIONS = ["CROSS SELL DATA", "UPSELL DATA", "RETAIN", "MAINTAIN",
           "UPSELL", "CROSS SELL VOICE", "UPSELL VOICE"]
ACTION_TOKENS = [a.replace(" ", "_") for a in ACTIONS]          # VW tokens cannot contain spaces
ACTION_LINES = [f"|Action product={t}" for t in ACTION_TOKENS]
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}

# Historical share of each action in the targeting (fallback logging propensity)
ACTION_PRIOR = {
    "CROSS SELL DATA": 0.0083128,
    "CROSS SELL VOICE": 0.0528180,
    "MAINTAIN": 0.1015183,
    "RETAIN": 0.5086577,
    "UPSELL": 0.2624428,
    "UPSELL DATA": 0.0422645,
    "UPSELL VOICE": 0.0239859,
}

# Segment -> action. Keys are lower-case so "RGS_..." and "Rgs_..." match the same entry.
# The original dictionary had duplicate keys with different values; Python kept the LAST one.
# Lines marked CHECK are the ones where the original code was ambiguous - confirm with the business.
SEGMENT_TO_ACTION = {
    "retention_low_probability_to_reconnect": "RETAIN",
    "retention_med_probability_to_reconnect": "RETAIN",
    "retention_high_probability_to_reconnect": "RETAIN",                 # CHECK: was UPSELL then RETAIN
    "retention_rgs_at_risk": "RETAIN",
    "retention_rgs_fragile": "UPSELL",
    "rgs_not_at_risk_dualplay_upgrade": "MAINTAIN",
    "rgs_not_at_risk_data_only_voice_appetency": "UPSELL VOICE",          # CHECK: was UPSELL VOICE then CROSS SELL VOICE
    "rgs_not_at_risk_data_only_no_voice_appetency": "CROSS SELL VOICE",
    "rgs_not_at_risk_voice_only_data_appetency": "UPSELL DATA",
    "rgs_not_at_risk_voice_only_no_data_appetency": "CROSS SELL DATA",
    "onboarding_early_churn_at_risk": "UPSELL",
    "onboarding_early_churn_not_at_risk": "UPSELL",
}
EXCLUDED_SEGMENTS = {"nursering_early_churn_at_risk", "nursering_early_churn_not_at_risk"}

# Model-score datasets: dataset -> feature name (all expose PROBA_1)
PROBA_MODELS = {
    "DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_TD.Scored": "PROBA_REACT_30",
    "PROPENSITY_REACTIVATION_15D_401_SCORING.Scored": "PROBA_REACT_15",
    "APPETENCY_DATA_204_SCORING_TD.prediction": "PROBA_APP_DATA",
    "APPETENCY_VOICE_204_SCORING_TD.Scored": "PROBA_APP_VOICE",
    "CHURN_ONE_DAY_ACTIVE_107_SCORING.Churn_One_Day_Scoring": "PROBA_CHURN_1D",
    "REV_DECREASE_204_SCORING_TD.predictions": "PROBA_REV_DECR",
    "DAIM_FRAGILITY_DATA_SCORING_TD.prediction": "PROBA_FRAG_DATA",
    "DAIM_FRAGILITY_VOICE_SCORING_TD.prediction": "PROBA_FRAG_VOICE",
}
# The three churn models are stacked into one PROBA_CHURN (highest risk kept)
CHURN_MODELS = [
    "DAIM_CHURN_GLOBAL_SCORING_TD.prediction",
    "DAIM_EARLY_CHURN_0_30_SCORING_TD.Early_Churn_0_30_Prediction",
    "DAIM_EARLY_CHURN_30_60_SCORING_TD.scoring_data_scored",
]
PROBA_COLS = ["PROBA_CHURN"] + list(PROBA_MODELS.values())

NATIONALITY_KEEP = ["Oman", "India", "Bangladesh", "Pakistan"]

WEEKS = ["W1", "W2", "W3", "W4"]      # W1 = most recent week
MONTHS = ["M1", "M2", "M3"]           # M1 = most recent month
PERIODS = WEEKS + MONTHS
USAGE_COLS = ["LOCAL_VOICE_MIN", "DATA_VOLUME_MB", "PG_DATA_AMOUNT"]

# Trend pairs as (recent, older). Variation = change FROM older TO recent.
# (The original code had the direction inverted: it treated W1 as the older week.)
TREND_PAIRS = {
    "REV":   [("W1", "W2"), ("W2", "W3"), ("W3", "W4"), ("M1", "M2"), ("M2", "M3")],
    "VOICE": [("W1", "W2"), ("W2", "W3"), ("W3", "W4"), ("M1", "M2")],
    "DATA":  [("W1", "W2"), ("W2", "W3"), ("W3", "W4"), ("M1", "M2")],
}
TREND_SOURCE = {"REV": "REV_{p}", "VOICE": "LOCAL_VOICE_MIN_{p}", "DATA": "DATA_VOLUME_MB_{p}"}
PCT_CLIP = (-100.0, 500.0)            # caps exploding % changes when the older period is tiny


def trend_feature_names():
    names = []
    for k, pairs in TREND_PAIRS.items():
        for recent, older in pairs:
            names += [f"VAR_{k}_{recent}_{older}", f"LOG_{k}_{recent}_{older}", f"{k}_{older}_ZERO"]
    return names


NUMERIC_FEATURES = PROBA_COLS + trend_feature_names()

KEY_COLS = ["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]


# =============================================================================
# 1. READING / CLEANING
# =============================================================================
def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def normalize_keys(df):
    """Same numeric type for every join key in every table (avoids string/int merge errors)."""
    for c in KEY_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def read_ds(name):
    df = dataiku.Dataset(name).get_dataframe()
    df.columns = [str(c).upper() for c in df.columns]
    return normalize_keys(df)


def downcast(df):
    """float64 -> float32 only (ids are left untouched)."""
    f = df.select_dtypes(include=["float64"]).columns
    df[f] = df[f].astype("float32")
    return df


def load_party(ds_name):
    df = read_ds(ds_name)
    return df[["PARTY_ID", "NATIONALITY"]].dropna(subset=["PARTY_ID"]).drop_duplicates("PARTY_ID")


def load_model_scores():
    """One row per ACCESS_METHOD_VAL with the 9 probabilities."""
    out = None
    for ds, newcol in PROBA_MODELS.items():
        df = read_ds(ds)[["ACCESS_METHOD_VAL", "PROBA_1"]].rename(columns={"PROBA_1": newcol})
        df = df.dropna(subset=["ACCESS_METHOD_VAL"]).groupby("ACCESS_METHOD_VAL", as_index=False)[newcol].max()
        out = df if out is None else out.merge(df, on="ACCESS_METHOD_VAL", how="outer")

    churn = pd.concat([read_ds(d)[["ACCESS_METHOD_VAL", "PROBA_1"]] for d in CHURN_MODELS], ignore_index=True)
    churn = (churn.dropna(subset=["ACCESS_METHOD_VAL"])
                  .groupby("ACCESS_METHOD_VAL", as_index=False)["PROBA_1"].max()
                  .rename(columns={"PROBA_1": "PROBA_CHURN"}))
    return downcast(out.merge(churn, on="ACCESS_METHOD_VAL", how="outer"))


def load_usage(dataset_by_period):
    """dataset_by_period: {"W1": "activities_w1_auto_hdfs", ..., "M3": "..."}.
    Keeps only the columns the model needs (voice minutes, data MB, PAYG data amount)."""
    out = None
    for p, ds in dataset_by_period.items():
        df = read_ds(ds)
        cols = [f"{u}_{p}" for u in USAGE_COLS if f"{u}_{p}" in df.columns]
        df = (df[["ACCESS_METHOD_VAL"] + cols].dropna(subset=["ACCESS_METHOD_VAL"])
                .groupby("ACCESS_METHOD_VAL", as_index=False).sum())
        out = df if out is None else out.merge(df, on="ACCESS_METHOD_VAL", how="outer")
        log(f"usage {p} loaded from {ds}: {len(df):,} lines")
    return downcast(out)


def load_subs(ds_name):
    df = read_ds(ds_name)
    cols = [f"SUB_AMT_{p}" for p in PERIODS]
    return downcast(df[["ACCESS_METHOD_VAL"] + cols].dropna(subset=["ACCESS_METHOD_VAL"])
                    .groupby("ACCESS_METHOD_VAL", as_index=False).sum())


# =============================================================================
# 2. FEATURE ENGINEERING (identical for training and scoring)
# =============================================================================
def build_features(pop, party, scores, usage, subs, seed=SEED):
    """pop: one row per line with at least ACCESS_METHOD_VAL and PARTY_ID."""
    df = pop.drop(columns=[c for c in ["NATIONALITY"] if c in pop.columns])
    df = df.dropna(subset=["ACCESS_METHOD_VAL"]).drop_duplicates("ACCESS_METHOD_VAL").copy()
    df = (df.merge(party, on="PARTY_ID", how="left")
            .merge(scores, on="ACCESS_METHOD_VAL", how="left")
            .merge(usage, on="ACCESS_METHOD_VAL", how="left")
            .merge(subs, on="ACCESS_METHOD_VAL", how="left"))

    # 2.1 Missing model scores -> small random value, drawn PER ROW
    rng = np.random.default_rng(seed)
    for c in PROBA_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        miss = df[c].isna()
        df.loc[miss, c] = rng.uniform(0, 0.02, int(miss.sum()))

    # 2.2 Usage / subscriptions: missing = no activity = 0 ; revenue = subscriptions + PAYG data
    for p in PERIODS:
        for c in [f"SUB_AMT_{p}", f"PG_DATA_AMOUNT_{p}", f"LOCAL_VOICE_MIN_{p}", f"DATA_VOLUME_MB_{p}"]:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float64")
        df[f"REV_{p}"] = df[f"SUB_AMT_{p}"] + df[f"PG_DATA_AMOUNT_{p}"]

    # 2.3 Trends: % change, log change, "older period was zero" flag
    for k, pairs in TREND_PAIRS.items():
        for recent, older in pairs:
            cur = df[TREND_SOURCE[k].format(p=recent)]
            prev = df[TREND_SOURCE[k].format(p=older)]
            zero = prev == 0
            df[f"{k}_{older}_ZERO"] = zero.astype(int)
            pct = np.where(zero, 0.0, (cur - prev) / prev.where(~zero, 1.0) * 100.0)
            df[f"VAR_{k}_{recent}_{older}"] = np.clip(pct, *PCT_CLIP)
            df[f"LOG_{k}_{recent}_{older}"] = np.log1p(cur.clip(lower=0)) - np.log1p(prev.clip(lower=0))

    # 2.4 Nationality group
    df["NATIONALITY_GROUP"] = np.where(df["NATIONALITY"].isin(NATIONALITY_KEEP), df["NATIONALITY"], "others")
    return df.reset_index(drop=True)


# =============================================================================
# 3. VOWPAL WABBIT FORMAT
# =============================================================================
def _fmt(v):
    return f"{v:.6g}"


def shared_lines(df):
    """One 'shared |User ...' line per customer.
    Numeric features use name:value (real numbers), not name=value (which VW reads as a category)."""
    s = "shared |User nationality_group=" + df["NATIONALITY_GROUP"].astype(str).str.replace(" ", "_")
    for c in NUMERIC_FEATURES:
        v = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        tok = " " + c.lower() + ":" + v.map(_fmt)
        s = s + tok.where(v != 0, "")
    return s.tolist()


def make_example(shared, label=None):
    """label = (action_index, cost, probability) or None for prediction."""
    lines = [shared]
    for i, al in enumerate(ACTION_LINES):
        if label is not None and i == label[0]:
            lines.append(f"0:{label[1]}:{label[2]:.6f} {al}")
        else:
            lines.append(al)
    return lines


def predict_pmf(vw, shared):
    ex = vw.parse(make_example(shared), vowpalwabbit.LabelType.CONTEXTUAL_BANDIT)
    pmf = np.asarray(vw.predict(ex), dtype=float)
    vw.finish_example(ex)
    return pmf / pmf.sum()


def learn_one(vw, shared, action_idx, cost, prob):
    ex = vw.parse(make_example(shared, (action_idx, cost, prob)), vowpalwabbit.LabelType.CONTEXTUAL_BANDIT)
    vw.learn(ex)
    vw.finish_example(ex)


# =============================================================================
# 4. SCORING A POPULATION
# =============================================================================
def score_population(vw, feat, policy="sample", id_cols=("ACCESS_METHOD_VAL",), seed=SEED):
    """policy='sample' -> draw the action from the model distribution (exploration, as in the original code)
       policy='greedy' -> take the most probable action (pure exploitation)."""
    shared = shared_lines(feat)
    n = len(shared)
    pmfs = np.zeros((n, len(ACTIONS)))
    for i, s in enumerate(shared):
        pmfs[i] = predict_pmf(vw, s)
        if i and i % 200000 == 0:
            log(f"scored {i:,}/{n:,}")

    greedy = pmfs.argmax(axis=1)
    rng = np.random.default_rng(seed)
    u = rng.random(n)[:, None]
    sampled = np.minimum((pmfs.cumsum(axis=1) < u).sum(axis=1), len(ACTIONS) - 1)
    chosen = sampled if policy == "sample" else greedy

    out = feat[list(id_cols)].copy().reset_index(drop=True)
    acts = np.array(ACTIONS)
    out["ACTION"] = acts[chosen]
    out["PROBA"] = pmfs[np.arange(n), chosen]
    out["ACTION_GREEDY"] = acts[greedy]
    out["PROBA_GREEDY"] = pmfs[np.arange(n), greedy]
    for i, t in enumerate(ACTION_TOKENS):
        out[f"P_{t}"] = pmfs[:, i]
    out["POLICY"] = policy
    return out


# =============================================================================
# 5. MODEL STORAGE (Dataiku managed folder)
# =============================================================================
def save_model(vw, meta, folder_name=MODEL_FOLDER):
    folder = dataiku.Folder(folder_name)
    path = os.path.join(tempfile.mkdtemp(), MODEL_FILE)
    vw.save(path)
    with open(path, "rb") as fh:
        model_bytes = fh.read()
    meta_bytes = json.dumps(meta, indent=2, default=str).encode("utf-8")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    for name, data in [(MODEL_FILE, model_bytes), (META_FILE, meta_bytes),
                       (f"archive/{stamp}_{MODEL_FILE}", model_bytes),
                       (f"archive/{stamp}_{META_FILE}", meta_bytes)]:
        with folder.get_writer(name) as w:
            w.write(data)
    log(f"model saved in folder '{folder_name}' (archive stamp {stamp})")


def load_model(folder_name=MODEL_FOLDER):
    folder = dataiku.Folder(folder_name)
    with folder.get_download_stream(MODEL_FILE) as s:
        model_bytes = s.read()
    with folder.get_download_stream(META_FILE) as s:
        meta = json.loads(s.read())
    if meta.get("features") != NUMERIC_FEATURES or meta.get("actions") != ACTIONS:
        raise ValueError("Saved model was trained with a different feature list or action list. Retrain it.")
    path = os.path.join(tempfile.mkdtemp(), MODEL_FILE)
    with open(path, "wb") as fh:
        fh.write(model_bytes)
    vw = vowpalwabbit.Workspace(f"{meta['vw_args']} -i {path}")
    log(f"model loaded (trained at {meta.get('trained_at')})")
    return vw, meta
