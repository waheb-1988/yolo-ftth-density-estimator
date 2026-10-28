# -*- coding: utf-8 -*-
"""
FSDR - PIPELINE 2 : NBA SCORING ONLY  (at campaign start, with the model saved by 01_nba_train.py)
===================================================================================================
No training happens here. The model is loaded from the managed folder, fresh features are built
from the *_score tables (anchored on CURRENT_DATE), and every customer receives an action.

Run: at campaign start (or any day you need fresh actions).

Dataiku inputs:
    managed folder  nba_model
    party_scoring_hdfs, subs_amt_score_hdfs,
    activities_w1_score_hfds, activities_w2/w3/w4_score_hdfs, activities_m1/m2/m3_score_hdfs,
    + the 11 model-score datasets listed in nba_lib.py
    POPULATION = "target" : sample_cg_tg_new_7oct2026 + consumer_aact_account_hdfs
    POPULATION = "base"   : base_prepaid_hdfs
Dataiku outputs:
    results_scoring_nba                         (all scored lines)
    TG_40k_free_scored_nba, TG_40K_constraint_scored_nba   (only when POPULATION="target"; replaces old Code 2)
These outputs feed the offer recommendation recipes (old Code 3 = constraint arm, old Code 4 = free arm) unchanged.
"""
import numpy as np
import pandas as pd
import dataiku

from nba_lib import (log, read_ds, load_party, load_model_scores, load_usage, load_subs,
                     build_features, score_population, load_model)

# ------------------------------------------------------------------ settings
POPULATION = "target"            # "target" = campaign test sample (free + constraint arms) | "base" = full prepaid base
TARGET_DATASET = "sample_cg_tg_new_7oct2026"
ACTION_POLICY = "sample"         # "sample" = keep exploration (as the original code) | "greedy" = best action only
GROUP_TO_FLAG = {"free": "free", "cstr": "with_constraint"}

# =============================================================================
# STEP 1 - LOAD THE TRAINED MODEL
# =============================================================================
log("STEP 1 - loading model")
vw, meta = load_model()

# =============================================================================
# STEP 2 - POPULATION TO SCORE
# =============================================================================
log(f"STEP 2 - population = {POPULATION}")
if POPULATION == "target":
    s = read_ds(TARGET_DATASET)
    s["TG_FLAG"] = s["GROUP"].astype(str).str.strip().str.lower().map(GROUP_TO_FLAG)
    s = s.dropna(subset=["TG_FLAG"])
    acct = read_ds("consumer_aact_account_hdfs")[["ACCOUNT_ID", "PARTY_ID"]].dropna().drop_duplicates("ACCOUNT_ID")
    pop = s[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "TG_FLAG"]].merge(acct, on="ACCOUNT_ID", how="left")
elif POPULATION == "base":
    pop = read_ds("base_prepaid_hdfs")[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]]
    pop["TG_FLAG"] = "base"
else:
    raise ValueError("POPULATION must be 'target' or 'base'")
log(f"lines to score: {pop.ACCESS_METHOD_VAL.nunique():,}")

# =============================================================================
# STEP 3 - FRESH FEATURES (windows relative to CURRENT_DATE)
# =============================================================================
log("STEP 3 - building features")
party = load_party("party_scoring_hdfs")
scores = load_model_scores()
usage = load_usage({
    "W1": "activities_w1_score_hfds", "W2": "activities_w2_score_hdfs",
    "W3": "activities_w3_score_hdfs", "W4": "activities_w4_score_hdfs",
    "M1": "activities_m1_score_hdfs", "M2": "activities_m2_score_hdfs", "M3": "activities_m3_score_hdfs",
})
subs = load_subs("subs_amt_score_hdfs")
feat = build_features(pop, party, scores, usage, subs)
del usage, subs, scores

# =============================================================================
# STEP 4 - PREDICT THE ACTION
# =============================================================================
log(f"STEP 4 - scoring with policy={ACTION_POLICY}")
res = score_population(vw, feat, policy=ACTION_POLICY,
                       id_cols=("ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "TG_FLAG"))
vw.finish()

res["ACCESS_METHOD_VAL"] = res["ACCESS_METHOD_VAL"].astype("string")   # same type as the fulfillment tables used by LightFM
res["SCORE_DATE"] = pd.Timestamp.now().normalize()
res["MODEL_TRAINED_AT"] = meta["trained_at"]
res.columns = res.columns.str.lower()
log(f"action distribution:\n{pd.crosstab(res.action, res.tg_flag, normalize='columns')}")

# =============================================================================
# STEP 5 - WRITE OUTPUTS
# =============================================================================
dataiku.Dataset("results_scoring_nba").write_with_schema(res)

if POPULATION == "target":
    dataiku.Dataset("TG_40k_free_scored_nba").write_with_schema(res[res.tg_flag == "free"])
    dataiku.Dataset("TG_40K_constraint_scored_nba").write_with_schema(res[res.tg_flag != "free"])

log("SCORING PIPELINE DONE")
