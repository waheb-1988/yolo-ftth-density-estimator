# -*- coding: utf-8 -*-
"""
FSDR - PIPELINE 1 : NBA TRAINING  (data extraction -> features -> labels -> train -> save -> score full base)
=========================================================================================================
Run: weekly, on the campaign anchor day (Thursday), after the targeting of the previous campaign
     and the BTL fulfillments of the last 7 days are loaded.

Dataiku inputs (built upstream by the SQL recipes, all anchored on the Thursday "T"):
    base_prepaid_hdfs, party_auto_hdfs, access_method_id_auto_hdfs, btl_catalog_hdfs,
    targeting_dataiku_auto_hdfs, subs_btl_auto_hdfs (fulfillment last 7 days),
    activities_w1/w2/w3/w4_auto_hdfs, activities_m1_auto_hdfs, activities_auto_m2_hdfs, activities_m3_auto_hdfs,
    subs_amt_auto_hdfs, + the 11 model-score datasets listed in nba_lib.py
Dataiku outputs:
    managed folder  nba_model           (model + metadata + archive)
    nba_training_set                    (labelled training rows - audit)
    nba_training_report                 (1 row of training KPIs)
    nba_scored_base                     (action for every line of the prepaid base)
"""
import random
import numpy as np
import pandas as pd
import dataiku
import vowpalwabbit

from nba_lib import (VW_ARGS, SEED, ACTIONS, ACTION_INDEX, ACTION_PRIOR, SEGMENT_TO_ACTION,
                     EXCLUDED_SEGMENTS, NUMERIC_FEATURES, log, read_ds, load_party,
                     load_model_scores, load_usage, load_subs, build_features, shared_lines,
                     predict_pmf, learn_one, score_population, save_model)

# ------------------------------------------------------------------ settings
TRAIN_MODE = "logged"       # "logged"     : learn from every targeted customer (takers AND non-takers) - recommended
                            # "simulation" : original approach, takers only, model must guess the logged action
N_PASSES = 2                # passes over the data in "logged" mode
SIM_ITERATIONS = 20000      # iterations in "simulation" mode
HOLDOUT_SHARE = 0.2         # share of targeted customers kept aside to evaluate the model
SCORE_FULL_BASE = True
BASE_SCORING_POLICY = "greedy"

random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# STEP 1 - READ EXTRACTED DATA
# =============================================================================
log("STEP 1 - reading data")
base = read_ds("base_prepaid_hdfs")[["ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"]]
party = load_party("party_auto_hdfs")
scores = load_model_scores()
usage = load_usage({
    "W1": "activities_w1_auto_hdfs", "W2": "activities_w2_auto_hdfs",
    "W3": "activities_w3_auto_hdfs", "W4": "activities_w4_auto_hdfs",
    "M1": "activities_m1_auto_hdfs", "M2": "activities_auto_m2_hdfs", "M3": "activities_m3_auto_hdfs",
})
subs = load_subs("subs_amt_auto_hdfs")
catalog = read_ds("btl_catalog_hdfs")
fulfil = read_ds("subs_btl_auto_hdfs")
targeting = read_ds("targeting_dataiku_auto_hdfs")
am = read_ds("access_method_id_auto_hdfs")
log(f"base: {len(base):,} lines | targeting: {len(targeting):,} rows | fulfillments: {len(fulfil):,}")

# =============================================================================
# STEP 2 - FEATURES FOR THE WHOLE BASE
# =============================================================================
log("STEP 2 - building features")
feat = build_features(base, party, scores, usage, subs)
del usage, subs, scores
log(f"features: {len(feat):,} lines x {len(NUMERIC_FEATURES) + 1} model features")

# =============================================================================
# STEP 3 - LABELS: ACTION (from segment), REWARD (BTL taken), PROPENSITY
# =============================================================================
log("STEP 3 - building labels")
am["ACCESS_METHOD_VAL"] = pd.to_numeric(am["MSISDN"], errors="coerce").astype("Int64")
am = am.sort_values("ACTIVATION_DATE").drop_duplicates("ACCESS_METHOD_ID", keep="last")
tg = targeting.merge(am[["ACCESS_METHOD_ID", "ACCESS_METHOD_VAL"]], on="ACCESS_METHOD_ID", how="left")

tg["SEG"] = tg["SEGMENT"].astype(str).str.strip().str.lower()
tg = tg[~tg["SEG"].isin(EXCLUDED_SEGMENTS)]                       # drop Nursering (< 45 days tenure)
tg["ACTION"] = tg["SEG"].map(SEGMENT_TO_ACTION)
unmapped = tg.loc[tg["ACTION"].isna(), "SEGMENT"].value_counts()
if len(unmapped):
    log(f"WARNING segments without action (dropped):\n{unmapped}")
tg = tg.dropna(subset=["ACTION", "ACCESS_METHOD_VAL"])
tg = tg.sort_values("DATE1").drop_duplicates("ACCESS_METHOD_VAL", keep="last")   # latest targeting per line

btl_names = set(catalog["BTL_NAME"].astype(str))
takers = set(fulfil.loc[fulfil["PRODUCTNAME"].astype(str).isin(btl_names), "ACCESS_METHOD_VAL"].dropna())
tg["REWARD"] = tg["ACCESS_METHOD_VAL"].isin(takers).astype(int)
tg["COST"] = -tg["REWARD"]                                          # VW minimises cost: taker = -1

share = tg["ACTION"].value_counts(normalize=True)                   # logging propensity = this week's action share
tg["PROPENSITY"] = tg["ACTION"].map(share).fillna(tg["ACTION"].map(ACTION_PRIOR)).clip(0.01, 1.0)

train_all = (tg[["ACCESS_METHOD_VAL", "ACTION", "REWARD", "COST", "PROPENSITY"]]
             .merge(feat, on="ACCESS_METHOD_VAL", how="inner"))
train_all["ACTION_IDX"] = train_all["ACTION"].map(ACTION_INDEX)
log(f"labelled rows: {len(train_all):,} | takers: {train_all.REWARD.sum():,} "
    f"({train_all.REWARD.mean():.2%})\n{train_all.groupby('ACTION').REWARD.agg(['count', 'mean'])}")

# train / holdout split (random)
rng = np.random.default_rng(SEED)
train_all["HOLDOUT"] = rng.random(len(train_all)) < HOLDOUT_SHARE
train = train_all[~train_all.HOLDOUT].reset_index(drop=True)
hold = train_all[train_all.HOLDOUT].reset_index(drop=True)

# =============================================================================
# STEP 4 - TRAIN THE CONTEXTUAL BANDIT
# =============================================================================
log(f"STEP 4 - training ({TRAIN_MODE})")
vw = vowpalwabbit.Workspace(VW_ARGS)

if TRAIN_MODE == "logged":
    sh = shared_lines(train)
    a_idx, cost, prop = train.ACTION_IDX.values, train.COST.values, train.PROPENSITY.values
    for p in range(N_PASSES):
        for i in np.random.permutation(len(train)):
            learn_one(vw, sh[i], int(a_idx[i]), int(cost[i]), float(prop[i]))
        log(f"pass {p + 1}/{N_PASSES} done")

elif TRAIN_MODE == "simulation":
    tk = train[train.REWARD == 1].reset_index(drop=True)
    sh = shared_lines(tk)
    true_idx = tk.ACTION_IDX.values
    cum_cost = 0
    for it in range(1, SIM_ITERATIONS + 1):
        j = np.random.randint(len(tk))
        pmf = predict_pmf(vw, sh[j])
        a = int(np.random.choice(len(ACTIONS), p=pmf))
        c = -1 if a == true_idx[j] else 0
        learn_one(vw, sh[j], a, c, float(pmf[a]))
        cum_cost += c
        if it % 5000 == 0:
            log(f"iteration {it:,}: running match rate {-cum_cost / it:.3f}")
else:
    raise ValueError("TRAIN_MODE must be 'logged' or 'simulation'")

# =============================================================================
# STEP 5 - EVALUATE ON THE HOLDOUT
# =============================================================================
log("STEP 5 - evaluation on holdout")
hold_pmf = np.vstack([predict_pmf(vw, s) for s in shared_lines(hold)]) if len(hold) else np.zeros((0, len(ACTIONS)))
greedy = hold_pmf.argmax(axis=1) if len(hold) else np.array([], dtype=int)
match = greedy == hold.ACTION_IDX.values
logged_take_rate = float(hold.REWARD.mean()) if len(hold) else float("nan")
ips_take_rate = float(np.mean(hold.REWARD.values * match / hold.PROPENSITY.values)) if len(hold) else float("nan")
taker_match = float(match[hold.REWARD.values == 1].mean()) if hold.REWARD.sum() else float("nan")
log(f"holdout take rate (current policy): {logged_take_rate:.4f} | IPS estimate model policy: {ips_take_rate:.4f} "
    f"| takers' action reproduced: {taker_match:.3f}")

# =============================================================================
# STEP 6 - SAVE MODEL + METADATA
# =============================================================================
meta = {
    "trained_at": pd.Timestamp.now().isoformat(),
    "vw_args": VW_ARGS,
    "train_mode": TRAIN_MODE,
    "actions": ACTIONS,
    "features": NUMERIC_FEATURES,
    "n_train": int(len(train)),
    "n_takers_train": int(train.REWARD.sum()),
    "n_holdout": int(len(hold)),
    "holdout_take_rate": logged_take_rate,
    "holdout_ips_take_rate": ips_take_rate,
    "holdout_taker_match": taker_match,
}
save_model(vw, meta)

report = pd.DataFrame([meta]).drop(columns=["actions", "features"])
dataiku.Dataset("nba_training_report").write_with_schema(report)

train_out = train_all[["ACCESS_METHOD_VAL", "ACTION", "REWARD", "PROPENSITY", "HOLDOUT"] + NUMERIC_FEATURES].copy()
train_out["ACCESS_METHOD_VAL"] = train_out["ACCESS_METHOD_VAL"].astype("string")
train_out.columns = train_out.columns.str.lower()
dataiku.Dataset("nba_training_set").write_with_schema(train_out)

# =============================================================================
# STEP 7 - SCORE THE FULL BASE WITH THE NEW MODEL
# =============================================================================
if SCORE_FULL_BASE:
    log(f"STEP 7 - scoring full base ({len(feat):,} lines, policy={BASE_SCORING_POLICY})")
    scored = score_population(vw, feat, policy=BASE_SCORING_POLICY,
                              id_cols=("ACCESS_METHOD_VAL", "ACCESS_METHOD_ID"))
    scored["ACCESS_METHOD_VAL"] = scored["ACCESS_METHOD_VAL"].astype("string")
    scored["SCORE_DATE"] = pd.Timestamp.now().normalize()
    scored["MODEL_TRAINED_AT"] = meta["trained_at"]
    scored.columns = scored.columns.str.lower()
    log(f"action distribution:\n{scored.action.value_counts(normalize=True)}")
    dataiku.Dataset("nba_scored_base").write_with_schema(scored)

vw.finish()          # only now - the model is no longer needed
log("TRAINING PIPELINE DONE")
