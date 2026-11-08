# -*- coding: utf-8 -*-
"""
FSDR - dummy data generator (all input tables, JSON, same structure as production)
==================================================================================
Writes one JSON file per table, with the exact production table names, field names,
field case and raw types (numbers stored as strings where production does so).

Window values are computed from a hidden DAILY history, using the same windows as the
production extractions (business rules section 4). So they are consistent with each other:
weeks add up to months, and the W2 of date D equals the W1 of date D-7.

Usage
-----
    python fsdr_generate_dummy.py --date 2026-10-22 --n 3000
    python fsdr_generate_dummy.py --date 2026-10-22 --n 3000 --weeks 4 --scenario churn_wave,data_growth
    python fsdr_generate_dummy.py --date 2026-10-22 --n 2000 --scenario data_quality_issues --format jsonl

Output
------
    <out>/<YYYY-MM-DD>/_manifest.json + one <table_name>.json per table
    With --weeks N, N folders are produced: D, D+7, ..., D+7(N-1) from ONE continuous history.

Scenarios (combinable, comma separated)
---------------------------------------
    baseline             normal behaviour
    churn_wave           churn risk of one region rises over time; those customers fade out
    data_growth          voice-only customers progressively start using data
    price_sensitive      customers mostly buy BTL offers priced near their usual spend
    new_offers           extra catalogue offers with no purchase history (cold start)
    campaign_fatigue     BTL take-up of targeted customers falls with repeated targeting
    data_quality_issues  duplicate lines, null keys, stacked churn duplicates (validation must catch them)

Only numpy and pandas are needed.
"""
import argparse
import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

REF_THURSDAY = date(2026, 3, 12)          # reference Thursday of the anchor formula
SUBS_TYPES_IN_SCOPE = ["CVM", "DATA", "Other Base Plans", "DIGITAL SERVICE", "EASY CREDIT", "EXPIRED CREDIT",
                       "Flexi", "Hala", "INTERNATIONAL VOICE", "New Shababiah", "Roaming", "SMS", "VOICE",
                       "Hala Flex"]
NATIONALITIES = (["Oman"] * 50 + ["India"] * 18 + ["Bangladesh"] * 12 + ["Pakistan"] * 8 + ["Egypt"] * 3 +
                 ["Philippines"] * 3 + ["Sri Lanka"] * 2 + ["Nepal"] * 2 + ["Sudan"] + ["Yemen"])
NAT_ID = {"Oman": "OMN", "India": "IND", "Bangladesh": "BGD", "Pakistan": "PAK", "Egypt": "EGY",
          "Philippines": "PHL", "Sri Lanka": "LKA", "Nepal": "NPL", "Sudan": "SDN", "Yemen": "YEM"}
REGIONS = {"Muscat": ["Bawshar", "Seeb", "Muttrah", "Al Amrat"], "North Batinah": ["Sohar", "Saham", "Shinas"],
           "South Batinah": ["Barka", "Rustaq", "Nakhal"], "Dhofar": ["Salalah", "Taqah", "Mirbat"],
           "Ad Dakhiliyah": ["Nizwa", "Bahla", "Samail"], "North Sharqiyah": ["Ibra", "Al Mudaybi"],
           "South Sharqiyah": ["Sur", "Jaalan Bani Bu Ali"], "Ad Dhahirah": ["Ibri", "Yanqul"],
           "Al Buraimi": ["Al Buraimi", "Mahdah"], "Al Wusta": ["Haima", "Duqm"]}
PLANS = ["Prepaid Plan A", "Prepaid Plan B", "Prepaid Plan C", "Prepaid Youth", "Prepaid Visitor",
         "Prepaid Flexi", "Prepaid Data Plus", "Prepaid Smart"]
DV_SEGMENTS = ["Smartphone_Heavy_Data", "Smartphone_Light", "Feature_Phone", "Dual_SIM_Switcher", "Tablet_Router"]
SEGMENTS = {
    "retain_low": "Retention_Low_Probability_To_Reconnect",
    "retain_med": "Retention_Med_Probability_To_Reconnect",
    "retain_high": "Retention_High_Probability_To_Reconnect",
    "rgs_at_risk": "Retention_RGS_At_Risk",
    "rgs_fragile": "Retention_Rgs_Fragile",
    "dualplay": "RGS_Not_At_Risk_DualPlay_Upgrade",
    "data_voice_app": "Rgs_Not_At_Risk_Data_Only_Voice_Appetency",
    "data_no_voice_app": "RGS_Not_At_Risk_Data_Only_No_Voice_Appetency",
    "voice_data_app": "RGS_Not_At_Risk_Voice_Only_Data_Appetency",
    "voice_no_data_app": "Rgs_Not_At_Risk_Voice_Only_No_Data_Appetency",
    "onb_risk": "Onboarding_Early_Churn_At_Risk",
    "onb_ok": "Onboarding_Early_Churn_Not_At_Risk",
    "nurs_risk": "Nursering_Early_Churn_At_Risk",
    "nurs_ok": "Nursering_Early_Churn_Not_At_Risk",
}
MODEL_FILES = {   # file name -> (hidden signal, coverage, tenure filter in days or None)
    "DAIM_CHURN_GLOBAL_SCORING_TD.prediction": ("churn", 0.85, None),
    "DAIM_EARLY_CHURN_0_30_SCORING_TD.Early_Churn_0_30_Prediction": ("churn", 0.95, (0, 30)),
    "DAIM_EARLY_CHURN_30_60_SCORING_TD.scoring_data_scored": ("churn", 0.95, (30, 60)),
    "DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_TD.Scored": ("react", 0.75, None),
    "PROPENSITY_REACTIVATION_15D_401_SCORING.Scored": ("react", 0.75, None),
    "APPETENCY_DATA_204_SCORING_TD.prediction": ("data_app", 0.85, None),
    "APPETENCY_VOICE_204_SCORING_TD.Scored": ("voice_app", 0.85, None),
    "CHURN_ONE_DAY_ACTIVE_107_SCORING.Churn_One_Day_Scoring": ("churn", 0.80, None),
    "REV_DECREASE_204_SCORING_TD.predictions": ("rev_decr", 0.80, None),
    "DAIM_FRAGILITY_DATA_SCORING_TD.prediction": ("frag_data", 0.70, None),
    "DAIM_FRAGILITY_VOICE_SCORING_TD.prediction": ("frag_voice", 0.70, None),
}
HIST_DAYS = 100           # daily usage / subscription history needed by the windows (max offset 91 + 6)
BTL_HIST_DAYS = 270       # fulfillment history (sub_btl_3months)


# =============================================================================
# helpers
# =============================================================================
def anchor_thursday(d):
    """T = most recent Thursday on or before d (same as the production SQL formula)."""
    return d - timedelta(days=(((d - REF_THURSDAY).days % 7) + 7) % 7)


def window_sum(arr, run_offset, start, end):
    """Sum of a daily array over [anchor-start, anchor-end] (start >= end, days before anchor).
    arr[:, j] = value of day (LAST_DAY - j); run_offset = days between LAST_DAY and the anchor."""
    return arr[:, run_offset + end: run_offset + start + 1].sum(axis=1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def r2(x):
    return np.round(np.asarray(x, dtype=float), 3)


def write_table(folder, name, df, fmt):
    path = os.path.join(folder, f"{name}.{fmt}")
    df = df.astype(object).where(pd.notna(df), None)
    records = df.to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as fh:
        if fmt == "json":
            json.dump(records, fh, ensure_ascii=False, default=str)
        else:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return len(records)


# =============================================================================
# 1. hidden world
# =============================================================================
def build_world(n, rng, first_day, n_days, scenarios, start_idx, weeks):
    cust = pd.DataFrame(index=range(n))
    vals = rng.choice(np.arange(70_000_000, 99_999_999), size=n, replace=False)
    cust["val"] = vals.astype(str)
    cust["am_id"] = 1_000_000 + np.arange(n)
    cust["acc_id"] = 5_000_000 + np.arange(n)
    cust["party_id"] = 8_000_000 + np.arange(n)
    cust["sub_id"] = ["S" + str(10_000_000 + i) for i in range(n)]
    cust["nationality"] = rng.choice(NATIONALITIES, n)
    cust["age"] = np.clip(rng.normal(34, 10, n), 18, 80).astype(int)
    cust["gender"] = rng.choice(["M", "F"], n, p=[0.68, 0.32])
    cust["region"] = rng.choice(list(REGIONS), n, p=[0.32, 0.14, 0.12, 0.1, 0.09, 0.06, 0.06, 0.05, 0.03, 0.03])
    cust["wilayat"] = [rng.choice(REGIONS[r]) for r in cust["region"]]
    cust["area"] = [f"{w}_Area_{rng.integers(1, 6)}" for w in cust["wilayat"]]
    cust["plan"] = rng.choice(PLANS, n)
    cust["dv_segment"] = rng.choice(DV_SEGMENTS, n, p=[0.35, 0.3, 0.12, 0.15, 0.08])
    # tenure at the first day of history: mix of new and old lines
    tenure0 = np.where(rng.random(n) < 0.12, rng.integers(-n_days, 60, n), rng.integers(60, 4000, n))
    cust["activation_date"] = [first_day - timedelta(days=int(t)) for t in tenure0]

    usage = rng.choice(["dual", "voice_only", "data_only"], n, p=[0.6, 0.22, 0.18])
    cust["usage_type"] = usage
    cust["voice_user"] = usage != "data_only"
    cust["data_user"] = usage != "voice_only"
    cust["spend"] = rng.lognormal(np.log(6), 0.7, n)                       # monthly spend level
    cust["churn"] = rng.beta(1.6, 6, n)                                    # base churn risk
    cust["data_app"] = np.clip(rng.beta(2, 3, n) + np.where(usage == "data_only", 0.3, 0), 0, 1)
    cust["voice_app"] = np.clip(rng.beta(2, 3, n) + np.where(usage == "voice_only", 0.3, 0), 0, 1)
    cust["price_sens"] = rng.beta(2, 2, n) * (1.6 if "price_sensitive" in scenarios else 1.0)

    # time-varying hidden state (n x n_days), day index 0 = first_day
    # scenario ramp: 0 until one week before the first run date, 1 at the last run date
    t = np.clip((np.arange(n_days)[None, :] - (start_idx - 7)) / (7.0 * weeks), 0, 1)
    churn_t = np.repeat(cust["churn"].values[:, None], n_days, axis=1)
    data_user_t = np.repeat(cust["data_user"].values[:, None], n_days, axis=1).astype(float)
    if "churn_wave" in scenarios:
        hit = (cust["region"].values == "North Batinah")[:, None]
        churn_t = np.where(hit, np.clip(churn_t + 0.5 * t, 0, 0.98), churn_t)
    if "data_growth" in scenarios:
        start = rng.random(n)[:, None]                                     # ramp level at which a voice user adopts data
        adopt = (cust["usage_type"].values == "voice_only")[:, None] & (t > start) & (t > 0) & (rng.random((n, 1)) < 0.6)
        data_user_t = np.maximum(data_user_t, adopt * 0.5)
    return cust, churn_t, data_user_t


# =============================================================================
# 2. daily usage + subscriptions
# =============================================================================
def build_daily(cust, churn_t, data_user_t, rng, first_day, n_days):
    n = len(cust)
    days = np.array([first_day + timedelta(days=i) for i in range(n_days)])
    alive = np.array([[d >= a for d in days] for a in cust["activation_date"]])   # line exists
    p_active = np.clip(0.92 - 0.8 * churn_t ** 1.5, 0.02, 0.98) * alive
    active = rng.random((n, n_days)) < p_active
    voice = cust["voice_user"].values[:, None] & active
    data = (rng.random((n, n_days)) < data_user_t) & active
    sp = cust["spend"].values[:, None]

    d = {}
    d["calls"] = rng.poisson(np.clip(2.5 * cust["voice_app"].values[:, None] + 0.5, 0.1, None), (n, n_days)) * voice
    d["voice_min"] = d["calls"] * rng.gamma(2.0, 1.3, (n, n_days))
    intl = np.clip(rng.beta(1.2, 6, n), 0, 0.9)[:, None]
    onnet_share = rng.beta(4, 3, n)[:, None]
    d["local_calls"] = np.round(d["calls"] * (1 - intl)).astype(int)
    d["local_min"] = d["voice_min"] * (1 - intl)
    d["onnet_calls"] = np.round(d["local_calls"] * onnet_share).astype(int)
    d["offnet_calls"] = d["local_calls"] - d["onnet_calls"]
    d["onnet_min"] = d["local_min"] * onnet_share
    d["offnet_min"] = d["local_min"] * (1 - onnet_share)
    d["sessions"] = rng.poisson(np.clip(6 * cust["data_app"].values[:, None], 0.2, None), (n, n_days)) * data
    d["data_mb"] = d["sessions"] * rng.gamma(1.5, 40, (n, n_days)) * (sp / 6) ** 0.5
    bundle_cover = rng.beta(5, 2, n)[:, None]
    d["pg_voice"] = d["voice_min"] * 0.035 * (1 - bundle_cover) * (rng.random((n, n_days)) < 0.5)
    d["pg_data"] = d["data_mb"] * 0.0008 * (1 - bundle_cover) * (rng.random((n, n_days)) < 0.25)
    d["pg_other"] = active * rng.gamma(0.5, 0.02, (n, n_days)) * (rng.random((n, n_days)) < 0.1)
    d["pg_total"] = d["pg_voice"] + d["pg_data"] + d["pg_other"]
    d["active"] = active.astype(int)
    d["active_voice"] = (d["calls"] > 0).astype(int)
    d["active_data"] = (d["sessions"] > 0).astype(int)

    # subscriptions: daily purchase probability ~ spend; amount = SUBSCRIPTION_AMT, TOT = AMT incl. 5% VAT
    buy = rng.random((n, n_days)) < np.clip(sp / 40, 0.01, 0.6) * active
    cls = rng.choice(["Daily Data", "Weekly Data", "Monthly Data", "Local Voice", "Other"], (n, n_days),
                     p=[0.22, 0.2, 0.18, 0.25, 0.15])
    cls = np.where(~cust["data_user"].values[:, None] & np.isin(cls, ["Daily Data", "Weekly Data", "Monthly Data"]),
                   "Local Voice", cls)
    amt = buy * rng.choice([0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0], (n, n_days), p=[0.15, 0.2, 0.25, 0.15, 0.1, 0.1, 0.05])
    d["sub_amt"] = amt
    d["sub_tot_amt"] = amt * 1.05
    d["sub_data_amt"] = amt * np.isin(cls, ["Daily Data", "Weekly Data", "Monthly Data"])
    d["sub_voice_amt"] = amt * (cls == "Local Voice")
    d["sub_count"] = buy.astype(int)
    # reverse to "days before last day" order: arr[:, j] = day (last_day - j)
    return {k: v[:, ::-1].astype(np.float32) for k, v in d.items()}, active[:, ::-1]


# =============================================================================
# 3. BTL catalogue
# =============================================================================
def build_catalog(rng, scenarios):
    rows = []
    gb_by_gen = {"L0": 1, "L1": 2, "L2": 3.5}
    for service in ["DATA", "VOICE", "COMBO"]:
        for validity in [7, 30]:
            for gen in ["L0", "L1", "L2"]:
                for size in [1, 2]:
                    base = (0.6 if validity == 7 else 2.0) * size * (1 + 0.35 * ["L0", "L1", "L2"].index(gen))
                    price = round(base * (1.3 if service == "COMBO" else 1.0), 1)
                    gb = round(gb_by_gen[gen] * size * (1 if validity == 7 else 4), 1) if service != "VOICE" else 0
                    mins = int((30 if validity == 7 else 120) * size * (1 + ["L0", "L1", "L2"].index(gen))) \
                        if service != "DATA" else 0
                    name = f"BTL_{service}_{gen}_{validity}D_{gb:g}GB_{mins}MIN"
                    p_data = price if service == "DATA" else (round(price * 0.7, 2) if service == "COMBO" else 0)
                    rows.append(dict(btl_name=name, type=f"{service.title()} Bundle", generosity=gen, gb=str(gb),
                                     min=str(mins), price=str(price), **{"Price data": str(p_data),
                                     "Price voice": str(round(price - p_data, 2))},
                                     service=service, validity=str(validity), _new=False))
    if "new_offers" in scenarios:
        for i, (svc, val, gen, price, gb, mins) in enumerate([("DATA", 30, "L2", 6.5, 40, 0), ("COMBO", 30, "L1", 7.0, 20, 300),
                                                              ("DATA", 7, "L2", 2.2, 10, 0)]):
            rows.append(dict(btl_name=f"BTL_NEW_{svc}_{gen}_{val}D_{i}", type=f"{svc.title()} Bundle", generosity=gen,
                             gb=str(gb), min=str(mins), price=str(price), **{"Price data": str(price if svc == "DATA" else 4.5),
                             "Price voice": str(0 if svc == "DATA" else 2.5)}, service=svc, validity=str(val), _new=True))
    return pd.DataFrame(rows)


def choose_offers(rng, cust_idx, cat, cust, price_sens_on):
    """Pick one BTL per purchase event, preferring the customer's services and price level."""
    cat_old = cat[~cat["_new"]].reset_index(drop=True)
    price = cat_old["price"].astype(float).values
    svc = cat_old["service"].values
    out = []
    for i in cust_idx:
        c = cust.iloc[i]
        w = np.ones(len(cat_old))
        w *= np.where(svc == "DATA", 0.3 + c["data_app"], 1)
        w *= np.where(svc == "VOICE", 0.3 + c["voice_app"], 1)
        if not c["data_user"]:
            w *= np.where(svc == "DATA", 0.05, 1)
        ref = c["spend"] / 4
        w *= np.exp(-c["price_sens"] * (2.5 if price_sens_on else 1.0) * np.maximum(price - ref, 0) / max(ref, 0.5))
        out.append(cat_old["btl_name"].iloc[rng.choice(len(cat_old), p=w / w.sum())])
    return out


# =============================================================================
# 4. main
# =============================================================================
def generate(n, start_date, weeks, scenarios, out_dir, fmt, seed, arms):
    rng = np.random.default_rng(seed)
    last_run = start_date + timedelta(days=7 * (weeks - 1))
    n_days = HIST_DAYS + 7 * (weeks - 1) + 7
    first_day = last_run - timedelta(days=n_days - 1)
    start_idx = (start_date - first_day).days
    cust, churn_t, data_user_t = build_world(n, rng, first_day, n_days, scenarios, start_idx, weeks)
    daily, active_rev = build_daily(cust, churn_t, data_user_t, rng, first_day, n_days)
    cat = build_catalog(rng, scenarios)

    # fulfillment events over BTL_HIST_DAYS + simulated weeks (timestamped)
    btl_days = BTL_HIST_DAYS + 7 * (weeks - 1) + 7
    btl_first = last_run - timedelta(days=btl_days - 1)
    rate = np.clip(cust["spend"].values / 500, 0.0015, 0.02)   # BTL purchases per day
    n_ev = rng.poisson(rate * btl_days)
    ev_cust = np.repeat(np.arange(n), n_ev)
    ev_day = rng.integers(0, btl_days, len(ev_cust))
    ev = pd.DataFrame({"i": ev_cust, "day": [btl_first + timedelta(days=int(x)) for x in ev_day]})
    ev["product"] = choose_offers(rng, ev["i"].values, cat, cust, "price_sensitive" in scenarios)
    ev["ts"] = [datetime.combine(dd, datetime.min.time()) + timedelta(seconds=int(s))
                for dd, s in zip(ev["day"], rng.integers(0, 86400, len(ev)))]

    times_targeted = np.zeros(n)
    written = []
    for w in range(weeks):
        D = start_date + timedelta(days=7 * w)
        T = anchor_thursday(D)
        offD = (last_run - D).days                  # offset of D in the reversed daily arrays
        offT = (last_run - T).days
        hidx = n_days - 1 - offD                    # index of day D in the forward hidden-state arrays
        folder = os.path.join(out_dir, D.isoformat())
        os.makedirs(folder, exist_ok=True)
        tables = {}
        lines_alive = np.array([a <= D - timedelta(days=2) for a in cust["activation_date"]])
        # lines that churned for good (no activity in the last 30 days) leave the base
        recent_active = active_rev[:, offD + 2: offD + 32].any(axis=1)
        in_base = lines_alive & (recent_active | (np.array([(D - a).days for a in cust["activation_date"]]) < 30))
        b = cust[in_base].copy()
        bi = b.index.values
        tenure = np.array([(D - a).days for a in b["activation_date"]])

        # ---------- base & reference
        tables["base_prepaid"] = pd.DataFrame({
            "UBSCRIPTION_ID": b["sub_id"].values, "ACCESS_METHOD_VAL": b["val"].values,
            "ACCESS_METHOD_ID": b["am_id"].values, "ACCOUNT_ID": b["acc_id"].values, "PARTY_ID": b["party_id"].values,
            "access_method_subs_start_dt": [a.isoformat() for a in b["activation_date"]], "CURRENT_PLAN": b["plan"].values})
        tables["consumer_aact_account"] = pd.DataFrame({"ACCOUNT_ID": b["acc_id"].values, "PARTY_ID": b["party_id"].values})
        tables["access_method_id_auto"] = pd.DataFrame({
            "ACCESS_METHOD_ID": b["am_id"].values, "MSISDN": b["val"].values,   # = ACCESS_METHOD_VAL (no prefix)
            "ACTIVATION_DATE": [a.isoformat() for a in b["activation_date"]]})
        party = pd.DataFrame({"party_id": b["party_id"].values, "AGE": b["age"].values, "GENDER": b["gender"].values,
                              "NATIONALITY": b["nationality"].values,
                              "nationality_identifier": b["nationality"].map(NAT_ID).values})
        tables["party_auto"] = party
        tables["party_scoring"] = party.copy()
        prev_m = (D - timedelta(days=2)).replace(day=1) - timedelta(days=1)
        tables["location_auto"] = pd.DataFrame({
            "Access_Method_id": b["am_id"].values, "Calling_Month": prev_m.month, "Calling_Year": prev_m.year,
            "REGION": b["region"].values, "WILAYAT": b["wilayat"].values, "AREA": b["area"].values})
        tables["DEVICEBEHAVIORSEGMENTATION.dv_segment"] = pd.DataFrame(
            {"ACCESS_METHOD_ID": b["am_id"].values, "DV_SEGMENT": b["dv_segment"].values})
        tables["btl_catalog"] = cat.drop(columns="_new")
        tables["BTL_script_english_arabic_ok"] = pd.DataFrame({
            "Sl No.": [str(i + 1) for i in range(len(cat))], "destination_offer_name": cat["btl_name"].values,
            "English Script": [f"Special offer for you: {r.btl_name.replace('_', ' ')} for {r.price} OMR, valid {r.validity} days. "
                               f"Reply YES to activate." for r in cat.itertuples()],
            "Arabic Script": [f"عرض خاص لك: {r.btl_name.replace('_', ' ')} بسعر {r.price} ريال، صالح لمدة {r.validity} يوم. "
                              f"أرسل نعم للتفعيل." for r in cat.itertuples()]})
        conf = []
        cr = cat.reset_index(drop=True)
        for a in range(len(cr)):
            for c2 in range(a + 1, len(cr)):
                A, B = cr.iloc[a], cr.iloc[c2]
                if A.service == B.service and A.validity == B.validity and abs(float(A.price) - float(B.price)) <= 1.5:
                    conf.append({"Offer_A": A.btl_name, "Gen_Level_A": A.generosity, "Price_A": A.price, "Data_A": A.gb,
                                 "Min_A": A["min"], "Validity_A": A.validity, "Offer_Type_A": A.service,
                                 "Offer_B": B.btl_name, "Gen_Level_B": B.generosity, "Price_B": B.price, "Data_B": B.gb,
                                 "Min_B": B["min"], "Validity_B": B.validity, "Offer_Type_B": B.service,
                                 "Price_Difference": str(round(abs(float(A.price) - float(B.price)), 2)),
                                 "Data_Difference": str(round(abs(float(A.gb) - float(B.gb)), 2)),
                                 "SAME_VALIDITY": "YES", "SAME_OFFER_TYPE": "YES"})
        tables["conflicting_offers_h"] = pd.DataFrame(conf)

        # ---------- subscription amounts (window INCLUDES the anchor day)
        def subs_amt(off):
            df = pd.DataFrame({"ACCESS_METHOD_VAL": b["val"].values, "ACCESS_METHOD_ID": b["am_id"].values,
                               "ACCOUNT_ID": b["acc_id"].values.astype(float)})
            for name, (s, e) in {"w1": (6, 0), "w2": (13, 7), "w3": (20, 14), "w4": (27, 21),
                                 "m1": (29, 0), "m2": (59, 30), "m3": (89, 60)}.items():
                df[f"Sub_Amt_{name}"] = r2(window_sum(daily["sub_tot_amt"], off, s, e)[bi])
            return df
        tables["subs_amt_auto"] = subs_amt(offT)
        tables["subs_amt_score"] = subs_amt(offD)

        # ---------- activities (window EXCLUDES the anchor day)
        act_windows = {"W1": (7, 1), "W2": (14, 8), "W3": (21, 15), "W4": (28, 22),
                       "M1": (28, 1), "M2": (56, 29), "M3": (85, 57)}

        def activities(off, P):
            s, e = act_windows[P]
            ws = lambda k: window_sum(daily[k], off, s, e)[bi]
            return pd.DataFrame({
                "ACCESS_METHOD_VAL": b["val"].values, "ACCESS_METHOD_ID": b["am_id"].values, "ACCOUNT_ID": b["acc_id"].values,
                f"Call_Count_{P}": ws("calls").astype(int), f"Data_Session_Count_{P}": ws("sessions").astype(int),
                f"Voice_Traffic_min_{P}": r2(ws("voice_min")), f"Data_Volume_MB_{P}": r2(ws("data_mb")),
                f"PG_Total_Amount_{P}": r2(ws("pg_total")), f"PG_Voice_Amount_{P}": r2(ws("pg_voice")),
                f"PG_Data_Amount_{P}": r2(ws("pg_data")),
                f"Onnet_Call_Count_{P}": ws("onnet_calls").astype(int), f"Onnet_Voice_min_{P}": r2(ws("onnet_min")),
                f"Offnet_Call_Count_{P}": ws("offnet_calls").astype(int), f"Offnet_Voice_min_{P}": r2(ws("offnet_min")),
                f"Local_Call_Count_{P}": ws("local_calls").astype(int), f"Local_Voice_min_{P}": r2(ws("local_min")),
                f"Active_Days_Count_{P}": ws("active").astype(int), f"Active_Days_Voice_Count_{P}": ws("active_voice").astype(int),
                f"Active_Days_Data_Count_{P}": ws("active_data").astype(int)})
        for P in act_windows:
            p = P.lower()
            tables["activities_auto_m2" if P == "M2" else f"activities_{p}_auto"] = activities(offT, P)
            tables[f"activities_{p}_score"] = activities(offD, P)

        # ---------- PAYG and pricing tables
        pay_w = {"m1": (30, 1), "m2": (60, 31), "m3": (90, 61), "w1": (7, 1), "w2": (14, 8), "w3": (21, 15), "w4": (28, 22)}

        def payg(off):
            df = pd.DataFrame({"ACCESS_METHOD_ID": b["am_id"].values, "ACCOUNT_ID": b["acc_id"].values,
                               "ACCESS_METHOD_VAL": b["val"].values})
            for k, (s, e) in pay_w.items():
                df[f"PG_Total_Amount_{k}"] = r2(window_sum(daily["pg_total"], off, s, e)[bi])
            return df
        tables["paygo_activity_auto"] = payg(offT)
        up = payg(offD)
        for i, (s, e) in enumerate([(31, 2), (61, 32), (91, 62)], start=1):
            up.insert(2 + i, f"local_Voice_min_p{i}", r2(window_sum(daily["local_min"], offD, s, e)[bi]))
        for i, (s, e) in enumerate([(31, 2), (61, 32), (91, 62)], start=1):
            up.insert(5 + i, f"data_volume_mb_p{i}", r2(window_sum(daily["data_mb"], offD, s, e)[bi]))
        tables["usage_data_voice_payg_tg"] = up

        st = pd.DataFrame({"ACCESS_METHOD_VAL": b["val"].values, "ACCESS_METHOD_ID": b["am_id"].values,
                           "ACCOUNT_ID": b["acc_id"].values.astype(float)})
        for k, (s, e) in {"w1": (8, 2), "w2": (15, 9), "w3": (22, 16), "w4": (29, 23)}.items():
            st[f"Total_Sub_Amt_{k}"] = r2(window_sum(daily["sub_amt"], offD, s, e)[bi])
            st[f"Data_Sub_Amt_{k}"] = r2(window_sum(daily["sub_data_amt"], offD, s, e)[bi])
            st[f"Voice_Sub_Amt_{k}"] = r2(window_sum(daily["sub_voice_amt"], offD, s, e)[bi])
        for k, (s, e) in {"M1": (31, 2), "M2": (61, 32), "M3": (91, 62)}.items():
            st[f"Sub_Count_{k}"] = window_sum(daily["sub_count"], offD, s, e)[bi].astype(int)
            st[f"Total_Sub_Amt_{k}"] = r2(window_sum(daily["sub_amt"], offD, s, e)[bi])
            st[f"Data_Sub_Amt_{k}"] = r2(window_sum(daily["sub_data_amt"], offD, s, e)[bi])
            st[f"Voice_Sub_Amt_{k}"] = r2(window_sum(daily["sub_voice_amt"], offD, s, e)[bi])
        tables["subscriptions_tg"] = st

        # ---------- model scores (one shared structure)
        churn_now = churn_t[bi, hidx]
        data_user_now = data_user_t[bi, hidx]
        rev_m1 = window_sum(daily["sub_tot_amt"], offD, 29, 0)[bi] + window_sum(daily["pg_total"], offD, 30, 1)[bi]
        rev_m2 = window_sum(daily["sub_tot_amt"], offD, 59, 30)[bi] + window_sum(daily["pg_total"], offD, 60, 31)[bi]
        inactive7 = 1 - np.clip(window_sum(daily["active"], offD, 7, 1)[bi] / 7, 0, 1)
        signal = {
            "churn": churn_now,
            "react": np.clip(inactive7 * (1 - churn_now) + 0.1, 0, 1),
            "data_app": np.clip(b["data_app"].values * (0.6 + 0.4 * (data_user_now < 0.5)), 0, 1),
            "voice_app": b["voice_app"].values * (~b["voice_user"].values * 0.5 + 0.5),
            "rev_decr": sigmoid(4 * (rev_m2 - rev_m1) / (rev_m2 + 1)),
            "frag_data": np.clip(churn_now * b["data_user"].values + 0.1 * rng.random(len(b)), 0, 1),
            "frag_voice": np.clip(churn_now * b["voice_user"].values + 0.1 * rng.random(len(b)), 0, 1),
        }
        lif = pd.cut(tenure, [-1, 30, 60, 90, 180, 365, 100000],
                     labels=["0-30", "31-60", "61-90", "91-180", "181-365", "365+"]).astype(str)
        for fname, (sig, cov, tfilter) in MODEL_FILES.items():
            p1 = sigmoid(3.5 * (signal[sig] - 0.5) + rng.normal(0, 0.6, len(b)))
            keep = rng.random(len(b)) < cov
            if tfilter:
                keep &= (tenure >= tfilter[0]) & (tenure <= tfilter[1])
            tables[fname] = pd.DataFrame({
                "subscription_id": b["sub_id"].values[keep], "access_method_val": b["val"].values[keep].astype(np.int64),
                "access_method_id": b["am_id"].values[keep], "tenure": tenure[keep], "life_in_nw_range": lif[keep],
                "proba_0": r2(1 - p1[keep]), "proba_1": r2(p1[keep]),
                "prediction": np.where(p1[keep] >= 0.5, "1", "0")})

        # ---------- segment of each line (what the segmentation engine would give)
        seg = np.empty(len(b), dtype=object)
        dl = b["data_user"].values | (data_user_now >= 0.5)
        vo = b["voice_user"].values
        for j in range(len(b)):
            if tenure[j] < 45:
                seg[j] = SEGMENTS["nurs_risk" if churn_now[j] > 0.3 else "nurs_ok"]
            elif tenure[j] < 90:
                seg[j] = SEGMENTS["onb_risk" if churn_now[j] > 0.3 else "onb_ok"]
            elif inactive7[j] > 0.7:
                seg[j] = SEGMENTS[["retain_low", "retain_med", "retain_high"][min(int((1 - churn_now[j]) * 3), 2)]]
            elif churn_now[j] > 0.45:
                seg[j] = SEGMENTS["rgs_at_risk"]
            elif churn_now[j] > 0.3:
                seg[j] = SEGMENTS["rgs_fragile"]
            elif dl[j] and vo[j]:
                seg[j] = SEGMENTS["dualplay"]
            elif dl[j]:
                seg[j] = SEGMENTS["data_voice_app" if b["voice_app"].values[j] > 0.5 else "data_no_voice_app"]
            else:
                seg[j] = SEGMENTS["voice_data_app" if b["data_app"].values[j] > 0.5 else "voice_no_data_app"]

        # ---------- targeting (previous campaign, date1 in D-8 .. D-2)
        tgt = rng.random(len(b)) < 0.45
        ti = np.where(tgt)[0]
        times_targeted[bi[ti]] += 1
        names = cat["btl_name"].values
        dests = rng.choice(names, (len(ti), 6))
        dests = np.where(rng.random((len(ti), 6)) < np.array([0, 0, 0.3, 0.5, 0.7, 0.85]), None, dests)
        date1 = [(D - timedelta(days=int(x))).isoformat() for x in rng.integers(2, 9, len(ti))]
        tdi = pd.DataFrame({"date1": date1, "access_method_id": b["am_id"].values[ti],
                            "account_id": b["acc_id"].values[ti], "segment": seg[ti]})
        for k in range(6):
            tdi[f"destination_offer_name_{k + 1}"] = dests[:, k]
        tables["targeting_dataiku_auto"] = tdi
        ta = pd.DataFrame({"DATE1": date1, "START_DATE": [(D - timedelta(days=int(x))).isoformat() for x in rng.integers(0, 8, len(ti))]})
        ta["END_DATE"] = [(date.fromisoformat(s) + timedelta(days=7)).isoformat() for s in ta["START_DATE"]]
        ta["MSISDN"] = b["val"].values[ti]
        for k in range(1, 7):
            ta[f"CHANNEL_{k}"] = np.where(dests[:, k - 1] == None, None, "SMS")  # noqa: E711
            ta[f"OPPORTUNITY_NAME_{k}"] = np.where(dests[:, k - 1] == None, None, seg[ti])  # noqa: E711
            ta[f"DESTINATION_OFFER_NAME_{k}"] = dests[:, k - 1]
        ta["FLAG_SPLIO"] = rng.choice(["Y", "N"], len(ti))
        for k in range(1, 7):
            ta[f"OFFER_SCRIPT_EN_{k}"] = [None if x is None else f"Special offer: {x}" for x in dests[:, k - 1]]
            ta[f"OFFER_SCRIPT_AR_{k}"] = [None if x is None else f"عرض خاص: {x}" for x in dests[:, k - 1]]
        ta["SEGMENT"] = seg[ti]
        ta["FILENAME"] = f"targeting_{D:%Y%m%d}.csv"
        ta["ETL_CREATE_DT"] = (D - timedelta(days=1)).isoformat()
        ta["JOB_ID"] = 1000 + w
        for k in range(5, 7):  # columns 5-6 come after JOB_ID in production order
            for col in [f"CHANNEL_{k}", f"OPPORTUNITY_NAME_{k}", f"DESTINATION_OFFER_NAME_{k}",
                        f"OFFER_SCRIPT_EN_{k}", f"OFFER_SCRIPT_AR_{k}"]:
                ta[col] = ta.pop(col)
        ta["SEGMENT_BTL_TAKER"] = np.where(rng.random(len(ti)) < 0.3, "BTL_TAKER", "NON_TAKER")
        ta["SEGMENT_BTL_TAKER_UPSELL"] = np.where(rng.random(len(ti)) < 0.2, "UPSELL", None)
        tables["targeting_auto"] = ta

        # ---------- fulfillment: last 7 days (+ extra take-up for targeted customers = campaign effect)
        f7 = ev[(ev["day"] >= D - timedelta(days=7)) & (ev["day"] <= D - timedelta(days=1)) & ev["i"].isin(bi)]
        fatigue = 0.5 ** np.maximum(times_targeted[bi[ti]] - 1, 0) if "campaign_fatigue" in scenarios else 1.0
        extra = ti[rng.random(len(ti)) < 0.06 * (1 - 0.5 * signal["churn"][ti]) * fatigue]
        if len(extra):
            ex = pd.DataFrame({"i": bi[extra], "day": [D - timedelta(days=int(x)) for x in rng.integers(1, 8, len(extra))]})
            ex["product"] = [dests[np.where(ti == j)[0][0], 0] for j in extra]
            ex["ts"] = [datetime.combine(dd, datetime.min.time()) + timedelta(seconds=int(s))
                        for dd, s in zip(ex["day"], rng.integers(0, 86400, len(ex)))]
            ev = pd.concat([ev, ex], ignore_index=True)       # becomes part of the history for later weeks
            f7 = pd.concat([f7, ex], ignore_index=True)
        tables["subs_btl_auto"] = pd.DataFrame({
            "access_method_val": cust["val"].values[f7["i"].values], "PRODUCTNAME": f7["product"].values,
            "FULFILMENTSUCCESSDATE_TIME": [t.isoformat(sep=" ") for t in f7["ts"]], "sub_flag": 1, "nb_sub": 1})
        h = ev[(ev["day"] >= D - timedelta(days=270)) & (ev["day"] <= D - timedelta(days=1))]
        h = h.assign(val=cust["val"].values[h["i"].values]).groupby(["val", "product"]).size().reset_index(name="nb_sub")
        tables["sub_btl_3months"] = pd.DataFrame({"access_method_val": h["val"], "PRODUCTNAME": h["product"],
                                                  "sub_flag": 1, "nb_sub": h["nb_sub"].astype(int)})

        # ---------- test sample (every field is a string), stratified groups
        s_idx = np.where(rng.random(len(b)) < 0.4)[0]
        rev3 = (window_sum(daily["sub_amt"], offD, 91, 2) + window_sum(daily["pg_total"], offD, 90, 1))[bi][s_idx]
        payg_m = window_sum(daily["pg_total"], offD, 30, 1)[bi][s_idx]
        subs_m = window_sum(daily["sub_amt"], offD, 31, 2)[bi][s_idx]
        vmin = window_sum(daily["local_min"], offD, 31, 2)[bi][s_idx]
        dmb = window_sum(daily["data_mb"], offD, 31, 2)[bi][s_idx]
        vgrp = pd.cut(vmin, [-1, 0, 60, 300, 1e9], labels=["No Voice", "Low Voice", "Mid Voice", "High Voice"]).astype(str)
        dgrp = pd.cut(dmb, [-1, 0, 1024, 10240, 1e12], labels=["No Data", "Low Data", "Mid Data", "High Data"]).astype(str)
        rgrp = pd.cut(rev3, [-1, 5, 15, 40, 1e9], labels=["0-5", "5-15", "15-40", "40+"]).astype(str)
        natg = np.where(np.isin(b["nationality"].values[s_idx], ["Oman", "India", "Bangladesh", "Pakistan"]),
                        b["nationality"].values[s_idx], "others")
        final = pd.Series([f"{a}|{r}|{v}|{d}|{p}" for a, r, v, d, p in
                           zip(natg, rgrp, vgrp, dgrp, b["plan"].values[s_idx])])
        groups = np.empty(len(s_idx), dtype=object)
        for _, idx in final.groupby(final).groups.items():           # stratified: shuffle inside each stratum
            idx = rng.permutation(np.asarray(idx))
            off = int(rng.integers(len(arms)))                     # random start so small strata stay balanced
            groups[idx] = [arms[(k + off) % len(arms)] for k in range(len(idx))]
        tables["sample_cg_tg_new_7oct2026"] = pd.DataFrame({
            "access_method_val": b["val"].values[s_idx], "access_method_id": b["am_id"].values[s_idx].astype(str),
            "account_id": b["acc_id"].values[s_idx].astype(str), "tenure": tenure[s_idx].astype(str),
            "activation_date": [a.isoformat() for a in b["activation_date"].values[s_idx]],
            "plan_name": b["plan"].values[s_idx], "payg_revenue": r2(payg_m).astype(str),
            "subscription_revenue": r2(subs_m).astype(str), "total_revenue": r2(payg_m + subs_m).astype(str),
            "total_revenue_3m": rgrp, "voice_traffic": r2(vmin).astype(str), "data_volume": r2(dmb).astype(str),
            "nationality": b["nationality"].values[s_idx], "region": b["region"].values[s_idx],
            "ARPU": r2(payg_m + subs_m).astype(str), "ARPU_3m": r2(rev3 / 3).astype(str),
            "voice_trf_group": vgrp, "data_trf_group": dgrp, "Final": final.values, "channel": "SMS", "group": groups})

        # ---------- data quality issues (on purpose)
        if "data_quality_issues" in scenarios:
            bp = tables["base_prepaid"]
            dup = bp.sample(frac=0.01, random_state=seed)
            bp = pd.concat([bp, dup], ignore_index=True)
            bp.loc[bp.sample(frac=0.005, random_state=seed + 1).index, "PARTY_ID"] = None
            tables["base_prepaid"] = bp
            ch = tables["DAIM_CHURN_GLOBAL_SCORING_TD.prediction"]
            tables["DAIM_CHURN_GLOBAL_SCORING_TD.prediction"] = pd.concat([ch, ch.sample(frac=0.05, random_state=seed)])
            a1 = tables["activities_w1_score"]
            a1["ACCESS_METHOD_ID"] = a1["ACCESS_METHOD_ID"].astype(str)
            a1.loc[a1.sample(frac=0.005, random_state=seed + 2).index, "ACCESS_METHOD_VAL"] = None

        counts = {name: write_table(folder, name, df, fmt) for name, df in tables.items()}
        manifest = {"run_date": D.isoformat(), "anchor_date": T.isoformat(), "source": "fsdr_generate_dummy.py",
                    "scenarios": sorted(scenarios), "seed": seed, "n_customers_world": n, "format": fmt,
                    "schema_version": "1.0", "row_counts": counts,
                    "note": "Dummy data: fake MSISDNs, ids and offers. Same tables, fields and types as production."}
        with open(os.path.join(folder, "_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        written.append(folder)
        print(f"[{D}] T={T} base={len(b):,} targeted={len(ti):,} takers7d={tables['subs_btl_auto'].access_method_val.nunique():,} "
              f"sample={len(s_idx):,} tables={len(tables)} -> {folder}")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FSDR dummy JSON data generator")
    ap.add_argument("--date", default=date.today().isoformat(), help="first run date D (YYYY-MM-DD)")
    ap.add_argument("--n", type=int, default=3000, help="number of customers in the world")
    ap.add_argument("--weeks", type=int, default=1, help="number of consecutive weekly snapshots")
    ap.add_argument("--scenario", default="baseline", help="comma separated scenarios")
    ap.add_argument("--out", default="data/fsdr", help="output root folder")
    ap.add_argument("--format", default="json", choices=["json", "jsonl"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", default="cg,free,cstr", help="groups of the test sample, e.g. cg,free,cstr,llm")
    a = ap.parse_args()
    generate(a.n, date.fromisoformat(a.date), a.weeks, set(a.scenario.split(",")), a.out, a.format, a.seed,
             a.arms.split(","))
