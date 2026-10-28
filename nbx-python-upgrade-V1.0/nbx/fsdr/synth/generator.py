"""Dummy-data generator: writes exactly the JSON files of the real data contract.

Every usage / revenue table is aggregated from the world's daily history with
the windows of nbx/fsdr/calendar.py (BR §4), so the values are consistent
with each other and with the run date: activities M1 = W1 + W2 + W3 + W4,
Sub_Amt_m1 >= the four weeks inside it, and W2 of D = W1 of D-7.

Last week's campaign is simulated too: targeting (D-8..D-2) by the segment
policy, offers chosen like production's constraint arm (all-L2), acceptances
drawn from the hidden response function, fulfilments in D-7..D-1 plus organic
purchases and non-BTL noise, and 270 days of purchase history for the ranker.

    run = generate_run(World.create(5000, seed=42, scenarios=["baseline"], as_of=d), d)
    write_run(run, Path("data/fsdr") / d.isoformat())
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nbx.fsdr import calendar as cal
from nbx.fsdr import config, schemas
from nbx.fsdr.calendar import MONTHS, WEEKS
from nbx.fsdr.synth import scenarios as scen
from nbx.fsdr.synth.world import GENEROSITY_BONUS, World

NON_BTL_PRODUCTS = ["DAILY_DATA_PASS_1GB", "WEEKLY_SOCIAL_PASS", "INTL_CALLING_PACK_5", "HALA_FLEX_RENEWAL"]
NON_BTL_SHARE = 0.06


@dataclass
class GeneratedRun:
    run_date: date
    anchor_date: date
    tables: dict[str, pd.DataFrame]       # raw frames: python values, schema column order, None = null
    manifest: dict
    world: World


# ---------------------------------------------------------------------------
# Raw conversion (typed frame -> JSON-ready python values)
# ---------------------------------------------------------------------------
def to_raw(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Typed frame -> object frame holding exactly what the JSON file will hold."""
    schema = schemas.TABLES[table]
    n = len(df)
    cols = {}
    for f in schema.fields:
        if f.name not in df.columns:
            cols[f.name] = np.full(n, None, dtype=object)
            continue
        s = df[f.name]
        mask = s.isna().to_numpy()
        if f.raw_type in schemas.INT_TYPES:
            vals = pd.to_numeric(s, errors="coerce").fillna(0).round().astype("int64").astype(object)
        elif f.raw_type == "double":
            vals = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float).round(3).astype(object)
        else:
            vals = s.astype(str).astype(object)
        arr = vals.to_numpy(dtype=object)
        arr[mask] = None
        cols[f.name] = arr
    return pd.DataFrame(cols, columns=schema.field_names)


def _fmt_num(x: float) -> str:
    return f"{x:g}"


# ---------------------------------------------------------------------------
# Usage / revenue tables from the daily history
# ---------------------------------------------------------------------------
def _usage_tables(world: World, run_date: date) -> dict[str, pd.DataFrame]:
    t_anchor = cal.anchor_date(run_date)
    start = cal.usage_history_start(run_date)
    days = [start + timedelta(days=i) for i in range(config.HISTORY_DAYS_USAGE)]
    parts: dict[str, list[pd.DataFrame]] = defaultdict(list)

    def ix(d: date) -> int:
        i = (d - start).days
        if i < 0 or i >= len(days):
            raise ValueError(f"window day {d} outside the generated history {start}..{days[-1]}")
        return i

    for chunk_id, lo in enumerate(range(0, world.n, config.GENERATOR_CHUNK)):
        sl = slice(lo, min(lo + config.GENERATOR_CHUNK, world.n))
        daily = [world.daily(d, sl, chunk_id) for d in days]
        m = sl.stop - sl.start
        cum = {k: np.vstack([np.zeros((1, m)), np.cumsum(np.vstack([x[k] for x in daily]), axis=0)]) for k in daily[0]}

        def win(k: str, w: cal.Window) -> np.ndarray:
            return cum[k][ix(w.end) + 1] - cum[k][ix(w.start)]

        def span(k: str, a: date, b: date) -> np.ndarray:
            return cum[k][ix(min(b, days[-1])) + 1] - cum[k][ix(max(a, start))]

        c = world.c.iloc[sl]
        val, amid, acc = c["msisdn"].to_numpy(), c["access_method_id"].to_numpy(), c["account_id"].to_numpy()

        # activities_* (BR 4.2): one table per period and anchor
        for anchor_name, anchor in (("T", t_anchor), ("D", run_date)):
            for p, w in cal.activity_windows(anchor).items():
                cols = {
                    "ACCESS_METHOD_VAL": val, "ACCESS_METHOD_ID": amid, "ACCOUNT_ID": acc,
                    f"Call_Count_{p}": win("voice_calls", w), f"Data_Session_Count_{p}": win("data_sessions", w),
                    f"Voice_Traffic_min_{p}": win("voice_min_rounded", w), f"Data_Volume_MB_{p}": win("data_mb_rounded", w),
                    f"PG_Total_Amount_{p}": win("pg_all", w), f"PG_Voice_Amount_{p}": win("pg_voice", w),
                    f"PG_Data_Amount_{p}": win("pg_data", w), f"Onnet_Call_Count_{p}": win("onnet_calls", w),
                    f"Onnet_Voice_min_{p}": win("onnet_min", w), f"Offnet_Call_Count_{p}": win("offnet_calls", w),
                    f"Offnet_Voice_min_{p}": win("offnet_min", w), f"Local_Call_Count_{p}": win("local_calls", w),
                    f"Local_Voice_min_{p}": win("local_min", w), f"Active_Days_Count_{p}": win("any_day", w),
                    f"Active_Days_Voice_Count_{p}": win("voice_day", w), f"Active_Days_Data_Count_{p}": win("data_day", w),
                }
                present = (cols[f"Active_Days_Count_{p}"] > 0) | (np.abs(cols[f"PG_Total_Amount_{p}"]) > 0)
                parts[schemas.activity_table_name(p, anchor_name)].append(pd.DataFrame(cols)[present])

        # subs_amt_* (BR 4.1, SUBSCRIPTION_TOT_AMT): rows for lines with any subscription
        has_sub = span("sub_tot_amt", start, run_date) > 0
        for table, anchor in (("subs_amt_auto", t_anchor), ("subs_amt_score", run_date)):
            cols = {"ACCESS_METHOD_VAL": val, "ACCESS_METHOD_ID": amid, "ACCOUNT_ID": acc.astype(float)}
            cols.update({f"Sub_Amt_{p.lower()}": win("sub_tot_amt", w) for p, w in cal.subs_windows(anchor).items()})
            parts[table].append(pd.DataFrame(cols)[has_sub])

        # paygo_activity_auto (T, positive charges only)
        cols = {"ACCESS_METHOD_ID": amid, "ACCOUNT_ID": acc, "ACCESS_METHOD_VAL": val}
        cols.update({f"PG_Total_Amount_{p.lower()}": win("pg_pos", w) for p, w in cal.payg_windows(t_anchor).items()})
        present = span("pg_pos", start, t_anchor) > 0
        parts["paygo_activity_auto"].append(pd.DataFrame(cols)[present])

        # subscriptions_tg (D, SUBSCRIPTION_AMT, BR-17/18/19)
        cols = {"ACCESS_METHOD_VAL": val, "ACCESS_METHOD_ID": amid, "ACCOUNT_ID": acc.astype(float)}
        for p, w in cal.subscriptions_tg_windows(run_date).items():
            sfx = p.lower() if p in WEEKS else p
            if p in MONTHS:
                cols[f"Sub_Count_{p}"] = win("sub_count", w)
            cols[f"Total_Sub_Amt_{sfx}"] = win("sub_amt", w)
            cols[f"Data_Sub_Amt_{sfx}"] = win("sub_amt_data", w)
            cols[f"Voice_Sub_Amt_{sfx}"] = win("sub_amt_voice", w)
        present = span("sub_amt", run_date - timedelta(days=95), run_date - timedelta(days=1)) > 0
        parts["subscriptions_tg"].append(pd.DataFrame(cols)[present])

        # usage_data_voice_payg_tg (D, actual durations, positive PAYG, BR-33/34)
        cols = {"ACCESS_METHOD_ID": amid, "ACCOUNT_ID": acc, "ACCESS_METHOD_VAL": val}
        for p, w in cal.usage_tg_volume_windows(run_date).items():
            cols[f"local_Voice_min_{p}"] = win("local_min", w)
            cols[f"data_volume_mb_{p}"] = win("data_mb", w)
        cols.update({f"PG_Total_Amount_{p.lower()}": win("pg_pos", w) for p, w in cal.payg_windows(run_date).items()})
        present = span("any_day", run_date - timedelta(days=95), run_date - timedelta(days=2)) > 0
        parts["usage_data_voice_payg_tg"].append(pd.DataFrame(cols)[present])

    return {t: pd.concat(v, ignore_index=True) for t, v in parts.items()}


# ---------------------------------------------------------------------------
# Base and reference tables
# ---------------------------------------------------------------------------
def _catalog_strings(cat: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "btl_name": cat["btl_name"], "type": cat["type"], "generosity": cat["generosity"],
        "gb": cat["gb"].map(_fmt_num), "min": cat["min"].map(_fmt_num), "price": cat["price"].map(_fmt_num),
        "Price data": cat["price_data"].map(_fmt_num), "Price voice": cat["price_voice"].map(_fmt_num),
        "service": cat["service"], "validity": cat["validity"].map(str),
    })


def _offer_text(o) -> tuple[str, str]:
    bits_en, bits_ar = [], []
    if o["gb"]:
        bits_en.append(f"{o['gb']:g}GB data")
        bits_ar.append(f"{o['gb']:g} جيجابايت")
    if o["min"]:
        bits_en.append(f"{int(o['min'])} local minutes")
        bits_ar.append(f"{int(o['min'])} دقيقة محلية")
    en = f"{config.OPERATOR_NAME}: enjoy {' + '.join(bits_en)} for {o['price']:g} {config.CURRENCY}, valid {o['validity']} days. Reply YES to activate."
    ar = f"{config.OPERATOR_NAME}: استمتع بـ {' + '.join(bits_ar)} مقابل {o['price']:g} ر.ع لمدة {o['validity']} يوم. أرسل نعم للتفعيل."
    return en, ar


def _conflicts(cat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, _), g in cat.sort_values("price").groupby(["service", "validity"]):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            if b["price"] - a["price"] > 1.5:
                continue
            rows.append({
                "Offer_A": a["btl_name"], "Gen_Level_A": a["generosity"], "Price_A": _fmt_num(a["price"]),
                "Data_A": _fmt_num(a["gb"]), "Min_A": _fmt_num(a["min"]), "Validity_A": str(a["validity"]),
                "Offer_Type_A": a["service"],
                "Offer_B": b["btl_name"], "Gen_Level_B": b["generosity"], "Price_B": _fmt_num(b["price"]),
                "Data_B": _fmt_num(b["gb"]), "Min_B": _fmt_num(b["min"]), "Validity_B": str(b["validity"]),
                "Offer_Type_B": b["service"],
                "Price_Difference": _fmt_num(round(b["price"] - a["price"], 3)),
                "Data_Difference": _fmt_num(round(b["gb"] - a["gb"], 3)),
                "SAME_VALIDITY": "Y", "SAME_OFFER_TYPE": "Y",
            })
    return pd.DataFrame(rows)


def _base_tables(world: World, run_date: date, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    c = world.c
    act = [(run_date - timedelta(days=int(t))).isoformat() for t in c["tenure"]]
    month, year = cal.location_month(run_date)
    party = pd.DataFrame({"party_id": c["party_id"], "AGE": c["age"], "GENDER": c["gender"], "NATIONALITY": c["nationality"],
                          "nationality_identifier": np.where(c["nationality"] == "Oman", "OMANI", "EXPAT")})
    loc = pd.DataFrame({"Access_Method_id": c["access_method_id"], "Calling_Month": month, "Calling_Year": year,
                        "REGION": c["region"], "WILAYAT": c["wilayat"], "AREA": c["area"]})
    extra = loc.sample(frac=0.03, random_state=int(rng.integers(1 << 30))).copy()     # BR-14: several locations
    extra["AREA"] = [f"AREA_{x:04d}" for x in rng.integers(400, 800, len(extra))]
    cat = world.catalog
    scripts = [_offer_text(o) for _, o in cat.iterrows()]
    dv = c.sample(frac=0.92, random_state=int(rng.integers(1 << 30)))
    return {
        "base_prepaid": pd.DataFrame({
            "UBSCRIPTION_ID": c["subscription_id"], "ACCESS_METHOD_VAL": c["msisdn"], "ACCESS_METHOD_ID": c["access_method_id"],
            "ACCOUNT_ID": c["account_id"], "PARTY_ID": c["party_id"], "access_method_subs_start_dt": act,
            "CURRENT_PLAN": c["plan"]}),
        "consumer_aact_account": pd.DataFrame({"ACCOUNT_ID": c["account_id"], "PARTY_ID": c["party_id"]}),
        "access_method_id_auto": pd.DataFrame({"ACCESS_METHOD_ID": c["access_method_id"], "MSISDN": c["msisdn"],
                                               "ACTIVATION_DATE": act}),
        "party_auto": party,
        "party_scoring": party.copy(),
        "location_auto": pd.concat([loc, extra], ignore_index=True),
        "btl_catalog": _catalog_strings(cat),
        "BTL_script_english_arabic_ok": pd.DataFrame({
            "Sl No.": [str(i + 1) for i in range(len(cat))], "destination_offer_name": cat["btl_name"],
            "English Script": [s[0] for s in scripts], "Arabic Script": [s[1] for s in scripts]}),
        "conflicting_offers_h": _conflicts(cat),
        "DEVICEBEHAVIORSEGMENTATION.dv_segment": pd.DataFrame({"ACCESS_METHOD_ID": dv["access_method_id"],
                                                                "DV_SEGMENT": dv["dv_segment"]}),
    }


# ---------------------------------------------------------------------------
# Model scores (one shared structure)
# ---------------------------------------------------------------------------
def _life_range(t: np.ndarray) -> np.ndarray:
    return np.select([t <= 30, t <= 90, t <= 180, t <= 365, t <= 730], ["0-30", "31-90", "91-180", "181-365", "1Y-2Y"], "2Y+")


def generate_model_score(world: World, m: config.ModelScore, rng: np.random.Generator, day: date | None = None) -> pd.DataFrame:
    """Any model file in MODEL_SCORE_SCHEMA: driven by the matching hidden signal + noise, partial coverage."""
    t = world.c["tenure"].to_numpy()
    mask = np.ones(world.n, bool) if m.tenure_range is None else (t >= m.tenure_range[0]) & (t <= m.tenure_range[1])
    cand = np.flatnonzero(mask)
    k = int(len(cand) * rng.uniform(*m.coverage))
    idx = np.sort(rng.choice(cand, size=k, replace=False)) if k else np.array([], dtype=int)
    p1 = np.round(world.score(m.signal or m.feature.lower(), rng, idx, day), 6) if k else np.array([])
    c = world.c.iloc[idx]
    return pd.DataFrame({
        "subscription_id": c["subscription_id"].to_numpy(), "access_method_val": c["msisdn"].astype("int64").to_numpy(),
        "access_method_id": c["access_method_id"].to_numpy(), "tenure": c["tenure"].to_numpy(),
        "life_in_nw_range": _life_range(c["tenure"].to_numpy()), "proba_0": np.round(1 - p1, 6), "proba_1": p1,
        "prediction": np.where(p1 >= 0.5, "1", "0"),
    })


# ---------------------------------------------------------------------------
# Segments, last week's campaign, purchase history
# ---------------------------------------------------------------------------
def _segments(world: World, rng: np.random.Generator) -> np.ndarray:
    c = world.c
    churn = world.score("churn", rng)
    react = world.score("reactivation_30", rng)
    da, va = world.score("data_appetite", rng), world.score("voice_appetite", rng)
    frag = np.maximum(world.score("fragility_data", rng), world.score("fragility_voice", rng))
    tenure, usage = c["tenure"].to_numpy(), c["usage_type"].to_numpy()
    low = (c["churn_day"].to_numpy() <= 0) | (usage == "inactive")
    seg = np.select(
        [(tenure < config.NURSERING_MAX_TENURE_DAYS) & (churn >= 0.5), tenure < config.NURSERING_MAX_TENURE_DAYS,
         (tenure < config.ONBOARDING_MAX_TENURE_DAYS) & (churn >= 0.5), tenure < config.ONBOARDING_MAX_TENURE_DAYS,
         low & (react < 0.3), low & (react < 0.6), low,
         churn >= 0.6, frag >= 0.55, usage == "dual",
         (usage == "data_only") & (va >= 0.4), usage == "data_only", da >= 0.4],
        ["Nursering_Early_Churn_At_Risk", "Nursering_Early_Churn_Not_At_Risk",
         "Onboarding_Early_Churn_At_Risk", "Onboarding_Early_Churn_Not_At_Risk",
         "Retention_Low_Probability_To_Reconnect", "Retention_Med_Probability_To_Reconnect",
         "Retention_High_Probability_To_Reconnect",
         "Retention_RGS_At_Risk", "Retention_RGS_Fragile", "RGS_Not_At_Risk_DualPlay_Upgrade",
         "RGS_Not_At_Risk_Data_Only_Voice_Appetency", "RGS_Not_At_Risk_Data_Only_No_Voice_Appetency",
         "RGS_Not_At_Risk_Voice_Only_Data_Appetency"],
        "RGS_Not_At_Risk_Voice_Only_No_Data_Appetency").astype(object)
    # production emits both spellings ("RGS_..." and "Rgs_..."): mix them
    mixed = (rng.random(len(seg)) < 0.5) & np.array(["RGS" in s for s in seg])
    seg[mixed] = [s.replace("RGS", "Rgs").replace("DualPlay", "Dualplay") for s in seg[mixed]]
    return seg


def _preference_weights(world: World, idx: np.ndarray, cat: pd.DataFrame) -> np.ndarray:
    c = world.c.iloc[idx]
    da, va = c["data_appetite"].to_numpy()[:, None], c["voice_appetite"].to_numpy()[:, None]
    svc = cat["service"].to_numpy()[None, :]
    fit = np.where(svc == "DATA", da, np.where(svc == "VOICE", va, 0.5 * (da + va)))
    gen = np.vectorize(GENEROSITY_BONUS.get)(cat["generosity"].to_numpy())[None, :]
    validity = cat["validity"].to_numpy()[None, :]
    spend = c["spend"].to_numpy()[:, None]
    anchor = np.where(validity == 7, spend / 4.3, spend)
    gap = np.maximum(0, cat["price"].to_numpy()[None, :] - anchor) / np.maximum(anchor, 0.5)
    w = np.exp(1.4 * fit + 0.5 * gen - 0.8 * c["price_sensitivity"].to_numpy()[:, None] * gap)
    # add-on families: roaming is bought by travellers, international packs by people who call abroad
    travel = np.where(c["traveller"].to_numpy(), 1.0, 0.01)[:, None]
    intl = np.clip(c["intl_share"].to_numpy() * 6, 0.01, 2.0)[:, None]
    w = w * np.where(svc == "ROAMING", travel, 1.0) * np.where(svc == "INTERNATIONAL", intl, 1.0)
    w = w * (~cat["is_new"].to_numpy())[None, :]                              # new offers: no history (cold start)
    return w.astype(np.float32)


def _sample_offers(rng: np.random.Generator, weights: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(weights / weights.sum(axis=1, keepdims=True), axis=1)
    return np.minimum((cdf < rng.random(len(weights))[:, None]).sum(axis=1), weights.shape[1] - 1)


def _legacy_offers(world: World, idx: np.ndarray, actions: np.ndarray) -> list[np.ndarray]:
    """Offers last week's campaign sent: production constraint arm with all-L2 (BR-66 active), the
    cheapest family offer above the customer's spend, per validity -> up to 4 destination offers."""
    cat = world.catalog[~world.catalog["is_new"]]
    spend = world.c["spend"].to_numpy()[idx]
    slots = [np.full(len(idx), None, dtype=object) for _ in range(4)]
    for action, (svc_a, svc_b) in config.ACTION_FAMILIES.items():
        sel = np.flatnonzero(actions == action)
        if not len(sel):
            continue
        for v_i, validity in enumerate((30, 7)):
            anchor = spend[sel] / (4.3 if validity == 7 else 1.0)
            for s_i, svc in enumerate((svc_a, svc_b)):
                fam = cat[(cat["service"] == svc) & (cat["validity"] == validity) & (cat["generosity"] == "L2")].sort_values("price")
                if fam.empty:
                    continue
                prices = fam["price"].to_numpy()
                pos = np.minimum(np.searchsorted(prices, anchor, side="right"), len(prices) - 1)
                slots[v_i * 2 + s_i][sel] = fam["btl_name"].to_numpy()[pos]
    return slots


def _ts(rng: np.random.Generator, days: list[date]) -> list[str]:
    secs = rng.integers(0, 86_400, len(days))
    return [(datetime.combine(d, datetime.min.time()) + timedelta(seconds=int(s))).isoformat() for d, s in zip(days, secs)]


def _our_campaign(world: World, campaign: pd.DataFrame, segments: np.ndarray):
    """Last week's campaign as logged by the app (campaign_log rows with offers): who, which action, which
    offers, which day -> the same arrays as the simulated legacy campaign."""
    pos = {m: i for i, m in enumerate(world.c["msisdn"].astype(str))}
    sent = campaign[campaign["offers"].map(lambda o: isinstance(o, list) and len(o) > 0)]
    sent = sent[sent["msisdn"].astype(str).isin(pos)].drop_duplicates("msisdn")
    targeted = np.array([pos[str(m)] for m in sent["msisdn"]], dtype=int)
    actions = sent["action"].astype(str).to_numpy(dtype=object)
    date1 = [date.fromisoformat(str(d)) for d in sent["run_date"]]
    slots = [np.full(len(sent), None, dtype=object) for _ in range(4)]
    for r, offers in enumerate(sent["offers"]):
        for i, o in enumerate(offers[:4]):
            slots[i][r] = o["code"]
    return targeted, segments[targeted].copy(), actions, date1, slots


def _campaign_tables(world: World, run_date: date, segments: np.ndarray, rng: np.random.Generator,
                     campaign: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Last week's campaign (targeting, responses, organic purchases, history). campaign = the app's own
    log of last week (closed-loop simulation); None = a simulated production campaign."""
    c = world.c
    n = world.n
    cat = world.catalog.set_index("btl_name")
    scripts = {name: _offer_text(row) for name, row in world.catalog.set_index("btl_name", drop=False).iterrows()}
    if campaign is not None:
        targeted, seg, actions, date1, slots = _our_campaign(world, campaign, segments)
    else:
        targeted = np.flatnonzero(rng.random(n) < config.TARGETED_SHARE)
        seg = segments[targeted].copy()
        seg[rng.random(len(seg)) < 0.01] = None                              # BR-37 rows without segment
        lower = np.array([s.lower() if s else "" for s in seg])
        actions = np.array([config.SEGMENT_TO_ACTION.get(s, "UPSELL" if s else "") for s in lower], dtype=object)
        d1_off = rng.integers(config.TARGETING_WINDOW[0], config.TARGETING_WINDOW[1] + 1, len(targeted))
        date1 = [run_date + timedelta(days=int(o)) for o in d1_off]
        slots = _legacy_offers(world, targeted, actions)

    targeting = pd.DataFrame({"date1": [d.isoformat() for d in date1], "access_method_id": c["access_method_id"].to_numpy()[targeted],
                              "account_id": c["account_id"].to_numpy()[targeted], "segment": seg})
    for i in range(6):
        targeting[f"destination_offer_name_{i + 1}"] = slots[i] if i < 4 else None

    # acceptances among the targeted (hidden response function)
    probs = np.zeros((len(targeted), 4))
    for i, s in enumerate(slots):
        ok = np.array([x is not None for x in s])
        if ok.any():
            o = cat.loc[s[ok]]
            probs[ok, i] = world.accept_probability(targeted[ok], actions[ok].astype(str), o["service"].to_numpy(),
                                                    o["generosity"].to_numpy(), o["price"].to_numpy(),
                                                    o["validity"].to_numpy(), day=run_date)
    p_any = 1 - np.prod(1 - probs, axis=1)
    takers = np.flatnonzero(rng.random(len(targeted)) < p_any)
    fulfil_rows = []
    btl_names = world.catalog.loc[~world.catalog["is_new"], "btl_name"].to_numpy()
    for t in takers:
        p = probs[t] / probs[t].sum()
        product = slots[int(rng.choice(4, p=p))][t]
        if rng.random() < 0.05:                                              # took another BTL than the one offered
            product = rng.choice(btl_names)
        first = max(date1[t], run_date + timedelta(days=config.REWARD_WINDOW[0]))
        span = (run_date + timedelta(days=config.REWARD_WINDOW[1]) - first).days
        fulfil_rows.append((targeted[t], product, first + timedelta(days=int(rng.integers(0, span + 1)))))

    # organic BTL purchases and non-BTL noise, D-7..D-1: every line that did not take a campaign offer -- being
    # contacted must not remove a customer's own buying (else contacted arms fall below the control)
    not_takers = np.setdiff1d(np.arange(n), targeted[takers])
    organic = not_takers[rng.random(len(not_takers)) < world.organic_probability(not_takers)]
    if len(organic):
        picks = _sample_offers(rng, _preference_weights(world, organic, world.catalog))
        for i, j in zip(organic, picks):
            fulfil_rows.append((i, world.catalog["btl_name"].iat[j], run_date - timedelta(days=int(rng.integers(1, 8)))))
    for i in np.flatnonzero(rng.random(n) < NON_BTL_SHARE):
        fulfil_rows.append((i, rng.choice(NON_BTL_PRODUCTS), run_date - timedelta(days=int(rng.integers(1, 8)))))

    f_idx = np.array([r[0] for r in fulfil_rows], dtype=int)
    fulfil = pd.DataFrame({"access_method_val": c["msisdn"].to_numpy()[f_idx], "PRODUCTNAME": [r[1] for r in fulfil_rows],
                           "FULFILMENTSUCCESSDATE_TIME": _ts(rng, [r[2] for r in fulfil_rows]), "sub_flag": 1, "nb_sub": 1})
    fulfil = fulfil.groupby(["access_method_val", "PRODUCTNAME", "FULFILMENTSUCCESSDATE_TIME"], as_index=False).agg(
        sub_flag=("sub_flag", "max"), nb_sub=("nb_sub", "sum"))

    # arbitration table (BR-22): a subset of the targeted, start_date in D-7..D
    arb = np.flatnonzero(rng.random(len(targeted)) < 0.35)
    start = [run_date + timedelta(days=int(o)) for o in rng.integers(config.ARBITRATION_WINDOW[0], config.ARBITRATION_WINDOW[1] + 1, len(arb))]
    arb_df = pd.DataFrame({"DATE1": [d.isoformat() for d in start], "START_DATE": [d.isoformat() for d in start],
                           "END_DATE": [(d + timedelta(days=6)).isoformat() for d in start],
                           "MSISDN": c["msisdn"].to_numpy()[targeted[arb]]})
    for i in range(1, 7):
        names = slots[i - 1][arb] if i <= 4 else np.full(len(arb), None, dtype=object)
        arb_df[f"CHANNEL_{i}"] = np.where([x is not None for x in names], "SMS", None)
        arb_df[f"OPPORTUNITY_NAME_{i}"] = [f"OPP_{a.replace(' ', '_')}" if x is not None else None for a, x in zip(actions[arb], names)]
        arb_df[f"DESTINATION_OFFER_NAME_{i}"] = names
        arb_df[f"OFFER_SCRIPT_EN_{i}"] = [scripts[x][0] if x is not None else None for x in names]
        arb_df[f"OFFER_SCRIPT_AR_{i}"] = [scripts[x][1] if x is not None else None for x in names]
    arb_df["FLAG_SPLIO"] = "N"
    arb_df["SEGMENT"] = seg[arb]
    arb_df["FILENAME"] = [f"arbitration_{d.isoformat()}.csv" for d in start]
    arb_df["ETL_CREATE_DT"] = [d.isoformat() for d in start]
    arb_df["JOB_ID"] = 10_000 + (run_date.toordinal() % 10_000)
    arb_df["SEGMENT_BTL_TAKER"] = rng.choice(["Y", "N"], len(arb))
    arb_df["SEGMENT_BTL_TAKER_UPSELL"] = rng.choice(["Y", "N"], len(arb))

    # 270-day purchase history (BR-24) for the ranker: preference-driven BTL purchases + last week's
    usage = c["usage_type"].to_numpy()
    lam = 3.0 * (0.4 + np.clip(c["spend"].to_numpy() / 8, 0, 1.2)) * np.where(usage == "inactive", 0.3, 1.0)
    counts = rng.poisson(lam)
    hist_rows = []
    for lo in range(0, n, config.GENERATOR_CHUNK):
        sel = np.arange(lo, min(lo + config.GENERATOR_CHUNK, n))
        ev = np.repeat(sel, counts[sel])
        if len(ev):
            w = _preference_weights(world, sel, world.catalog)
            picks = _sample_offers(rng, w[ev - lo])
            hist_rows.append(pd.DataFrame({"i": ev, "PRODUCTNAME": world.catalog["btl_name"].to_numpy()[picks]}))
    hist = pd.concat(hist_rows + [pd.DataFrame({"i": f_idx, "PRODUCTNAME": [r[1] for r in fulfil_rows]})], ignore_index=True)
    hist["access_method_val"] = c["msisdn"].to_numpy()[hist["i"].to_numpy()]
    history = hist.groupby(["access_method_val", "PRODUCTNAME"], as_index=False).size().rename(columns={"size": "nb_sub"})
    history["sub_flag"] = 1
    return {"targeting_dataiku_auto": targeting, "targeting_auto": arb_df, "subs_btl_auto": fulfil,
            "sub_btl_3months": history[["access_method_val", "PRODUCTNAME", "sub_flag", "nb_sub"]]}


# ---------------------------------------------------------------------------
# Test sample (BR-59..62): control + 3 arms, stratified
# ---------------------------------------------------------------------------
def _bucket(x: np.ndarray, edges: list[float], labels: list[str]) -> np.ndarray:
    return np.array(labels, dtype=object)[np.searchsorted(edges, x, side="left")]


def _arm_pattern(length: int, offset: int = 0, arms: list[str] | None = None) -> list[str]:
    """Interleaved arm sequence respecting ARM_SHARES over the chosen arms (stride scheduling). A random
    offset per stratum keeps small strata from all starting with the same arm."""
    shares = {a: s for a, s in config.ARM_SHARES.items() if a in (arms or config.ARMS)}
    total = sum(shares.values())
    marks = []
    for arm, share in ((a, s / total) for a, s in shares.items()):
        k = max(1, round(share * 100))
        marks += [((i + 0.5) / k, arm) for i in range(k)]
    seq = [a for _, a in sorted(marks)]
    return [seq[(offset + i) % len(seq)] for i in range(length)]


def _sample_table(world: World, usage: dict[str, pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    c = world.c
    eligible = c[~c["plan"].isin(config.EXCLUDED_PLANS)]
    k = int(len(eligible) * config.TARGET_SAMPLE_SHARE)
    s = eligible.sample(n=k, random_state=int(rng.integers(1 << 30))).copy()
    sub = usage["subscriptions_tg"].set_index("ACCESS_METHOD_VAL")
    use = usage["usage_data_voice_payg_tg"].set_index("ACCESS_METHOD_VAL")

    def col(frame, name):
        return frame[name].reindex(s["msisdn"]).fillna(0.0).to_numpy()

    sub_m1, payg_m1 = col(sub, "Total_Sub_Amt_M1"), col(use, "PG_Total_Amount_m1")
    total = sub_m1 + payg_m1
    total_3m = sum(col(sub, f"Total_Sub_Amt_M{i}") + col(use, f"PG_Total_Amount_m{i}") for i in (1, 2, 3))
    voice, data = col(use, "local_Voice_min_p1"), col(use, "data_volume_mb_p1")
    nat_group = np.where(s["nationality"].isin(config.NATIONALITY_KEEP), s["nationality"], config.NATIONALITY_OTHERS)
    rev_band = _bucket(total_3m, [0, 5, 15, 40], ["0", "0-5", "5-15", "15-40", "40+"])
    voice_g = _bucket(voice, [0, 100, 500], ["0", "1-100", "100-500", "500+"])
    data_g = _bucket(data, [0, 1000, 10000], ["0", "1-1000", "1000-10000", "10000+"])
    strata = np.array(["|".join(map(str, t)) for t in zip(nat_group, rev_band, voice_g, data_g, s["plan"])], dtype=object)

    group = np.empty(len(s), dtype=object)
    counts = np.empty(len(s), dtype=object)
    for st in pd.unique(strata):
        pos = np.flatnonzero(strata == st)
        rng.shuffle(pos)
        group[pos] = _arm_pattern(len(pos), int(rng.integers(0, 100)), world.params.get("arms"))
        counts[pos] = str(len(pos))

    def f3(x):
        return [f"{v:.3f}" for v in x]

    return pd.DataFrame({
        "access_method_val": s["msisdn"].to_numpy(), "access_method_id": s["access_method_id"].astype(str).to_numpy(),
        "account_id": s["account_id"].astype(str).to_numpy(), "tenure": s["tenure"].astype(str).to_numpy(),
        "activation_date": [(world.as_of - timedelta(days=int(t))).isoformat() for t in s["tenure"]],
        "plan_name": s["plan"].to_numpy(), "payg_revenue": f3(payg_m1), "subscription_revenue": f3(sub_m1),
        "total_revenue": f3(total), "total_revenue_3m": f3(total_3m), "voice_traffic": f3(voice), "data_volume": f3(data),
        "nationality": s["nationality"].to_numpy(), "region": s["region"].to_numpy(), "ARPU": f3(total),
        "ARPU_3m": f3(total_3m / 3), "voice_trf_group": voice_g, "data_trf_group": data_g,
        "Final": np.where(group == "cg", "CG", "TG"), "channel": "SMS", "group": group, "strata": strata, "counts": counts,
    })


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _consent_table(world: World, run_date: date) -> pd.DataFrame:
    """V1.0: ~3 % do-not-contact, ~2 % marketing opt-out, fixed per customer. Own random stream, so every other
    table is identical to the one generated without consent data (same world for V0 / V1.0 comparisons)."""
    rng = np.random.default_rng(np.random.SeedSequence([world.seed, 9]))
    n = world.n
    return pd.DataFrame({"access_method_val": world.c["msisdn"].to_numpy(), "dnd": (rng.random(n) < 0.03).astype(int),
                         "marketing_opt_in": (rng.random(n) >= 0.02).astype(int),
                         "updated_at": [(run_date - timedelta(days=int(d))).isoformat() for d in rng.integers(1, 400, n)]})


def generate_run(world: World, run_date: date, campaign: pd.DataFrame | None = None) -> GeneratedRun:
    """Builds every table of the contract for run date D from the world. campaign: the app's logged campaign
    of D-7 (closed-loop simulation); its responses become this run's fulfilments."""
    t0 = time.time()
    rng = np.random.default_rng(np.random.SeedSequence([world.seed, 5, run_date.toordinal()]))
    typed: dict[str, pd.DataFrame] = {}
    typed.update(_base_tables(world, run_date, rng))
    usage = _usage_tables(world, run_date)
    typed.update(usage)
    for k, m in enumerate(config.MODEL_SCORE_REGISTRY):
        typed[m.table] = generate_model_score(world, m, np.random.default_rng(np.random.SeedSequence([world.seed, 3, k])), run_date)
    typed.update(_campaign_tables(world, run_date, _segments(world, rng), rng, campaign))
    typed[config.TARGET_SAMPLE_TABLE] = _sample_table(world, usage, rng)
    typed[config.CONSENT_TABLE] = _consent_table(world, run_date)

    missing = set(schemas.TABLES) - set(typed)
    if missing:
        raise RuntimeError(f"generator did not build: {sorted(missing)}")
    tables = {name: to_raw(typed[name], name) for name in schemas.TABLES}
    issues = scen.inject_quality_issues(tables, world.seed) if world.has("data_quality_issues") else []

    manifest = {
        "run_date": run_date.isoformat(),
        "anchor_date": cal.anchor_date(run_date).isoformat(),
        "source": "generator",
        "operator": config.OPERATOR_NAME,
        "schema_version": config.SCHEMA_VERSION,
        "generator": {"n": world.n, "seed": world.seed, "scenarios": world.scenarios, "world_as_of": world.as_of.isoformat(),
                      "catalog_size": len(world.catalog), "new_offers": int(world.catalog["is_new"].sum())},
        "row_counts": {name: len(df) for name, df in tables.items()},
        "quality_issues_injected": issues,
        "windows": cal.all_windows(run_date),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generation_seconds": round(time.time() - t0, 2),
    }
    return GeneratedRun(run_date, cal.anchor_date(run_date), tables, manifest, world)


def write_run(run: GeneratedRun, out_dir: Path, fmt: str = "json") -> Path:
    """One file per table (<table>.json array, or .jsonl one record per line) + _manifest.json."""
    if fmt not in ("json", "jsonl"):
        raise ValueError("fmt must be 'json' or 'jsonl'")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in run.tables.items():
        records = df.to_dict("records")
        path = out_dir / f"{name}.{fmt}"
        with path.open("w", encoding="utf-8") as fh:
            if fmt == "json":
                json.dump(records, fh, ensure_ascii=False, allow_nan=False)
            else:
                for r in records:
                    fh.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    manifest = {**run.manifest, "format": fmt}
    (out_dir / config.MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def generate(n: int = config.GENERATOR_DEFAULT_N, run_date: date | None = None, scenarios=None,
             seed: int = config.GENERATOR_SEED, out_dir: Path | None = None, fmt: str = "json") -> GeneratedRun:
    """Convenience entry point (CLI / API): new world as of D, generate, optionally write."""
    run_date = run_date or date.today()
    world = World.create(n, seed=seed, scenarios=scen.parse(scenarios), as_of=run_date)
    run = generate_run(world, run_date)
    if out_dir is not None:
        write_run(run, out_dir, fmt)
    return run


def generate_weeks(n: int, first_date: date, weeks: int, scenarios=None, seed: int = config.GENERATOR_SEED,
                   out_root: Path | None = None, fmt: str = "json", arms: list[str] | None = None) -> list[GeneratedRun]:
    """Consecutive weekly runs D, D+7, ... from ONE world anchored on the last run date, so the shared days
    agree (W2 of D+7 = W1 of D) and scenario drifts grow week after week. Writes <out_root>/<date>/."""
    if weeks < 1:
        raise ValueError("weeks must be >= 1")
    dates = [first_date + timedelta(days=7 * k) for k in range(weeks)]
    world = World.create(n, seed=seed, scenarios=scen.parse(scenarios), as_of=dates[-1], arms=arms)
    runs = [generate_run(world, d) for d in dates]
    if out_root is not None:
        for run in runs:
            write_run(run, Path(out_root) / run.run_date.isoformat(), fmt)
    return runs
