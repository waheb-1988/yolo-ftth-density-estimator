"""Generates realistic synthetic Ooredoo Oman customer/offer/purchase data.

Three MSISDNs are pinned to specific behavioural signatures so that the
needle-in-haystack benchmark tests in scripts/benchmark_llms.py are
deterministic:

    msisdn=1  -> REACTIVATION  (98 days inactive)
    msisdn=15 -> INTERNATIONAL (heavy international caller)
    msisdn=25 -> RENEWAL       (purchased 5 days ago, bundle about to expire)

Everything else is randomized with a fixed seed for reproducibility.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta

from nbx.core import config

random.seed(42)

DV_SEGMENTS = [
    "new device old version",
    "old version 2nd hand",
    "high value smart",
    "new smart",
    "basic feature",
]
USAGE_TYPES = ["DATA_DOMINANT", "VOICE_DOMINANT", "BALANCED"]
DATA_TIERS = ["MINIMAL_DATA", "LOW_DATA", "MEDIUM_DATA", "HIGH_DATA", "HEAVY_DATA"]
ACTIVITY_LEVELS = ["LOW_ACTIVITY", "MEDIUM_ACTIVITY", "HIGH_ACTIVITY"]
CHANNELS = ["APP", "USSD", "RETAIL_STORE", "WEBSITE", "CALL_CENTER"]
NATIONALITIES = ["OMN"] * 50 + ["IND"] * 20 + ["PAK"] * 15 + ["BGD"] * 10 + ["EGY"] * 5
REVENUE_SEGMENTS = ["Low Value", "Mid Value", "High Value", "Premium"]
NWK_TYPES = ["3G", "4G", "5G"]
CATEGORIES_FAVORITE = ["NBO", "Data", "Voice", "Bundle", "International", "Roaming"]


def _rfm_score() -> int:
    # RFM digits are each 1-9 (Recency, Frequency, Monetary), e.g. 111-999
    return int(f"{random.randint(1,9)}{random.randint(1,9)}{random.randint(1,9)}")


def _weighted_usage_type() -> str:
    return random.choices(USAGE_TYPES, weights=[0.4, 0.3, 0.3])[0]


def _build_customer(msisdn: int) -> dict:
    usage_type = _weighted_usage_type()
    is_intl = random.random() < 0.20
    is_roamer = random.random() < 0.10
    arpu = round(random.uniform(0.5, 25.0), 2)
    tenure_days = random.randint(30, 2500)
    days_since_purchase = random.randint(0, 120)
    return {
        "msisdn": msisdn,
        "tenure_days": tenure_days,
        "arpu": arpu,
        "dv_segment": random.choice(DV_SEGMENTS),
        "revenue_segment": random.choice(REVENUE_SEGMENTS),
        "rfm_score": _rfm_score(),
        "usage_type": usage_type,
        "data_usage_tier": random.choice(DATA_TIERS),
        "activity_level": random.choice(ACTIVITY_LEVELS),
        "is_intl_caller": is_intl,
        "is_roamer": is_roamer,
        "days_since_last_purchase": days_since_purchase,
        "days_since_last_network_activity": random.randint(0, 100),
        "avg_offer_price": round(random.uniform(1.0, 15.0), 2),
        "preferred_recharge_channel": random.choice(CHANNELS),
        "data_volume_mb_p1": round(random.uniform(500, 40000), 0),
        "voice_traffic_min_p1": round(random.uniform(10, 400), 0),
        "active_days_count_p1": random.randint(1, 30),
        "favorite_category": random.choice(CATEGORIES_FAVORITE),
        "last_product_name": f"PS_{random.randint(1,9)}D_{random.randint(1,9)}R_UNLIMITED_OT",
        "nationality": random.choice(NATIONALITIES),
        "nwk_type": random.choice(NWK_TYPES),
    }


def generate_customers(n: int = 50) -> list[dict]:
    customers = [_build_customer(m) for m in range(1, n + 1)]

    # --- Pinned test signatures -------------------------------------------------
    customers[0].update(
        {
            "msisdn": 1,
            "tenure_days": 414,
            "arpu": 3.50,
            "dv_segment": "new device old version",
            "revenue_segment": "Low Value",
            "rfm_score": 999,
            "usage_type": "BALANCED",
            "data_usage_tier": "MINIMAL_DATA",
            "activity_level": "LOW_ACTIVITY",
            "is_intl_caller": False,
            "is_roamer": False,
            "days_since_last_purchase": 98,
            "days_since_last_network_activity": 98,
            "avg_offer_price": 5.0,
            "preferred_recharge_channel": "APP",
            "data_volume_mb_p1": 22450,
            "voice_traffic_min_p1": 180,
            "active_days_count_p1": 12,
            "favorite_category": "NBO",
            "last_product_name": "PS_8D_6R_UNLIMITED_OT",
            "nationality": "OMN",
            "nwk_type": "4G",
        }
    )

    if n >= 15:
        customers[14].update(
            {
                "msisdn": 15,
                "dv_segment": "high value smart",
                "revenue_segment": "High Value",
                "usage_type": "VOICE_DOMINANT",
                "is_intl_caller": True,
                "is_roamer": False,
                "days_since_last_purchase": 12,
                "days_since_last_network_activity": 2,
                "voice_traffic_min_p1": 180,
                "favorite_category": "International",
                "nationality": "IND",
            }
        )

    if n >= 25:
        customers[24].update(
            {
                "msisdn": 25,
                "dv_segment": "new smart",
                "revenue_segment": "Mid Value",
                "usage_type": "BALANCED",
                "is_intl_caller": False,
                "is_roamer": False,
                "days_since_last_purchase": 5,
                "days_since_last_network_activity": 1,
                "favorite_category": "NBO",
                "last_product_name": "PS_7D_5R_UNLIMITED_OT",
            }
        )

    return customers


# ---------------------------------------------------------------------------
# Offers
# ---------------------------------------------------------------------------
def generate_offers() -> list[dict]:
    offers: list[dict] = []

    def add(program_rule, category, offer_type, price, data_gb, voice_mins,
             validity, is_data, is_voice, is_intl, is_roaming, discount, desc):
        offers.append(
            {
                "program_rule": program_rule,
                "offer_name": program_rule,
                "category": category,
                "offer_type": offer_type,
                "price_omr": price,
                "data_gb": data_gb,
                "voice_mins": voice_mins,
                "validity_days": validity,
                "is_data": is_data,
                "is_voice": is_voice,
                "is_intl": is_intl,
                "is_roaming": is_roaming,
                "discount_pct": discount,
                "description": desc,
            }
        )

    # NBO (8)
    add("PS_8D_6R_UNLIMITED_OT", "NBO", "Prepaid Add-on", 6.0, "Unlimited", "Unlimited", 8,
        True, True, False, False, 0,
        "8-day unlimited data and voice bundle at 6 OMR. Best for heavy users needing full connectivity.")
    add("PS_7D_5R_UNLIMITED_OT", "NBO", "Prepaid Add-on", 5.0, "Unlimited", "Unlimited", 7,
        True, True, False, False, 0,
        "7-day unlimited data and voice renewal bundle at 5 OMR. Ideal for customers about to expire their current plan.")
    add("PS_DW_5R_UNLIMITED_OT", "NBO", "Prepaid Add-on", 5.0, "Unlimited", "Unlimited", 7,
        True, True, False, False, 0,
        "Low-cost weekly unlimited bundle designed to re-engage dormant customers with full connectivity.")
    add("PS_3D_2R_DATA_OT", "NBO", "Prepaid Add-on", 2.0, "3GB", "0", 3,
        True, False, False, False, 0,
        "Short 3-day light data top-up for occasional browsing needs.")
    add("PS_15D_10R_MIX_OT", "NBO", "Prepaid Add-on", 10.0, "10GB", "500", 15,
        True, True, False, False, 0,
        "15-day mixed data and voice bundle balancing cost and usage for regular customers.")
    add("PS_30D_15R_SAVER_OT", "NBO", "Prepaid Add-on", 15.0, "20GB", "1000", 30,
        True, True, False, False, 5,
        "Monthly saver bundle with 5% discount, good value for steady balanced usage.")
    add("PS_1D_1R_MICRO_OT", "NBO", "Prepaid Add-on", 1.0, "500MB", "50", 1,
        True, True, False, False, 0,
        "Micro 1-day bundle for low-value, price-sensitive customers needing minimal connectivity.")
    add("PS_10D_8R_SMART_OT", "NBO", "Prepaid Add-on", 8.0, "8GB", "300", 10,
        True, True, False, False, 0,
        "10-day smart bundle tailored for new smartphone users transitioning from feature phones.")

    # Data (8)
    add("DT_1D_1R_1GB", "Data", "Data Add-on", 1.0, "1GB", "0", 1, True, False, False, False, 0,
        "1-day 1GB data top-up for light browsing.")
    add("DT_7D_4R_5GB", "Data", "Data Add-on", 4.0, "5GB", "0", 7, True, False, False, False, 0,
        "7-day 5GB data bundle for weekly streaming and social media use.")
    add("DT_30D_12R_20GB", "Data", "Data Add-on", 12.0, "20GB", "0", 30, True, False, False, False, 0,
        "Monthly 20GB data bundle for heavy data users, best value per GB.")
    add("DT_30D_20R_UNLIMITED", "Data", "Data Add-on", 20.0, "Unlimited", "0", 30, True, False, False, False, 0,
        "Monthly unlimited data bundle for the heaviest streamers and gamers.")
    add("DT_3D_2R_2GB", "Data", "Data Add-on", 2.0, "2GB", "0", 3, True, False, False, False, 0,
        "3-day 2GB data booster for quick top-ups between cycles.")
    add("DT_15D_8R_10GB", "Data", "Data Add-on", 8.0, "10GB", "0", 15, True, False, False, False, 0,
        "15-day 10GB data bundle balancing cost and volume for medium users.")
    add("DT_1D_05R_300MB", "Data", "Data Add-on", 0.5, "300MB", "0", 1, True, False, False, False, 0,
        "Cheapest micro data top-up for minimal, urgent connectivity.")
    add("DT_30D_25R_40GB", "Data", "Data Add-on", 25.0, "40GB", "0", 30, True, False, False, False, 0,
        "Premium monthly 40GB data bundle for high-value data-dominant customers.")

    # Voice (5)
    add("VC_7D_3R_200MIN", "Voice", "Voice Add-on", 3.0, "0", "200", 7, False, True, False, False, 0,
        "7-day 200-minute local voice bundle for voice-dominant customers.")
    add("VC_30D_8R_UNLIMITED", "Voice", "Voice Add-on", 8.0, "0", "Unlimited", 30, False, True, False, False, 0,
        "Monthly unlimited local voice bundle for heavy callers.")
    add("VC_1D_1R_50MIN", "Voice", "Voice Add-on", 1.0, "0", "50", 1, False, True, False, False, 0,
        "1-day 50-minute voice top-up for urgent short-term needs.")
    add("VC_15D_5R_500MIN", "Voice", "Voice Add-on", 5.0, "0", "500", 15, False, True, False, False, 0,
        "15-day 500-minute voice bundle for regular callers.")
    add("VC_30D_6R_1000MIN", "Voice", "Voice Add-on", 6.0, "0", "1000", 30, False, True, False, False, 0,
        "Monthly 1000-minute voice bundle, cost-effective for frequent callers.")

    # Bundle (5)
    add("BN_7D_7R_MIX", "Bundle", "Combo Bundle", 7.0, "6GB", "400", 7, True, True, False, False, 0,
        "7-day combo bundle mixing data and voice for balanced usage customers.")
    add("BN_30D_18R_MIX", "Bundle", "Combo Bundle", 18.0, "25GB", "1500", 30, True, True, False, False, 0,
        "Monthly premium combo bundle for high value balanced customers.")
    add("BN_3D_3R_MIX", "Bundle", "Combo Bundle", 3.0, "2GB", "100", 3, True, True, False, False, 0,
        "3-day light combo bundle for occasional balanced users.")
    add("BN_15D_10R_MIX", "Bundle", "Combo Bundle", 10.0, "12GB", "600", 15, True, True, False, False, 0,
        "15-day mid-tier combo bundle covering both data and voice needs.")
    add("BN_1D_1R_MIX", "Bundle", "Combo Bundle", 1.0, "500MB", "30", 1, True, True, False, False, 0,
        "1-day micro combo bundle for minimal price-sensitive top-ups.")

    # International (2)
    add("IN_7D_10R_INTL300", "International", "International Add-on", 10.0, "0", "300", 7, False, True, True, False, 0,
        "7-day 300-minute international calling bundle for frequent overseas callers.")
    add("IN_30D_20R_INTL_UNLIMITED", "International", "International Add-on", 20.0, "0", "Unlimited", 30,
        False, True, True, False, 0,
        "Monthly unlimited international voice bundle, best for expat customers calling home frequently.")

    # Roaming (2)
    add("RM_7D_15R_ROAM2GB", "Roaming", "Roaming Add-on", 15.0, "2GB", "60", 7, True, True, False, True, 0,
        "7-day roaming bundle with 2GB data and 60 minutes for travel.")
    add("RM_3D_9R_ROAM500MB", "Roaming", "Roaming Add-on", 9.0, "500MB", "30", 3, True, True, False, True, 0,
        "3-day short-trip roaming bundle for light travelers.")

    return offers


# ---------------------------------------------------------------------------
# Purchases
# ---------------------------------------------------------------------------
# Segment -> monthly (30-day), non-voice offer affinity. These are exactly
# the program_rules generate_offers() emits with validity=30 and
# category != "Voice" -- the same filter core/collab_filter.py applies before
# training, and the same universe the production LightFM NBO job's monthly
# upsell logic scores. Weighting a portion of purchases by this table (rather
# than drawing every purchase uniformly) gives the collaborative-filtering
# model learnable per-segment structure instead of pure noise.
SEGMENT_MONTHLY_AFFINITY = {
    "new device old version": {"PS_30D_15R_SAVER_OT": 3, "DT_30D_12R_20GB": 3, "BN_30D_18R_MIX": 1},
    "old version 2nd hand": {"DT_30D_12R_20GB": 3, "PS_30D_15R_SAVER_OT": 2, "BN_30D_18R_MIX": 1},
    "high value smart": {"DT_30D_25R_40GB": 3, "IN_30D_20R_INTL_UNLIMITED": 3, "BN_30D_18R_MIX": 2},
    "new smart": {"BN_30D_18R_MIX": 3, "DT_30D_20R_UNLIMITED": 3, "PS_30D_15R_SAVER_OT": 1},
    "basic feature": {"PS_30D_15R_SAVER_OT": 3, "DT_30D_12R_20GB": 2},
}
MONTHLY_AFFINITY_SHARE = 0.65  # fraction of purchases drawn from the affinity table vs. fully uniform


def _weighted_offer_choice(segment: str, offers: list[dict]) -> dict:
    affinity = SEGMENT_MONTHLY_AFFINITY.get(segment)
    if not affinity or random.random() > MONTHLY_AFFINITY_SHARE:
        return random.choice(offers)
    by_rule = {o["program_rule"]: o for o in offers}
    rules = [r for r in affinity if r in by_rule]
    if not rules:
        return random.choice(offers)
    weights = [affinity[r] for r in rules]
    return by_rule[random.choices(rules, weights=weights)[0]]


def generate_purchases(customers: list[dict], offers: list[dict], n: int = 260) -> list[dict]:
    purchases = []
    now = datetime.now()
    for i in range(n):
        customer = random.choice(customers)
        offer = _weighted_offer_choice(customer["dv_segment"], offers)
        is_affinity_pick = offer["program_rule"] in SEGMENT_MONTHLY_AFFINITY.get(customer["dv_segment"], {})
        # Bias affinity-driven purchases into the last 90 days so
        # core/collab_filter.py's training window (which mirrors
        # sub_btl_90days_catalog) actually has density to learn from; keep
        # the wider 0-180 day spread for everything else (unchanged
        # behavior for the vector-store "purchase pattern" documents).
        days_ago = random.randint(0, 89) if is_affinity_pick else random.randint(0, 180)
        purchase_date = now - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        purchases.append(
            {
                "transaction_id": f"TXN-{10001 + i}",
                "msisdn": customer["msisdn"],
                "purchase_date": purchase_date.isoformat(timespec="seconds"),
                "offer_name": offer["offer_name"],
                "program_rule": offer["program_rule"],
                "category": offer["category"],
                "price_omr": offer["price_omr"],
                "purchase_count": 1,
            }
        )
    return purchases


# ---------------------------------------------------------------------------
# Revenue history -- simulates subscriptions_tg_hdfs + usage_data_voice_payg_tg_hdfs
# closely enough to reproduce calculate_rev() from the production NBO job:
# rev_mX = subscription amount + PAYG amount; avg_rev = mean(rev) + std(rev)
# (a deliberately generous affordability estimate, not a plain average),
# then split 90% data / 10% voice by the same fixed assumption.
# ---------------------------------------------------------------------------
def generate_revenue_history(customers: list[dict]) -> list[dict]:
    history = []
    for c in customers:
        base = c["arpu"]
        rev_m = [round(max(0.0, random.gauss(base, base * 0.25)), 2) for _ in range(3)]
        rev_w = [round(max(0.0, random.gauss(base / 4, base * 0.12)), 2) for _ in range(4)]

        avg_m = sum(rev_m) / len(rev_m)
        std_m = (sum((x - avg_m) ** 2 for x in rev_m) / len(rev_m)) ** 0.5
        avg_rev_monthly = round(avg_m + std_m, 2)

        avg_w = sum(rev_w) / len(rev_w)
        std_w = (sum((x - avg_w) ** 2 for x in rev_w) / len(rev_w)) ** 0.5
        avg_rev_weekly = round(avg_w + std_w, 2)

        history.append(
            {
                "msisdn": c["msisdn"],
                "rev_m1": rev_m[0], "rev_m2": rev_m[1], "rev_m3": rev_m[2],
                "rev_w1": rev_w[0], "rev_w2": rev_w[1], "rev_w3": rev_w[2], "rev_w4": rev_w[3],
                "avg_rev_monthly": avg_rev_monthly,
                "avg_rev_weekly": avg_rev_weekly,
                "avg_rev_data_monthly": round(0.9 * avg_rev_monthly, 2),
                "avg_rev_voice_monthly": round(0.1 * avg_rev_monthly, 2),
                "avg_rev_data_weekly": round(0.9 * avg_rev_weekly, 2),
                "avg_rev_voice_weekly": round(0.1 * avg_rev_weekly, 2),
            }
        )
    return history


# ---------------------------------------------------------------------------
# NBA source data -- simulates the inputs the production Vowpal Wabbit NBA
# script consumes, split into the same 7 file groupings core/config.py's
# NBA_SOURCE_FILE_ATTRS documents (mirroring the production script's own
# per-Dataiku-dataset boundaries, not one merged blob). See core/config.py's
# NBA_* constants for the ported business rules (dict_segm_actions,
# segment_prop_dict, the context feature list) and core/nba_bandit.py's
# _load_scores() for where these 7 files get joined back together --
# including the 15 var_rev/voice/data_* features, which are NOT generated
# here: production computes them (calcul_evol()) after merging, from raw
# usage, so this module only produces the raw usage, not the derived %.
#
# Two things here are genuinely inferred rather than given:
#   - assign_nba_segment(): the upstream SEGMENT field arrives pre-computed
#     in production, from a job not included in the script provided. This
#     derivation follows the naming convention and the scores this function
#     does compute -- flagged as inference, not as given business logic.
#   - The 9 model-score probabilities themselves: production reads them from
#     9 separately-trained scoring pipelines; here they're simulated with a
#     mild, documented correlation to existing customer fields so the bandit
#     has learnable structure instead of pure noise.
# ---------------------------------------------------------------------------
NATIONALITY_GROUP_MAP = {"OMN": "Oman", "IND": "India", "BGD": "Bangladesh", "PAK": "Pakistan"}


def _usage_series(baseline: float, periods: int, zero_prob: float = 0.12) -> list[float]:
    """Noisy multi-period series around a single-period baseline, with a
    real chance of a zero period so core.nba_bandit._pct_change's epsilon
    branch actually gets exercised at merge time, same as it would be in
    production for a customer who went quiet for a week/month."""
    series = []
    for _ in range(periods):
        if baseline <= 0 or random.random() < zero_prob:
            series.append(0.0)
        else:
            series.append(round(max(0.0, random.gauss(baseline, baseline * 0.3)), 2))
    return series


def _nba_probas(customer: dict) -> dict:
    inactive = customer.get("days_since_last_network_activity", 0) >= 60
    activity_low = customer.get("activity_level") == "LOW_ACTIVITY"
    data_tier_low = customer.get("data_usage_tier") in ("MINIMAL_DATA", "LOW_DATA")
    usage_type = customer.get("usage_type")

    def boosted(low_range, high_range, condition):
        return round(random.uniform(*(high_range if condition else low_range)), 4)

    return {
        "proba_churn": boosted((0.0, 0.25), (0.4, 0.85), activity_low or inactive),
        "proba_react_30": boosted((0.0, 0.15), (0.35, 0.8), inactive),
        "proba_react_15": boosted((0.0, 0.12), (0.3, 0.7), inactive),
        "proba_app_data": boosted((0.0, 0.3), (0.4, 0.9), data_tier_low and usage_type != "VOICE_DOMINANT"),
        "proba_app_voice": boosted((0.0, 0.3), (0.4, 0.9), usage_type == "DATA_DOMINANT"),
        "proba_churn_1d": boosted((0.0, 0.1), (0.2, 0.5), activity_low),
        "proba_rev_decr": round(random.uniform(0.0, 0.6), 4),
        "proba_frag_data": boosted((0.0, 0.3), (0.35, 0.75), data_tier_low),
        "proba_frag_voice": boosted((0.0, 0.3), (0.35, 0.75), usage_type == "VOICE_DOMINANT" and activity_low),
    }


def assign_nba_segment(customer: dict, probas: dict) -> str:
    """INFERRED, not given -- see module docstring above. Follows the
    Nursering/Onboarding/Retention/RGS_Not_At_Risk naming convention against
    tenure and the probability scores this module computes."""
    tenure = customer.get("tenure_days", 0)
    if tenure < 45:
        return "Nursering_Early_Churn_At_Risk" if probas["proba_churn"] > 0.5 else "Nursering_Early_Churn_Not_At_Risk"

    if tenure < 180:
        return "Onboarding_Early_Churn_At_Risk" if probas["proba_churn"] > 0.5 else "Onboarding_Early_Churn_Not_At_Risk"

    if customer.get("days_since_last_network_activity", 0) >= 60:
        return "Retention_High_Probability_To_Reconnect" if probas["proba_react_30"] > 0.5 else "Retention_Low_Probability_To_Reconnect"

    if probas["proba_churn"] > 0.5:
        return "Retention_RGS_At_Risk"

    if max(probas["proba_frag_data"], probas["proba_frag_voice"]) > 0.6:
        return "Retention_RGS_Fragile"

    usage_type = customer.get("usage_type")
    if usage_type == "DATA_DOMINANT":
        return "RGS_Not_At_Risk_Data_Only_Voice_Appetency" if probas["proba_app_voice"] > 0.5 else "RGS_Not_At_Risk_Data_Only_No_Voice_Appetency"
    if usage_type == "VOICE_DOMINANT":
        return "RGS_Not_At_Risk_Voice_Only_Data_Appetency" if probas["proba_app_data"] > 0.5 else "RGS_Not_At_Risk_Voice_Only_No_Data_Appetency"
    return "RGS_Not_At_Risk_DualPlay_Upgrade"


def generate_nba_source_data(customers: list[dict], purchases: list[dict]) -> dict[str, list[dict]]:
    """Returns {source_key: rows} for the 7 files core.config.NBA_SOURCE_FILE_ATTRS
    names -- raw per-source data, split the way production's own Dataiku
    datasets are split. Revenue lives in revenue_history.json (already
    generated, shared with NBO); the 15 var_* variation features are NOT
    computed here -- see core/nba_bandit.py::_pct_change, run at merge time."""
    reward_cutoff = datetime.now() - timedelta(days=config.NBA_REWARD_WINDOW_DAYS)
    recent_takers = {
        p["msisdn"] for p in purchases if datetime.fromisoformat(p["purchase_date"]) >= reward_cutoff
    }

    churn, reactivation, appetency, fragility, revenue_decrease = [], [], [], [], []
    activity_usage, targeting = [], []

    for c in customers:
        msisdn = c["msisdn"]
        probas = _nba_probas(c)
        segment = assign_nba_segment(c, probas)
        excluded = segment in config.NBA_EXCLUDED_SEGMENTS
        action = None if excluded else config.NBA_SEGMENT_ACTION_MAP.get(segment)
        reward = 1 if msisdn in recent_takers else 0
        propensity = None
        if action:
            propensity = round(
                config.NBA_ACTION_PROPENSITY[action] + random.uniform(0, config.NBA_PROPENSITY_JITTER_MAX), 6
            )

        churn.append({"msisdn": msisdn, "proba_churn": probas["proba_churn"], "proba_churn_1d": probas["proba_churn_1d"]})
        reactivation.append({"msisdn": msisdn, "proba_react_30": probas["proba_react_30"], "proba_react_15": probas["proba_react_15"]})
        appetency.append({"msisdn": msisdn, "proba_app_data": probas["proba_app_data"], "proba_app_voice": probas["proba_app_voice"]})
        fragility.append({"msisdn": msisdn, "proba_frag_data": probas["proba_frag_data"], "proba_frag_voice": probas["proba_frag_voice"]})
        revenue_decrease.append({"msisdn": msisdn, "proba_rev_decr": probas["proba_rev_decr"]})

        voice_w = _usage_series(c.get("voice_traffic_min_p1", 0), 4)
        voice_m = _usage_series(c.get("voice_traffic_min_p1", 0), 3)
        data_w = _usage_series(c.get("data_volume_mb_p1", 0), 4)
        data_m = _usage_series(c.get("data_volume_mb_p1", 0), 3)
        activity_usage.append(
            {
                "msisdn": msisdn,
                "local_voice_min_w1": voice_w[0], "local_voice_min_w2": voice_w[1],
                "local_voice_min_w3": voice_w[2], "local_voice_min_w4": voice_w[3],
                "local_voice_min_m1": voice_m[0], "local_voice_min_m2": voice_m[1], "local_voice_min_m3": voice_m[2],
                "data_volume_mb_w1": data_w[0], "data_volume_mb_w2": data_w[1],
                "data_volume_mb_w3": data_w[2], "data_volume_mb_w4": data_w[3],
                "data_volume_mb_m1": data_m[0], "data_volume_mb_m2": data_m[1], "data_volume_mb_m3": data_m[2],
            }
        )

        targeting.append(
            {
                "msisdn": msisdn,
                "nationality_group": NATIONALITY_GROUP_MAP.get(c.get("nationality"), "others"),
                "nba_segment": segment,
                "excluded": excluded,
                "action": action,
                "reward": reward,
                "cost": -1 if reward == 1 else 0,
                "propensity": propensity,
            }
        )

    return {
        "churn": churn,
        "reactivation": reactivation,
        "appetency": appetency,
        "fragility": fragility,
        "revenue_decrease": revenue_decrease,
        "activity_usage": activity_usage,
        "targeting": targeting,
    }


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------
def generate_segments() -> list[dict]:
    return [
        {
            "segment": "new device old version",
            "typical_behavior": "Customers with a new device but an older software version. Usually price-sensitive, "
                                 "moderate data usage, respond well to unlimited low-cost bundles and reactivation offers.",
        },
        {
            "segment": "old version 2nd hand",
            "typical_behavior": "Second-hand device owners on older OS versions. Low ARPU, minimal data usage, "
                                 "high churn risk, best targeted with micro bundles and voice top-ups.",
        },
        {
            "segment": "high value smart",
            "typical_behavior": "High-value smartphone customers with strong ARPU. Heavy data and international usage, "
                                 "respond well to premium bundles, international, and roaming offers.",
        },
        {
            "segment": "new smart",
            "typical_behavior": "Recently acquired smartphone customers. Growing data usage, exploring app-based channels, "
                                 "good candidates for renewal and mid-tier combo bundles.",
        },
        {
            "segment": "basic feature",
            "typical_behavior": "Feature phone customers with minimal data usage. Voice-dominant, price-sensitive, "
                                 "best targeted with voice bundles and micro top-ups.",
        },
    ]


# ---------------------------------------------------------------------------
# Need detection (8-signal logic)
# ---------------------------------------------------------------------------
def detect_need(customer: dict) -> str:
    if customer.get("is_roamer"):
        return "ROAMING"
    if customer.get("is_intl_caller") and customer.get("voice_traffic_min_p1", 0) >= 100:
        return "INTERNATIONAL"
    if customer.get("days_since_last_network_activity", 0) >= 60:
        return "REACTIVATION"
    if 0 <= customer.get("days_since_last_purchase", 999) <= 7:
        return "RENEWAL"
    if customer.get("usage_type") == "DATA_DOMINANT" or customer.get("data_usage_tier") in ("HIGH_DATA", "HEAVY_DATA"):
        return "DATA_UPSELL"
    if customer.get("usage_type") == "VOICE_DOMINANT" or customer.get("voice_traffic_min_p1", 0) >= 250:
        return "VOICE_UPSELL"
    if customer.get("preferred_recharge_channel") != "APP" and customer.get("revenue_segment") in ("High Value", "Premium"):
        return "CHANNEL_MIGRATION"
    return "GENERAL_NBO"


def save_all(data_dir=None) -> dict:
    data_dir = data_dir or config.DATA_DIR
    customers = generate_customers(50)
    offers = generate_offers()
    purchases = generate_purchases(customers, offers, 260)
    segments = generate_segments()
    revenue_history = generate_revenue_history(customers)
    nba_source = generate_nba_source_data(customers, purchases)

    (data_dir / "customers.json").write_text(json.dumps(customers, indent=2))
    (data_dir / "offers.json").write_text(json.dumps(offers, indent=2))
    (data_dir / "purchases.json").write_text(json.dumps(purchases, indent=2))
    (data_dir / "segments.json").write_text(json.dumps(segments, indent=2))
    (data_dir / "revenue_history.json").write_text(json.dumps(revenue_history, indent=2))
    for key, attr in config.NBA_SOURCE_FILE_ATTRS.items():
        path = getattr(config, attr)
        (data_dir / path.name).write_text(json.dumps(nba_source[key], indent=2))

    return {
        "customers": len(customers),
        "offers": len(offers),
        "purchases": len(purchases),
        "segments": len(segments),
        "revenue_history": len(revenue_history),
        **{f"nba_{key}": len(rows) for key, rows in nba_source.items()},
    }


if __name__ == "__main__":
    stats = save_all()
    print(f"Generated: {stats}")
