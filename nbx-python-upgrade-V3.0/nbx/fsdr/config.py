"""FSDR configuration: every business rule constant (BR-xx), the model-score
registry, arms, paths and generator defaults.

Business rules come from docs/fsdr_reference/FSDR_NBA_business_rules.md;
each constant names its BR number. Logic modules import from here and never
hard-code these values.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from nbx.core import config as nbx_config

OPERATOR_NAME = os.getenv("OPERATOR_NAME", "FSDR")
CURRENCY = os.getenv("FSDR_CURRENCY", "OMR")
SCHEMA_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FSDR_DATA_DIR = Path(os.getenv("FSDR_DATA_DIR", str(nbx_config.DATA_DIR / "fsdr")))
FSDR_STATE_DIR = FSDR_DATA_DIR / "state"          # registry, campaign log, rewards
CAMPAIGN_DIR = "campaign"                         # <data root>/state/campaign: nba_campaign_log.jsonl, nba_rewards.jsonl
JSON_SCHEMA_DIR = nbx_config.BASE_DIR / "docs" / "fsdr_json_schemas"
MANIFEST_FILE = "_manifest.json"
VALIDATION_JSON = "_validation.json"
VALIDATION_MD = "_validation.md"

# V2.0 data layout (plan §13-§15)
FSDR_INPUT_DIR = Path(os.getenv("FSDR_INPUT_DIR", str(nbx_config.DATA_DIR / "input")))    # <D>/<folder>/<table>.*
FSDR_CACHE_DIR = Path(os.getenv("FSDR_CACHE_DIR", str(nbx_config.DATA_DIR / "cache")))    # <D>/<table>/part-NN.parquet
FSDR_RUNS_DIR = Path(os.getenv("FSDR_RUNS_DIR", str(nbx_config.DATA_DIR / "runs")))       # <D>/{train,score,final}/<step>/
INPUT_FOLDERS = ("usage", "models", "subscription", "catalog", "others")
INPUT_FORMATS = (".csv.gz", ".json", ".jsonl", ".csv", ".parquet")   # V3.0: .csv.gz = production (master §2)
PRODUCTION_FORMAT = ".csv.gz"
# master §2: plain .csv / .json / .jsonl / .parquet only for tests and the dummy generator (its manifest says so)
ALLOW_TEST_FORMATS = os.getenv("FSDR_ALLOW_TEST_FORMATS", "false").lower() == "true"
CSV_GZIP_LEVEL = 6                                # FSDR CSV standard
ID_TEXT_COLUMNS = {"ACCESS_METHOD_VAL", "MSISDN", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID", "SUBSCRIPTION_ID",
                   "UBSCRIPTION_ID"}              # always read and written as text
PUSH_COMPRESSION = os.getenv("FSDR_PUSH_COMPRESSION", "gzip")    # "gzip" | "none" (also writes input_flytxt.csv)
CACHE_PARTITIONS = int(os.getenv("FSDR_CACHE_PARTITIONS", "16"))     # plan §15: hash of ACCESS_METHOD_ID
TABLE_FOLDERS = {                                 # plan §13; the 11 model-score tables go to "models"
    "usage": ["subs_amt_auto", "subs_amt_score", "activities_w1_auto", "activities_w2_auto", "activities_w3_auto",
              "activities_w4_auto", "activities_m1_auto", "activities_auto_m2", "activities_m3_auto",
              "activities_w1_score", "activities_w2_score", "activities_w3_score", "activities_w4_score",
              "activities_m1_score", "activities_m2_score", "activities_m3_score", "paygo_activity_auto",
              "usage_data_voice_payg_tg"],
    "subscription": ["base_prepaid", "consumer_aact_account", "access_method_id_auto", "subscriptions_tg",
                     "subs_btl_auto", "sub_btl_3months"],
    "catalog": ["btl_catalog", "BTL_script_english_arabic_ok", "conflicting_offers_h"],
    "others": ["party_auto", "party_scoring", "customer_profile", "location_auto", "consent_contact",
               "DEVICEBEHAVIORSEGMENTATION.dv_segment", "targeting_dataiku_auto", "targeting_auto"],
}                                                 # any other table (e.g. the test sample) -> "others"
STEP_FILE = "_step.json"
STEP_PREVIEW_ROWS = 1000                          # training outputs (1M rows): Parquet + the first rows as preview.csv
STEP_FULL_CSV_MAX_ROWS = 100_000                  # scoring outputs (<= 50k rows): full CSV next to the Parquet

# customer_profile (master prompt §2.1, §5): age and tenure of every line, provided weekly
PROFILE_TABLE = "customer_profile"
PROFILE_TENURE_UNIT = os.getenv("FSDR_PROFILE_TENURE_UNIT", "days")          # open item 5
PROFILE_AGE_RANGE = (15, 100)                     # outside -> warning
PROFILE_AGE_TOLERANCE_YEARS = 2                   # vs party_*.AGE
PROFILE_TENURE_TOLERANCE_DAYS = 30                # vs run date - access_method_subs_start_dt
PROFILE_MIN_COVERAGE = 0.95                       # share of the base with a profile row (below -> warning)

# ---------------------------------------------------------------------------
# Calendar (BR notation: D = run date, T = campaign anchor)
# ---------------------------------------------------------------------------
REFERENCE_THURSDAY = date(2026, 3, 12)            # T = D - (((D - 2026-03-12) MOD 7) + 7) MOD 7
SNAPSHOT_LAG_DAYS = 2                             # BR-01 ACTIVITY_DATE = D-2
HISTORY_DAYS_USAGE = 97                           # daily history D-96..D: deepest window is paygo M3 = T-90, T >= D-6
HISTORY_DAYS_FULFILMENT = 270                     # BR-24 interactions D-270..D-1

# ---------------------------------------------------------------------------
# Eligibility (BR-01 .. BR-12)
# ---------------------------------------------------------------------------
MSISDN_MIN_EXCLUSIVE = "69999999"                 # BR-02 string comparison
# BR-12 (SUBSTR(MSISDN, 4)) is applied upstream in the SQL exports: every MSISDN in the JSON files,
# including access_method_id_auto.MSISDN, is already the 8-digit ACCESS_METHOD_VAL. No prefix handling here.
EXCLUDED_PLANS = [                                # BR-10 internal / employee / business plans (13)
    "FSDR Internal",
    "Employee Data with Voice",
    "Shahry Business",
    "FSDR Shahry",
    "Employee superfast 100 Mbps plan",
    "Employee superfast 200 Mbps plan",
    "Employee superfast 400 Mbps plan",
    "5G Employee Plan 100 Mbps",
    "5G Employee Plan 50 Mbps",
    "Employee Fast HBB 150 GB Plan",
    "Employee Fast HBB 300 GB Plan",
    "Employee Fast HBB 600 GB Plan",
    "Fast  Home Broadband 150GB (Employee)",      # double space as in production
]

# ---------------------------------------------------------------------------
# Demographics (BR-13 .. BR-15)
# ---------------------------------------------------------------------------
NATIONALITY_KEEP = ["Oman", "India", "Bangladesh", "Pakistan"]   # BR-15; the rest -> "others"
NATIONALITY_OTHERS = "others"

# ---------------------------------------------------------------------------
# Segments and actions (BR-36 .. BR-45)
# ---------------------------------------------------------------------------
# BR-38. Fixed order: the bandit's probability vector is returned in this order. Never shuffle it.
ACTIONS = ["CROSS SELL DATA", "UPSELL DATA", "RETAIN", "MAINTAIN", "UPSELL", "CROSS SELL VOICE", "UPSELL VOICE"]
NO_CONTACT = "NO_CONTACT"                         # llm-arm only, never a VW action (plan Q10)

EXCLUDED_SEGMENTS = {"nursering_early_churn_at_risk", "nursering_early_churn_not_at_risk"}   # BR-36

# BR-39, lower-case keys so "RGS_..." and "Rgs_..." resolve to the same entry.
SEGMENT_TO_ACTION = {
    "retention_low_probability_to_reconnect": "RETAIN",
    "retention_med_probability_to_reconnect": "RETAIN",
    "retention_high_probability_to_reconnect": "RETAIN",                 # CHECK (plan Q1): was UPSELL then RETAIN
    "retention_rgs_at_risk": "RETAIN",
    "retention_rgs_fragile": "UPSELL",
    "rgs_not_at_risk_dualplay_upgrade": "MAINTAIN",
    "rgs_not_at_risk_data_only_voice_appetency": "UPSELL VOICE",          # CHECK (plan Q1b): or CROSS SELL VOICE
    "rgs_not_at_risk_data_only_no_voice_appetency": "CROSS SELL VOICE",
    "rgs_not_at_risk_voice_only_data_appetency": "UPSELL DATA",
    "rgs_not_at_risk_voice_only_no_data_appetency": "CROSS SELL DATA",
    "onboarding_early_churn_at_risk": "UPSELL",
    "onboarding_early_churn_not_at_risk": "UPSELL",
}

# Canonical segment spellings, and the mixed-case variants production also emits.
SEGMENTS = [
    "Retention_Low_Probability_To_Reconnect", "Retention_Med_Probability_To_Reconnect",
    "Retention_High_Probability_To_Reconnect", "Retention_RGS_At_Risk", "Retention_RGS_Fragile",
    "RGS_Not_At_Risk_DualPlay_Upgrade", "RGS_Not_At_Risk_Data_Only_Voice_Appetency",
    "RGS_Not_At_Risk_Data_Only_No_Voice_Appetency", "RGS_Not_At_Risk_Voice_Only_Data_Appetency",
    "RGS_Not_At_Risk_Voice_Only_No_Data_Appetency", "Onboarding_Early_Churn_At_Risk",
    "Onboarding_Early_Churn_Not_At_Risk",
    "Nursering_Early_Churn_At_Risk", "Nursering_Early_Churn_Not_At_Risk",
]
NURSERING_MAX_TENURE_DAYS = 45                    # BR-36
ONBOARDING_MAX_TENURE_DAYS = 90

ACTION_PRIOR = {                                  # BR-45 share of historical targeting volume
    "RETAIN": 0.5086577,
    "UPSELL": 0.2624428,
    "MAINTAIN": 0.1015183,
    "CROSS SELL VOICE": 0.0528180,
    "UPSELL DATA": 0.0422645,
    "UPSELL VOICE": 0.0239859,
    "CROSS SELL DATA": 0.0083128,
}
MISSING_SCORE_RANGE = (0.0, 0.02)                 # BR-47, drawn per row

# ---------------------------------------------------------------------------
# Model-score registry (BR-46). Adding / removing a model = editing this list.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelScore:
    table: str                      # exact file / table name
    feature: str                    # feature name in the NBA context
    combine: str = "max"            # how rows mapping to the same feature are merged
    signal: str = ""                # hidden-world signal the generator drives it with
    tenure_range: tuple[int, int] | None = None   # generator: only lines in this tenure band
    coverage: tuple[float, float] = (0.70, 0.90)  # generator: share of the base scored


MODEL_SCORE_REGISTRY: list[ModelScore] = [
    ModelScore("DAIM_CHURN_GLOBAL_SCORING_TD.prediction", "PROBA_CHURN", "max", "churn"),
    ModelScore("DAIM_EARLY_CHURN_0_30_SCORING_TD.Early_Churn_0_30_Prediction", "PROBA_CHURN", "max", "churn",
               tenure_range=(0, 30), coverage=(0.90, 0.98)),
    ModelScore("DAIM_EARLY_CHURN_30_60_SCORING_TD.scoring_data_scored", "PROBA_CHURN", "max", "churn",
               tenure_range=(30, 60), coverage=(0.90, 0.98)),
    ModelScore("DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_TD.Scored", "PROBA_REACT_30", "max", "reactivation_30"),
    ModelScore("PROPENSITY_REACTIVATION_15D_401_SCORING.Scored", "PROBA_REACT_15", "max", "reactivation_15"),
    ModelScore("APPETENCY_DATA_204_SCORING_TD.prediction", "PROBA_APP_DATA", "max", "data_appetite"),
    ModelScore("APPETENCY_VOICE_204_SCORING_TD.Scored", "PROBA_APP_VOICE", "max", "voice_appetite"),
    ModelScore("CHURN_ONE_DAY_ACTIVE_107_SCORING.Churn_One_Day_Scoring", "PROBA_CHURN_1D", "max", "churn_1d"),
    ModelScore("REV_DECREASE_204_SCORING_TD.predictions", "PROBA_REV_DECR", "max", "revenue_decrease"),
    ModelScore("DAIM_FRAGILITY_DATA_SCORING_TD.prediction", "PROBA_FRAG_DATA", "max", "fragility_data"),
    ModelScore("DAIM_FRAGILITY_VOICE_SCORING_TD.prediction", "PROBA_FRAG_VOICE", "max", "fragility_voice"),
]


def proba_features(registry: list[ModelScore] | None = None) -> list[str]:
    """Distinct feature names, in registry order (PROBA_CHURN first)."""
    out: list[str] = []
    for m in registry or MODEL_SCORE_REGISTRY:
        if m.feature not in out:
            out.append(m.feature)
    return out


# ---------------------------------------------------------------------------
# Experiment arms (BR-59 .. BR-62)
# ---------------------------------------------------------------------------
ARMS = ["cg", "free", "cstr", "llm"]              # cg = control (no contact); llm = new arm
ARM_SHARES = {"cg": 0.25, "free": 0.25, "cstr": 0.25, "llm": 0.25}
TARGET_SAMPLE_TABLE = os.getenv("FSDR_TARGET_SAMPLE_TABLE", "sample_cg_tg_new_7oct2026")   # plan Q16
TARGET_SAMPLE_SHARE = 0.40                        # generator: share of the base in the test sample
STRATA_FIELDS = ["nationality_group", "revenue_band", "voice_group", "data_group", "plan"]   # BR-62

# ---------------------------------------------------------------------------
# Offers, ARPU anchor and price bands (BR-63 .. BR-82); used from Phase 4
# ---------------------------------------------------------------------------
SERVICES = ["DATA", "VOICE", "COMBO"]
GENEROSITY_LEVELS = ["L0", "L1", "L2"]            # BR-65 L0 < L1 < L2
VALIDITIES = [7, 30]                              # BR-64
GENEROSITY_MAPPING = os.getenv("FSDR_GENEROSITY_MAPPING", "all_L2")   # BR-66 "all_L2" (active) | "normal" (plan Q3)
ACTION_FAMILIES = {                               # BR-66: (service A, service B) per action
    "MAINTAIN": ("COMBO", "DATA"),
    "UPSELL": ("COMBO", "DATA"),
    "RETAIN": ("COMBO", "DATA"),
    "UPSELL VOICE": ("DATA", "COMBO"),
    "CROSS SELL VOICE": ("DATA", "COMBO"),
    "CROSS SELL DATA": ("VOICE", "COMBO"),
    "UPSELL DATA": ("VOICE", "COMBO"),
}
ACTION_GENEROSITY_NORMAL = {                      # BR-66 "normal" (commented out in production): (gen A, gen B)
    "MAINTAIN": ("L0", "L0"),
    "UPSELL": ("L1", "L1"),
    "RETAIN": ("L2", "L2"),
    "UPSELL VOICE": ("L0", "L1"),
    "CROSS SELL VOICE": ("L0", "L2"),
    "CROSS SELL DATA": ("L0", "L2"),
    "UPSELL DATA": ("L0", "L2"),
}
FREE_ARM_GENEROSITY = "L2"                        # BR-67
FREE_ARM_SERVICES = ("COMBO", "DATA")             # BR-67 VOICE excluded
ARPU_DATA_SHARE = 0.9                             # BR-74
ARPU_VOICE_SHARE = 0.1                            # BR-75
PRICE_BAND_K = {                                  # BR-78 pass-1 upper margin
    "UPSELL": 3.0, "MAINTAIN": 2.0, "RETAIN": 2.0,
    "UPSELL VOICE": 3.0, "CROSS SELL DATA": 3.0, "CROSS SELL VOICE": 3.0, "UPSELL DATA": 3.0,
    "free": 3.0,
}
MAX_OFFERS_PER_VALIDITY = 2                       # BR-82
MAX_OFFERS_TOTAL = 4                              # BR-82

# ---------------------------------------------------------------------------
# Campaign windows (BR-21 .. BR-25), in days relative to D
# ---------------------------------------------------------------------------
TARGETING_WINDOW = (-8, -2)                       # BR-21 targeting_dataiku_auto.date1
ARBITRATION_WINDOW = (-7, 0)                      # BR-22 targeting_auto.start_date
REWARD_WINDOW = (-7, -1)                          # BR-23 fulfilment success date
INTERACTION_WINDOW = (-270, -1)                   # BR-24 sub_btl_3months

# ---------------------------------------------------------------------------
# Dummy-data generator defaults
# ---------------------------------------------------------------------------
GENERATOR_DEFAULT_N = 5000
GENERATOR_MAX_N = 2_000_000                       # V2.0: 1M designed, 2M tested on the P16 (plan §16)
GENERATOR_MODES = {"small": 2_000, "medium": 100_000, "full": 1_000_000}
GENERATOR_SEED = 42
GENERATOR_CHUNK = 20_000                          # customers per daily-history chunk (memory bound)
SCENARIOS = ["baseline", "churn_wave", "data_growth", "price_sensitive", "new_offers",
             "campaign_fatigue", "data_quality_issues"]
SCENARIO_RAMP_DAYS = 28                           # scenario drifts build up over the last 4 weeks
TARGETED_SHARE = 0.55                             # share of the base targeted in last week's campaign

# ---------------------------------------------------------------------------
# Features (BR-46 .. BR-53)
# ---------------------------------------------------------------------------
USAGE_COLS = ["LOCAL_VOICE_MIN", "DATA_VOLUME_MB", "PG_DATA_AMOUNT"]   # columns the model reads from activities_*
# BR-53 trend pairs as (recent, older): change FROM older TO recent (W1 = most recent week).
TREND_PAIRS = {
    "REV": [("W1", "W2"), ("W2", "W3"), ("W3", "W4"), ("M1", "M2"), ("M2", "M3")],
    "VOICE": [("W1", "W2"), ("W2", "W3"), ("W3", "W4"), ("M1", "M2")],
    "DATA": [("W1", "W2"), ("W2", "W3"), ("W3", "W4"), ("M1", "M2")],
}
TREND_SOURCE = {"REV": "REV_{p}", "VOICE": "LOCAL_VOICE_MIN_{p}", "DATA": "DATA_VOLUME_MB_{p}"}   # BR-49 REV = SUB_AMT + PG_DATA
PCT_CLIP = (-100.0, 500.0)                        # BR-50 % change, capped when the older period is tiny
# M3 last-campaign context, from the campaign log (campaign_log.last_campaign_features). Part of the model's
# feature list, so the compatibility check refuses a model trained without them (and the reverse).
LAST_CAMPAIGN_FEATURES = os.getenv("FSDR_LAST_CAMPAIGN_FEATURES", "true").lower() == "true"
LAST_CAMPAIGN_NUMERIC = ["PREV_OFFER_TAKEN", "DAYS_SINCE_LAST_BTL", "NB_CONTACTS_LAST_4W"]
PREV_ACTION_NONE = "NONE"                          # PREV_ACTION of a line never contacted
# V1.0 review #7: nationality as a targeting feature is a legal question (review F4); false drops it from the
# model (the fairness report per nationality group is produced either way).
USE_NATIONALITY = os.getenv("FSDR_USE_NATIONALITY", "true").lower() == "true"
CATEGORICAL_FEATURES = (["NATIONALITY_GROUP"] if USE_NATIONALITY else []) + (["PREV_ACTION"] if LAST_CAMPAIGN_FEATURES else [])
# V1.0 review #11 / #12: one PROBA_<x>_MISSING flag per score (BR-47's random fill stays, the model now sees it)
# and LOG_TENURE (tenure of the model-score files). false = the production feature list (parity).
EXTENDED_FEATURES = os.getenv("FSDR_EXTENDED_FEATURES", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Bandit, training and scoring (BR-54 .. BR-58)
# ---------------------------------------------------------------------------
VW_EPSILON = 0.2                                  # BR-54 (allowed range for the training agent below)
VW_EPSILON_RANGE = (0.05, 0.3)
VW_ARGS_TEMPLATE = "--cb_explore_adf -q UA --quiet --epsilon {epsilon}"   # BR-54
TRAIN_MODE = "logged"                             # "logged" (all targeted, recommended) | "simulation" (BR-43/55/56)
TRAIN_PASSES = 2
TRAIN_PASSES_RANGE = (1, 5)
SIM_ITERATIONS = 20_000                           # BR-55 simulation mode
HOLDOUT_SHARE = 0.2
REWARD_DEFINITIONS = ["any_btl", "same_service", "exact_offer"]
REWARD_DEFINITION = "any_btl"                     # BR-40 production (plan Q4)
# BR-40: a taker is a targeted line with ANY catalogue BTL fulfilled in D-7..D-1 (production). Set true to also
# require the fulfilment on or after the line's targeting date1 (plan Q2 option; the two windows overlap).
REWARD_AFTER_TARGETING = os.getenv("FSDR_REWARD_AFTER_TARGETING", "false").lower() == "true"
PROPENSITY_CLIP = (0.01, 1.0)
SCORING_POLICY = "sample"                         # BR-57 sample (production) | greedy
BASE_SCORING_POLICY = "greedy"
MAX_MODEL_AGE_DAYS = 14                           # plan Q18
PROMOTION_TOLERANCE = 0.0                         # V0 rule: challenger must reach champion IPS - tolerance
# V1.0 review #2 / #3: the holdout is evaluated for the policy that is deployed (sampling from the PMF, BR-57), and
# the challenger is compared with the champion on the SAME holdout (out-of-time for the champion) by a paired
# bootstrap of the SNIPS gain. "paired_ci": refused only when the challenger is significantly worse (upper bound of
# the gain < -PROMOTION_MARGIN); significantly better -> promoted; inconclusive -> promoted (the fresher model wins
# when the data cannot separate them; scoring refuses a champion older than 14 days anyway). "point" = V0 (point
# estimates, which promote and refuse on noise: 0.215 vs 0.133 on 221 rows).
PROMOTION_RULE = os.getenv("FSDR_PROMOTION_RULE", "paired_ci").lower()     # paired_ci | point (V0)
PROMOTION_MARGIN = float(os.getenv("FSDR_PROMOTION_MARGIN", "0.0"))
BOOTSTRAP_SAMPLES = 500
CONFIDENCE = 0.95
# V1.0 review #1: the legacy campaign assigns the action from the segment (BR-39), so P(action | context) = 1 for
# the action sent. "deterministic" logs 1.0 (and flags the week as not usable for off-policy estimates);
# "share" is V0 (this week's action share, which the review found biased).
TARGETING_PROPENSITY = os.getenv("FSDR_TARGETING_PROPENSITY", "deterministic").lower()
# V1.0 review #4: uplift reward. The control group's organic take rate is modelled from the features (logistic
# regression on the cg rows of the campaign log); the bandit's cost becomes -(reward - baseline), so it learns the
# lift of a contact instead of "who buys anyway". false = V0 (cost = -reward).
UPLIFT_REWARD = os.getenv("FSDR_UPLIFT_REWARD", "true").lower() == "true"
PSI_BINS = 10
PSI_ALERT = 0.25                                  # PSI above this on a feature = drift alert
GROUP_TO_FLAG = {"free": "free", "cstr": "with_constraint", "llm": "llm"}   # BR-59; cg is not scored (control)

# ---------------------------------------------------------------------------
# Offer ranker (BR-83 .. BR-86)
# ---------------------------------------------------------------------------
RANKER = os.getenv("FSDR_RANKER", "bpr")          # bpr (implicit, default) | lightfm (if installed) (plan Q12)
RANKER_TOP_N = 100                                # BR-85
# Which offers the ranker may propose (BR-83/85). "history" = production Codes 3/4: only offers bought in
# the 270-day history after the action's family + validity filter (item_id = interaction1.BTL_NAME.unique()).
# "catalogue" = every catalogue offer of the family (the commented-out production line); needed for brand-new
# offers, which have no history yet (scenario new_offers).
RANKER_ITEMS = os.getenv("FSDR_RANKER_ITEMS", "history").lower()
BPR_PARAMS = {"factors": 32, "iterations": 60, "learning_rate": 0.05, "regularization": 0.01, "random_state": 42,
              "num_threads": 1}               # single thread: same data -> same ranking (offline llm arm = cstr arm)
CONTACT_MIN_GAP_DAYS = 2                          # see CONTACT_* below
# V1.0 review #10 / #14: offer arbitration. "value" ranks each family's offers on expected revenue
# P(accept | customer, offer) x price x OFFER_MARGIN, P from an acceptance model trained on last week's
# campaign (offers sent -> fulfilments), on offer ATTRIBUTES, so new offers get a score too (cold start); a share
# OFFER_EXPLORATION of the picks is a random family offer, so new offers get logged data. Families (BR-66/67),
# max offers (BR-82) and conflicts stay hard rules. "price_band" = production's 3 passes (BR-78..81), used as
# the fallback while the acceptance model has too little data.
OFFER_POLICY = os.getenv("FSDR_OFFER_POLICY", "value").lower()
OFFER_MARGIN = float(os.getenv("FSDR_OFFER_MARGIN", "1.0"))       # no cost data in the contract yet (review Q2)
OFFER_EXPLORATION = float(os.getenv("FSDR_OFFER_EXPLORATION", "0.05"))
ACCEPTANCE_MIN_POSITIVES = 20
# Price floor of the value policy. "anchor" keeps production's floor: every pick is priced above the line's ARPU
# anchor (as in the 3 passes, BR-78..80), so a line whose anchor is above every family price gets no offer (BR-82)
# and only the choice among the allowed offers changes. "none" lets the value policy also offer cheaper than the
# anchor (more lines contacted, more revenue in the simulation) -- a change of BR-78..82 that needs business sign-off.
OFFER_PRICE_FLOOR = os.getenv("FSDR_OFFER_PRICE_FLOOR", "anchor").lower()
LIGHTFM_PARAMS = {"no_components": 400, "loss": "warp", "k": 15, "epochs": 30}   # BR-84, used only by LightFMRanker

# ---------------------------------------------------------------------------
# Recommender agent: freedom, hard and soft rules (prompt §5.2)
# ---------------------------------------------------------------------------
EPSILON_LLM = float(os.getenv("FSDR_EPSILON_LLM", "0.1"))   # share of llm-arm decisions replaced by a bandit sample
CHURN_BLOCK_THRESHOLD = 0.60                      # plan Q9: PROBA_CHURN >= this blocks upsell-family actions
UPSELL_FAMILY = {"UPSELL", "UPSELL DATA", "UPSELL VOICE"}
# Contact cap (plan Q8, adjusted): campaigns are weekly and last week's targeting is D-8..D-2,
# so "1 contact per 7 days" would block most of the base. Hard rule = no new contact within
# CONTACT_MIN_GAP_DAYS of the last one, and at most CONTACT_CAP_28D in 28 days (campaign log, Phase 6).
CONTACT_CAP_28D = 3                               # L4 hard rule: at most 3 contacts logged in the last 28 days
CONTACT_CAP_DAYS = 28
# Training labels (H5): "auto" uses the campaign log when it holds last week's campaign (action actually sent,
# logged propensity), else the targeting tables with this week's action shares; "targeting" / "campaign_log" force one.
LABEL_SOURCE = os.getenv("FSDR_LABEL_SOURCE", "auto").lower()
# M4 price-escalation guard: cap the ARPU anchor at (1 + g) x the anchor logged for the line last time
# (campaign purchases raise next week's spend, so anchors could climb week after week).
# V1.0 review #9: on by default (10 % a week); "off" disables it (V0).
_GROWTH = os.getenv("FSDR_ANCHOR_MAX_GROWTH", "0.10").strip().lower()
ANCHOR_MAX_WEEKLY_GROWTH = None if _GROWTH in ("", "off", "none") else float(_GROWTH)
# V1.0 review #6: consent / do-not-contact (consent_contact table). A line with dnd = 1 or marketing_opt_in = 0 is
# never contacted, in any arm: hard rule, like eligibility.
CONSENT_TABLE = "consent_contact"
# Display labels of the competitors (prompt rule 2). The shipped snapshot already uses COMPETITOR_A / COMPETITOR_B;
# map the operator names of your own snapshot here, e.g. FSDR_COMPETITOR_LABELS='{"<name A>": "COMPETITOR_A"}'.
COMPETITOR_LABELS = json.loads(os.getenv("FSDR_COMPETITOR_LABELS", "{}") or "{}")
COMPETITOR_UNCOMPETITIVE_DELTA = 1.0              # our monthly price above the closest plan by more than this = at risk

# ---------------------------------------------------------------------------
# Batch add-on offers: roaming and international (not in the BR catalogue; added for the batch flow)
# They are recommended BESIDE the main offers: BR-82 (max 4) applies to the main offers only,
# and an add-on may never conflict with a main offer.
# ---------------------------------------------------------------------------
ROAMING_SERVICES = [s.strip().upper() for s in os.getenv("FSDR_ROAMING_SERVICES", "ROAMING").split(",") if s.strip()]
INTERNATIONAL_SERVICES = [s.strip().upper() for s in os.getenv("FSDR_INTERNATIONAL_SERVICES", "INTERNATIONAL").split(",") if s.strip()]
ADDON_TOP_ROAMING = 2
ADDON_TOP_INTERNATIONAL = 2
INTL_MIN_CALLS_M1 = 3          # international calls in M1 (Call_Count - Local_Call_Count) to offer international packs
BATCH_SIZE = 25                # customers per batch in the batch flow (UI / API)
BATCH_MAX = 2000
# Customers decided in parallel by the agents. vLLM batches concurrent requests on the GPU, so it gets more.
# The code runs each agent's required evidence tools before the first LLM call and hands the results
# over as prior evidence: same tools, same trace, but the LLM only reasons and decides (fewer, faster
# calls, and small models stop wasting steps on refused or missing tool calls). 0 = the LLM calls them.
AGENT_PREFETCH = os.getenv("FSDR_AGENT_PREFETCH", "1") not in ("0", "false", "no")
AGENT_CONCURRENCY = int(os.getenv("FSDR_AGENT_CONCURRENCY", "8" if nbx_config.LLM_BACKEND == "vllm" else "4"))

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
BLOCKING_KEY_ISSUE_SHARE = 0.01                   # null / duplicate / orphan keys above 1 % block the run
MIN_SCORE_COVERAGE = 0.70                         # below -> warning

# ---------------------------------------------------------------------------
# Weekly completeness per pipeline (master prompt §17.9, "which table is used where")
# required: missing = the run is blocked; optional: missing = warning (a fallback exists)
# ---------------------------------------------------------------------------
_ACT_AUTO = ["activities_w1_auto", "activities_w2_auto", "activities_w3_auto", "activities_w4_auto",
             "activities_m1_auto", "activities_auto_m2", "activities_m3_auto"]
_ACT_SCORE = ["activities_w1_score", "activities_w2_score", "activities_w3_score", "activities_w4_score",
              "activities_m1_score", "activities_m2_score", "activities_m3_score"]
_SCORES = [m.table for m in MODEL_SCORE_REGISTRY]
PIPELINE_TABLES = {
    "train_week0": {
        "required": ["base_prepaid", "consumer_aact_account", "access_method_id_auto", "party_auto", "party_scoring",
                     "subs_amt_auto", *_ACT_AUTO, "subs_amt_score", *_ACT_SCORE, *_SCORES, "subscriptions_tg",
                     "usage_data_voice_payg_tg", "targeting_dataiku_auto", "targeting_auto", "subs_btl_auto",
                     "sub_btl_3months", "btl_catalog"],
        # customer_profile missing -> AGE from party_scoring, tenure from access_method_subs_start_dt (§2.1)
        "optional": [PROFILE_TABLE, "location_auto", "DEVICEBEHAVIORSEGMENTATION.dv_segment", "paygo_activity_auto",
                     CONSENT_TABLE],
    },
    "train_weekly": {
        "required": ["base_prepaid", "consumer_aact_account", "party_scoring", "subs_amt_score", *_ACT_SCORE, *_SCORES,
                     "subscriptions_tg", "usage_data_voice_payg_tg", "targeting_auto", "subs_btl_auto",
                     "sub_btl_3months", "btl_catalog"],
        "optional": [PROFILE_TABLE, "location_auto", "DEVICEBEHAVIORSEGMENTATION.dv_segment", "targeting_dataiku_auto",
                     CONSENT_TABLE],
    },
    "score": {   # + the validated training bundle (checked by fsdr-score itself)
        "required": ["subscriptions_tg", "usage_data_voice_payg_tg", "targeting_auto", "sub_btl_3months", "btl_catalog",
                     "BTL_script_english_arabic_ok", "conflicting_offers_h"],
        "optional": [TARGET_SAMPLE_TABLE, CONSENT_TABLE],
    },
}

# ---------------------------------------------------------------------------
# V2.0 Phase 3: features (B-01 .. B-12), S-rules, bands, micro-segments, the two actions (master prompt §6 - §8)
# ---------------------------------------------------------------------------
FEATURE_WORKERS = int(os.getenv("FSDR_FEATURE_WORKERS", "2"))     # partitions in parallel (8 on the P16, plan §15)
# B-04: the 3 churn tables are stacked in registry order (global, 0-30, 30-60); the first value per line is kept
CHURN_FEATURE = "PROBA_CHURN"
# B-05: a missing score is flagged and imputed with the median of the line's nationality group (full base)
MISSING_FLAG_SUFFIX = "_MISSING"
# B-11 / §9: the context the LLM NBA agent reads besides the scores and trends (customer_360), fixed order
CUSTOMER_360_CONTEXT = ["NATIONALITY_GROUP", "AGE_BAND", "TENURE_BAND", "MICRO_SEGMENT", "VALUE_BAND", "USAGE_TYPE",
                        "REGION", "DV_SEGMENT", "SEGMENT", "RULE_ACTION", "ACTION_ORIGINAL", "ACTION_AGENT",
                        "ACTION_AGENT_RANK2", "ARPU_M1", "TENURE_DAYS", "AGE"]

# S-rules (§7): current segment of every line, first match wins
S_TENURE_NURSERING = 45                           # S-1/2: tenure < 45 days -> excluded
S_TENURE_ONBOARDING = 90                          # S-3/4
S_CHURN_AT_RISK = 0.5                             # S-1..4 and S-8 (PROBA_CHURN)
S_CHURN_1D_AT_RISK = 0.5                          # S-8 (PROBA_CHURN_1D)
S_RECONNECT_HIGH = 0.6                            # S-5
S_RECONNECT_MED = 0.3                             # S-6 (below -> S-7)
S_FRAGILITY = 0.5                                 # S-9
S_REV_DROP_PCT = -30.0                            # S-9 revenue M1 vs M2
S_VOICE_USER_MIN = 1.0                            # Local_Voice_min_M1 >= 1
S_DATA_USER_MB = 10.0                             # Data_Volume_MB_M1 >= 10
S_APPETENCY = 0.5                                 # S-11/12 (PROBA_APP_VOICE), S-13/14 (PROBA_APP_DATA)
# (rule id, segment, rule action); None = excluded (B-02). The order is the evaluation order.
S_RULES = [
    ("S-1", "Nursering_Early_Churn_At_Risk", None),
    ("S-2", "Nursering_Early_Churn_Not_At_Risk", None),
    ("S-3", "Onboarding_Early_Churn_At_Risk", "UPSELL"),
    ("S-4", "Onboarding_Early_Churn_Not_At_Risk", "UPSELL"),
    ("S-5", "Retention_High_Probability_To_Reconnect", "RETAIN"),
    ("S-6", "Retention_Med_Probability_To_Reconnect", "RETAIN"),
    ("S-7", "Retention_Low_Probability_To_Reconnect", "RETAIN"),
    ("S-8", "Retention_RGS_At_Risk", "RETAIN"),
    ("S-9", "Retention_RGS_Fragile", "UPSELL"),
    ("S-10", "RGS_Not_At_Risk_DualPlay_Upgrade", "MAINTAIN"),
    ("S-11", "Data_Only_Voice_Appetency", "UPSELL VOICE"),
    ("S-12", "Data_Only_No_Voice_Appetency", "CROSS SELL VOICE"),
    ("S-13", "Voice_Only_Data_Appetency", "UPSELL DATA"),
    ("S-14", "Voice_Only_No_Data_Appetency", "CROSS SELL DATA"),
]
S_SEGMENTS = [s for _, s, _ in S_RULES]

# B-03: the original dict_segm_actions of code 1, entry by entry in source order. It is a Python dict literal
# with repeated keys, so the LAST value wins (open item 1) and keys are case-sensitive (RGS_ vs Rgs_).
DICT_SEGM_ACTIONS_SOURCE = [
    ("Retention_Low_Probability_To_Reconnect", "RETAIN"), ("RGS_Not_At_Risk_DualPlay_Upgrade", "MAINTAIN"),
    ("Retention_RGS_At_Risk", "RETAIN"), ("Retention_RGS_Fragile", "UPSELL"),
    ("RGS_Not_At_Risk_Data_Only_Voice_Appetency", "UPSELL VOICE"), ("Retention_High_Probability_To_Reconnect", "UPSELL"),
    ("RGS_Not_At_Risk_Data_Only_No_Voice_Appetency", "CROSS SELL VOICE"), ("Onboarding_Early_Churn_At_Risk", "UPSELL"),
    ("RGS_Not_At_Risk_Voice_Only_Data_Appetency", "UPSELL DATA"), ("Onboarding_Early_Churn_Not_At_Risk", "UPSELL"),
    ("RGS_Not_At_Risk_Voice_Only_No_Data_Appetency", "CROSS SELL DATA"), ("Retention_Rgs_Fragile", "UPSELL"),
    ("Retention_Rgs_At_Risk", "RETAIN"), ("Rgs_Not_At_Risk_Data_Only_No_Voice_Appetency", "CROSS SELL VOICE"),
    ("Rgs_Not_At_Risk_Dualplay_Upgrade", "MAINTAIN"), ("Rgs_Not_At_Risk_Voice_Only_No_Data_Appetency", "CROSS SELL DATA"),
    ("Rgs_Not_At_Risk_Voice_Only_Data_Appetency", "UPSELL DATA"), ("Rgs_Not_At_Risk_Data_Only_Voice_Appetency", "UPSELL VOICE"),
    ("Retention_Med_Probability_To_Reconnect", "RETAIN"), ("Retention_Low_Probability_To_Reconnect", "RETAIN"),
    ("Retention_High_Probability_To_Reconnect", "RETAIN"), ("Rgs_Not_At_Risk_Data_Only_No_Voice_Appetency", "CROSS SELL VOICE"),
    ("Rgs_Not_At_Risk_Dualplay_Upgrade", "MAINTAIN"), ("Retention_Rgs_Fragile", "UPSELL"),
    ("Onboarding_Early_Churn_Not_At_Risk", "UPSELL"), ("Rgs_Not_At_Risk_Data_Only_Voice_Appetency", "CROSS SELL VOICE"),
    ("Onboarding_Early_Churn_At_Risk", "UPSELL"), ("Retention_Rgs_At_Risk", "RETAIN"),
    ("Rgs_Not_At_Risk_Voice_Only_Data_Appetency", "UPSELL DATA"), ("Rgs_Not_At_Risk_Voice_Only_No_Data_Appetency", "CROSS SELL DATA"),
]
DICT_SEGM_ACTIONS = dict(DICT_SEGM_ACTIONS_SOURCE)                 # last value wins, as in code 1
# §7.1: S-rule segment -> original segment name, for lines without a segment-engine segment this week.
# Every line is a CHECK item (plan §11): the S-rules re-create the segment engine, they are not the engine.
SEGMENT_CROSSWALK = {
    "Nursering_Early_Churn_At_Risk": ("Nursering_Early_Churn_At_Risk", "same name; excluded (B-02)"),
    "Nursering_Early_Churn_Not_At_Risk": ("Nursering_Early_Churn_Not_At_Risk", "same name; excluded (B-02)"),
    "Onboarding_Early_Churn_At_Risk": ("Onboarding_Early_Churn_At_Risk", "same name"),
    "Onboarding_Early_Churn_Not_At_Risk": ("Onboarding_Early_Churn_Not_At_Risk", "same name"),
    "Retention_High_Probability_To_Reconnect": ("Retention_High_Probability_To_Reconnect",
                                                "same name; original action RETAIN by last-value-wins (was UPSELL)"),
    "Retention_Med_Probability_To_Reconnect": ("Retention_Med_Probability_To_Reconnect", "same name"),
    "Retention_Low_Probability_To_Reconnect": ("Retention_Low_Probability_To_Reconnect", "same name"),
    "Retention_RGS_At_Risk": ("Retention_RGS_At_Risk", "same name (RGS_ spelling)"),
    "Retention_RGS_Fragile": ("Retention_RGS_Fragile", "same name (RGS_ spelling)"),
    "RGS_Not_At_Risk_DualPlay_Upgrade": ("RGS_Not_At_Risk_DualPlay_Upgrade", "same name (RGS_ spelling)"),
    "Data_Only_Voice_Appetency": ("RGS_Not_At_Risk_Data_Only_Voice_Appetency",
                                  "RGS_ spelling -> UPSELL VOICE (the Rgs_ spelling gives CROSS SELL VOICE)"),
    "Data_Only_No_Voice_Appetency": ("RGS_Not_At_Risk_Data_Only_No_Voice_Appetency", "RGS_ spelling"),
    "Voice_Only_Data_Appetency": ("RGS_Not_At_Risk_Voice_Only_Data_Appetency",
                                  "S-13 also takes lines with no usage; the engine may not"),
    "Voice_Only_No_Data_Appetency": ("RGS_Not_At_Risk_Voice_Only_No_Data_Appetency",
                                     "S-14 also takes lines with no usage; the engine may not"),
}

# §8 bands. Micro-segment = nationality group x age band x tenure band (5 x 4 x 6 = 120 cells)
NATIONALITY_GROUPS = ["Oman", "Pakistan", "India", "Bangladesh", "others"]
AGE_MIN, AGE_MAX = 15, 100                        # outside -> unknown
AGE_BANDS = [("young", 0, 40), ("middle", 40, 60), ("senior", 60, 200)]   # [lo, hi)
AGE_BAND_NAMES = [b for b, _, _ in AGE_BANDS] + ["unknown"]
YOUNG_SPLIT_AGE = 25                              # open item 5: under 25 / 25-39, tested in Phase 4
TENURE_BANDS = [("t0_lt3m", 0, 90), ("t1_3_6m", 90, 183), ("t2_6_12m", 183, 366), ("t3_1_2y", 366, 731),
                ("t4_gt2y", 731, 10**9)]          # days, [lo, hi)
TENURE_BAND_NAMES = [b for b, _, _ in TENURE_BANDS] + ["unknown"]
VALUE_BANDS = ["zero", "low", "mid", "high"]      # ARPU_M1 = 0, then tertiles of the paying lines (context only)
USAGE_TYPES = ["dual", "data_only", "voice_only", "none"]
AGENT_DICTIONARY_PREFIX = "dict_segm_actions_agent_v"

# ---------------------------------------------------------------------------
# V3.0: 100k contacted in four groups of 25k + control (update prompt PROMPT_FSDR_update_100k_targets.md)
# 2 x 2 design: rules (production / challenger) x offer universe (constraint = code 3 / free = code 4)
# ---------------------------------------------------------------------------
T1_CONSTRAINT_SIZE = int(os.getenv("FSDR_T1_CONSTRAINT_SIZE", "25000"))
T1_FREE_SIZE = int(os.getenv("FSDR_T1_FREE_SIZE", "25000"))
T2_CONSTRAINT_SIZE = int(os.getenv("FSDR_T2_CONSTRAINT_SIZE", "25000"))
T2_FREE_SIZE = int(os.getenv("FSDR_T2_FREE_SIZE", "25000"))
CONTROL_SIZE = int(os.getenv("FSDR_CONTROL_SIZE", "20000"))        # open item 1 (10k = less precise per 25k group)
T2_SPLIT = os.getenv("FSDR_T2_SPLIT", "true").lower() == "true"     # false -> one 50k T2 with the CC rules
GROUP_SIZE_TOLERANCE = 0.02                       # A-30: sizes within +-2 %
BALANCE_MAX_POINTS = 2.0                          # A-31: stratum shares differ by at most 2 points between groups

# One entry per contacted group. universe: "constraint" (code 3 families by action) | "free" (code 4: L2,
# DATA / COMBO, action not used). rules: "code3" | "code4" | "CC" | "CF". rule_action: the NBA hint and fallback.
GROUPS = {
    "t1c": {"panel_group": "t1_constraint", "target": "target1", "universe": "constraint", "rules": "code3",
            "agent": "6A-C", "tg_flag": "constraint", "rule_action": "ACTION_ORIGINAL", "nbo_sees_action": True,
            "consistency_loop": True, "rules_version": "production_constraint",
            "scored_table": "TG_40K_constraint_scored_nba", "recos_monthly": "recos_monthly_40k_constraint",
            "recos_weekly": "recos_weekly_40k_constraint", "action_column": True},
    "t1f": {"panel_group": "t1_free", "target": "target1", "universe": "free", "rules": "code4",
            "agent": "6A-F", "tg_flag": "free", "rule_action": "ACTION_ORIGINAL",
            "nbo_sees_action": os.getenv("FSDR_T1F_NBO_SEES_ACTION", "false").lower() == "true",
            "consistency_loop": False, "rules_version": "production_free",
            "scored_table": "TG_40k_free_scored_nba", "recos_monthly": "recos_monthly_tg_free",
            "recos_weekly": "recos_weekly_tg_free", "action_column": False},
    "t2c": {"panel_group": "t2_constraint", "target": "target2", "universe": "constraint", "rules": "CC",
            "agent": "6B-C", "tg_flag": "challenger_constraint", "rule_action": "ACTION_AGENT", "nbo_sees_action": True,
            "consistency_loop": True, "rules_version": "CC-v1",
            "scored_table": "TG_challenger_constraint_scored_nba", "recos_monthly": "recos_monthly_challenger_constraint",
            "recos_weekly": "recos_weekly_challenger_constraint", "action_column": True},
    "t2f": {"panel_group": "t2_free", "target": "target2", "universe": "free", "rules": "CF",
            "agent": "6B-F", "tg_flag": "challenger_free", "rule_action": "ACTION_AGENT",
            "nbo_sees_action": os.getenv("FSDR_T2F_NBO_SEES_ACTION", "false").lower() == "true",
            "consistency_loop": False, "rules_version": "CF-v1",
            "scored_table": "TG_challenger_free_scored_nba", "recos_monthly": "recos_monthly_challenger_free",
            "recos_weekly": "recos_weekly_challenger_free", "action_column": False},
}
CONTROL_GROUP = "control"
NOT_SELECTED = "not_selected"
PANEL_GROUPS = [g["panel_group"] for g in GROUPS.values()] + [CONTROL_GROUP, NOT_SELECTED]
# the provided test sample (master §2.4): its "group" column -> our groups; T2C / T2F come from the rest of the base
SAMPLE_GROUP_MAP = {"cstr": "t1c", "free": "t1f", "cg": "control"}
# Tables of the earlier update prompts, replaced by the 4-group names above (kept out of every new output)
SUPERSEDED_TABLES = ["TG_40k_challenger_scored_nba", "recos_monthly_challenger", "recos_weekly_challenger",
                     "TG_40k_challenger_constraint_scored_nba", "TG_40k_challenger_free_scored_nba"]


def group_sizes() -> dict[str, int]:
    """Configured size of every panel group (t2_split false: one T2 of both sizes, with the CC rules)."""
    sizes = {"t1c": T1_CONSTRAINT_SIZE, "t1f": T1_FREE_SIZE}
    if T2_SPLIT:
        sizes.update(t2c=T2_CONSTRAINT_SIZE, t2f=T2_FREE_SIZE)
    else:
        sizes["t2c"] = T2_CONSTRAINT_SIZE + T2_FREE_SIZE
    sizes[CONTROL_GROUP] = CONTROL_SIZE
    return sizes


# Code 4 / FR rules (T1F) and the free universe of T2F
FREE_UNIVERSE_GENEROSITY = "L2"                   # FR-02
FREE_UNIVERSE_SERVICES = ("COMBO", "DATA")        # FR-03 no VOICE; order = (type A, type B) as code 4
FREE_PRICE_K = 3.0                                # FR-09
RECOMMENDABLE_HISTORY_DAYS = 270                  # B-20 / B-20b (sub_btl_3months)
ONLY_OWN_PAST_OFFERS = os.getenv("FSDR_ONLY_OWN_PAST_OFFERS", "false").lower() == "true"   # B-20b switch
MAX_CANDIDATES_PER_VALIDITY = 25                  # §10: candidates per validity sent to the NBO agent
TARGET1_FREE_SHARE = 0.0                          # B-23: the code-4 part now lives in its own group (T1F)

# Challenger CC-v1 (T2C) and CF-v1 (T2F): open item 4, approved before the first push
CHALLENGER_BAND = (1.0, 1.5)                      # CC-02 / CF-02: price between 1.0x and 1.5x the anchor
CHALLENGER_BAND_WIDENED = (1.0, 2.0)              # CC-04 / CF-04: widened once, upwards only
CHALLENGER_UPSELL_STEP = 0.5                      # minimum +0.5 OMR above the anchor for the UPSELL actions
UPSELL_ACTIONS = ["UPSELL", "UPSELL DATA", "UPSELL VOICE"]
DEFAULT_DATA_SHARE = 0.9                          # CC-03 / CF-03: no spend history -> 90 / 10
DO_NOT_DISTURB_IGNORED = 3                        # CC-05 / CF-05
GUARD_MODE_TARGET1 = os.getenv("FSDR_GUARD_MODE_TARGET1", "repair")   # open item 3: "repair" | "report"

# LLM budget at 100k (update §6)
LLM_WEEKLY_BUDGET_H = float(os.getenv("FSDR_LLM_WEEKLY_BUDGET_H", "10"))
LEVEL2_MAX_OUTPUT_TOKENS = 120
LEVEL2_MAX_INPUT_TOKENS = 1500

# ---------------------------------------------------------------------------
# V3.0 target upload folders and the final output (code 5, F-01 .. F-19)
# ---------------------------------------------------------------------------
TARGETS_FOLDER = "targets"                        # data/input/<D>/targets/<target>/<group>.csv.gz
TARGET_FILES = {"t1c": "target1/t1_constraint", "t1f": "target1/t1_free",
                "t2c": "target2/t2_constraint", "t2f": "target2/t2_free", "control": "control/control"}
TARGETS_MANIFEST = "_targets_manifest.json"
# F-08 forced bundles in slot 6, active version (code 5 "Change the forced offers above to L2 offers"), in order:
# (anchor column, low, high, offer, popular list that blocks the forcing)
_POP_CM = ["CM_3R_8GB_240MIN", "CM_4R_10GB_300MIN", "CM_5R_14GB_360MIN"]
_POP_DM = _POP_CM + ["DM_3R_10GB", "DM_4R_14GB", "DM_5R_18GB"]
FORCED_BUNDLES = [("F05_MONTHLY", 0, 3, "CM_3R_8GB_240MIN", _POP_CM), ("F05_MONTHLY", 3, 4, "CM_4R_10GB_300MIN", _POP_CM),
                  ("F05_MONTHLY", 4, 5, "CM_5R_14GB_360MIN", _POP_CM), ("F05_DATA_MONTHLY", 0, 3, "DM_3R_10GB", _POP_DM),
                  ("F05_DATA_MONTHLY", 3, 4, "DM_4R_14GB", _POP_DM), ("F05_DATA_MONTHLY", 4, 5, "DM_5R_18GB", _POP_DM)]
FAVOURITE_ARPU_FILTER = False                     # F-07 (code 5 commented it out)
OPPORTUNITY_PREFIX = "nbo_LFM_"                   # F-18, expected by the campaign platform
INPUT_FLYTXT_COLUMNS = (["date", "start_date", "end_date", "msisdn"] + [f"channel_{i}" for i in range(1, 7)]
                        + [f"opportunity_name_{i}" for i in range(1, 7)] + [f"destination_offer_name_{i}" for i in range(1, 7)]
                        + ["flag_splio"] + [f"offer_script_en_{i}" for i in range(1, 7)]
                        + [f"offer_script_ar_{i}" for i in range(1, 7)] + ["segment", "segment_btl_taker", "segment_btl_taker_upsell"])
