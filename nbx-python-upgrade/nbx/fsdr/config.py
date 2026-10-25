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
SCHEMA_VERSION = "1.0"

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
GENERATOR_MAX_N = 200_000
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
CATEGORICAL_FEATURES = ["NATIONALITY_GROUP"] + (["PREV_ACTION"] if LAST_CAMPAIGN_FEATURES else [])

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
PROMOTION_TOLERANCE = 0.0                         # challenger must reach champion IPS - tolerance
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
# (campaign purchases raise next week's spend, so anchors could climb week after week). Empty = off.
ANCHOR_MAX_WEEKLY_GROWTH = float(os.getenv("FSDR_ANCHOR_MAX_GROWTH")) if os.getenv("FSDR_ANCHOR_MAX_GROWTH") else None
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
