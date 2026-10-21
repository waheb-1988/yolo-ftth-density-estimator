"""Global configuration: settings, LLM roster, vector store roster."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Project root (nbx-python/): this file lives in nbx/core/.
BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "mistral:7b")

# Which server answers LLM chat calls (llms/llm_client.py dispatches on this):
#   "offline" -- pure Python, no server (default): every agent runs its
#               deterministic policy; see llms/offline_client.py.
#   "ollama" -- local dev / Windows / multi-model benchmarking.
#   "vllm"   -- Linux GPU production: one model, continuous batching for many
#               concurrent /recommend calls and nightly batch scoring.
# Embeddings are unaffected: Qwen3-Embedding models always go through
# OLLAMA_URL and MiniLM through sentence-transformers, whatever this says.
LLM_BACKEND = os.getenv("LLM_BACKEND", "offline").lower()
LLM_BACKENDS = ["offline", "ollama", "vllm"]
# vLLM's own default port is 8000, which FastAPI already uses here -- so 8001.
# 127.0.0.1, not "localhost": vLLM binds IPv4 only, and "localhost" tries ::1
# first -- on Windows that costs ~2 s per new connection before falling back.
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8001")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
# Cap per completion. vLLM otherwise defaults to the whole remaining context,
# and one runaway generation then holds a batch slot for its entire length.
VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "1024"))
# Per LLM call. A 7B model partly offloaded to a 4 GB GPU takes ~25-95 s per
# agent step, and longer when two runs share the server; 120 s cut runs short.
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "300"))
# "graph": agents/nbo_agent.py step 2 answers "what fits this need" by
# traversing core.knowledge_graph.need_based_offers() -- real peer purchases
# / segment affinity, no embeddings. Matches production telco data, which is
# entirely relational (msisdn, item_id, price, revenue) with no free text to
# search. "vector" keeps the original embedded-vector-store path available
# for comparison -- see RETRIEVAL_MODES.
DEFAULT_RETRIEVAL_MODE = os.getenv("DEFAULT_RETRIEVAL_MODE", "graph")
RETRIEVAL_MODES = ["graph", "vector"]
DEFAULT_VECTORSTORE = os.getenv("DEFAULT_VECTORSTORE", "chroma")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DATA_DIR.mkdir(parents=True, exist_ok=True)

CUSTOMERS_FILE = DATA_DIR / "customers.json"
OFFERS_FILE = DATA_DIR / "offers.json"
PURCHASES_FILE = DATA_DIR / "purchases.json"
SEGMENTS_FILE = DATA_DIR / "segments.json"

CHROMA_DIR = str(DATA_DIR / "chroma")
FAISS_DIR = str(DATA_DIR / "faiss_index")
MILVUS_DB_PATH = str(DATA_DIR / "milvus_lite.db")
LANCEDB_DIR = str(DATA_DIR / "lancedb")

# ---------------------------------------------------------------------------
# The 7 Ollama LLMs, ranked best -> worst for this NBO reasoning task.
# ---------------------------------------------------------------------------
OLLAMA_MODELS = [
    {
        "rank": 1,
        "name": "mistral:7b",
        "display": "Mistral 7B",
        "strength": "Best instruction following, structured JSON output",
        "recommended": True,
    },
    {
        "rank": 2,
        "name": "llama3.1:8b",
        "display": "LLaMA 3.1 8B",
        "strength": "Strong reasoning, good for multi-step agent tasks",
        "recommended": True,
    },
    {
        "rank": 3,
        "name": "gemma2:9b",
        "display": "Gemma 2 9B",
        "strength": "Excellent context understanding, Google-trained",
        "recommended": False,
    },
    {
        "rank": 4,
        "name": "qwen2.5:7b",
        "display": "Qwen 2.5 7B",
        "strength": "Strong multilingual, good for Arabic/English mix",
        "recommended": False,
    },
    {
        "rank": 5,
        "name": "phi3.5:3.8b",
        "display": "Phi 3.5 Mini",
        "strength": "Fastest inference, good for simple lookups",
        "recommended": False,
    },
    {
        "rank": 6,
        "name": "neural-chat:7b",
        "display": "Neural Chat 7B",
        "strength": "Conversational, good for customer-facing explanations",
        "recommended": False,
    },
    {
        "rank": 7,
        "name": "deepseek-r1:7b",
        "display": "DeepSeek R1 7B",
        "strength": "Best chain-of-thought reasoning, slower but thorough",
        "recommended": False,
    },
    {
        "rank": 8,
        "name": "deepseek-llm:7b-base",
        "display": "DeepSeek LLM 7B (base)",
        "strength": "Base model, not instruction-tuned — needs few-shot prompting to hold "
                    "the JSON schema reliably; useful as a raw-reasoning quality baseline",
        "recommended": False,
    },
    {
        "rank": 9,
        "name": "hf.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M",
        "display": "Ministral 3 8B Instruct",
        "strength": "Edge-optimized Mistral family model (8.4B LM + vision encoder, "
                    "256k context), pulled directly from its HF GGUF repo since it is "
                    "not in the official Ollama library",
        "recommended": False,
    },
]
# mistral:7b (rank 1) already resolves to Mistral-7B-Instruct-v0.3 upstream —
# that has been Ollama's "mistral" tag since 2024-05-22, no separate entry needed.

OLLAMA_MODEL_NAMES = [m["name"] for m in OLLAMA_MODELS]


def get_model_info(name: str) -> dict | None:
    for m in OLLAMA_MODELS:
        if m["name"] == name:
            return m
    return None


# ---------------------------------------------------------------------------
# Embedding models. Each vector store keeps one collection per embedding
# model (dimensions are not interchangeable), so switching models here does
# not corrupt an existing index — it just indexes a parallel collection.
#
# NV-Embed-v2 (nvidia, 4096-dim, 8B) was evaluated but deliberately excluded:
# it ships under CC-BY-NC-4.0, which forbids the commercial/business use this
# engine is built for. Qwen3-Embedding covers the same quality/size range
# under Apache 2.0 and has GGUF builds, so it runs through the same Ollama
# path as the LLMs instead of requiring a second serving stack.
# ---------------------------------------------------------------------------
EMBEDDING_MODELS = [
    {
        "key": "all-MiniLM-L6-v2",
        "display": "MiniLM-L6-v2 (sentence-transformers)",
        "backend": "sentence-transformers",
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "strength": "Default. Tiny, CPU-friendly, no Ollama dependency — good baseline.",
        "recommended": True,
    },
    {
        "key": "qwen3-embedding:0.6b",
        "display": "Qwen3-Embedding 0.6B",
        "backend": "ollama",
        "model_name": "qwen3-embedding:0.6b",
        "dimension": 1024,
        "mrl_min_dimension": 32,
        "strength": "Small multilingual embedder, MRL-truncatable 32-1024 dims.",
        "recommended": False,
    },
    {
        "key": "qwen3-embedding:4b",
        "display": "Qwen3-Embedding 4B",
        "backend": "ollama",
        "model_name": "qwen3-embedding:4b",
        "dimension": 2560,
        "mrl_min_dimension": 32,
        "strength": "Mid-size, stronger retrieval quality, still CPU-runnable via Ollama.",
        "recommended": False,
    },
    {
        "key": "qwen3-embedding:8b",
        "display": "Qwen3-Embedding 8B",
        "backend": "ollama",
        "model_name": "qwen3-embedding:8b",
        "dimension": 4096,
        "mrl_min_dimension": 32,
        "strength": "Largest Qwen3 embedder, top MTEB-tier quality, needs a GPU for speed.",
        "recommended": False,
    },
]

EMBEDDING_MODEL_KEYS = [e["key"] for e in EMBEDDING_MODELS]


def get_embedding_info(key: str) -> dict | None:
    for e in EMBEDDING_MODELS:
        if e["key"] == key:
            return e
    return None


# ---------------------------------------------------------------------------
# The 5 local vector stores.
# ---------------------------------------------------------------------------
VECTOR_STORES = [
    {"key": "chroma", "display": "ChromaDB", "module": "nbx.vectorstores.chroma_store", "class_name": "ChromaStore"},
    {"key": "faiss", "display": "FAISS", "module": "nbx.vectorstores.faiss_store", "class_name": "FaissStore"},
    {"key": "qdrant", "display": "Qdrant (local)", "module": "nbx.vectorstores.qdrant_store", "class_name": "QdrantStore"},
    {"key": "milvus", "display": "Milvus Lite", "module": "nbx.vectorstores.milvus_store", "class_name": "MilvusStore"},
    {"key": "lancedb", "display": "LanceDB", "module": "nbx.vectorstores.lancedb_store", "class_name": "LanceDBStore"},
]

VECTOR_STORE_KEYS = [v["key"] for v in VECTOR_STORES]

# ---------------------------------------------------------------------------
# Need-detection signals (8-signal logic).
# ---------------------------------------------------------------------------
NEED_SIGNALS = [
    "REACTIVATION",
    "RENEWAL",
    "INTERNATIONAL",
    "ROAMING",
    "DATA_UPSELL",
    "VOICE_UPSELL",
    "CHANNEL_MIGRATION",
    "GENERAL_NBO",
]

RERANK_WEIGHTS = {
    "llm_score": 0.35,
    "cf_score": 0.25,
    "retrieval_score": 0.15,
    "graph_score": 0.10,
    "price_fit": 0.15,
}
# Rebalanced from the pre-CF weights (llm .45 / retrieval .25 / graph .15 /
# price .15) when core/collab_filter.py was added: cf_score is now a
# first-class behavioral signal (a real matrix-factorization model trained on
# simulated 90-day subscriptions, mirroring the production LightFM NBO job),
# so it earns the second-highest weight after the LLM. llm_score stays
# highest since its job is now explicitly to reason over — and be able to
# override — what the behavioral model says, not just read retrieved text.

# ---------------------------------------------------------------------------
# Collaborative filtering (core/collab_filter.py) — implicit-feedback matrix
# factorization over a simulated 90-day subscription window, standing in for
# the production Dataiku/LightFM NBO job. LightFM itself (PyPI, last released
# 2020) fails to build on Python 3.12/Windows here (broken setup.py, no
# wheel) — `implicit`'s BayesianPersonalizedRanking is the maintained
# equivalent: same problem class (implicit-feedback top-N ranking via
# pairwise loss), same job, ships a working win_amd64/cp312 wheel.
# ---------------------------------------------------------------------------
CF_INTERACTION_WINDOW_DAYS = 90          # matches sub_btl_90days_catalog
CF_MONTHLY_VALIDITY_DAYS = 30            # matches items_feat.validity == 30
CF_EXCLUDED_CATEGORY = "Voice"           # matches items_feat.service == "VOICE"
CF_MODEL_PARAMS = {
    "factors": 32,
    "iterations": 60,
    "learning_rate": 0.05,
    "regularization": 0.01,
    "random_state": 42,
}

REVENUE_HISTORY_FILE = DATA_DIR / "revenue_history.json"

# The production job's price-affordability band: an upsell must price between
# the customer's estimated average spend and (average spend + this many OMR)
# to qualify for the tightest preference tier.
AFFORDABILITY_BAND_OMR = 3.0

# ---------------------------------------------------------------------------
# NBA contextual bandit (core/nba_bandit.py, agents/nba_agent.py) — ported
# from the production Vowpal Wabbit NBA script (--cb_explore_adf -q UA
# --epsilon 0.2). Real vowpalwabbit installs cleanly here (unlike lightfm),
# so this is the actual library, not a substitute.
#
# NBA_SEGMENT_ACTION_MAP and NBA_ACTION_PROPENSITY are read verbatim from
# that script's dict_segm_actions / segment_prop_dict. The source dict has
# case-duplicate segment keys (e.g. "Retention_RGS_At_Risk" and
# "Retention_Rgs_At_Risk" both -> RETAIN) because two upstream feeds disagreed
# on casing; collapsed here to one canonical spelling each, since this repo
# generates its own segment values and controls their casing.
#
# Two corrections vs. the source, flagged rather than silently applied:
#   1. Its context_ list has "proba_frag_voice" twice and omits
#      "proba_app_voice" despite computing it -- fixed here (dedup + include).
#   2. Action names contain spaces ("CROSS SELL VOICE"), but VW's text
#      format splits features on whitespace -- a literal "|Action
#      product=CROSS SELL VOICE" line would parse as three tokens, not one
#      categorical value. core/nba_bandit.py underscores them
#      (CROSS_SELL_VOICE) only inside VW example strings; the human-readable
#      name with spaces is what the API/UI actually shows.
#
# The upstream SEGMENT field itself (Retention_RGS_At_Risk, etc.) arrives
# pre-computed in the production pipeline from a job not included in the
# script given here. core/data_simulator.py::assign_nba_segment() infers a
# plausible derivation from the naming convention and the scores this script
# does compute (tenure, churn/reactivation/fragility/appetency) -- documented
# there as an inference, not as given business logic.
# ---------------------------------------------------------------------------
NBA_ACTIONS = ["CROSS SELL VOICE", "UPSELL VOICE", "CROSS SELL DATA", "UPSELL DATA", "MAINTAIN", "UPSELL", "RETAIN"]

# The point of these two actions isn't pushing a new purchase -- NBO's
# offers still come back as context (agents/final_agent.py still calls it),
# but is_push_action tells the caller not to treat this as a sales push.
NBA_NON_PUSH_ACTIONS = {"RETAIN", "MAINTAIN"}

NBA_EXCLUDED_SEGMENTS = ["Nursering_Early_Churn_At_Risk", "Nursering_Early_Churn_Not_At_Risk"]

NBA_SEGMENT_ACTION_MAP = {
    "Retention_Low_Probability_To_Reconnect": "RETAIN",
    "RGS_Not_At_Risk_DualPlay_Upgrade": "MAINTAIN",
    "Retention_RGS_At_Risk": "RETAIN",
    "Retention_RGS_Fragile": "UPSELL",
    "RGS_Not_At_Risk_Data_Only_Voice_Appetency": "UPSELL VOICE",
    "Retention_High_Probability_To_Reconnect": "UPSELL",
    "RGS_Not_At_Risk_Data_Only_No_Voice_Appetency": "CROSS SELL VOICE",
    "Onboarding_Early_Churn_At_Risk": "UPSELL",
    "RGS_Not_At_Risk_Voice_Only_Data_Appetency": "UPSELL DATA",
    "Onboarding_Early_Churn_Not_At_Risk": "UPSELL",
    "RGS_Not_At_Risk_Voice_Only_No_Data_Appetency": "CROSS SELL DATA",
}

# Fixed empirical action base-rates from the production population, kept as
# given (historical frequencies, not something to re-derive):
NBA_ACTION_PROPENSITY = {
    "CROSS SELL DATA": 0.0083128,
    "CROSS SELL VOICE": 0.0528180,
    "MAINTAIN": 0.1015183,
    "RETAIN": 0.5086577,
    "UPSELL": 0.2624428,
    "UPSELL DATA": 0.0422645,
    "UPSELL VOICE": 0.0239859,
}
NBA_PROPENSITY_JITTER_MAX = 0.02

NBA_PROBA_FEATURES = [
    "proba_churn", "proba_react_30", "proba_react_15", "proba_app_data",
    "proba_app_voice", "proba_churn_1d", "proba_rev_decr", "proba_frag_data", "proba_frag_voice",
]
NBA_VARIATION_FEATURES = [
    "var_rev_w1_w2", "var_rev_w2_w3", "var_rev_w3_w4", "var_rev_m1_m2", "var_rev_m2_m3",
    "var_voice_w1_w2", "var_voice_w2_w3", "var_voice_w3_w4", "var_voice_m1_m2", "var_voice_m2_m3",
    "var_data_w1_w2", "var_data_w2_w3", "var_data_w3_w4", "var_data_m1_m2", "var_data_m2_m3",
]
NBA_CONTEXT_FEATURES = ["nationality_group"] + NBA_PROBA_FEATURES + NBA_VARIATION_FEATURES

NBA_VW_ARGS = "--cb_explore_adf -q UA --quiet --epsilon 0.2"
NBA_TRAIN_ITERATIONS = 4000  # scaled down from production's 20,000 for this dataset's size
NBA_PCT_CHANGE_EPSILON = 0.00002  # matches the source's zero-baseline substitution
# REWARD in the source is "did this msisdn appear in subs_btl_auto_hdfs" --
# not defined there as a time window, but a lifetime "ever purchased" read
# is uninformative here (260 simulated purchases over 50 customers means
# ~98% have bought *something* in the full 180-day history). 14 days reads
# as "took the current campaign's offer," giving a ~54/46 split instead --
# an inferred window, not a given one, chosen for a learnable reward signal.
NBA_REWARD_WINDOW_DAYS = 14

# ---------------------------------------------------------------------------
# NBA source files -- split to mirror the production script's own dataset
# boundaries, not merged into one file. The original script reads ~15
# separately-scored Dataiku datasets and joins them; this collapses that to
# 7 files at the same semantic boundaries (grouping only where production
# itself already concatenates before the main merge, e.g. the 3 churn
# scorers -- DAIM_CHURN_GLOBAL_SCORING_TD + DAIM_EARLY_CHURN_0_30 +
# DAIM_EARLY_CHURN_30_60 -- land in one churn_df there too):
#
#   nba_scores_churn.json           <- DAIM_CHURN_GLOBAL/EARLY_CHURN_0_30/30_60 + CHURN_ONE_DAY_ACTIVE_107
#   nba_scores_reactivation.json    <- DAIM_PROPENSITY_REACTIVATION_30D_404 + PROPENSITY_REACTIVATION_15D_401
#   nba_scores_appetency.json       <- APPETENCY_DATA_204 + APPETENCY_VOICE_204
#   nba_scores_fragility.json       <- DAIM_FRAGILITY_DATA_SCORING_TD + DAIM_FRAGILITY_VOICE_SCORING_TD
#   nba_scores_revenue_decrease.json <- REV_DECREASE_204_SCORING_TD
#   nba_activity_usage.json         <- activities_w1-4/m1-3_auto_hdfs (raw usage, BEFORE the % change calc)
#   nba_targeting.json              <- targeting_auto_hdfs + targeting_dataiku_auto_hdfs + party_auto
#                                       (nationality) + subs_btl_auto (reward) + derived action/cost/propensity
#
# revenue (SUB_AMT + PG_DATA_AMOUNT -> REV_M/W) is NOT re-split here -- it's
# core/data_simulator.py's existing revenue_history.json, already shared
# with NBO's affordability rule (agents/rules_engine.py), the same file
# doing the same job as the source's subs_amt_auto_hdfs + paygo_activity_auto_hdfs.
#
# The 15 var_rev/var_voice/var_data_* features are NOT stored in any file --
# core/nba_bandit.py computes them at merge/read time, same order of
# operations as the source script (calcul_evol() runs on the already-merged
# dataframe, after these per-source datasets are joined, not before).
# ---------------------------------------------------------------------------
NBA_CHURN_FILE = DATA_DIR / "nba_scores_churn.json"
NBA_REACTIVATION_FILE = DATA_DIR / "nba_scores_reactivation.json"
NBA_APPETENCY_FILE = DATA_DIR / "nba_scores_appetency.json"
NBA_FRAGILITY_FILE = DATA_DIR / "nba_scores_fragility.json"
NBA_REV_DECREASE_FILE = DATA_DIR / "nba_scores_revenue_decrease.json"
NBA_ACTIVITY_USAGE_FILE = DATA_DIR / "nba_activity_usage.json"
NBA_TARGETING_FILE = DATA_DIR / "nba_targeting.json"

# key -> config attribute name, iterated by agents/nba_ingestion_pipeline.py
# and core/data_simulator.py::save_all() so both stay in lockstep with the
# file list above.
NBA_SOURCE_FILE_ATTRS = {
    "churn": "NBA_CHURN_FILE",
    "reactivation": "NBA_REACTIVATION_FILE",
    "appetency": "NBA_APPETENCY_FILE",
    "fragility": "NBA_FRAGILITY_FILE",
    "revenue_decrease": "NBA_REV_DECREASE_FILE",
    "activity_usage": "NBA_ACTIVITY_USAGE_FILE",
    "targeting": "NBA_TARGETING_FILE",
}

# API-facing action_type values -- underscored (CROSS_SELL_VOICE) since VW's
# own text format already forced the same underscoring internally (see
# core/nba_bandit.py); NBA_ACTIONS above keeps the human-readable spelling
# for anything that isn't a raw API value (UI labels, business-rule text).
NBA_ACTION_TYPES = [a.replace(" ", "_") for a in NBA_ACTIONS] + ["NO_ACTION"]
DEFAULT_NBA_ACTION_TYPE = "NO_ACTION"  # excluded (Nursering) or no-signal customers

# ---------------------------------------------------------------------------
# Competitor intelligence (core/competitors.py) -- public plan pages of the
# two other Omani mobile operators, scraped politely (robots.txt checked:
# omantel.om allows crawlers, vodafone.om has no robots.txt), stored as
# structured offers + text passages and indexed in their own Chroma
# collection so the agents can cite competitor prices in their reasoning.
# Prices are a snapshot: every record carries its source URL and scrape time.
# ---------------------------------------------------------------------------
# Native-language SMS (Hindi / Urdu / Bengali, agents/agentic/languages.py):
#   "template" -- reviewed templates (default). A 3B model produced garbled
#                 Hindi/Bengali and Persian instead of Urdu in testing, which
#                 must never reach a customer.
#   "llm"      -- model-written, still script- and price-checked in code;
#                 enable only with a stronger model and after native review.
MESSENGER_NATIVE_SOURCE = os.getenv("MESSENGER_NATIVE_SOURCE", "template").lower()

COMPETITOR_DIR = DATA_DIR / "competitors"
COMPETITOR_OFFERS_FILE = COMPETITOR_DIR / "competitor_offers.json"
COMPETITOR_PASSAGES_FILE = COMPETITOR_DIR / "competitor_passages.json"
COMPETITOR_USER_AGENT = "NBX-competitor-research/1.0 (+public price pages; contact: CVM analytics team)"
COMPETITOR_REQUEST_DELAY_S = 1.5
# Omantel's site loads its plan catalog from this public JSON endpoint (seen
# in the page's own network calls); it is far more reliable than parsing the
# rendered cards. Only mass-market categories are kept.
COMPETITOR_API_SOURCES = [
    ("Omantel", "https://portal.omantel.om/api/omantel-config/portal-plans/web-prepaid"),
    ("Omantel", "https://portal.omantel.om/api/omantel-config/portal-plans/web-postpaid"),
]
OMANTEL_API_CATEGORIES = {
    # categoryCode prefix -> (category, page_type, segment)
    "WEB_NEW_HAYYAK_PLANS": ("Bundle", "prepaid", "mass"),
    "WEB_HAYYAK_DATA_ONLY": ("Data", "prepaid", "mass"),
    "WEB_TOURIST_PACK": ("Bundle", "visitors", "visitors"),
    "WEB_BAQATI_POSTPAID_12": ("Bundle", "postpaid", "mass"),
    "WEB_BAQATI_POSTPAID_NO": ("Bundle", "postpaid", "mass"),
}
COMPETITOR_SOURCES = [
    # (operator, page type, url)
    ("Omantel", "prepaid", "https://www.omantel.om/en/personal/mobile-prepaid"),
    ("Omantel", "prepaid", "https://www.omantel.om/en/personal/mobile-prepaid/plan-hayyak"),
    ("Omantel", "data", "https://www.omantel.om/en/personal/mobile-prepaid/plan-hayyak/add-on-mobile-internet"),
    ("Omantel", "data", "https://www.omantel.om/en/personal/mobile-prepaid/plan-hayyak/add-on-unlimited-social"),
    ("Omantel", "voice", "https://www.omantel.om/en/personal/mobile-prepaid/plan-hayyak/add-on-national-calls"),
    ("Omantel", "international", "https://www.omantel.om/en/personal/mobile-prepaid/plan-hayyak/add-on-international-calls"),
    ("Omantel", "postpaid", "https://www.omantel.om/en/personal/mobile-postpaid/plan-new-baqati"),
    ("Omantel", "postpaid", "https://www.omantel.om/en/personal/mobile-postpaid/plan-baqati-data-only"),
    ("Vodafone Oman", "prepaid", "https://www.vodafone.om/plans/prepaid"),
    ("Vodafone Oman", "postpaid", "https://www.vodafone.om/plans/post-pay"),
    ("Vodafone Oman", "students", "https://www.vodafone.om/plans/students"),
    ("Vodafone Oman", "visitors", "https://www.vodafone.om/plans/visitors"),
]
