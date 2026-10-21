"""Pydantic request/response schemas shared across the API and agent pipeline."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain records
# ---------------------------------------------------------------------------
class Customer(BaseModel):
    msisdn: int
    tenure_days: int
    arpu: float
    dv_segment: str
    revenue_segment: str
    rfm_score: int
    usage_type: str
    data_usage_tier: str
    activity_level: str
    is_intl_caller: bool
    is_roamer: bool
    days_since_last_purchase: int
    days_since_last_network_activity: int
    avg_offer_price: float
    preferred_recharge_channel: str
    data_volume_mb_p1: float
    voice_traffic_min_p1: float
    active_days_count_p1: int
    favorite_category: str
    last_product_name: str
    nationality: str
    nwk_type: str


class Offer(BaseModel):
    program_rule: str
    offer_name: str
    category: str
    offer_type: str
    price_omr: float
    data_gb: str
    voice_mins: str
    validity_days: int
    is_data: bool
    is_voice: bool
    is_intl: bool
    is_roaming: bool
    discount_pct: float
    description: str


class Purchase(BaseModel):
    transaction_id: str
    msisdn: int
    purchase_date: str
    offer_name: str
    program_rule: str
    category: str
    price_omr: float
    purchase_count: int


class Segment(BaseModel):
    """Mirrors the shape actually consumed by knowledge_graph.py / seed_data.py
    (core/data_simulator.py::generate_segments()). No richer segment schema is
    specified anywhere in the codebase today — TODO: extend if real segment
    uploads carry more fields than this."""

    segment: str
    typical_behavior: str


# ---------------------------------------------------------------------------
# NBA source data -- 7 models, one per file in core.config.NBA_SOURCE_FILE_ATTRS,
# split to mirror the production Vowpal Wabbit script's own per-Dataiku-
# dataset boundaries rather than one merged blob. core/nba_bandit.py's
# _load_scores() joins these (plus revenue_history.json) back into one
# per-customer row and computes the 15 var_rev/voice/data_* features at that
# merge time -- production runs its own calcul_evol() the same way, after
# merging, not before, so those derived features have no upload model here.
# ---------------------------------------------------------------------------
class NBAChurnScore(BaseModel):
    """Mirrors DAIM_CHURN_GLOBAL_SCORING_TD + DAIM_EARLY_CHURN_0_30/30_60
    (production concatenates these 3 into one churn_df before this point)
    + CHURN_ONE_DAY_ACTIVE_107_SCORING."""

    msisdn: int
    proba_churn: float
    proba_churn_1d: float


class NBAReactivationScore(BaseModel):
    """Mirrors DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_TD +
    PROPENSITY_REACTIVATION_15D_401_SCORING."""

    msisdn: int
    proba_react_30: float
    proba_react_15: float


class NBAAppetencyScore(BaseModel):
    """Mirrors APPETENCY_DATA_204_SCORING_TD + APPETENCY_VOICE_204_SCORING_TD."""

    msisdn: int
    proba_app_data: float
    proba_app_voice: float


class NBAFragilityScore(BaseModel):
    """Mirrors DAIM_FRAGILITY_DATA_SCORING_TD + DAIM_FRAGILITY_VOICE_SCORING_TD."""

    msisdn: int
    proba_frag_data: float
    proba_frag_voice: float


class NBARevDecreaseScore(BaseModel):
    """Mirrors REV_DECREASE_204_SCORING_TD."""

    msisdn: int
    proba_rev_decr: float


class NBAActivityUsage(BaseModel):
    """Mirrors activities_w1-4_auto_hdfs + activities_m1-3_auto_hdfs -- raw
    per-period usage, BEFORE the % change calculation (that's computed at
    merge time, not stored)."""

    msisdn: int
    local_voice_min_w1: float
    local_voice_min_w2: float
    local_voice_min_w3: float
    local_voice_min_w4: float
    local_voice_min_m1: float
    local_voice_min_m2: float
    local_voice_min_m3: float
    data_volume_mb_w1: float
    data_volume_mb_w2: float
    data_volume_mb_w3: float
    data_volume_mb_w4: float
    data_volume_mb_m1: float
    data_volume_mb_m2: float
    data_volume_mb_m3: float


class NBATargeting(BaseModel):
    """Mirrors targeting_auto_hdfs + targeting_dataiku_auto_hdfs (SEGMENT) +
    party_auto_hdfs (NATIONALITY -> nationality_group) + subs_btl_auto_hdfs
    (REWARD ground truth), plus the derived action/cost/propensity
    (dict_segm_actions / segment_prop_dict, see core.config.NBA_*).
    excluded/action/propensity apply to Nursering (tenure < 45 days)
    customers, who never enter bandit training."""

    msisdn: int
    nationality_group: str
    nba_segment: str
    excluded: bool
    action: Optional[str] = None
    reward: int
    cost: int
    propensity: Optional[float] = None


# ---------------------------------------------------------------------------
# Pipeline / agent schemas
# ---------------------------------------------------------------------------
class Recommendation(BaseModel):
    rank: int
    offer_name: str
    category: str
    price_omr: float
    data_gb: str
    voice_mins: str
    validity_days: int
    final_score: float
    llm_score: float
    cf_score: float
    retrieval_score: float
    graph_score: float
    why: str


class StepTimes(BaseModel):
    profile_ms: int
    retrieval_ms: int
    llm_ms: int
    rerank_ms: int


class RecommendResponse(BaseModel):
    msisdn: int
    detected_need: str
    retrieval_mode_used: str
    vector_store_used: str
    embedding_model_used: str
    llm_used: str
    pipeline_time_ms: int
    step_times: StepTimes
    recommendations: list[Recommendation]
    channel_suggestion: str
    llm_reasoning: str
    retrieved_context_count: int
    graph_cohort_hits: int
    cf_candidate_count: int


# ---------------------------------------------------------------------------
# API request bodies
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    msisdn: int
    llm: Optional[str] = None
    vectorstore: Optional[str] = None
    embedding_model: Optional[str] = None
    retrieval_mode: Optional[str] = None  # "graph" (default) | "vector" -- see core/config.py


class BenchmarkRequest(BaseModel):
    msisdn: int
    vectorstore: Optional[str] = None


class NBADecideRequest(BaseModel):
    msisdn: int
    llm: Optional[str] = None
    vectorstore: Optional[str] = None
    embedding_model: Optional[str] = None
    retrieval_mode: Optional[str] = None  # "graph" (default) | "vector" -- see core/config.py


class VectorStoreSwitchRequest(BaseModel):
    store: str


class VectorStoreCompareRequest(BaseModel):
    query: str
    top_k: int = 5


# ---------------------------------------------------------------------------
# API response bodies
# ---------------------------------------------------------------------------
class ModelBenchmarkResult(BaseModel):
    llm: str
    detected_need: str
    recommendations: list[Recommendation]
    avg_final_score: float
    pipeline_time_ms: int
    valid_json: bool


class BenchmarkResponse(BaseModel):
    msisdn: int
    results: list[ModelBenchmarkResult]
    winner: str
    ranking: list[str]


class SeedResponse(BaseModel):
    customers: int
    offers: int
    purchases: int
    indexed: dict[str, int]


class IndexEmbeddingRequest(BaseModel):
    embedding_model: str


class IndexEmbeddingResponse(BaseModel):
    embedding_model: str
    indexed: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool  # = configured LLM backend reachable (see core/readiness.py)
    vectorstores: dict[str, str]
    loaded_models: list[str]
    llm_backend: str = "ollama"
    llm_backend_url: str = ""


class ModelInfo(BaseModel):
    rank: int
    name: str
    display: str
    strength: str
    recommended: bool
    available: bool


class EmbeddingModelInfo(BaseModel):
    key: str
    display: str
    backend: str
    dimension: int
    strength: str
    recommended: bool
    available: bool


class VectorStoreInfo(BaseModel):
    key: str
    display: str
    status: str
    document_count: int
    index_size_bytes: int


class RuleApplication(BaseModel):
    rule_name: str
    effect: str


class BanditContext(BaseModel):
    nba_segment: str
    historical_action: Optional[str] = None
    need_override_applied: Optional[str] = None
    llm_overrode_bandit: Optional[bool] = None


class NBADecideResponse(BaseModel):
    """agents/final_agent.py's response -- composes agents/nba_agent.py
    (bandit + LLM -> action) and agents/nbo_agent.py (retrieval/CF/graph +
    LLM -> offers), then applies agents/rules_engine.py."""

    msisdn: int
    action_type: str
    is_push_action: bool
    bandit_action_probability: Optional[float] = None
    nba_llm_reasoning: Optional[str] = None
    bandit_context: Optional[BanditContext] = None
    rules_applied: list[RuleApplication]
    nbo: RecommendResponse


class NBAActionRequest(BaseModel):
    msisdn: int
    llm: Optional[str] = None


class NBAActionResponse(BaseModel):
    """agents/nba_agent.py::decide_action()'s own response -- NBA alone,
    no NBO call. See core/nba_bandit.py for bandit_action's origin and
    agents/nba_agent.py's SYSTEM_PROMPT for how the LLM can override it."""

    msisdn: int
    action_type: str
    final_action: Optional[str] = None
    bandit_action: Optional[str] = None
    bandit_action_probability: Optional[float] = None
    llm_reasoning: str
    llm_overrode_bandit: bool
    is_push_action: bool
    bandit_context: Optional[dict] = None
    step_times: dict


class GraphStatsResponse(BaseModel):
    nodes: int
    edges: int
    by_kind: dict[str, int]


class SimilarCustomer(BaseModel):
    msisdn: int
    shared_segment: str
    shared_offer_count: int


class CohortOfferHit(BaseModel):
    program_rule: str
    graph_score: float
    peer_count: int
    reason: str


class CompareResultRow(BaseModel):
    store: str
    results: list[dict]
    time_ms: int
    top_result: Optional[str] = None


# ---------------------------------------------------------------------------
# Upload & Setup ingestion pipeline (agents/ingestion_pipeline.py)
# ---------------------------------------------------------------------------
class UploadStartResponse(BaseModel):
    job_id: str


class UploadStepStatus(BaseModel):
    name: str
    display: str
    status: str  # "pending" | "running" | "done" | "failed"
    message: str
    count: Optional[int] = None
    error: Optional[str] = None


class UploadJobStatus(BaseModel):
    job_id: str
    overall_status: str  # "pending" | "running" | "done" | "failed"
    steps: list[UploadStepStatus]


class AgenticRequest(BaseModel):
    msisdn: int
    scope: str = "full"  # "nba" (action only) | "nbo" / "full" (NBA first, then offers for that action)
    llm: Optional[str] = None
    use_last_nba: bool = False  # offers only: reuse this customer's last agentic NBA decision


class CopilotRequest(BaseModel):
    question: str
    history: list[dict] = Field(default_factory=list)  # [{"role": "user"|"assistant", "content": str}]
    llm: Optional[str] = None
