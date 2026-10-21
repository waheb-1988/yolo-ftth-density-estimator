"""The agentic NBO RAG pipeline: profile -> retrieve -> reason -> rerank -> assemble."""
from __future__ import annotations

import json
import time

from nbx.core import collab_filter, config, knowledge_graph
from nbx.core.data_simulator import detect_need
from nbx.llms import llm_client
from nbx.agents.retriever import HybridRetriever
from nbx.agents.reranker import rerank
from nbx.vectorstores import get_store

SYSTEM_PROMPT = (
    "You are an NBO recommendation engine for Ooredoo Oman.\n"
    "Given the customer profile and retrieved offer context,\n"
    "reason step by step about which 4 offers best match this customer.\n"
    "Consider: need signal, usage pattern, price fit, cohort behavior.\n"
    "Return ONLY valid JSON — no prose before or after."
)

CHANNEL_FALLBACK = {
    "REACTIVATION": "OUTBOUND_CALL",
    "RENEWAL": "SMS",
    "INTERNATIONAL": "APP",
    "ROAMING": "APP",
    "DATA_UPSELL": "APP",
    "VOICE_UPSELL": "SMS",
    "CHANNEL_MIGRATION": "APP",
    "GENERAL_NBO": "APP",
}

OUTPUT_SCHEMA_HINT = """Return exactly this JSON shape (no other text):
{
  "channel_suggestion": "APP" | "SMS" | "OUTBOUND_CALL" | "CALL_CENTER" | "USSD",
  "reasoning": "<2-4 sentences explaining your overall reasoning>",
  "offers": [
    {"offer_name": "<program_rule from context>", "llm_score": <0.0-1.0>, "why": "<1 sentence>"},
    {"offer_name": "<program_rule from context>", "llm_score": <0.0-1.0>, "why": "<1 sentence>"},
    {"offer_name": "<program_rule from context>", "llm_score": <0.0-1.0>, "why": "<1 sentence>"},
    {"offer_name": "<program_rule from context>", "llm_score": <0.0-1.0>, "why": "<1 sentence>"}
  ]
}"""


def _load_json(path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_customer(msisdn: int) -> dict | None:
    customers = _load_json(config.CUSTOMERS_FILE)
    return next((c for c in customers if c["msisdn"] == msisdn), None)


def _offers_by_name() -> dict[str, dict]:
    offers = _load_json(config.OFFERS_FILE)
    return {o["program_rule"]: o for o in offers}


def _build_queries(customer: dict, detected_need: str) -> list[str]:
    return [
        f"{customer['dv_segment']} {customer['usage_type']} {detected_need} offers",
        f"customers like {customer['dv_segment']} purchase history",
        f"{detected_need} best offers {customer['nationality']}",
    ]


def _graph_hits_as_retrieved(need_hits: list[dict]) -> list[dict]:
    """Reshapes knowledge_graph.need_based_offers() output into the same
    hit shape agents/retriever.py's HybridRetriever produces ({id, text,
    type, metadata, retrieval_score}), so the LLM prompt block and
    agents/reranker.py's retrieval_score extraction work unchanged
    regardless of which retrieval_mode produced the candidates. The "text"
    here is assembled from catalog attributes, not retrieved via embedding
    similarity -- retrieval_score is a graph signal (peer purchases /
    segment affinity), not cosine similarity."""
    out = []
    for offer in need_hits:
        text = (
            f"{offer['offer_name']} ({offer['category']}): {offer['description']} "
            f"Price {offer['price_omr']} OMR, data {offer['data_gb']}, voice {offer['voice_mins']} minutes, "
            f"validity {offer['validity_days']} days. Graph signal: {offer['basis']}."
        )
        out.append(
            {
                "id": f"offer::{offer['program_rule']}",
                "text": text,
                "type": "offer",
                "metadata": offer,
                "retrieval_score": offer["graph_retrieval_score"],
            }
        )
    return out


def _build_user_prompt(
    customer: dict, detected_need: str, retrieved: list[dict], graph_hits: list[dict], cf_hits: list[dict]
) -> str:
    context_lines = []
    for i, hit in enumerate(retrieved, start=1):
        context_lines.append(f"[{i}] ({hit['type']}) {hit['text']}")
    context_block = "\n".join(context_lines) if context_lines else "(no context retrieved)"

    graph_lines = [f"- {g['program_rule']}: {g['reason']}" for g in graph_hits]
    graph_block = "\n".join(graph_lines) if graph_lines else "(no cohort purchase history found)"

    cf_lines = [f"- {c['program_rule']}: score {c['cf_score']:.2f} ({c['cf_reason']})" for c in cf_hits]
    cf_block = "\n".join(cf_lines) if cf_lines else "(no behavioral model signal for this customer)"

    return (
        f"CUSTOMER PROFILE:\n{json.dumps(customer, indent=2)}\n\n"
        f"DETECTED NEED: {detected_need}\n\n"
        f"RETRIEVED CONTEXT:\n{context_block}\n\n"
        f"COHORT PURCHASE GRAPH (what similar customers actually bought):\n{graph_block}\n\n"
        f"BEHAVIORAL MODEL SIGNAL (implicit-feedback collaborative filtering, monthly non-voice offers only):\n{cf_block}\n"
        f"Use this signal as evidence, not a verdict — weigh it against the retrieved context and cohort graph above, "
        f"and say in your reasoning if you're overriding it.\n\n"
        f"{OUTPUT_SCHEMA_HINT}"
    )


async def run_pipeline(
    msisdn: int,
    llm: str | None = None,
    vectorstore: str | None = None,
    embedding_model: str | None = None,
    retrieval_mode: str | None = None,
    need_override: str | None = None,
) -> dict:
    """retrieval_mode picks how step 2 answers "what offers fit this need":
      - "graph" (default): pure relational traversal over
        core.knowledge_graph.need_based_offers() -- need -> category ->
        candidate offers, scored by real peer purchases / segment affinity.
        No embeddings, no vector store call. Matches production telco data,
        which has no free text to search (see core/knowledge_graph.py).
      - "vector": the original path -- agents/retriever.py's HybridRetriever
        over the chosen embedded vector store (dense + sparse text match).
    Either way, cohort purchase graph and collaborative filtering (step 2's
    other two signals) run the same; only the "retrieval_score" source
    changes.
    """
    pipeline_start = time.time()
    llm_name = llm or config.DEFAULT_LLM
    mode = retrieval_mode or config.DEFAULT_RETRIEVAL_MODE

    # --- Step 1: Profile analysis -------------------------------------------------
    t0 = time.time()
    customer = _get_customer(msisdn)
    if customer is None:
        raise ValueError(f"No customer found for msisdn={msisdn}")
    # need_override lets the NBA contextual bandit (agents/nba_agent.py) steer
    # NBO toward a specific category when its chosen action implies one
    # (e.g. UPSELL DATA / CROSS SELL VOICE) -- overrides the deterministic
    # 8-signal cascade for this one request; detect_need() still runs so the
    # override is visible against what NBO would have picked on its own.
    need = need_override or detect_need(customer)
    profile_ms = int((time.time() - t0) * 1000)

    # --- Step 2: "what fits this need" (graph or vector) + cohort graph + -----------
    #             collaborative-filtering retrieval (implicit BPR, LightFM-equivalent)
    t0 = time.time()
    store = None
    store_key = None
    if mode == "vector":
        store_key = vectorstore or config.DEFAULT_VECTORSTORE
        store = get_store(store_key, embedding_model)
        queries = _build_queries(customer, need)
        retrieved = HybridRetriever(store).retrieve(queries, top_k=10)
    else:
        need_hits = knowledge_graph.need_based_offers(customer, need, top_k=10)
        retrieved = _graph_hits_as_retrieved(need_hits)

    graph_hits = knowledge_graph.cohort_offers(customer, top_k=10)
    graph_scores = {g["program_rule"]: g["graph_score"] for g in graph_hits}
    cf_hits = collab_filter.score_customer(customer, top_k=10)
    cf_scores = {c["program_rule"]: c["cf_score"] for c in cf_hits}
    retrieval_ms = int((time.time() - t0) * 1000)

    # --- Step 3: LLM reasoning (agentic) --------------------------------------------
    t0 = time.time()
    user_prompt = _build_user_prompt(customer, need, retrieved, graph_hits, cf_hits)
    resolved_model = await llm_client.resolve_model(llm_name)

    async def _call(prompt: str) -> tuple[dict, dict | None]:
        result = await llm_client.generate(model=resolved_model, prompt=prompt, system=SYSTEM_PROMPT, force_json=True)
        return result, llm_client.extract_json(result["content"])

    llm_result, parsed = await _call(user_prompt)
    llm_offers = parsed.get("offers", []) if isinstance(parsed, dict) else []
    retried = False

    # format=json grammar-constrains syntactically valid JSON, but not that
    # it matches OUR schema -- a model can return valid, parseable JSON with
    # no "offers" key at all (this is what was actually happening, not a raw
    # parse failure -- the old message said "could not be parsed as JSON"
    # even when parsing succeeded and only the schema didn't match). One
    # retry with an explicit reminder of the required shape resolves most of
    # these without falling all the way back to CF/retrieval/graph backfill.
    if not llm_result.get("error") and not llm_offers:
        retried = True
        retry_prompt = (
            f"{user_prompt}\n\nYour previous response did not include a valid \"offers\" array of "
            f"exactly 4 items. Return ONLY the JSON object described above, with a non-empty \"offers\" array."
        )
        llm_result, parsed = await _call(retry_prompt)
        llm_offers = parsed.get("offers", []) if isinstance(parsed, dict) else []

    llm_reasoning = parsed.get("reasoning", "") if isinstance(parsed, dict) else ""
    llm_channel = parsed.get("channel_suggestion") if isinstance(parsed, dict) else None

    if llm_result.get("error"):
        llm_reasoning = f"LLM call failed ({llm_result['error']}); falling back to CF/retrieval/graph ranking."
    elif not llm_offers:
        raw_preview = (llm_result.get("content") or "").strip()[:160].replace("\n", " ")
        why_empty = "returned valid JSON with no usable \"offers\" array" if parsed is not None else "did not return valid JSON"
        llm_reasoning = (
            llm_reasoning
            or f"LLM {why_empty} after {'a retry' if retried else '1 attempt'} "
               f"(raw: \"{raw_preview}\"); falling back to CF/retrieval/graph ranking."
        )
    llm_ms = int((time.time() - t0) * 1000)

    # --- Step 4: Reranking (LLM + collab-filter + retrieval + graph + price fit) ----
    t0 = time.time()
    offers_by_name = _offers_by_name()
    recommendations = rerank(llm_offers, retrieved, customer, offers_by_name, graph_scores, cf_scores)
    rerank_ms = int((time.time() - t0) * 1000)

    channel_suggestion = llm_channel or CHANNEL_FALLBACK.get(need, customer.get("preferred_recharge_channel", "APP"))
    pipeline_time_ms = int((time.time() - pipeline_start) * 1000)

    # --- Step 5: Response assembly -----------------------------------------------------
    return {
        "msisdn": msisdn,
        "detected_need": need,
        "retrieval_mode_used": mode,
        "vector_store_used": store_key if mode == "vector" else "none (graph mode)",
        "embedding_model_used": store.embedding_model if mode == "vector" else "none (graph mode)",
        "llm_used": llm_name,
        "pipeline_time_ms": pipeline_time_ms,
        "step_times": {
            "profile_ms": profile_ms,
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "rerank_ms": rerank_ms,
        },
        "recommendations": recommendations,
        "channel_suggestion": channel_suggestion,
        "llm_reasoning": llm_reasoning,
        "retrieved_context_count": len(retrieved),
        "graph_cohort_hits": len(graph_hits),
        "cf_candidate_count": len(cf_hits),
    }
