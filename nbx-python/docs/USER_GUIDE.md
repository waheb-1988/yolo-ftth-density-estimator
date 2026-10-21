# User guide: the web UI

Start the app (`python run.py start`) and open **http://localhost:7860**.
The UI talks to the API on port 8000, so both must be running. `run.py start`
starts both.

Customers are identified by **MSISDN** (in the sample data, the numbers 1–50).

---

## Home

Portfolio overview: customer count, revenue at risk, risk mix, recent
decisions, and shortcuts to the two recommenders. The header shows the LLM
status. On the offline backend it reads `offline · fallback mode`.

## Agentic Best Action Recommender (step 1)

Chooses **one action** for a customer: `RETAIN`, `MAINTAIN`, `UPSELL`,
`UPSELL DATA`, `UPSELL VOICE`, `CROSS SELL DATA` or `CROSS SELL VOICE`.

1. Type an MSISDN (e.g. `15`), or pick a customer from the list. The list is
   sorted by churn risk, highest first.
2. Click **Recommend best action**.
3. Read the result:
   - the live **trace**: every tool the agent called (risk scores, usage
     trends, bandit, contact policy, competitor prices) and what it returned;
   - the **decision**: action, channel, send window, confidence;
   - the **critic's verdict** and any automatic fixes;
   - the **customer message**: SMS in English, Arabic and the customer's own
     language, plus a call-centre script;
   - the **explanation**: which signals drove the action and why the
     alternatives were rejected.
4. Click **Step 2 → Recommend offers for this action** to get offers that serve
   this decision, without re-running step 1.

**Batch:** paste several MSISDNs (`15, 7, 1, 31`) and click **Recommend for
all**. You get a CSV to download. The UI caps a batch. For full runs use
`python -m nbx.scripts.batch_score`.

## Agentic Offer Recommender (steps 1 + 2)

Runs the action decision, then the **offer specialist**: it searches the
catalog, collaborative filtering and the cohort graph; checks eligibility;
simulates the expected value; and compares each candidate with the closest
Omantel and Vodafone Oman plans. Same inputs and batch mode as above.

## Competitor Intelligence

- Browse the competitor plan snapshot by **operator** and **segment**.
- **Benchmark one of our offers**: pick an offer and click **Compare** to see
  the like-for-like competitor plans, with price differences.
- **Semantic search** over the competitor pages (needs `python run.py setup`).
- **Re-scrape now** fetches the public pages again (needs internet, about 30 s).

## Business Rules

The offer rules (`nbx/agents/rules_engine.py`) and the action policy, read live
from the running code.

## Graph Insights

For one MSISDN: **Cohort offers** (what similar customers bought), **Similar
customers**, and **Graph stats** (node and edge counts).

## Customer lookup

Profile and purchase history for one MSISDN.

## Benchmark LLMs

Runs the offer pipeline on every installed LLM and ranks them. It only works
with a real LLM backend (Ollama or vLLM). On the offline backend no models are
installed, so there is nothing to compare.

## Vector store compare

Runs one query on all 5 vector stores side by side (results, scores,
latency), and sets the active store. Needs `python run.py setup`. Stores that
are not installed show as unavailable.

## Upload & Setup / Upload NBA Data

Replace the dataset with your own JSON files. Each step (validate → write →
index → graph → readiness) is shown as it runs, and **Retry** resumes from the
step that failed. **⚡ Quick load from data/** re-ingests the files already in
`data/`. File formats: [DATA.md](DATA.md).

## System status

LLM backend, vector-store status, and recent API calls. **Index this embedding
model** builds a parallel index for another embedding model (the Qwen3 models
need Ollama).
