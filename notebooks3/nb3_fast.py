"""Cells of notebooks3 (FAST_MODE): the notebooks2 agents notebook, with agents 3 / 4 asked once per customer
PROFILE (the facts rounded into bands) instead of once per customer. Every other step is the notebooks2 cell."""

TITLE = r'''# {num} · FAST_MODE: all the agents on the GPU ({server}), one LLM answer per customer profile

The same run as `notebooks2/{nb2}` (train on `*_auto` at T, score on `*_score` at D, every agent on **your laptop
GPU** through **{server}**, only `{url}` is called), with one change: **agents 3 (NBA) and 4 (NBO) answer once per
customer profile, not once per customer.**

- A **profile** is the customer's facts rounded into bands: segment, value band, usage, revenue band and trend, data /
  voice bands, active days, tenure, churn and appetency bands, last week. Customers with the same profile get the
  same prompt, so they need only one LLM answer.
- **Step 5b** counts the profiles at four levels (`fine`, `medium`, `coarse`, `minimal`) **before any LLM call**, gives the GPU
  time of each with the speed G2 measured, and picks the finest that fits `TIME_BUDGET_MIN` (`PROFILE_LEVEL = "auto"`).
- **Agent 3** still draws the action **per customer** from the profile's scores (ε draw, the logged propensity is
  the customer's own), so the uplift stays measurable.
- **Agent 4**: one answer per (profile, action, short list of offers, price anchor band). The passes (price band,
  one per family, EXT-6) still choose per customer.
- **Quality check** (`FAST_CHECK_N`): a sample of customers is also asked with their **exact** facts; step 6 prints how
  often the top action agrees.
- `PROFILE_LEVEL = "off"` gives exactly the notebooks2 run (one call per customer).

| Step | Agent | Runs on | What it decides |
|---|---|---|---|
| G1 / G2 | — | GPU | the GPU, the model, the speed of one call |
| 1 – 5 | Agents 1, 2, 2b | CPU + GPU | as notebooks2: the data check, segments (T), the agent dictionary, the scoring view (D), the panel |
| **5b** | — | CPU | **the customer profiles, their count and the GPU time, before any call** |
| 6 | **Agent 3** NBA | GPU | one answer per T2C profile, the action drawn per customer |
| 7 | **Agent 4** NBO + **6B** | GPU + CPU | one answer per offer profile; the challenger rules choose |
| 8 – 11 | Agents 7, 9, 8 | CPU + GPU | as notebooks2: code 5, the auditor, the strategy, the final file |
| G3 | — | — | was this an LLM + GPU run? plus the FAST_MODE figures |

**Run order:** `notebooks2/02_prepare_week` → **this notebook**. Use the same `RUN_DATE` and `TEST_10K` in both.
Start Jupyter with `python -m jupyterlab`.'''

FAST_SETTINGS = r'''
# --- which Target 2 lists to score -------------------------------------------------------------------------------------
SCORE_GROUPS = ["t2c", "t2f"]   # ["t2c"] = group a only (target2_a, ~50k) | ["t2f"] = group b only (target2_b, ~50k)
                                #   | ["t2c", "t2f"] = both (~100k). The other list is not read, not scored, not in the file.

# --- FAST_MODE (notebooks3): agents 3 / 4 answer once per customer profile ------------------------------------------
PROFILE_LEVEL = "auto"          # "fine" | "medium" | "coarse" | "minimal" | "off" (= one call per customer, as notebooks2)
                                #   "auto" = the finest level whose estimate (step 5b) fits TIME_BUDGET_MIN
FAST_CHECK_N = 100              # agent 3: this many customers are also asked with their exact facts (0 = no check)
SHARED_LLM_CACHE = True         # keep the answers across weeks (data/runs/_llm_cache.jsonl, no MSISDN in it):
                                #   a profile already answered with the same model and prompt costs nothing
'''

TIME_SWAP = ("TIME_BUDGET_MIN = 20            # minutes for the per-customer agents 3 + 4; customers past it get the cell policy",
             "TIME_BUDGET_MIN = 45            # GPU minutes for agents 3 + 4; step 5b picks the profile level that fits, profiles past it get the cell policy")

STEP_5B = [("md", r'''## Step 5b · The customer profiles (FAST_MODE), before any LLM call

Each Target-2 customer's facts are rounded into bands, and the customers with the same rounded facts make one
**profile**. A profile is sent once, and its answer is used for every customer in it.

- **Bands:** revenue, data and voice use quantile bands over the target customers (shown as their value range, for
  example `revenue M1 2.10-3.40`). The probabilities use fixed bands (`0.2-0.4`), tenure and active days use fixed
  ranges, and the revenue trend is `down` / `flat` / `up` (M1 against the mean of M2 and M3, ±10 %).
- **Levels:** `fine` (5 bands, every fact), `medium` (3 bands, every fact), `coarse` (3 bands; leaves out data,
  voice, active days and tenure) and `minimal` (as coarse, also without the appetencies; last week is only
  "targeted, took an offer" / "targeted, no offer taken" / "not targeted").
- **Unit of work:** a profile × group (T2C: × the cell policy). It costs one NBA answer plus one NBO answer for T2C,
  and one NBO answer for T2F.
- **Budget:** profiles are taken in priority order (tier of the segment, then the biggest profiles first, with the
  groups interleaved) until `TIME_BUDGET_MIN`. Customers of the profiles past it get the cell policy.'''),
           ("code", r'''llm_budget = use("nbx.fsdr", "llm_budget")

VAL = "ACCESS_METHOD_VAL"
feat = pd.concat([pd.read_parquet(p, columns=[VAL, "PROBA_CHURN", "PROBA_APP_DATA", "PROBA_APP_VOICE", "PROBA_FRAG_DATA",
                                              "REV_M1", "REV_M2", "REV_M3", "DATA_VOLUME_MB_M1", "LOCAL_VOICE_MIN_M1",
                                              "ACTIVE_DAYS_COUNT_M1", "TENURE_DAYS"])
                  for p in sorted(FEAT_DIR.glob("*.parquet"))])          # scoring view (D) or training (T)
ctx = panel.merge(feat, on=VAL, how="left").merge(segs_score[[VAL, "W0_TARGETED", "W0_ACTION", "W0_REWARD", "S_AT_RISK"]],
                                                  on=VAL, how="left").drop_duplicates(VAL).set_index(VAL)
t2 = panel[panel["TARGET"] == AGENT_TARGET].drop_duplicates(VAL).assign(DECIDED_BY="-")
queue = llm_budget.llm_queue(t2)[VAL].tolist()                     # priority tier first, groups interleaved
T = ctx.loc[t2[VAL]]


def facts_exact(v):
    """The exact facts of one customer: the notebooks2 prompt (PROFILE_LEVEL "off" and the quality check)."""
    r = ctx.loc[v]
    return (f"segment {r['SEGMENT']}, value band {r['VALUE_BAND']}, usage {r['USAGE_TYPE']}, tenure {fmt(r['TENURE_DAYS'], 0)} days, "
            f"revenue M1/M2/M3 {fmt(r['REV_M1'])}/{fmt(r['REV_M2'])}/{fmt(r['REV_M3'])}, data {fmt(r['DATA_VOLUME_MB_M1'], 0)} MB, "
            f"voice {fmt(r['LOCAL_VOICE_MIN_M1'], 0)} min, active days {fmt(r['ACTIVE_DAYS_COUNT_M1'], 0)}, churn {fmt(r['PROBA_CHURN'])}, "
            f"data appetency {fmt(r['PROBA_APP_DATA'])}, voice appetency {fmt(r['PROBA_APP_VOICE'])}, "
            f"last week: {'targeted with ' + str(r['W0_ACTION']) + (', took an offer' if r['W0_REWARD'] == 1 else ', no offer taken') if r['W0_TARGETED'] == 1 else 'not targeted'}")


def _num(col):
    return pd.to_numeric(T[col], errors="coerce") if col in T.columns else pd.Series(np.nan, index=T.index)


def _qband(x, k, nd=2):
    """k quantile bands of x over the target customers, written as their range ('-' = missing)."""
    out = pd.Series("-", index=x.index, dtype=object)
    ok = x.notna()
    if not ok.any():
        return out
    e = np.unique(np.quantile(x[ok], np.linspace(0, 1, k + 1)))
    if len(e) < 2:
        out[ok] = fmt(float(e[0]), nd)
        return out
    j = np.clip(np.searchsorted(e, x[ok].to_numpy(), side="right") - 1, 0, len(e) - 2)
    labels = [f"{fmt(float(e[i]), nd)}-{fmt(float(e[i + 1]), nd)}" for i in range(len(e) - 1)]
    out[ok] = [labels[i] for i in j]
    return out


def _fband(x, edges, names):
    """Fixed bands: edges[i] <= x < edges[i + 1] -> names[i] ('-' = missing)."""
    out = pd.Series("-", index=x.index, dtype=object)
    ok = x.notna()
    j = np.clip(np.searchsorted(np.asarray(edges[1:-1], float), x[ok].to_numpy(), side="right"), 0, len(names) - 1)
    out[ok] = [names[i] for i in j]
    return out


def _pband(x, k):
    e = np.linspace(0, 1, k + 1)
    return _fband(x.clip(0, 1), e, [f"{e[i]:.1f}-{e[i + 1]:.1f}" for i in range(k)])


_m1, _prev = _num("REV_M1"), pd.concat([_num("REV_M2"), _num("REV_M3")], axis=1).mean(axis=1)
_ratio = _m1 / _prev.where(_prev > 0)
TREND = pd.Series(np.select([_ratio < 0.9, _ratio > 1.1, _ratio.notna()], ["down", "up", "flat"], "-"), index=T.index)
TREND[_ratio.isna() & (_m1 > 0)] = "new revenue"
_lw = np.where(_num("W0_TARGETED") == 1, "targeted with " + T["W0_ACTION"].astype(str)
               + np.where(_num("W0_REWARD") == 1, ", took an offer", ", no offer taken"), "not targeted")
LAST_WEEK = pd.Series(_lw, index=T.index)
LAST_WEEK_SHORT = pd.Series(np.where(_num("W0_TARGETED") == 1, np.where(_num("W0_REWARD") == 1, "targeted, took an offer",
                                                                         "targeted, no offer taken"), "not targeted"), index=T.index)
LEVELS = {   # bands per fact (0 / None = the fact is left out of the prompt)
    "fine":    {"rev": 5, "data": 5, "voice": 5, "proba": 5, "apps": True, "lw": "full",
                "days": ([0, 8, 16, 24, 1e9], ["0-7", "8-15", "16-23", "24+"]),
                "tenure": ([0, 90, 365, 1095, 1e9], ["under 3 months", "3-12 months", "1-3 years", "over 3 years"])},
    "medium":  {"rev": 3, "data": 3, "voice": 3, "proba": 3, "apps": True, "lw": "full",
                "days": ([0, 11, 21, 1e9], ["0-10", "11-20", "21+"]),
                "tenure": ([0, 180, 730, 1e9], ["under 6 months", "6-24 months", "over 2 years"])},
    "coarse":  {"rev": 3, "data": 0, "voice": 0, "proba": 3, "apps": True, "lw": "full", "days": None, "tenure": None},
    "minimal": {"rev": 3, "data": 0, "voice": 0, "proba": 3, "apps": False, "lw": "short", "days": None, "tenure": None},
}
LEVEL_ORDER = list(LEVELS)                             # finest first


def profile_facts(level):
    """The rounded facts of every target customer at this level (a Series indexed by the line)."""
    if level == "off":
        return pd.Series([facts_exact(v) for v in T.index], index=T.index)
    L = LEVELS[level]
    s = ("segment " + T["SEGMENT"].astype(str) + ", value band " + T["VALUE_BAND"].astype(str) + ", usage "
         + T["USAGE_TYPE"].astype(str) + ", revenue M1 " + _qband(_m1, L["rev"]) + " (trend " + TREND + ")")
    if L["data"]:
        s = s + ", data " + _qband(_num("DATA_VOLUME_MB_M1"), L["data"], 0) + " MB"
    if L["voice"]:
        s = s + ", voice " + _qband(_num("LOCAL_VOICE_MIN_M1"), L["voice"], 0) + " min"
    if L["days"]:
        s = s + ", active days " + _fband(_num("ACTIVE_DAYS_COUNT_M1"), *L["days"])
    if L["tenure"]:
        s = s + ", tenure " + _fband(_num("TENURE_DAYS"), *L["tenure"])
    s = s + ", churn " + _pband(_num("PROBA_CHURN"), L["proba"])
    if L["apps"]:
        s = s + ", data appetency " + _pband(_num("PROBA_APP_DATA"), L["proba"]) + ", voice appetency " + _pband(_num("PROBA_APP_VOICE"), L["proba"])
    return s + ", last week: " + (LAST_WEEK if L["lw"] == "full" else LAST_WEEK_SHORT)


def units(level):
    """One row per target customer: its facts at this level and its unit of work (profile x group x cell policy)."""
    u = pd.DataFrame({"FACTS": profile_facts(level), "GROUP_KEY": T["GROUP_KEY"], "CELL": T["ACTION_AGENT"].astype(str),
                      "SEGMENT": T["SEGMENT"].astype(str)}, index=T.index)
    u["ACT"] = np.where(u["GROUP_KEY"] == "t2c", u["CELL"], "any")
    u["UNIT"] = u.groupby(["FACTS", "GROUP_KEY", "ACT"], sort=False).ngroup()
    return u


def cost(u):
    """Estimated GPU seconds of the units: T2C = NBA + NBO (the epsilon draw adds a few NBO answers), T2F = NBO."""
    n_c = u.loc[u["GROUP_KEY"] == "t2c", "UNIT"].nunique()
    n_f = u.loc[u["GROUP_KEY"] != "t2c", "UNIT"].nunique()
    return n_c, n_f, n_c * (s_nba + s_nbo * (1 + EPSILON)) + n_f * s_nbo


n_cust = len(T)
n_t2c = int((T["GROUP_KEY"] == "t2c").sum())
rows, UNITS = [], {}
for lv in LEVEL_ORDER:
    UNITS[lv] = units(lv)
    n_c, n_f, sec = cost(UNITS[lv])
    rows.append({"level": lv, "customers": n_cust, "T2C profiles": n_c, "T2F profiles": n_f,
                 "customers per profile": round(n_cust / max(n_c + n_f, 1), 1), "estimated GPU min": round(sec / 60, 1)})
_off = n_t2c * (s_nba + s_nbo) + (n_cust - n_t2c) * s_nbo
rows.append({"level": "off", "customers": n_cust, "T2C profiles": n_t2c, "T2F profiles": n_cust - n_t2c,
             "customers per profile": 1.0, "estimated GPU min": round(_off / 60, 1)})
LEVEL_TABLE = pd.DataFrame(rows)
show(LEVEL_TABLE, n=4, title=f"profiles of the {n_cust:,} Target-2 customers ({MODEL}, PARALLEL {PARALLEL}, speed of G2)")

if PROFILE_LEVEL == "auto":
    LEVEL = next((r["level"] for r in rows[:len(LEVEL_ORDER)] if r["estimated GPU min"] <= TIME_BUDGET_MIN), LEVEL_ORDER[-1])
else:
    assert PROFILE_LEVEL in LEVEL_ORDER + ["off"], f"PROFILE_LEVEL {PROFILE_LEVEL!r}"
    LEVEL = PROFILE_LEVEL
PROF = UNITS[LEVEL] if LEVEL in UNITS else units("off")

# --- the budget: units in priority order until TIME_BUDGET_MIN (or MAX_LLM_LINES customers) -------------------------
_rank = pd.Series(np.arange(len(queue)), index=queue)
U = PROF.assign(RANK=_rank.reindex(PROF.index).to_numpy()).groupby("UNIT").agg(
    n=("RANK", "size"), rank=("RANK", "min"), group=("GROUP_KEY", "first"), segment=("SEGMENT", "first"))
U["tier"] = U["segment"].map(llm_budget.TIER).fillna(len(llm_budget.PRIORITY_TIERS) + 1).astype(int)
U = U.sort_values(["tier", "n", "rank"], ascending=[True, False, True], kind="mergesort")
U["_k"] = U.groupby("group").cumcount()
U = U.sort_values(["_k", "group"], kind="mergesort")
U["sec"] = np.where(U["group"] == "t2c", s_nba + s_nbo * (1 + EPSILON), s_nbo)
keep = U["sec"].cumsum() <= TIME_BUDGET_MIN * 60
if MAX_LLM_LINES is not None:
    keep &= U["n"].cumsum() <= MAX_LLM_LINES
LLM_UNITS = set(U.index[keep])
LLM_LINES = set(PROF.index[PROF["UNIT"].isin(LLM_UNITS)])
FACTS = PROF["FACTS"]
print(f"PROFILE_LEVEL {PROFILE_LEVEL} -> {LEVEL}: {PROF['UNIT'].nunique():,} units of work for {n_cust:,} customers; "
      f"{len(LLM_UNITS):,} units = {len(LLM_LINES):,} customers fit {TIME_BUDGET_MIN} min "
      f"(about {U.loc[keep, 'sec'].sum() / 60:.1f} min); the rest ({n_cust - len(LLM_LINES):,}): cell policy")
show(PROF.loc[PROF["UNIT"].isin(LLM_UNITS)].groupby("UNIT").agg(customers=("FACTS", "size"), group=("GROUP_KEY", "first"),
                                                                 facts=("FACTS", "first"))
     .sort_values("customers", ascending=False), n=5, title="the biggest profiles (no MSISDN in a prompt)")

# --- the answers are kept across weeks: the same model + prompt = the same answer (temperature 0) -----------------
if SHARED_LLM_CACHE:
    _shared = config.FSDR_RUNS_DIR / "_llm_cache.jsonl"
    if _shared.exists():
        _n0 = len(_cache)
        for line in _shared.read_text(encoding="utf-8").splitlines():
            try:
                _row = json.loads(line)
                _cache.setdefault(_row["key"], _row["answer"])
            except (ValueError, KeyError):
                pass
        print(f"shared LLM cache: {len(_cache) - _n0:,} answers of earlier weeks loaded from {_shared.name}")
    LLM_CACHE = _shared                                # the new answers of agents 3 / 4 go there too''')]

STEP_6 = [("md", r'''## Step 6 · Agent 3 (③) NBA: one answer per T2C profile, the action drawn per customer

For each **T2C profile** of the budget, the LLM scores the allowed actions from the profile's facts. No MSISDN is sent.
- **Allowed:** the 7 actions, minus an upsell for an at-risk segment (H-06). The segment and the cell policy are part
  of the profile, so the allowed list is the same for every customer in it.
- **Choice, per customer:** p = (1 − ε)·score/Σscores + ε/k, one draw per customer with its own seed. The
  **propensity** p of the chosen action is the customer's own, so the uplift stays measurable.
- **Fallback:** a customer past the budget, or with an invalid answer, keeps the cell policy of agent 2b.
- **Quality check:** `FAST_CHECK_N` customers are also asked with their **exact** facts (the notebooks2 prompt), and
  the cell prints how often both give the same top action.'''),
          ("code", r'''def allowed_of(seg, cell):
    return [a for a in ACTIONS if not (seg in AT_RISK_SEGMENTS and a in UPSELL_ACTIONS and a != cell)]


SYS3 = ("You are the next-best-action agent of a telecom campaign engine. Score each allowed action 0-100 for this "
        "prepaid customer: the chance it makes the customer stay and spend more.")


def nba_job(facts_text, cell, allowed):
    return (SYS3, f"Customer: {facts_text}. Cell policy: {cell}. Allowed actions, in order: {allowed}. "
                  "Return one score per allowed action, in that order, and a short reason.",
            {"type": "object", "properties": {"scores": ints(len(allowed)), "reason": {"type": "string", "maxLength": 100}},
             "required": ["scores", "reason"]}, 120)


sel = PROF[PROF.index.isin(LLM_LINES) & (PROF["GROUP_KEY"] == "t2c")]
first = sel.groupby("UNIT", sort=False).head(1)
keys = [(u, allowed_of(s, c)) for u, s, c in zip(first["UNIT"], first["SEGMENT"], first["CELL"])]
answers = ask_many("Agent 3 NBA", [nba_job(f, c, al) for f, c, (_, al) in zip(first["FACTS"], first["CELL"], keys)])
ANS3 = {u: (al, ans) for (u, al), ans in zip(keys, answers)}


def ok3(ans, allowed):
    return bool(ans) and len(ans.get("scores", [])) == len(allowed)


rows = []
for v, unit, cell in zip(sel.index, sel["UNIT"], sel["CELL"]):
    allowed, ans = ANS3[unit]
    if not ok3(ans, allowed):
        rows.append({VAL: v, "ACTION": cell, "PROBA": 1.0, "DECIDED_BY": "fallback_cell_policy", "PROFILE": unit,
                     "REASON": "invalid answer"})
        continue
    s = np.clip(np.asarray(ans["scores"], float), 0, None)
    p = (1 - EPSILON) * (s / s.sum() if s.sum() > 0 else np.full(len(s), 1 / len(s))) + EPSILON / len(s)
    rng = np.random.default_rng([SEED, int(v) % 1_000_000_007])
    k = int(rng.choice(len(allowed), p=p / p.sum()))
    rows.append({VAL: v, "ACTION": allowed[k], "PROBA": round(float(p[k] / p.sum()), 6), "PROFILE": unit,
                 "DECIDED_BY": "llm_nba" if k == int(np.argmax(s)) else "llm_nba_sampled", "REASON": ans.get("reason", "")})
NBA = pd.DataFrame(rows, columns=[VAL, "ACTION", "PROBA", "DECIDED_BY", "PROFILE", "REASON"])
NBA.drop(columns="REASON").to_csv(AGENTS_DIR / "agent3_nba.csv", index=False)
if len(NBA):
    cmp = NBA.merge(ctx[["ACTION_AGENT"]], left_on=VAL, right_index=True)
    print(f"agent 3: {len(NBA):,} T2C customers from {len(ANS3):,} profile answers "
          f"({len(NBA) / max(len(ANS3), 1):.1f} customers per answer); same action as the cell policy for "
          f"{(cmp['ACTION'] == cmp['ACTION_AGENT']).mean():.0%}; {NBA['DECIDED_BY'].value_counts().to_dict()}")
    display(pd.crosstab(cmp["ACTION_AGENT"], cmp["ACTION"]).rename_axis(index="cell policy (2b)", columns="agent 3"))
    show(NBA, n=5, title="agent 3 decisions (first rows)")

# --- quality check: the profile's answer against the customer's exact facts ---------------------------------------
CHECK3 = {}
if FAST_CHECK_N and LEVEL != "off" and len(sel):
    smp = sel.sample(min(FAST_CHECK_N, len(sel)), random_state=SEED)
    exact = ask_many("Agent 3 check (exact facts)", [nba_job(facts_exact(v), c, ANS3[u][0])
                                                      for v, c, u in zip(smp.index, smp["CELL"], smp["UNIT"])])
    same, diff, n_ok = 0, [], 0
    for u, ex in zip(smp["UNIT"], exact):
        allowed, pa = ANS3[u]
        if ok3(ex, allowed) and ok3(pa, allowed):
            n_ok += 1
            a, b = np.asarray(pa["scores"], float), np.asarray(ex["scores"], float)
            same += int(np.argmax(a) == np.argmax(b))
            diff.append(float(np.abs(a - b).mean()))
    CHECK3 = {"customers": int(len(smp)), "compared": n_ok, "same_top_action": round(same / max(n_ok, 1), 3),
              "mean_score_gap": round(float(np.mean(diff)), 1) if diff else None}
    print(f"quality check ({LEVEL}): the profile and the exact facts give the same top action for "
          f"{CHECK3['same_top_action']:.0%} of {n_ok} customers (mean score gap {CHECK3['mean_score_gap']} points of 100)")
    if n_ok and CHECK3["same_top_action"] < 0.7:
        print("WARNING: the profiles lose too much of the customer's facts: set PROFILE_LEVEL to a finer level "
              "(or raise TIME_BUDGET_MIN so that 'auto' picks one), then run again from step 5b")

# --- guard: did the LLM really decide? -------------------------------------------------------------------------
if len(NBA):
    _fb = (NBA["DECIDED_BY"] == "fallback_cell_policy").mean()
    print(f"agent 3 by the LLM: {1 - _fb:.0%} of {len(NBA):,} T2C customers sent to it ({_fb:.0%} fallback)")
    if _fb > MAX_FALLBACK_SHARE:
        _msg = (f"agent 3: {_fb:.0%} of the answers failed or were invalid ({STATS.get('Agent 3 NBA', {}).get('last_error')}). "
                "Check the LLM server, lower PARALLEL, or pick a stronger MODEL; then run from G1 again")
        if STOP_ON_FALLBACK:
            raise SystemExit(_msg)
        print("WARNING:", _msg)
elif LLM_LINES:
    print("agent 3: no T2C customer in the LLM queue (only T2F lines?): agent 4 still scores the offers")''')]

STEP_7 = [("md", r'''## Step 7 · Agent 4 (④) NBO: one answer per (profile, action), with the challenger rules (Agent 6B)

The **rules first**: for each Target 2 customer the universe rules (CC-v1 / CF-v1: families, catalogue, the
customer's relative price band) give the allowed offers, and the best `NBO_SHORTLIST` per validity (by the rule
score) are short-listed.

- **One answer per (profile, action):** the customers of a profile with the same action (T2F: "any") are grouped.
  The LLM scores the offers that are most often on their short lists, `NBO_SHORTLIST` per validity. Every customer of
  the group gets those scores for the offers the customer is allowed; the other allowed offers rank after them (rule score × 0.01).
- **Why not per short list:** each customer's short list and its order depend on his own rule scores, so almost
  no two customers share one. Grouping by short list saves next to nothing (measured: 1.2 customers per answer).
- **Price:** the rules have already kept only offers in the customer's price band, so in profile mode the prompt gives
  no per-customer price anchor (the revenue band of the profile stands for it).
- The passes **still choose** the offers per customer (price band, one per family, EXT-6 for slots 5 / 6).
  Customers past the budget keep the rule score. `PROFILE_LEVEL = "off"` = the notebooks2 prompt, per customer.
- **Coverage** (printed): the share of the customers' own short-listed offers that the group's answer scored.'''),
          ("code", r'''NBA_ACTION = NBA.set_index(VAL)["ACTION"] if len(NBA) else pd.Series(dtype=object)
EXACT = LEVEL == "off"
SYS4 = ("You are the next-best-offer agent of a telecom campaign engine. Score each offer 0-100: the chance "
        "this prepaid customer buys it and it raises his spend without overpricing.")
NBO_PROFILES = {}


def _offer_text(rows, with_score):
    return "; ".join(f"{i}: {r.SERVICE} {r.GENEROSITY}, {int(r.VALIDITY)} days, price {r.PRICE:.2f}, "
                     f"{fmt(getattr(r, 'GB', None), 1)} GB, {fmt(getattr(r, 'MIN', None), 0)} min"
                     + (f", rule score {r.SCORE:.2f}" if with_score else "") for i, r in enumerate(rows.itertuples()))


def _job(user, n):
    return (SYS4, user, {"type": "object", "properties": {"scores": ints(n)}, "required": ["scores"]}, 40 + 5 * n)


def nbo(group_key, cands):
    """Agent 4, called by P6 with the candidates the rules allowed (both validities): a score per row, NaN = rule score."""
    out = pd.Series(np.nan, index=cands.index)
    mine = cands[cands[VAL].isin(LLM_LINES)]
    acc = NBO_PROFILES.setdefault(group_key, {"customers": 0, "answers": 0, "own_short": 0, "covered": 0})
    if not len(mine):
        return out
    short = mine.sort_values("SCORE", ascending=False, kind="mergesort").groupby([VAL, "VALIDITY"], group_keys=False).head(NBO_SHORTLIST)
    act = {v: (NBA_ACTION.get(v, ctx.loc[v, "ACTION_AGENT"]) if group_key == "t2c" else "any (free universe)")
           for v in short[VAL].unique()}
    if EXACT:                                       # notebooks2: one prompt per customer, with his anchor and rule scores
        work, jobs = [], []
        for v, c in short.groupby(VAL, sort=False):
            anchor = fmt(c["ANCHOR"].iloc[0]) if "ANCHOR" in c.columns else "-"
            work.append((mine.index[mine[VAL] == v], c.index))
            jobs.append(_job(f"Customer: {facts_exact(v)}. Action: {act[v]}. Price anchor {anchor}. Offers: "
                             f"{_offer_text(c, True)}. Return one score per offer, in order.", len(c)))
        for (all_idx, short_idx), ans in zip(work, ask_many("Agent 4 NBO", jobs)):
            if ans and len(ans.get("scores", [])) == len(short_idx):
                out.loc[all_idx] = cands.loc[all_idx, "SCORE"] * 0.01
                out.loc[short_idx] = np.asarray(ans["scores"], float) / 100.0
        acc["customers"] += len(work); acc["answers"] += len(work)
        acc["own_short"] += len(short); acc["covered"] += len(short)
        return out
    # profile mode: one answer per (profile, action) over the offers most short-listed by its customers
    short = short.assign(_K=short[VAL].map(lambda v: (FACTS[v], act[v])))
    keys, jobs, lists = [], [], []
    for k, s in short.groupby("_K", sort=False):
        top = (s.groupby(["VALIDITY", "BTL_NAME"]).agg(n=(VAL, "size"), sc=("SCORE", "mean")).reset_index()
                .sort_values(["VALIDITY", "n", "sc"], ascending=[True, False, False], kind="mergesort")
                .groupby("VALIDITY").head(NBO_SHORTLIST))
        rows = s.drop_duplicates(["VALIDITY", "BTL_NAME"]).set_index(["VALIDITY", "BTL_NAME"]).loc[
            list(zip(top["VALIDITY"], top["BTL_NAME"]))].reset_index()
        keys.append(k); lists.append(rows[["VALIDITY", "BTL_NAME"]])
        jobs.append(_job(f"Customer: {k[0]}. Action: {k[1]}. Offers (all in the customer's price band, best first by the "
                         f"rules): {_offer_text(rows, False)}. Return one score per offer, in order.", len(rows)))
    scored = []
    for k, lst, ans in zip(keys, lists, ask_many("Agent 4 NBO", jobs)):
        if ans and len(ans.get("scores", [])) == len(lst):
            scored.append(lst.assign(_K=[k] * len(lst), LLM=np.asarray(ans["scores"], float) / 100.0))
    m = mine.assign(_K=mine[VAL].map(lambda v: (FACTS[v], act.get(v))))
    if scored:
        sc = pd.concat(scored, ignore_index=True)
        ok = m["_K"].isin(set(sc["_K"]))
        out.loc[m.index[ok]] = cands.loc[m.index[ok], "SCORE"] * 0.01          # allowed, not scored: after the LLM's
        hit = m.reset_index().merge(sc, on=["_K", "VALIDITY", "BTL_NAME"], how="inner").set_index("index")
        out.loc[hit.index] = hit["LLM"]
        own = short.reset_index().merge(sc, on=["_K", "VALIDITY", "BTL_NAME"], how="left")
        acc["own_short"] += len(own); acc["covered"] += int(own["LLM"].notna().sum())
    acc["customers"] += short[VAL].nunique(); acc["answers"] += len(jobs)
    return out


AGENTS = {"actions": NBA.drop(columns=["PROFILE", "REASON"]), "nbo": nbo, "groups": set(config.TARGETS[AGENT_TARGET]),
          "label": MODEL}
p6_t2 = scoring_groups.run_p6_offers(D, force=True, target=AGENT_TARGET, agents=AGENTS)
P6 = RUN / "score" / f"P6_offers_{AGENT_TARGET}"
print(f"P6 {AGENT_TARGET}: group audit {p6_t2['audit_status']} -- {p6_t2['decisions']}")
for _g, _s in NBO_PROFILES.items():
    print(f"agent 4, {_g}: {_s['customers']:,} customer requests (P6 batches) from {_s['answers']:,} answers "
          f"({_s['customers'] / max(_s['answers'], 1):.1f} per answer); the answer scored "
          f"{_s['covered'] / max(_s['own_short'], 1):.0%} of the customers' own short-listed offers")
show(pd.read_csv(P6 / "group_report.csv")[["group", "rules", "contacted", "with_offers", "llm_scored_offers", "decided_by"]],
     title="who decided, per group")
_st4 = STATS.get("Agent 4 NBO", {})
_n4 = _st4.get("calls", 0) + _st4.get("cached", 0) + _st4.get("errors", 0)
if _n4:
    _fb4 = _st4.get("errors", 0) / _n4
    print(f"agent 4 by the LLM: {1 - _fb4:.0%} of {_n4:,} NBO answers ({_fb4:.0%} failed -> rule scores)")
    if _fb4 > MAX_FALLBACK_SHARE:
        _msg = f"agent 4: {_fb4:.0%} of the calls failed ({_st4.get('last_error')})"
        if STOP_ON_FALLBACK:
            raise SystemExit(_msg)
        print("WARNING:", _msg)''')]

FAST_TIME = [("md", r'''## FAST_MODE in figures

How many customers each LLM answer covered, and the GPU time against the per-customer estimate (`off` in step 5b).
Saved to `data/runs/<D>/agents/fast_mode.json`.'''),
             ("code", r'''_a = {a: STATS.get(a, {}) for a in ("Agent 3 NBA", "Agent 4 NBO")}
answers34 = sum(s.get("calls", 0) + s.get("cached", 0) + s.get("errors", 0) for s in _a.values())
wall34 = sum(s.get("wall_seconds", 0.0) for s in _a.values())
_c4 = sum(g["customers"] for g in NBO_PROFILES.values())
_n4 = sum(g["answers"] for g in NBO_PROFILES.values())
per3 = round(len(NBA) / max(len(ANS3), 1), 2)
per4 = round(_c4 / max(_n4, 1), 2)
FAST = {"profile_level": LEVEL, "profile_level_setting": PROFILE_LEVEL, "target_customers": n_cust,
        "customers_sent_to_llm": len(LLM_LINES), "units_sent": len(LLM_UNITS), "answers_agents_3_4": answers34,
        "agent3_customers_per_answer": per3, "agent4_customers_per_answer": per4,
        "calls_saved_share": round(1 - answers34 / max(len(NBA) + _c4, 1), 3),
        "from_cache_agents_3_4": sum(s.get("cached", 0) for s in _a.values()), "gpu_wall_minutes_agents_3_4": round(wall34 / 60, 1),
        "estimate_minutes_per_customer_mode": float(LEVEL_TABLE.loc[LEVEL_TABLE["level"] == "off", "estimated GPU min"].iloc[0]),
        "levels": LEVEL_TABLE.to_dict("records"), "nbo_profiles": NBO_PROFILES, "quality_check_agent3": CHECK3}
(AGENTS_DIR / "fast_mode.json").write_text(json.dumps(FAST, indent=2, default=str), encoding="utf-8")
print(f"FAST_MODE {LEVEL}: {len(LLM_LINES):,} of {n_cust:,} customers decided by {answers34:,} LLM answers "
      f"(agent 3: {per3} customers per answer, agent 4: {per4}; {FAST['calls_saved_share']:.0%} of the per-customer calls saved; "
      f"{FAST['from_cache_agents_3_4']:,} from the cache) in "
      f"{FAST['gpu_wall_minutes_agents_3_4']} min; one call per customer would take about "
      f"{FAST['estimate_minutes_per_customer_mode']} min")''')]

G3_FAST = r'''_rep = json.loads((AGENTS_DIR / "llm_gpu_report.json").read_text(encoding="utf-8"))
_rep["fast_mode"] = FAST
(AGENTS_DIR / "llm_gpu_report.json").write_text(json.dumps(_rep, indent=2, default=str), encoding="utf-8")
print(f"FAST_MODE added to the report: level {LEVEL}, {FAST['calls_saved_share']:.0%} of the per-customer calls saved"
      + (f", quality check {CHECK3['same_top_action']:.0%} same top action" if CHECK3 else ""))'''

STEP_12 = [("md", r'''## Step 12 · The 360 view of every scored target line (for your checks)

Two files, written next to the final file in `data/output/<D>/target2/` (Parquet + `.csv.gz` each):
- `target2_scored_360`: **the scored MSISDNs only** (the lines of the final file), with all the data below, the flags,
  `MSISDN_FLYTXT` and two checks against `input_flytxt` (`CHK_NOT_IN_FLYTXT`, `CHK_SLOTS_DIFFER_FROM_FLYTXT`: both must be 0);
- `target2_360`: **every target line**, scored or not (`DROP_RULE` says why a line is not in the file).

One row per line, in this order (for the scored view, see the two flytxt checks just after the `CHK_*` columns):


| Columns | What |
|---|---|
| `ACCESS_METHOD_VAL`, `GROUP_KEY`, `IN_INPUT_FLYTXT`, `DROP_RULE` | the line, its group (`t2c` / `t2f`), whether it is in the final file, and why not |
| `SEGMENT_AT_T`, `SEGMENT`, `SEGMENT_MOVED_T_TO_D`, `RULE_ID`, `VALUE_BAND`, `USAGE_TYPE`, `NATIONALITY_GROUP` | the segment at training (T) and at scoring (D), and the bands |
| `ACTION_ORIGINAL`, `ACTION_AGENT`, `ACTION_SENT`, `DECIDED_BY` | the production action, the cell policy, the action sent, and who decided |
| `AGENT3_ACTION`, `AGENT3_SCORES`, `AGENT3_REASON`, `AGENT3_DECIDED_BY`, `PROPENSITY_LOGGED` | agent 3: its score per allowed action, its reason, and the real logged propensity |
| `SENT_TO_LLM`, `PROFILE_LEVEL`, `PROFILE_ID`, `PROFILE_FACTS` | FAST_MODE: whether the LLM decided the line, and the profile it was decided with (the prompt text, no MSISDN) |
| `SLOT_1` … `SLOT_6`, `REASON_i`, `REMOVED_i`, `SLOT_i_PRICE`, `SLOT_i_ANCHOR`, `SLOT_i_VS_ANCHOR` | the 6 offers of the file (= `destination_offer_name_i`), why each is there, the price, the anchor of the price band for that service and validity (combo = total spend, data = the customer's own data share × total, voice = the rest; mean + std), and price ÷ anchor |
| `CHK_*`, `N_OFFERS` | the check flags (below) |
| `reco_m*` / `reco_w*`, `B26_*`, `F05_*` | the recommended offers with score, anchor, pass and rule; the two anchors |
| W0 columns and every T2 feature | last week (targeted, action, reward), revenue / voice / data per period, model scores, … |

**The check flags:**
- `CHK_OFFERS_BELOW_ANCHOR`: the number of slots priced **below** the customer's usual spend (a cannibalization
  risk). The challenger band starts at 1.0 × the anchor, so the challenger offers are never below it. It can be > 0
  for favourites (code 5 adds them without a price floor: F-07 is off, as in production) and for the fallback passes.
- `CHK_FAVOURITE_BELOW_ANCHOR`: of which favourites.
- `CHK_FALLBACK_PASS_BELOW_ANCHOR`: of which offers of the **fallback passes** (production code 3 / 4, B-28 … B-30,
  used when the challenger band finds nothing). They price against the production anchor with a fixed 90 / 10
  data / voice split, so a voice or data offer can sit under the customer's own split. `SLOT_i_SELECTED_BY` says
  which rule picked each offer. Anything else below the anchor is printed as a WARNING (expected 0).
- `CHK_UPSELL_AT_RISK_SEGMENT`: an upsell sent to an at-risk segment (H-06). It must be 0.
- `CHK_UPSELL_CHURN_GE_060`: an upsell sent to a customer with churn ≥ 0.60 (for information).

The cell prints the summary per group. The file holds MSISDNs: read it with pandas, here or in another notebook on
this laptop, and do not open it in Excel.'''),
            ("code", r'''det = pd.read_parquet(OUTPUT / AGENT_TARGET / f"details_{AGENT_TARGET}.parquet")
v360 = det.copy()


def _c(name):
    return pd.to_numeric(v360[name], errors="coerce") if name in v360.columns else pd.Series(np.nan, index=v360.index)


# --- agent 3: the real decision, its scores, reason and logged propensity ------------------------------------------
a3 = NBA.set_index(VAL) if len(NBA) else pd.DataFrame(columns=["ACTION", "PROBA", "DECIDED_BY", "PROFILE", "REASON"])


def _scores(v):
    u = PROF.at[v, "UNIT"] if v in PROF.index else None
    if u is None or u not in ANS3:
        return None
    allowed, ans = ANS3[u]
    return "; ".join(f"{a}: {int(s)}" for a, s in zip(allowed, ans["scores"])) if ok3(ans, allowed) else None


v360["AGENT3_ACTION"] = v360[VAL].map(a3["ACTION"])
v360["AGENT3_SCORES"] = [(_scores(v) if v in a3.index else None) for v in v360[VAL]]
v360["AGENT3_REASON"] = v360[VAL].map(a3["REASON"])
v360["AGENT3_DECIDED_BY"] = v360[VAL].map(a3["DECIDED_BY"])
v360["PROPENSITY_LOGGED"] = v360[VAL].map(a3["PROBA"]).where(v360[VAL].isin(a3.index), v360.get("PROPENSITY"))
v360["SENT_TO_LLM"] = v360[VAL].isin(LLM_LINES)
v360["PROFILE_LEVEL"] = LEVEL
v360["PROFILE_ID"] = v360[VAL].map(PROF["UNIT"])
v360["PROFILE_FACTS"] = v360[VAL].map(FACTS)

# --- the segment at T (training, *_auto) against the segment at D (scoring view) -----------------------------------
_t = pd.concat([pd.read_parquet(p, columns=[VAL, "SEGMENT"]) for p in sorted((train_dir / T3_TRAIN / "segments").glob("*.parquet"))])
_t = _t[_t[VAL].isin(set(v360[VAL]))].drop_duplicates(VAL).set_index(VAL)["SEGMENT"]
v360["SEGMENT_AT_T"] = v360[VAL].map(_t)
v360["SEGMENT_MOVED_T_TO_D"] = v360["SEGMENT_AT_T"].notna() & (v360["SEGMENT_AT_T"] != v360["SEGMENT"])

# --- each slot: its price against the customer's usual spend (F-05 anchor: monthly for 30 days, weekly for 7 days) --
_cat = load_catalog(cache).drop_duplicates("BTL_NAME").set_index("BTL_NAME")
_share = use("nbx.fsdr.offers", "anchors").data_share(cache)                      # the customer's own data share
DATA_SHARE = v360[VAL].map(_share).fillna(config.DEFAULT_DATA_SHARE).astype(float).to_numpy()
_rec = [pd.read_csv(f, dtype=str, keep_default_na=False, usecols=lambda c: c in ("access_method_val", "btl_name", "selected_by"))
        for f in sorted(P6.glob("recos_*"))]                                   # which rule selected each recommended offer
_rec = pd.concat(_rec, ignore_index=True) if _rec else pd.DataFrame(columns=["access_method_val", "btl_name", "selected_by"])
_sel = _rec.assign(_K=_rec["access_method_val"].str.strip() + "|" + _rec["btl_name"].str.strip()).drop_duplicates("_K").set_index("_K")["selected_by"]
fb_below = pd.Series(0, index=v360.index)
below = pd.Series(0, index=v360.index)
fav_below = pd.Series(0, index=v360.index)
n_off = pd.Series(0, index=v360.index)
for i in range(1, 7):
    if f"SLOT_{i}" not in v360.columns:
        continue
    s = v360[f"SLOT_{i}"].where(v360[f"SLOT_{i}"].astype(str).str.strip().ne("") & v360[f"SLOT_{i}"].notna())
    price = pd.to_numeric(s.map(_cat["PRICE"]), errors="coerce")
    per = np.where(pd.to_numeric(s.map(_cat["VALIDITY"]), errors="coerce").astype(float).eq(7).to_numpy(), "WEEKLY", "MONTHLY")
    svc = s.map(_cat["SERVICE"]).astype(str).str.upper()
    # the anchor of the challenger price band (CC-03 / CF-03): total spend (mean + std) for COMBO, the customer's own
    # data share x total for DATA, (1 - share) x total for VOICE (no spend history: 90 / 10)
    total = _c("B26_AVG_REV_WEEKLY").where(per == "WEEKLY", _c("B26_AVG_REV_MONTHLY"))
    anc = total * np.select([svc == "DATA", svc == "VOICE"], [DATA_SHARE, 1 - DATA_SHARE], 1.0)
    anc = anc.where(s.notna() & (total > 0))
    v360[f"SLOT_{i}_PRICE"] = price
    v360[f"SLOT_{i}_ANCHOR"] = anc.round(3)
    v360[f"SLOT_{i}_VS_ANCHOR"] = (price / anc.where(anc > 0)).round(2)
    sel_by = (v360[VAL].astype(str) + "|" + s.fillna("").astype(str)).map(_sel).where(s.notna())
    v360[f"SLOT_{i}_SELECTED_BY"] = sel_by
    b = (price.astype(float) < anc.astype(float)).fillna(False).astype(bool)
    reason = v360[f"REASON_{i}"] if f"REASON_{i}" in v360.columns else pd.Series("", index=v360.index)
    below += b.astype(int)
    fav_below += (b & reason.astype(str).eq("favourite")).astype(int)
    fb_below += (b & sel_by.astype(str).str.startswith("fallback")).astype(int)
    n_off += s.notna().astype(int)
act = v360["ACTION_SENT"].astype(str)
v360["N_OFFERS"] = n_off
v360["CHK_OFFERS_BELOW_ANCHOR"] = below
v360["CHK_FAVOURITE_BELOW_ANCHOR"] = fav_below
v360["CHK_FALLBACK_PASS_BELOW_ANCHOR"] = fb_below
v360["CHK_UPSELL_AT_RISK_SEGMENT"] = act.isin(UPSELL_ACTIONS) & v360["SEGMENT"].isin(AT_RISK_SEGMENTS)
v360["CHK_UPSELL_CHURN_GE_060"] = act.isin(UPSELL_ACTIONS) & (_c("PROBA_CHURN") >= 0.6)

first = [VAL, "GROUP_KEY", "IN_INPUT_FLYTXT", "DROP_RULE", "SEGMENT_AT_T", "SEGMENT", "SEGMENT_MOVED_T_TO_D", "RULE_ID",
         "VALUE_BAND", "USAGE_TYPE", "NATIONALITY_GROUP", "ACTION_ORIGINAL", "ACTION_AGENT", "ACTION_SENT", "DECIDED_BY",
         "AGENT3_ACTION", "AGENT3_SCORES", "AGENT3_REASON", "AGENT3_DECIDED_BY", "PROPENSITY_LOGGED", "SENT_TO_LLM",
         "PROFILE_LEVEL", "PROFILE_ID", "PROFILE_FACTS",
         *[c for i in range(1, 7) for c in (f"SLOT_{i}", f"REASON_{i}", f"SLOT_{i}_SELECTED_BY", f"REMOVED_{i}", f"SLOT_{i}_PRICE",
                                         f"SLOT_{i}_ANCHOR", f"SLOT_{i}_VS_ANCHOR")],
         "N_OFFERS", "CHK_OFFERS_BELOW_ANCHOR", "CHK_FAVOURITE_BELOW_ANCHOR", "CHK_FALLBACK_PASS_BELOW_ANCHOR",
         "CHK_UPSELL_AT_RISK_SEGMENT",
         "CHK_UPSELL_CHURN_GE_060"]
first = [c for c in first if c in v360.columns]
v360 = v360[first + [c for c in v360.columns if c not in first]]
OUT360 = OUTPUT / AGENT_TARGET
v360.to_parquet(OUT360 / f"{AGENT_TARGET}_360.parquet", index=False)
v360.to_csv(OUT360 / f"{AGENT_TARGET}_360.csv.gz", index=False, compression="gzip")
print(f"360 view: {len(v360):,} lines x {v360.shape[1]} columns -> {OUT360 / (AGENT_TARGET + '_360.parquet')} (+ .csv.gz)")

# --- the scored MSISDNs only (the lines of the final file), with all the data, the flags and the flytxt check -------
scored = v360[v360["IN_INPUT_FLYTXT"]].copy()
_fp = OUT360 / f"input_flytxt_{AGENT_TARGET}.csv.gz"
if _fp.exists():
    _fly = pd.read_csv(_fp, dtype=str, keep_default_na=False)
    _fly = _fly.assign(_V=_fly["msisdn"].str.strip().str[-8:]).drop_duplicates("_V").set_index("_V")
    scored.insert(1, "MSISDN_FLYTXT", scored[VAL].map(_fly["msisdn"]))
    _diff = pd.Series(False, index=scored.index)
    for i in range(1, 7):
        a = scored[f"SLOT_{i}"].fillna("").astype(str).str.strip() if f"SLOT_{i}" in scored.columns else ""
        b = scored[VAL].map(_fly[f"destination_offer_name_{i}"]).fillna("").astype(str).str.strip()
        _diff |= a != b
    scored["CHK_NOT_IN_FLYTXT"] = scored["MSISDN_FLYTXT"].isna()
    scored["CHK_SLOTS_DIFFER_FROM_FLYTXT"] = _diff & ~scored["CHK_NOT_IN_FLYTXT"]
    _fc = ["CHK_NOT_IN_FLYTXT", "CHK_SLOTS_DIFFER_FROM_FLYTXT"]
    _head = [VAL, "MSISDN_FLYTXT"] + [c for c in first if c != VAL] + _fc
    scored = scored[_head + [c for c in scored.columns if c not in _head]]
    print(f"flytxt check: {len(_fly):,} lines in {_fp.name}; {int(scored['CHK_NOT_IN_FLYTXT'].sum()):,} scored lines "
          f"not found in it, {int(scored['CHK_SLOTS_DIFFER_FROM_FLYTXT'].sum()):,} with other offers (both must be 0)")
    _extra = len(set(_fly.index) - set(scored[VAL]))
    if _extra:
        print(f"WARNING: {_extra:,} flytxt lines are not in the scored view")
else:
    print(f"no {_fp.name}: the final file was not written (see steps 8 / 9), the scored view has no flytxt check")
scored.to_parquet(OUT360 / f"{AGENT_TARGET}_scored_360.parquet", index=False)
scored.to_csv(OUT360 / f"{AGENT_TARGET}_scored_360.csv.gz", index=False, compression="gzip")
print(f"scored view: {len(scored):,} scored MSISDNs x {scored.shape[1]} columns -> "
      f"{OUT360 / (AGENT_TARGET + '_scored_360.parquet')} (+ .csv.gz)")

summary = v360.groupby("GROUP_KEY").agg(
    lines=(VAL, "size"), in_final_file=("IN_INPUT_FLYTXT", "sum"), sent_to_llm=("SENT_TO_LLM", "sum"),
    offers_per_line=("N_OFFERS", "mean"), segment_moved_T_to_D=("SEGMENT_MOVED_T_TO_D", "sum"),
    lines_offer_below_anchor=("CHK_OFFERS_BELOW_ANCHOR", lambda s: int((s > 0).sum())),
    of_which_favourite=("CHK_FAVOURITE_BELOW_ANCHOR", lambda s: int((s > 0).sum())),
    of_which_fallback_pass=("CHK_FALLBACK_PASS_BELOW_ANCHOR", lambda s: int((s > 0).sum())),
    upsell_at_risk_segment=("CHK_UPSELL_AT_RISK_SEGMENT", "sum"), upsell_churn_ge_060=("CHK_UPSELL_CHURN_GE_060", "sum"))
show(summary.round(2).reset_index(), n=5, title="checks per group")
if int(v360["CHK_UPSELL_AT_RISK_SEGMENT"].sum()):
    print("WARNING: upsells sent to an at-risk segment (H-06): look at those lines in the 360 view")
_fb = int((v360["CHK_FALLBACK_PASS_BELOW_ANCHOR"] > 0).sum())
if _fb:
    print(f"note: {_fb:,} lines got an offer from the FALLBACK passes (production code 3 / 4, B-28 ... B-30) under the band "
          "anchor: those passes use the production anchor (fixed 90 / 10 data / voice split), not the customer's own "
          "split; filter CHK_FALLBACK_PASS_BELOW_ANCHOR > 0 and read SLOT_i_SELECTED_BY")
_bad = int(((v360["CHK_OFFERS_BELOW_ANCHOR"] - v360["CHK_FAVOURITE_BELOW_ANCHOR"] - v360["CHK_FALLBACK_PASS_BELOW_ANCHOR"]) > 0).sum())
print(f"{'WARNING' if _bad else 'OK'}: {_bad:,} lines with another offer below the band anchor (forced bundle or a pass): "
      + ("filter CHK_OFFERS_BELOW_ANCHOR > 0 and read REASON_i / SLOT_i_SELECTED_BY" if _bad else "expected 0"))
show(v360[first].head(3), n=3, title="the first columns (first rows)")''')]


GROUPS_ENV = r"""
# --- the groups to score: the other group's size is 0 (no size warning, nothing scored) -----------------------------
import os as _os
SCORE_GROUPS = [g for g in dict.fromkeys(SCORE_GROUPS)]
assert SCORE_GROUPS and set(SCORE_GROUPS) <= {"t2c", "t2f"}, 'SCORE_GROUPS: ["t2c"], ["t2f"] or ["t2c", "t2f"]'
if TEST_10K and _info["score_group"] not in SCORE_GROUPS:
    raise SystemExit(f"TEST_10K: the 10k copy holds only {_info['score_group']} (SCORE_GROUP of notebook 02), but "
                     f"SCORE_GROUPS = {SCORE_GROUPS}: put {_info['score_group']!r} in SCORE_GROUPS")
for _k, _env in (("t2c", "FSDR_T2_CONSTRAINT_SIZE"), ("t2f", "FSDR_T2_FREE_SIZE")):
    if _k not in SCORE_GROUPS:
        _os.environ[_env] = "0"
print("groups scored:", ", ".join({"t2c": "group a (T2C, target2_a)", "t2f": "group b (T2F, target2_b)"}[g] for g in SCORE_GROUPS))
"""

GROUPS_CELL = [("md", r'''## The groups to score (`SCORE_GROUPS`)

Only the lists of `SCORE_GROUPS` are read: `["t2c"]` scores group a (`target2_a`), `["t2f"]` group b (`target2_b`),
both = the two lists. The other list may be absent from the week folder. Every step below (the quality gate of 4b,
P1, the agents, P6, the final file, the 360 view) then works on the chosen groups only.'''),
               ("code", r'''_tg = use("nbx.fsdr", "targets")
_tg.read_targets = getattr(_tg, "_read_targets_orig", _tg.read_targets)
_tg._read_targets_orig = _tg.read_targets


def _read_chosen(folder):
    """The engine's read_targets, keeping only the lists of SCORE_GROUPS (and their issues)."""
    t, issues = _tg._read_targets_orig(folder)
    keep = t["GROUP_KEY"].isin(SCORE_GROUPS + [config.CONTROL_GROUP])
    return t[keep].reset_index(drop=True), [i for i in issues if i.get("group") in SCORE_GROUPS + [config.CONTROL_GROUP]]


_tg.read_targets = _read_chosen
print(f"target lists read: {SCORE_GROUPS} (sizes {config.group_sizes()})")''')]
