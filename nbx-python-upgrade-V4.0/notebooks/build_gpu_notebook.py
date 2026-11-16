"""Builds notebooks/03_gpu_agents_step_by_step.ipynb (V4.0): every agent, one by one, with the LLM agents on the
laptop GPU through Ollama (local only). Run it again after editing the cells below or the engine:

    uv run python notebooks/build_gpu_notebook.py

Standalone like 01 / 02 (notebooks/build_step_notebooks.py): the engine code is copied into Part A, nothing is
imported from the project folder, no API, no Gradio, no vowpalwabbit. The LLM is called with the standard library
(urllib) on http://127.0.0.1:11434. Stored WITHOUT outputs (tests/test_notebook.py checks it).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build_step_notebooks", HERE / "build_step_notebooks.py")
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)
md, code, nbf = sb.md, sb.code, sb.nbf

NAME = "03_gpu_agents_step_by_step.ipynb"
ROOTS = ["nbx.fsdr.synth.generator", "nbx.fsdr.datasource", "nbx.fsdr.schemas", "nbx.fsdr.contract.data_agent",
         "nbx.fsdr.contract.cache", "nbx.fsdr.pipelines.training", "nbx.fsdr.steps", "nbx.fsdr.csvstd",
         "nbx.fsdr.pipelines.scoring_groups", "nbx.fsdr.targets", "nbx.fsdr.llm_budget",
         "nbx.fsdr.segmentation.actions"]

TITLE = md("""# 03 · All the agents on the laptop GPU (Ollama), step by step (FSDR V4.0)

This notebook runs the whole week with **every agent**, one by one. The LLM agents run on **your laptop GPU**
through **Ollama** (already installed). Everything stays on this machine: the only server it calls is
`http://127.0.0.1:11434`. There is no API and no Gradio here.

**How to run it:** start Ollama (§ below), then **Run All**. The defaults use dummy data (3,000 lines) and a time
budget for the per-customer agents, so a first run takes a few minutes. For real data, change the settings cell.

**One weekly folder.** Each week you export the tables once, into `data/weekly/<D>/` (5 sub-folders + the target
lists). The features come from the fresh `*_auto` tables (week-0 mode every week); the `*_score` tables and the
separate scoring folder are not needed. Scoring = the trained base joined with your target lists, the offers by
the rules and the agents, then the output files.

| Step | Agent | Runs on | What it decides | LLM calls |
|---|---|---|---|---|
| G1–G2 | — | GPU | is Ollama on the GPU; how fast is it | a few |
| 1 | — | CPU | the weekly folder (your exports, or dummy data) | — |
| 2 | **Agent 1** ① data | CPU + GPU | rules validate and **block** bad data; the LLM explains the report | 1 |
| 3 | **Agent 2** ② segmentation | CPU + GPU | rules: features, S-segments; the LLM explains each segment | 1 per segment (14) |
| 4 | **Agent 2b** ②b dictionary | GPU | the action of each segment × nationality cell (Target 2), checked by H-06 | 1 per cell (≤ 60) |
| 5 | — | CPU | target lists → panel | — |
| 6 | **Agent 3** ③ NBA | GPU | the action of each **T2C** customer (ε exploration, logged propensity) | 1 per customer |
| 7 | **Agent 4** ④ NBO + **6B** | GPU + CPU | scores the offers the rules allow for each **T2C / T2F** customer; the challenger rules choose | 1 per customer |
| 8 | **Agent 7** ⑦ code 5 | CPU | the 38-column file of Target 2 | — |
| 9 | **Agent 9** ⑨ auditor | CPU + GPU | rules **block** the final file; the LLM explains the checks | 1 |
| 10 | **Agent 8** ⑧ strategy | GPU | group a vs group b, agents vs cell policy, what to change next week | 1 |
| 11 | — | CPU | control file + the final file of Target 2 | — |

**What runs on the GPU:** only the LLM (Ollama). The data steps (pandas, Parquet) run on the CPU, which is normal:
they take seconds to minutes. **The LLM proposes, the rules decide:** an action or offer the rules do not allow is
refused, the auditor can still block the output, and every row says who decided (`DECIDED_BY`, `score_source`).

**Target 2 only (V4.0):** 50,000 lines in group **a** (T2C = Target 2 Constraint: offers of the families of the
customer's action) and 50,000 in group **b** (T2F = Target 2 Free: the best L2 DATA / COMBO offers for the spend,
whatever the action). No control group by default (`FSDR_CONTROL_SIZE=0`). Target 1 (the production rules) is not scored.""")

OLLAMA_MD = md("""## Before you start: Ollama

1. **Start Ollama**: open the *Ollama* app (its icon appears next to the clock), or run `ollama serve` in a terminal.
2. **A model** (once, about 5–9 GB, the only download): `ollama pull qwen3:8b` (fast) or `ollama pull qwen3:14b`
   (better, about 2x slower). With 24 GB of GPU memory both fit entirely on the GPU. `MODEL = "auto"` picks the best
   one you have.
3. **Faster (recommended):** let Ollama answer several customers at once. In PowerShell, once:
   ```powershell
   setx OLLAMA_NUM_PARALLEL 8
   setx OLLAMA_FLASH_ATTENTION 1
   ```
   then quit Ollama (right-click its icon → Quit) and start it again. Set `PARALLEL = 8` below.
4. **Watch the GPU** while the agents run: `nvidia-smi -l 1` in a terminal (`GPU-Util` goes up, `ollama` is listed).""")

SETTINGS = '''# --- settings -------------------------------------------------------------------
RUN_DATE = "2026-11-19"         # the run date D of this week (the folder data/weekly/<D>/)
GENERATE = False                # False: your real exports. True: dummy data for a test (real exports are never overwritten)
N_LINES = 3000                  # dummy lines
GENERATE_TARGETS = False        # False: your real lists. True: dummy lists for a test (real lists are never overwritten)

OLLAMA_URL = "http://127.0.0.1:11434"   # local only
MODEL = "auto"                  # "auto" = the best installed of qwen3:14b, qwen3:8b, qwen2.5:14b, llama3.1:8b, ...
PARALLEL = 4                    # customers sent to Ollama at the same time (set OLLAMA_NUM_PARALLEL to the same)
TIME_BUDGET_MIN = 20            # minutes for the per-customer agents 3 + 4; customers past it get the cell policy
MAX_LLM_LINES = None            # or a number: at most this many customers for agents 3 + 4 (None = the time budget)
AGENT_TARGET = "target2"        # the agents decide Target 2 (groups a = T2C and b = T2F); Target 1 is not scored
EPSILON = 0.1                   # agent 3: exploration share (the logged propensity makes the uplift measurable)
NBO_SHORTLIST = 8               # agent 4: offers per validity the LLM scores per customer (the best by the rules)
SEED = 7
FORCE = True                    # re-run every step (False: steps already done with the same inputs are skipped)
PREVIEW_ROWS = 8
'''

GPU_SETUP = '''# --- paths and helpers (do not edit) ---------------------------------------------
import json, os, sys, time
from datetime import date
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from nbx.fsdr import config, csvstd, steps

pd.set_option("display.max_columns", 80, "display.width", 220, "display.max_colwidth", 80)
D = date.fromisoformat(RUN_DATE)
WEEK = ROOT / "data" / "weekly" / RUN_DATE          # the ONE weekly folder: 5 sub-folders + the target lists
RUN, CACHE, OUTPUT = config.FSDR_RUNS_DIR / RUN_DATE, config.FSDR_CACHE_DIR / RUN_DATE, config.output_folder(RUN_DATE)


''' + sb.SETUP[sb.SETUP.index("def show("):sb.SETUP.index('print(f"run date')] + '''print(f"run date : {RUN_DATE}")
for name, p in (("weekly", WEEK), ("runs", RUN), ("cache", CACHE), ("output", OUTPUT)):
    print(f"{name:8s}: {p}  ({'found' if p.exists() else 'not yet'})")
print(f"offers per line: {config.RECO_OFFERS}  (FSDR_RECO_OFFERS)")'''

LLM_HELPERS = '''# --- the LLM client: Ollama on this machine, JSON replies checked against a schema (do not edit) -------------
import hashlib, threading, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

assert urlparse(OLLAMA_URL).hostname in ("127.0.0.1", "localhost", "::1"), "local only: OLLAMA_URL must be this machine"
AGENTS_DIR = RUN / "agents"
AGENTS_DIR.mkdir(parents=True, exist_ok=True)
LLM_CACHE = AGENTS_DIR / "llm_cache.jsonl"       # every answer, keyed by (model, prompt): a re-run costs nothing
_cache, _lock = {}, threading.Lock()
if LLM_CACHE.exists():
    for line in LLM_CACHE.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            _cache[row["key"]] = row["answer"]
        except (ValueError, KeyError):
            pass
STATS = {}                                       # per agent: calls, cached, errors, seconds, tokens


def _http(path, body=None, timeout=600):
    req = urllib.request.Request(OLLAMA_URL + path, data=None if body is None else json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _stat(agent):
    return STATS.setdefault(agent, {"calls": 0, "cached": 0, "errors": 0, "seconds": 0.0, "prompt_tokens": 0,
                                    "output_tokens": 0})


def ask(agent, system, user, schema, num_predict=256, use_cache=True):
    """One JSON answer of the local LLM, or None (the caller then uses the rule fallback). Temperature 0."""
    key = hashlib.sha256(json.dumps([MODEL, system, user, schema], sort_keys=True).encode()).hexdigest()
    st = _stat(agent)
    with _lock:
        if use_cache and key in _cache:
            st["cached"] += 1
            return _cache[key]
    body = {"model": MODEL, "stream": False, "format": schema, "keep_alive": "30m",
            "options": {"temperature": 0, "seed": SEED, "num_ctx": 4096, "num_predict": num_predict},
            "messages": [{"role": "system", "content": system + " Reply with compact JSON only."},
                         {"role": "user", "content": user}]}
    if THINK_OFF:
        body["think"] = False
    t = time.time()
    try:
        r = _http("/api/chat", body)
        answer = json.loads(r["message"]["content"])
    except (OSError, ValueError, KeyError, urllib.error.URLError) as e:
        with _lock:
            st["errors"] += 1
            st["last_error"] = str(e)[:200]
        return None
    with _lock:
        st["calls"] += 1
        st["seconds"] += time.time() - t
        st["prompt_tokens"] += r.get("prompt_eval_count", 0)
        st["output_tokens"] += r.get("eval_count", 0)
        if use_cache:
            _cache[key] = answer
            with LLM_CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "agent": agent, "answer": answer}) + "\\n")
    return answer


def ask_many(agent, jobs):
    """jobs: (system, user, schema, num_predict[, use_cache]) tuples, sent PARALLEL at a time. Answers in order."""
    t = time.time()
    with ThreadPoolExecutor(max_workers=max(1, PARALLEL)) as ex:
        out = list(ex.map(lambda j: ask(agent, *j), jobs))
    _stat(agent)["wall_seconds"] = _stat(agent).get("wall_seconds", 0.0) + time.time() - t
    return out


def ints(n, lo=0, hi=100):
    """Schema: a list of exactly n integers in [lo, hi] (compact: one number per option)."""
    return {"type": "array", "items": {"type": "integer", "minimum": lo, "maximum": hi}, "minItems": n, "maxItems": n}


def fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and x != x) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


print("LLM client ready (answers cached in", LLM_CACHE.relative_to(ROOT), ")")'''

CELLS = [
    md("""## G1 · The GPU and Ollama

This cell only reads. It checks the GPU (`nvidia-smi`), that Ollama answers on this machine, which models you have,
picks `MODEL`, loads it, and shows **how much of the model sits in GPU memory** (it must be 100 %: a model that
does not fit runs partly on the CPU and is much slower)."""),
    code('''import shutil, subprocess

smi = shutil.which("nvidia-smi")
if smi:
    q = subprocess.run([smi, "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    print("GPU    :", q or "nvidia-smi gave no answer")
else:
    print("GPU    : nvidia-smi not found (no NVIDIA driver?): Ollama would run on the CPU, much slower")

try:
    print("Ollama :", _http("/api/version", timeout=5)["version"], "at", OLLAMA_URL)
except OSError as e:
    raise SystemExit(f"Ollama does not answer at {OLLAMA_URL}: start the Ollama app (or `ollama serve`), then run "
                     f"this cell again ({e})")
installed = [m["name"] for m in _http("/api/tags", timeout=10).get("models", [])]
print("models :", ", ".join(installed) or "none")
PREFERRED = ["qwen3:14b", "qwen2.5:14b", "qwen3:8b", "llama3.1:8b", "llama3.1:latest", "qwen2.5:7b", "mistral:7b",
             "qwen3:4b", "qwen3:1.7b"]
if MODEL == "auto":
    MODEL = next((m for m in PREFERRED if m in installed), installed[0] if installed else None)
if MODEL not in installed:
    raise SystemExit(f"model {MODEL!r} is not installed: run `ollama pull qwen3:8b` in a terminal, then this cell again")
THINK_OFF = MODEL.startswith("qwen3")              # qwen3 "thinks" by default: off, the answers are short JSON
print("MODEL  :", MODEL)

t = time.time()
_http("/api/generate", {"model": MODEL, "prompt": "ok", "stream": False, "keep_alive": "30m",
                        "options": {"num_predict": 1}, **({"think": False} if THINK_OFF else {})})
print(f"loaded in {time.time() - t:.1f} s")
for m in _http("/api/ps", timeout=10).get("models", []):
    if m["name"] == MODEL:
        share = m.get("size_vram", 0) / max(m.get("size", 1), 1)
        print(f"in GPU memory: {share:.0%} of {m['size'] / 1e9:.1f} GB"
              + ("" if share > 0.99 else "  <- not all on the GPU: choose a smaller model, or close other GPU programs"))'''),

    md("""## G2 · Speed test and the time estimate

Ten customers' worth of agent 3 (NBA) and agent 4 (NBO) prompts, `PARALLEL` at a time, on your GPU. From the measured
time it estimates this run and a production week (50k T2C + 50k T2F). The agents only need GPU time for the
customers of Target 2; everything else takes seconds to minutes on the CPU."""),
    code('''ACTIONS = list(config.ACTIONS)
_sys = "You are a telecom marketing agent. Score each option 0-100 for this prepaid customer."
# the same shape and length as the real prompts of steps 7 and 8, so the estimate holds
_facts = ("segment Retention_RGS_Fragile, value band mid, usage both, tenure 812 days, revenue M1/M2/M3 4.20/5.10/5.60, "
          "data 2310 MB, voice 143 min, active days 27, churn 0.4{i}, data appetency 0.31, voice appetency 0.12, "
          "last week: targeted with UPSELL, no offer taken")
_nba = [(_sys, f"Customer: {_facts.format(i=i)}. Cell policy: UPSELL. Allowed actions, in order: {ACTIONS}. "
               "Return one score per allowed action, in that order, and a short reason.",
         {"type": "object", "properties": {"scores": ints(len(ACTIONS)), "reason": {"type": "string", "maxLength": 100}},
          "required": ["scores", "reason"]}, 120, False) for i in range(10)]
_offers = "; ".join(f"{k}: DATA L2, {30 if k < NBO_SHORTLIST else 7} days, price {1.5 + k / 2:.2f}, {2 + k:.1f} GB, - min, "
                    f"rule score 0.{50 + k}" for k in range(2 * NBO_SHORTLIST))
_nbo = [(_sys, f"Customer: {_facts.format(i=i)}. Action: UPSELL. Price anchor 4.{i}0. Offers: {_offers}. "
               f"Return one score per offer, in order.",
         {"type": "object", "properties": {"scores": ints(2 * NBO_SHORTLIST)}, "required": ["scores"]}, 160, False)
        for i in range(10)]
t = time.time(); ok_nba = ask_many("G2 speed test", _nba); s_nba = (time.time() - t) / len(_nba)
t = time.time(); ok_nbo = ask_many("G2 speed test", _nbo); s_nbo = (time.time() - t) / len(_nbo)
st = STATS["G2 speed test"]
tok_s = st["output_tokens"] / max(st["seconds"], 1e-9) * min(PARALLEL, len(_nba))
SEC_T2C, SEC_T2F = s_nba + s_nbo, s_nbo                 # per customer: T2C = NBA + NBO, T2F = NBO only
print(f"{MODEL}, PARALLEL {PARALLEL}: {s_nba:.2f} s per NBA call, {s_nbo:.2f} s per NBO call (wall, parallel) | "
      f"~{tok_s:.0f} output tokens/s | {sum(a is None for a in ok_nba + ok_nbo)} failed")
failed = sum(a is None for a in ok_nba + ok_nbo)
if failed == len(ok_nba + ok_nbo):
    raise SystemExit(f"every test call failed ({st.get('last_error')}): the LLM server is not working. Check the server "
                     "(its terminal or log), restart it, then run G1 and this cell again")
if failed:
    print(f"WARNING: {failed} test calls failed ({st.get('last_error')}): lower PARALLEL, or check the server")


def estimate(n_t2c, n_t2f):
    return n_t2c * SEC_T2C + n_t2f * SEC_T2F


print("estimated GPU time of agents 3 + 4 (the other agents add about 1-3 minutes):")
print(f"production week (50k T2C + 50k T2F): about {estimate(50_000, 50_000) / 3600:.1f} h with {MODEL} at PARALLEL {PARALLEL}")
print(f"per 1,000 Target-2 customers (half T2C, half T2F): about {estimate(500, 500) / 60:.1f} min")'''),

    md("""## Step 1 · The weekly folder `data/weekly/<D>/`

One folder per week, the same 5 sub-folders as the exports, and the target lists inside it:

```
data/weekly/<D>/  usage/ models/ subscription/ catalog/ others/
                  target2/a_constraint/ target2/b_free/
```

**What to copy each week** (file name = table name + `.csv.gz`):
- the fresh **history** `activities_w1…w4_auto`, `activities_m1_auto`, `activities_auto_m2`, `activities_m3_auto`,
  `subs_amt_auto`, `party_auto`, `access_method_id_auto`, `targeting_dataiku_auto`, `subs_btl_auto`;
- the **base and models**: `base_prepaid`, `consumer_aact_account`, the 11 model tables;
- what the **offers** need: `btl_catalog`, `sub_btl_3months` (270-day purchases: the offer universe), `subscriptions_tg`
  and `usage_data_voice_payg_tg` (the price anchors), `targeting_auto` (campaigns already running), `BTL_script_english_arabic_ok`,
  `conflicting_offers_h`, `consent_contact`;
- **not needed**: the `*_score` tables and `party_scoring` (a week-0 run reads the `*_auto` history only).

With `GENERATE = True` this cell writes dummy data in that shape (and removes the `*_score` tables, to show they are
not needed)."""),
    code('''from nbx.fsdr.synth import generator
from nbx.fsdr.datasource import JsonSource

NOT_NEEDED = {"party_scoring", "subs_amt_score"} | {f"activities_{p}_score" for p in ("w1", "w2", "w3", "w4", "m1", "m2", "m3")}
if GENERATE:
    man = WEEK / config.MANIFEST_FILE
    if WEEK.exists() and any(WEEK.iterdir()) and (not man.exists() or json.loads(man.read_text())["source"] != "generator"):
        raise SystemExit(f"{WEEK} holds real data: set GENERATE = False")
    run = generator.generate(N_LINES, D, ["baseline"], config.GENERATOR_SEED, WEEK, "csv.gz", "folders")
    for f in WEEK.rglob("*"):
        if f.is_file() and f.name.split(".")[0] in NOT_NEEDED:
            f.unlink()
    print(f"dummy data: {len(run.tables) - len(NOT_NEEDED)} tables for {N_LINES:,} lines (no *_score tables)")
if not GENERATE and not (WEEK.exists() and any(p.is_file() for p in WEEK.rglob("*.csv*"))):
    raise SystemExit(f"no exports in {WEEK}: create the folders with `uv run python run.py fsdr-folders --date {RUN_DATE}`, "
                     "copy the files, then run again (or GENERATE = True for a test)")
src = JsonSource(WEEK)
found = pd.DataFrame([{"folder": src.path(t).parent.name, "table": t} for t in src.tables()])
print(f"{len(found)} tables in {WEEK.relative_to(ROOT)}")
display(found.groupby("folder")["table"].apply(lambda s: ", ".join(sorted(s))).to_frame("tables"))'''),

    md("""## Step 2 · Agent 1 (①) data agent

**The rules decide:** T0 checks every table against the data contract (columns, types, keys, ranges, windows) and
the completeness of the week; a *blocking* issue stops the run. It then writes the Parquet cache.
**The LLM explains:** it reads the report and writes, in plain words, whether the data is usable and what to fix.
It cannot unblock anything."""),
    code('''from nbx.fsdr.contract import data_agent

t0 = data_agent.run_t0(WEEK, run_date=D, force=FORCE)
print(f"T0 {t0['status'].upper()} -- pipeline {t0['pipeline']}, {t0['tables']} tables, {t0['rows_in']:,} rows, "
      f"validation {t0['validation']} {t0['issue_counts']}")
rep = json.loads((RUN / "train" / "T0_load" / "validation_report.json").read_text(encoding="utf-8"))
expected = [i["table"] for i in rep["issues"] if "not needed by the" in i["message"]]    # *_score in week-0 mode
top = [f"{i['level']}: {i['table']} {i.get('column') or ''} {i['message']}" for i in rep["issues"]
       if "not needed by the" not in i["message"]][:15]
a1 = ask("Agent 1 data", "You are the data agent of a telecom campaign engine. Explain a data validation report to a "
         "business user in plain words. Only report real problems.",
         f"Run {RUN_DATE}, status {t0['status']}, pipeline {t0['pipeline']}, {t0['tables']} tables, {t0['rows_in']} rows. "
         f"Issues to explain: {top or 'none'}. {len(expected)} tables are absent on purpose and are NOT a problem "
         "(this week-0 run reads the *_auto history only). "
         "Say if the data is usable, the most important points, and what to fix first (nothing if there is no issue).",
         {"type": "object", "properties": {"usable": {"type": "boolean"}, "summary": {"type": "string", "maxLength": 500},
                                           "fix_first": {"type": "array", "items": {"type": "string", "maxLength": 160}, "maxItems": 3}},
          "required": ["usable", "summary", "fix_first"]}, 300)
display(Markdown(f"**Agent 1 (LLM):** {a1['summary'] if a1 else '(no LLM answer: the rule text above stands)'}"))
if a1 and a1["fix_first"]:
    display(Markdown("**Fix first:** " + "; ".join(a1["fix_first"])))
if t0["status"] != "done":
    raise SystemExit("T0 is BLOCKED by the rules: fix the data (see validation_report.md), then run again")'''),

    md("""## Step 3 · Agent 2 (②) segmentation

**The rules decide (CPU):** T2 builds the features of every line of the base from the `*_auto` history; T3 applies
the S-rules (first match wins) and gives every line its segment, bands, micro-segment and the two actions.
**The LLM explains** each segment in one line for the business (one call per segment)."""),
    code('''from nbx.fsdr.pipelines import training

t2 = training.run_t2(D, workers=1, force=FORCE)
t3 = training.run_t3(D, workers=1, force=FORCE)
print(f"T2 {t2['lines']:,} lines, {t2['features']} features (anchor {t2['anchor']}) | T3 {t3['lines']:,} lines, "
      f"{t3['targetable']:,} targetable, {t3['excluded_nursering']:,} Nursering excluded")
seg = pd.read_csv(RUN / "train" / "T3_segments" / "segment_counts.csv")
jobs = [("You are the segmentation agent of a telecom campaign engine.",
         f"Segment {r.SEGMENT} (rule {r.RULE_ID}, action {r.RULE_ACTION}): {r.lines} lines ({r.share:.1%} of the base), "
         f"average ARPU {r.avg_arpu_m1}/month, {r.w0_targeted} targeted last week, {r.w0_takers} took an offer. "
         "In one sentence: who these customers are and what to watch.",
         {"type": "object", "properties": {"why": {"type": "string", "maxLength": 200}}, "required": ["why"]}, 90)
        for r in seg.itertuples()]
seg["agent_2_explains"] = [(a or {}).get("why", "-") for a in ask_many("Agent 2 segmentation", jobs)]
show(seg[["RULE_ID", "SEGMENT", "RULE_ACTION", "lines", "share", "avg_arpu_m1", "agent_2_explains"]], n=20)'''),

    md("""## Step 4 · Agent 2b (②b) the agent dictionary

For every **segment × nationality** cell the LLM proposes the action(s) of Target 2, from the evidence of the cell:
size, value, risk and fragility shares, and **last week's reward rate per action** (who took an offer after being
targeted). The **check** (`validate_agent_dictionary`) then refuses an unknown action, an action for an excluded
segment, and an upsell as the top action of an at-risk segment (H-06); a refused cell keeps its current action. The
new version `dict_segm_actions_agent_v<N>.json` is applied to every line and next week starts from it."""),
    code('''from nbx.fsdr.segmentation.actions import AT_RISK_SEGMENTS, UPSELL_ACTIONS

segs = pd.concat([pd.read_parquet(p) for p in sorted((RUN / "train" / "T3_segments" / "segments").glob("*.parquet"))])
current = training.current_agent_dictionary(D)
cells, jobs = [], []
for c in current["cells"]:
    lines = segs[(segs["SEGMENT"] == c["segment"]) & (segs["NATIONALITY_GROUP"] == c["nationality_group"])]
    if c["excluded"] or not len(lines):
        continue
    orig = c["actions"][0]["action"] if c["actions"] else None
    allowed = [a for a in ACTIONS if not (c["segment"] in AT_RISK_SEGMENTS and a in UPSELL_ACTIONS and a != orig)]
    tg = lines[lines["W0_TARGETED"] == 1]
    rew = tg.groupby("W0_ACTION")["W0_REWARD"].agg(["mean", "size"]) if len(tg) else pd.DataFrame()
    reward = "; ".join(f"{a}: {m:.0%} of {n}" for a, (m, n) in rew.iterrows()) or "no targeting last week"
    cells.append((c, allowed))
    jobs.append(("You are the action-dictionary agent of a telecom campaign engine. Choose the best marketing action "
                 "for a whole cell of prepaid customers, from the evidence.",
                 f"Cell: segment {c['segment']}, nationality group {c['nationality_group']}, {len(lines)} lines. "
                 f"Average ARPU {lines['ARPU_M1'].mean():.2f}/month; at risk {lines['S_AT_RISK'].mean():.0%}; fragile "
                 f"{lines['S_FRAGILITY'].mean():.0%}; data users {lines['S_DATA_USER'].mean():.0%}; voice users "
                 f"{lines['S_VOICE_USER'].mean():.0%}. Current action: {[a['action'] for a in c['actions']]}. "
                 f"Last week's reward rate by action: {reward}. Allowed actions: {allowed}. Rank 1 to 2 actions.",
                 {"type": "object", "properties": {"actions": {"type": "array", "items": {"type": "string", "enum": allowed},
                                                               "minItems": 1, "maxItems": 2},
                                                   "reason": {"type": "string", "maxLength": 160}},
                  "required": ["actions", "reason"]}, 120))
answers = ask_many("Agent 2b dictionary", jobs)
proposal = {"cells": [dict(c, actions=[{"action": a, "rank": i + 1, "score": round(1 - i / 4, 2), "evidence_ids": ["W0", "T3"],
                                        "reason": ans["reason"]} for i, a in enumerate(dict.fromkeys(ans["actions"]))])
                      for (c, _), ans in zip(cells, answers) if ans and ans.get("actions")]}
r2b = training.update_agent_dictionary(D, proposal, made_by=f"agent 2b ({MODEL})")
new_top = [c for c in r2b["changed_cells"] if (c["before"] or [None])[0] != (c["after"] or [None])[0]]
refused = [p for p in r2b["problems"] if "missing cell" not in p["problems"][0]]
print(f"{r2b['version']} (parent {r2b['parent']}): {len(jobs)} cells asked, {len(proposal['cells'])} answered; "
      f"{len(new_top)} with a NEW top action (what Target 2 sends), {len(r2b['changed_cells']) - len(new_top)} only a new "
      f"second choice; {len(refused)} refused by the check (they keep the current action)")
show([{**c, "before": c["before"][0] if c["before"] else None, "after": c["after"][0]} for c in new_top]
     or [{"segment": "-", "after": "no top action changed"}], n=20, title="cells with a new top action")
show(refused or [{"problems": "none"}], n=10, title="refused by the check (H-06, unknown action, excluded segment)")'''),

    md("""## Step 5 · Target lists and the panel

The lists are read from the weekly folder (`target2/a_constraint/target2_a.csv.gz`, `target2/b_free/target2_b.csv.gz`).
Your real lists are never overwritten by dummy lists. P1 joins each listed line with its segment and actions, so
Target 2 now uses the agent dictionary of step 4. Lists: `target2/a_constraint/target2_a.csv.gz` (group a, 50,000),
`target2/b_free/target2_b.csv.gz` (group b, 50,000). No control list by default."""),
    code('''from nbx.fsdr import targets
from nbx.fsdr.pipelines import scoring_groups

folder = scoring_groups.input_folder(D)
if GENERATE_TARGETS and scoring_groups.real_targets(folder):
    print("the folder holds YOUR lists: they are kept and used")
elif GENERATE_TARGETS:
    print("dummy lists:", scoring_groups.generate_targets(D)["sizes"])
p1 = scoring_groups.run_p1_panel(D, force=True)
panel = pd.read_parquet(RUN / "score" / "P1_panel" / "panel.parquet")
print(f"P1 {p1['status'].upper()} -- {len(panel):,} lines, sizes {p1['sizes']}, lists are "
      f"{'DUMMY' if targets.dummy_lists(folder) else 'YOURS'}")'''),

    md("""## Step 6 · Agent 3 (③) NBA: the action of each T2C customer

For each **T2C** customer (in priority order: retention first, the groups interleaved, until `TIME_BUDGET_MIN` or
`MAX_LLM_LINES`), the LLM scores the allowed actions from the customer's facts. No MSISDN is sent, only the facts.
- **Allowed:** the 7 actions, minus an upsell for an at-risk segment (H-06).
- **Choice by code:** p = (1 − ε)·score/Σscores + ε/k, one draw per customer (`llm_nba` = the agent's top action,
  `llm_nba_sampled` = another action drawn from its scores); the **propensity** p of the chosen action is logged (`proba` in `results_scoring_nba`), so next week's uplift can be measured without bias.
- **Fallback:** a customer past the budget, or an invalid answer, keeps the cell policy of agent 2b (`ACTION_AGENT`).
T2F customers need no action (the free universe does not use it)."""),
    code('''import numpy as np
from nbx.fsdr import llm_budget

VAL = "ACCESS_METHOD_VAL"
feat = pd.concat([pd.read_parquet(p, columns=[VAL, "PROBA_CHURN", "PROBA_APP_DATA", "PROBA_APP_VOICE", "PROBA_FRAG_DATA",
                                              "REV_M1", "REV_M2", "REV_M3", "DATA_VOLUME_MB_M1", "LOCAL_VOICE_MIN_M1",
                                              "ACTIVE_DAYS_COUNT_M1", "TENURE_DAYS"])
                  for p in sorted((RUN / "train" / "T2_features" / "base_features").glob("*.parquet"))])
ctx = panel.merge(feat, on=VAL, how="left").merge(segs[[VAL, "W0_TARGETED", "W0_ACTION", "W0_REWARD", "S_AT_RISK"]],
                                                  on=VAL, how="left").set_index(VAL)
t2 = panel[panel["TARGET"] == AGENT_TARGET].assign(DECIDED_BY="-")
queue = llm_budget.llm_queue(t2)[VAL].tolist()                     # priority tier first, groups interleaved
per = t2.set_index(VAL)["GROUP_KEY"].map({"t2c": SEC_T2C, "t2f": SEC_T2F}).fillna(SEC_T2F)
limit = MAX_LLM_LINES if MAX_LLM_LINES is not None else int(np.searchsorted(per.reindex(queue).cumsum().to_numpy(),
                                                                             TIME_BUDGET_MIN * 60, side="right"))
LLM_LINES = set(queue[:limit])
print(f"{AGENT_TARGET}: {len(t2):,} customers; agents 3 + 4 for {len(LLM_LINES):,} of them "
      f"(budget {TIME_BUDGET_MIN} min, about {per.reindex(queue[:limit]).sum() / 60:.1f} min), the rest: cell policy")


def facts(v):
    r = ctx.loc[v]
    return (f"segment {r['SEGMENT']}, value band {r['VALUE_BAND']}, usage {r['USAGE_TYPE']}, tenure {fmt(r['TENURE_DAYS'], 0)} days, "
            f"revenue M1/M2/M3 {fmt(r['REV_M1'])}/{fmt(r['REV_M2'])}/{fmt(r['REV_M3'])}, data {fmt(r['DATA_VOLUME_MB_M1'], 0)} MB, "
            f"voice {fmt(r['LOCAL_VOICE_MIN_M1'], 0)} min, active days {fmt(r['ACTIVE_DAYS_COUNT_M1'], 0)}, churn {fmt(r['PROBA_CHURN'])}, "
            f"data appetency {fmt(r['PROBA_APP_DATA'])}, voice appetency {fmt(r['PROBA_APP_VOICE'])}, "
            f"last week: {'targeted with ' + str(r['W0_ACTION']) + (', took an offer' if r['W0_REWARD'] == 1 else ', no offer taken') if r['W0_TARGETED'] == 1 else 'not targeted'}")


nba_lines = [v for v in queue[:limit] if ctx.loc[v, "GROUP_KEY"] == "t2c"]
nba_allowed = {v: [a for a in ACTIONS if not (ctx.loc[v, "SEGMENT"] in AT_RISK_SEGMENTS and a in UPSELL_ACTIONS
                                              and a != ctx.loc[v, "ACTION_AGENT"])] for v in nba_lines}
jobs = [("You are the next-best-action agent of a telecom campaign engine. Score each allowed action 0-100 for this "
         "prepaid customer: the chance it makes the customer stay and spend more.",
         f"Customer: {facts(v)}. Cell policy: {ctx.loc[v, 'ACTION_AGENT']}. Allowed actions, in order: {nba_allowed[v]}. "
         "Return one score per allowed action, in that order, and a short reason.",
         {"type": "object", "properties": {"scores": ints(len(nba_allowed[v])), "reason": {"type": "string", "maxLength": 100}},
          "required": ["scores", "reason"]}, 120) for v in nba_lines]
answers = ask_many("Agent 3 NBA", jobs)
rows = []
for v, ans in zip(nba_lines, answers):
    allowed, cell = nba_allowed[v], ctx.loc[v, "ACTION_AGENT"]
    if not ans or len(ans.get("scores", [])) != len(allowed):
        rows.append({VAL: v, "ACTION": cell, "PROBA": 1.0, "DECIDED_BY": "fallback_cell_policy", "REASON": "invalid answer"})
        continue
    s = np.clip(np.asarray(ans["scores"], float), 0, None)
    p = (1 - EPSILON) * (s / s.sum() if s.sum() > 0 else np.full(len(s), 1 / len(s))) + EPSILON / len(s)
    rng = np.random.default_rng([SEED, int(v) % 1_000_000_007])
    k = int(rng.choice(len(allowed), p=p / p.sum()))
    rows.append({VAL: v, "ACTION": allowed[k], "PROBA": round(float(p[k] / p.sum()), 6),
                 "DECIDED_BY": "llm_nba" if k == int(np.argmax(s)) else "llm_nba_sampled", "REASON": ans.get("reason", "")})
NBA = pd.DataFrame(rows, columns=[VAL, "ACTION", "PROBA", "DECIDED_BY", "REASON"])
NBA.drop(columns="REASON").to_csv(AGENTS_DIR / "agent3_nba.csv", index=False)
if len(NBA):
    cmp = NBA.merge(ctx[["ACTION_AGENT"]], left_on=VAL, right_index=True)
    print(f"agent 3: {len(NBA):,} T2C customers; same action as the cell policy for {(cmp['ACTION'] == cmp['ACTION_AGENT']).mean():.0%}; "
          f"{NBA['DECIDED_BY'].value_counts().to_dict()}")
    display(pd.crosstab(cmp["ACTION_AGENT"], cmp["ACTION"]).rename_axis(index="cell policy (2b)", columns="agent 3"))
    show(NBA, n=5, title="agent 3 decisions (first rows)")'''),

    md("""## Step 7 · Agent 4 (④) NBO with the challenger rules (Agent 6B)

The **rules first**: for each Target 2 customer the universe rules (CC-v1 / CF-v1: families, catalogue, the
customer's relative price band) give the allowed offers. The **LLM** then scores the best `NBO_SHORTLIST` per
validity (by the rule score) for the customers of the budget; the other allowed offers rank after them. The **passes
still choose** the offers (price band, one per family, EXT-6 for slots 5 / 6), and `score_source` says `llm:<model>`.
Customers past the budget keep the rule score."""),
    code('''NBA_ACTION = NBA.set_index(VAL)["ACTION"]


def nbo(group_key, cands):
    """Agent 4, called by P6 with the candidates the rules allowed (both validities): a score per row, NaN = rule score."""
    out = pd.Series(np.nan, index=cands.index)
    work, jobs = [], []
    for v, c in cands[cands[VAL].isin(LLM_LINES)].groupby(VAL, sort=False):
        short = c.sort_values("SCORE", ascending=False).groupby("VALIDITY", group_keys=False).head(NBO_SHORTLIST)
        offers = "; ".join(f"{i}: {r.SERVICE} {r.GENEROSITY}, {int(r.VALIDITY)} days, price {r.PRICE:.2f}, "
                           f"{fmt(getattr(r, 'GB', None), 1)} GB, {fmt(getattr(r, 'MIN', None), 0)} min, rule score {r.SCORE:.2f}"
                           for i, r in enumerate(short.itertuples()))
        anchor = fmt(short["ANCHOR"].iloc[0]) if "ANCHOR" in short.columns else "-"
        work.append((c.index, short.index))
        jobs.append(("You are the next-best-offer agent of a telecom campaign engine. Score each offer 0-100: the chance "
                     "this prepaid customer buys it and it raises his spend without overpricing.",
                     f"Customer: {facts(v)}. Action: {NBA_ACTION.get(v, ctx.loc[v, 'ACTION_AGENT']) if group_key == 't2c' else 'any (free universe)'}. "
                     f"Price anchor {anchor}. Offers: {offers}. Return one score per offer, in order.",
                     {"type": "object", "properties": {"scores": ints(len(short))}, "required": ["scores"]}, 40 + 5 * len(short)))
    for (all_idx, short_idx), ans in zip(work, ask_many("Agent 4 NBO", jobs)):
        if ans and len(ans.get("scores", [])) == len(short_idx):
            out.loc[all_idx] = cands.loc[all_idx, "SCORE"] * 0.01      # allowed, not short-listed: after the LLM's
            out.loc[short_idx] = np.asarray(ans["scores"], float) / 100.0
    return out


AGENTS = {"actions": NBA, "nbo": nbo, "groups": set(config.TARGETS[AGENT_TARGET]), "label": MODEL}
p6_t2 = scoring_groups.run_p6_offers(D, force=True, target=AGENT_TARGET, agents=AGENTS)
P6 = RUN / "score" / f"P6_offers_{AGENT_TARGET}"
print(f"P6 {AGENT_TARGET}: group audit {p6_t2['audit_status']} -- {p6_t2['decisions']}")
show(pd.read_csv(P6 / "group_report.csv")[["group", "rules", "contacted", "with_offers", "llm_scored_offers", "decided_by"]],
     title="who decided, per group")'''),

    md("""## Step 8 · Agent 7 (⑦) code 5: the final file of Target 2

Code 5 of production (F-01 … F-19): 38 columns, favourites and forced bundles in empty slots, conflicts and
scripts. Only on PASS is `input_flytxt_target2.csv.gz` written."""),
    code('''fin_t2 = scoring_groups.run_target_final(D, AGENT_TARGET, force=True)
print(f"{AGENT_TARGET}: final {'READY' if fin_t2['ready'] else 'BLOCKED'}, {fin_t2['input_flytxt_rows']:,} rows")
det2 = pd.read_parquet(OUTPUT / AGENT_TARGET / f"details_{AGENT_TARGET}.parquet")
show(det2[[c for c in ("GROUP_KEY", "SEGMENT", "ACTION_AGENT", "ACTION_SENT", "DECIDED_BY", "SLOT_1", "REASON_1", "SLOT_2")
           if c in det2.columns]], n=5, title="Target 2 details (first rows)")'''),

    md("""## Step 9 · Agent 9 (⑨) auditor

**The rules block:** the group audit (A-12 universe, A-12b catalogue, A-17 rules version, A-21 no control line,
A-30 / A-31) and the file checks (A-20 … A-24) of Target 2. Any FAIL and the final file is not written.
**The LLM explains** the result for the business; it cannot change it."""),
    code('''checks = {t: {"audit": json.loads((OUTPUT / t / "audit_groups.json").read_text(encoding="utf-8"))["status"],
              "file_checks": [f"{c['id']} {c['status']}" for c in json.loads((OUTPUT / t / "summary.json").read_text(encoding="utf-8"))["output_checks"]],
              "ready": json.loads((OUTPUT / t / "summary.json").read_text(encoding="utf-8"))["ready"]} for t in config.TARGETS}
a9 = ask("Agent 9 auditor", "You are the auditor agent of a telecom campaign engine. Explain audit results to the business.",
         f"Audit of {RUN_DATE}: {json.dumps(checks)}. Say whether the target files may be handed over, and why.",
         {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["READY", "BLOCKED"]},
                                           "explanation": {"type": "string", "maxLength": 400}},
          "required": ["verdict", "explanation"]}, 200)
rule_verdict = "READY" if all(c["ready"] for c in checks.values()) else "BLOCKED"
show([{"target": t, **c, "file_checks": ", ".join(c["file_checks"])} for t, c in checks.items()], title="the rules")
display(Markdown(f"**Verdict of the rules: {rule_verdict}.** Agent 9 (LLM): {a9['explanation'] if a9 else '-'}"
                 + ("" if not a9 or a9["verdict"] == rule_verdict else "  *(the LLM disagrees: the rules win)*")))'''),

    md("""## Step 10 · Agent 8 (⑧) strategy: group a vs group b, agents vs cell policy

The LLM compares what was sent to group **a** (T2C) and group **b** (T2F), and, inside Target 2, the customers the
agents decided one by one (agents 3 + 4) with those who got the cell policy (past the budget). It proposes what to
watch or change next week. A proposal only; nothing changes by itself. The real comparison is the reward at D+7
(who bought) between group a and group b."""),
    code('''def profile(det):
    sent = det[det["IN_INPUT_FLYTXT"]] if "IN_INPUT_FLYTXT" in det.columns else det
    act = sent["ACTION_SENT"] if "ACTION_SENT" in sent.columns else sent.get("ACTION_AGENT")
    n = sum(sent[f"SLOT_{i}"].notna() & sent[f"SLOT_{i}"].astype(str).ne("") for i in range(1, 7) if f"SLOT_{i}" in sent.columns)
    return {"lines": int(len(det)), "in_file": int(len(sent)), "offers_per_line": round(float(np.mean(n)), 2) if len(sent) else 0,
            "actions": act.value_counts(normalize=True).round(3).to_dict() if act is not None and len(sent) else {}}


cmp = {"group a (T2C)": profile(det2[det2["GROUP_KEY"] == "t2c"]), "group b (T2F)": profile(det2[det2["GROUP_KEY"] == "t2f"]),
       "decided by the agents": profile(det2[det2["ACCESS_METHOD_VAL"].isin(LLM_LINES)]),
       "cell policy (past the budget)": profile(det2[~det2["ACCESS_METHOD_VAL"].isin(LLM_LINES)])}
a8 = ask("Agent 8 strategy", "You are the strategy agent of a telecom campaign engine.",
         f"This week's Target 2: {json.dumps(cmp)}. Agent 3 kept the cell policy for "
         f"{(NBA['ACTION'] == NBA[VAL].map(ctx['ACTION_AGENT'])).mean() if len(NBA) else 0:.0%} of its customers. "
         "Compare group a and group b, and the agents with the cell policy, and give 3 things to watch or change next "
         "week (the reward of this week is measured at D+7: who bought, group a against group b).",
         {"type": "object", "properties": {"comparison": {"type": "string", "maxLength": 500},
                                           "next_week": {"type": "array", "items": {"type": "string", "maxLength": 160}, "maxItems": 3}},
          "required": ["comparison", "next_week"]}, 300)
show(pd.DataFrame(cmp).T.astype(str), title="group a vs group b, agents vs cell policy")
if a8:
    display(Markdown(f"**Agent 8 (LLM):** {a8['comparison']}\\n\\n" + "\\n".join(f"- {x}" for x in a8["next_week"])))'''),

    md("""## Step 11 · The final file

The final file `final/input_flytxt.csv.gz`
holds Target 2 (groups a and b), only if it passed. The engine only **saves** files; it sends nothing: you check the file and
hand it over yourself."""),
    code('''ctrl = scoring_groups.write_control(D)
fin = scoring_groups.run_final_file(D)
print(f"control: {ctrl['lines']:,} lines | final file {'READY' if fin['ready'] else 'BLOCKED'} -- "
      f"{fin['input_flytxt_rows']:,} rows {fin['rows_per_target']}" + (f" -- {fin['reasons']}" if fin["reasons"] else ""))
tree(OUTPUT, depth=1)'''),

    md("""## Time per agent on your GPU

`seconds` is the GPU time of the calls (summed over the parallel calls), `wall` the time you waited. Cached answers
(a re-run of the same week) cost nothing. The last line extrapolates agents 3 + 4 to a production week."""),
    code('''tt = pd.DataFrame([{"agent": a, **{k: v for k, v in s.items() if k != "last_error"}} for a, s in STATS.items()])
tt = tt.assign(wall=tt.get("wall_seconds", pd.Series(0.0, index=tt.index)).fillna(0).round(1),
               seconds=tt["seconds"].round(1)).drop(columns=[c for c in ("wall_seconds",) if c in tt.columns])
show(tt, n=20, title=f"LLM calls per agent ({MODEL}, PARALLEL {PARALLEL})")
n2c = sum(1 for v in LLM_LINES if ctx.loc[v, "GROUP_KEY"] == "t2c")
measured = sum(STATS.get(a, {}).get("wall_seconds", 0.0) for a in ("Agent 3 NBA", "Agent 4 NBO"))
fresh = sum(STATS.get(a, {}).get("calls", 0) for a in ("Agent 3 NBA", "Agent 4 NBO"))
errors = sum(STATS.get(a, {}).get("errors", 0) for a in ("Agent 3 NBA", "Agent 4 NBO"))
if errors:
    print(f"agents 3 + 4: {errors} calls FAILED (those customers got the cell policy / rule score): "
          f"{STATS.get('Agent 4 NBO', STATS.get('Agent 3 NBA', {})).get('last_error')}")
if LLM_LINES and fresh == 0 and not errors:
    print("agents 3 + 4: every answer came from the cache (a re-run of this week): see step G2 for the estimate")
elif LLM_LINES and measured > 0:
    per_line = measured / len(LLM_LINES)
    print(f"agents 3 + 4: {measured / 60:.1f} min for {len(LLM_LINES):,} customers ({n2c:,} T2C) = {per_line:.2f} s per customer; "
          f"a production week of 100,000 Target-2 customers (50k a + 50k b): about {100_000 * per_line / 3600:.1f} h")
json.dump({"model": MODEL, "parallel": PARALLEL, "stats": STATS, "llm_lines": len(LLM_LINES)},
          (AGENTS_DIR / "agent_times.json").open("w", encoding="utf-8"), indent=2, default=str)'''),
]


# ---------------------------------------------------------------------------------------------------------------
# 04: the same steps with vLLM (WSL2) instead of Ollama. Only the server set-up, the settings, the LLM client and the
# G1 check differ; every agent step is shared with 03, so the two notebooks cannot drift apart.
# ---------------------------------------------------------------------------------------------------------------
VLLM_NAME = "04_vllm_agents_step_by_step.ipynb"


def _swap(text, pairs):
    for a, b in pairs:
        assert a in text, a[:70]
        text = text.replace(a, b)
    return text


VLLM_TITLE = md(_swap(TITLE.source, [
    ("# 03 · All the agents on the laptop GPU (Ollama), step by step (FSDR V4.0)",
     "# 04 · All the agents on the laptop GPU with vLLM, step by step (FSDR V4.0)"),
    ("through **Ollama** (already installed). Everything stays on this machine: the only server it calls is\n"
     "`http://127.0.0.1:11434`. There is no API and no Gradio here.",
     "served by **vLLM** in WSL2 (faster than Ollama for many customers: it batches them on the GPU). Everything stays "
     "on this\nmachine: the only server it calls is `http://127.0.0.1:8001`. There is no API and no Gradio here. Same "
     "steps and\nsame agents as notebook 03 (Ollama); only the LLM server differs."),
    ("**How to run it:** start Ollama (§ below), then **Run All**.",
     "**How to run it:** start the vLLM server (§ below, in another terminal), then **Run All**."),
    ("is Ollama on the GPU; how fast is it", "is the vLLM server up; how fast is it"),
    ("**What runs on the GPU:** only the LLM (Ollama).", "**What runs on the GPU:** only the LLM (vLLM)."),
]))

VLLM_SETUP_MD = md("""## Before you start: the vLLM server (WSL2)

vLLM runs on Linux, so on Windows it runs inside **WSL2** (Ubuntu). The Windows NVIDIA driver already gives WSL2 the
GPU: do **not** install a driver inside Ubuntu. Steps 1-3 are done once; step 4 at each session.

1. **WSL2 + Ubuntu** (PowerShell as administrator, then restart the laptop and create the Ubuntu user):
   ```powershell
   wsl --install -d Ubuntu
   wsl -e nvidia-smi          # must show the RTX PRO 5000: the GPU is visible in WSL2
   ```
2. **Install vLLM** (from the project folder; about 10 minutes, ~5 GB). The script reads the GPU: compute capability
   8.0 or more -> **vLLM 0.32.0** (CUDA 12.8+, needed by Blackwell), in `~/vllm-env` inside WSL2:
   ```powershell
   uv run python run.py vllm-install
   ```
3. **`.env`** (optional, defaults shown): `VLLM_HF_MODEL=Qwen/Qwen3-8B-AWQ`, `VLLM_SERVED_NAME=qwen3:8b`,
   `VLLM_MAX_SEQS=32`, `VLLM_GPU_UTIL=0.85`, and `VLLM_API_KEY=<a long random secret>` (the notebook reads it from
   `.env` and never prints it). For better answers: `Qwen/Qwen3-14B-AWQ` / `qwen3:14b` (about 2x slower).
4. **Serve** (a second terminal, keep it open):
   ```powershell
   uv run python run.py vllm-serve
   ```
   The first time it downloads the model from Hugging Face (~6 GB, the only download); wait for
   `Application startup complete`. Then set `VLLM_OFFLINE=1` in `.env`: nothing leaves the machine any more.
5. **Watch the GPU** while the agents run: `nvidia-smi -l 1` (vLLM takes ~85 % of the GPU memory at start: normal,
   it is its KV cache).""")

VLLM_SETTINGS = _swap(SETTINGS, [
    ('OLLAMA_URL = "http://127.0.0.1:11434"   # local only',
     'LLM_URL = "http://127.0.0.1:8001"      # the vLLM server (uv run python run.py vllm-serve), local only'),
    ('MODEL = "auto"                  # "auto" = the best installed of qwen3:14b, qwen3:8b, qwen2.5:14b, llama3.1:8b, ...',
     'MODEL = "auto"                  # "auto" = the model the server serves (VLLM_SERVED_NAME in .env)'),
    ("PARALLEL = 4                    # customers sent to Ollama at the same time (set OLLAMA_NUM_PARALLEL to the same)",
     "PARALLEL = 32                   # customers sent at the same time; vLLM batches them (VLLM_MAX_SEQS = the same)"),
])

VLLM_HELPERS = _swap(LLM_HELPERS, [
    ("# --- the LLM client: Ollama on this machine,", "# --- the LLM client: vLLM on this machine (OpenAI API),"),
    ('assert urlparse(OLLAMA_URL).hostname in ("127.0.0.1", "localhost", "::1"), "local only: OLLAMA_URL must be this machine"',
     'assert urlparse(LLM_URL).hostname in ("127.0.0.1", "localhost", "::1"), "local only: LLM_URL must be this machine"\n'
     'try:                                            # the server key: from the environment or .env, never printed\n'
     '    from dotenv import dotenv_values\n'
     '    _KEY = os.environ.get("VLLM_API_KEY") or dotenv_values(ROOT / ".env").get("VLLM_API_KEY") or ""\n'
     'except ImportError:\n'
     '    _KEY = os.environ.get("VLLM_API_KEY", "")'),
    ('    req = urllib.request.Request(OLLAMA_URL + path, data=None if body is None else json.dumps(body).encode("utf-8"),\n'
     '                                 headers={"Content-Type": "application/json"})',
     '    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {_KEY}"} if _KEY else {})}\n'
     '    req = urllib.request.Request(LLM_URL + path, data=None if body is None else json.dumps(body).encode("utf-8"),\n'
     '                                 headers=headers)'),
    ('json.dumps([MODEL, system, user, schema], sort_keys=True)',
     'json.dumps(["vllm", MODEL, system, user, schema], sort_keys=True)'),
    ('    body = {"model": MODEL, "stream": False, "format": schema, "keep_alive": "30m",\n'
     '            "options": {"temperature": 0, "seed": SEED, "num_ctx": 4096, "num_predict": num_predict},\n',
     '    body = {"model": MODEL, "temperature": 0, "seed": SEED, "max_tokens": num_predict,\n'
     '            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": _plain(schema)}},\n'),
    ('def _stat(agent):',
     'def _plain(schema):\n'
     '    """The schema without number ranges and string lengths: every vLLM version compiles it with its fast grammar\n'
     '    engine (old ones fall back to a slow one and can crash). The code enforces them anyway (scores are clipped,\n'
     '    actions and offers checked against the allowed ids)."""\n'
     '    if isinstance(schema, dict):\n'
     '        return {k: _plain(v) for k, v in schema.items() if k not in ("minimum", "maximum", "maxLength", "minLength")}\n'
     '    return [_plain(v) for v in schema] if isinstance(schema, list) else schema\n'
     '\n\n'
     'def _stat(agent):'),
    ('        body["think"] = False', '        body["chat_template_kwargs"] = {"enable_thinking": False}'),
    ('        r = _http("/api/chat", body)\n        answer = json.loads(r["message"]["content"])',
     '        r = _http("/v1/chat/completions", body)\n        answer = json.loads(r["choices"][0]["message"]["content"])'),
    ('        st["prompt_tokens"] += r.get("prompt_eval_count", 0)\n        st["output_tokens"] += r.get("eval_count", 0)',
     '        st["prompt_tokens"] += r.get("usage", {}).get("prompt_tokens", 0)\n'
     '        st["output_tokens"] += r.get("usage", {}).get("completion_tokens", 0)'),
])

VLLM_G1 = [md("""## G1 · The GPU and the vLLM server

This cell only reads. It checks the GPU (`nvidia-smi`), that the vLLM server answers on this machine, which model it
serves (and its context length), picks `MODEL` and sends a first request."""),
           code('''import shutil, subprocess

smi = shutil.which("nvidia-smi")
if smi:
    q = subprocess.run([smi, "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    print("GPU    :", q or "nvidia-smi gave no answer")
else:
    print("GPU    : nvidia-smi not found: vLLM needs the NVIDIA driver")

try:
    served = _http("/v1/models", timeout=5)["data"]
except OSError as e:
    raise SystemExit(f"vLLM does not answer at {LLM_URL}: in another terminal run `uv run python run.py vllm-serve`, "
                     f"wait for 'Application startup complete', then run this cell again ({e})")
ids = [m["id"] for m in served]
print("vLLM   :", ", ".join(f"{m['id']} (context {m.get('max_model_len', '-')})" for m in served), "at", LLM_URL)
if MODEL == "auto":
    MODEL = ids[0]
if MODEL not in ids:
    raise SystemExit(f"the server serves {ids}, not {MODEL!r}: set MODEL = 'auto' (or VLLM_SERVED_NAME in .env)")
THINK_OFF = "qwen3" in MODEL.lower()               # qwen3 "thinks" by default: off, the answers are short JSON
print("MODEL  :", MODEL)
t = time.time()
ok = ask("G1 first request", "You check a server.", "Answer ok = true.",
         {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}, 10, False)
print(f"first answer in {time.time() - t:.1f} s: {ok}")
if ok is None:
    raise SystemExit(f"the server answered with an error: {STATS['G1 first request'].get('last_error')}")''')]


def build(out_dir: Path = HERE, backend: str = "ollama") -> Path:
    """backend "ollama" -> 03_gpu_agents_step_by_step.ipynb, "vllm" -> 04_vllm_agents_step_by_step.ipynb."""
    vllm = backend == "vllm"
    mods = sb.engine_modules(ROOTS)
    helpers = sb._use_imports(GPU_SETUP)
    part_b = md("## Part B · The steps\n\nThe engine is loaded. Each step below calls it and shows what it wrote.")
    cells = [VLLM_TITLE if vllm else TITLE, VLLM_SETUP_MD if vllm else OLLAMA_MD,
             code(VLLM_SETTINGS if vllm else SETTINGS), *sb._engine_cells(mods), part_b, code(helpers),
             code(VLLM_HELPERS if vllm else LLM_HELPERS)]
    for c in (VLLM_G1 + CELLS[2:] if vllm else CELLS):
        cells.append(code(sb._use_imports(c.source)) if c.cell_type == "code" else c)
    cells += [md("## Proof: nothing came from the project folder\n\nEvery engine module was defined by a cell of this "
                 "notebook, vowpalwabbit was never loaded, and no API / Gradio package was imported."),
              code(sb.PROOF + '\nprint("api / gradio loaded:", any(m in sys.modules for m in ("gradio", "fastapi", "uvicorn")))')]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    p = Path(out_dir) / (VLLM_NAME if vllm else NAME)
    nbf.write(nb, p)
    return p


if __name__ == "__main__":
    print("written", build())
    print("written", build(backend="vllm"))
