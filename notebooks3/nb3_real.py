"""Cells of notebooks3/04_agents_nba_fast_ollama_real25k.ipynb: notebook 03 (Ollama, FAST_MODE) on the REAL week of
notebook 02, with:

1. the dates read from the data: D = the most recent purchase day (subs_btl_auto), T = the campaign day (date1 of
   targeting_dataiku_auto); purchases counted from T to D; training (*_auto) anchored on T, scoring (*_score) on D;
2. training on the whole real prepaid base (every eligible line outside the lists; TRAIN_LINES can sample it), and
   RUN_SIZE target lines scored (1k / 10k / 25k / 50k / all), everything under data/real25k/;
3. an LLM that does not stop: no budget, retries with back-off, Ollama restarted, answers cached at once, Windows kept
   awake, a pre-flight check, a progress line every 100 answers (and a log);
4. 6 offers for every line: why some lines had fewer, then a top-up with the engine's own rules, and a blocking audit;
5. the Proof cell accepts a preloaded vowpalwabbit (never used by the engine); the timer keeps only sound times;
6. a final report.
Every change lives in the notebook; the engine (nbx/) is not modified.
"""

TITLE = r'''# 04 · The real run on the GPU (Ollama): the whole prepaid base to train, RUN_SIZE lines scored, every customer decided by the LLM

The same notebook as `03_agents_nba_fast_ollama_gpu.ipynb` (FAST_MODE, every agent on **your laptop GPU** through
**Ollama**, only `http://127.0.0.1:11434` is called), on **the real week prepared by notebook 02**
(`data/weekly/<D>/`), with these changes:

| | What changes |
|---|---|
| **Dates** | read from the data, never typed: **D** = the most recent purchase day (`subs_btl_auto`), **T** = the campaign day (`date1` of `targeting_dataiku_auto`). Purchases (takers) are counted **from T to D**; training (`*_auto`) is anchored on **T**, scoring (`*_score`) on **D**. The run stops when there is more than one `date1`, when fewer than 95 % of the purchases fall between T and D, or when an `*_auto` window goes past T. |
| **Data** | step 0 builds `data/real25k/weekly/<D>/` from the full week of notebook 02: **training on the whole real prepaid base** (every eligible line of `base_prepaid` outside the lists, `TRAIN_LINES = "all"`), and **`RUN_SIZE` target lines scored** (`1_000`, `10_000`, `25_000`, `50_000` or `"all"`, drawn at random from the lists of `SCORE_GROUPS` and split between them). Every output goes to `data/real25k/`. No generator. |
| **LLM** | **no time budget, no line limit**: every customer gets an LLM answer, nobody gets the cell policy. Each request is retried 6 times (timeouts, connection errors, invalid JSON; back-off 2 … 60 s), Ollama is restarted when it does not answer, every answer is cached at once (a re-run resumes where it stopped), Windows is kept awake, a **pre-flight check** runs first, and a progress line with the ETA is printed every 100 answers (and written to `agents/llm_progress.log`). |
| **6 offers** | step 7b tops up the empty slots with the engine's own rules (the short list, then the same action's offers, then the favourites, then the segment's other allowed actions); every added offer is in the catalogue, has both scripts, and does not conflict with or repeat another slot. The auditor blocks the file unless every row has **6 distinct, valid offers** (A-25). Step 8b shows why lines had fewer than 6 and what was added. |
| **Report** | the final report: dates, takers and taker rate, agent 3 propensities, offers per line before / after the top-up, LLM answers (fresh / cached / retried / failed), the cell-policy share (must be 0), customers per minute. |

Run it from Jupyter (**Restart & Run All**) or headless: `python notebooks3/run_notebook04.py` (no cell timeout).
**Run order:** `02_prepare_week` (the real `data_raw/`) → **this notebook**.'''

# ---------------------------------------------------------------------------------------------------------------------
# settings (ag.SETTINGS up to "import time as _time", then these lines)
# ---------------------------------------------------------------------------------------------------------------------
SETTINGS_SWAPS = [
    ('RUN_DATE = "2026-11-17"         # the run date D (the SAME as in notebook 02): data/weekly/<D>/',
     'RUN_DATE = None                 # set by step 0 from the data: D = the most recent purchase day (never typed)'),
    ("TIME_BUDGET_MIN = 20            # minutes for the per-customer agents 3 + 4; customers past it get the cell policy",
     'TIME_BUDGET_MIN = float("inf")  # no time budget: every customer gets an LLM answer, nobody gets the cell policy'),
]

REAL_SETTINGS = r'''
# --- notebook 04 real: the real week of notebook 02, the dates from the data, the sizes ---------------------------------
RUN_SIZE = 25_000                 # target lines SCORED: 1_000 | 10_000 | 25_000 | 50_000 | "all" (the full lists, ~50k each)
                                  #   drawn at random from the lists of SCORE_GROUPS, split between the groups
                                  #   (25_000 with ["t2c", "t2f"] = 12,500 + 12,500; with ["t2c"] = 25,000 of group a)
TRAIN_LINES = "all"               # training: "all" = every eligible line of the real base_prepaid outside the lists;
                                  #   a number (e.g. 25_000) = that many random eligible lines (a faster test)
REBUILD_COPY = False              # False = reuse data/real25k/weekly/<D> when it was built from the same week and settings
MIN_PURCHASES_IN_WINDOW = 0.95    # stop when fewer purchases of subs_btl_auto fall between T and D
STRICT_AUTO_WINDOW = False        # False: an *_auto window may end ON T (the subscription windows include their anchor
                                  #   day, BR 4.1) but never after it; True: every *_auto window must end BEFORE T
TOPUP_DROPPED = True              # the lines code 5 drops (F-04: no offer in slots 1-4) are topped up and kept in the file
LLM_RETRIES = 6                   # per answer: timeouts, connection errors, invalid JSON (back-off 2, 4, 8, 16, 32, 60 s)
LLM_TIMEOUT_S = 300               # seconds per request before a retry
PREFLIGHT_STRICT = True           # stop when the pre-flight check fails (Ollama up, model, 100 % GPU, OLLAMA_NUM_PARALLEL)
TEST_10K = True                   # internal: the base is a copy; agent 2b and the NBA training table read TRAIN_VALS only
'''

STEP0 = [("md", r'''## Step 0 · The dates from the data, and the real copy (whole base to train, `RUN_SIZE` lines to score)

**The dates** (never typed), from the week folder notebook 02 wrote (`data/weekly/<D>/`, the latest one):
- **D** = the most recent purchase day of `subs_btl_auto` (the scoring anchor);
- **T** = the campaign day = the `date1` of `targeting_dataiku_auto` (the training anchor);
- purchases (takers) are counted **from T to D**.

The run **stops** when `targeting_dataiku_auto` holds more than one `date1`, when fewer than `MIN_PURCHASES_IN_WINDOW`
(95 %) of the purchases fall between T and D, or when the week folder was prepared for another date.

**The copy** `data/real25k/weekly/<D>/`, from the full week of notebook 02 (run 02 with `RUN_SIZE = "all"`):
- **training:** `TRAIN_LINES = "all"` = every eligible line of the real `base_prepaid` (8 digits, outside both lists);
  the tables are linked whole (no copy, no filter). A number = that many random eligible lines, the tables filtered;
- **scoring:** `RUN_SIZE` lines drawn at random (seed `SEED`) from the lists of `SCORE_GROUPS`, split between the
  groups; `"all"` = every line of the lists. Its `_manifest.json` carries D and T. It is reused when it was built from
the same week with the same settings (`REBUILD_COPY = False`). Runs, cache and outputs go to `data/real25k/`.'''),
         ("code", r'''import csv, gzip, io, json, os, shutil, time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path.cwd()
while not (_root / "pyproject.toml").exists() and _root != _root.parent:
    _root = _root.parent
REAL = _root / "data" / "real25k"
_weeks = sorted(p for p in (_root / "data" / "weekly").glob("20??-??-??") if (p / "_manifest.json").exists())
if not _weeks:
    raise SystemExit("no week prepared by notebook 02 in data/weekly/: run notebooks3/02_prepare_week.ipynb on the real data_raw first")
SRC_WEEK = _weeks[-1]
SRC_MAN = json.loads((SRC_WEEK / "_manifest.json").read_text(encoding="utf-8"))


def _rd(p, **kw):
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig", **kw)


def _dates(p, name):
    cols = pd.read_csv(p, nrows=0, encoding="utf-8-sig").columns
    real = next((c for c in cols if c.strip().lower() == name.lower()), None)
    if real is None:
        raise SystemExit(f"{p.name}: no column {name}")
    return pd.to_datetime(_rd(p, usecols=[real])[real], errors="coerce").dt.date.dropna()


_p_btl = next(SRC_WEEK.rglob("subs_btl_auto.csv.gz"), None)
_p_tg = next(SRC_WEEK.rglob("targeting_dataiku_auto.csv.gz"), None)
if _p_btl is None or _p_tg is None:
    raise SystemExit(f"{SRC_WEEK}: subs_btl_auto and targeting_dataiku_auto are both needed to read D and T")
_buy, _d1 = _dates(_p_btl, "FULFILMENTSUCCESSDATE_TIME"), _dates(_p_tg, "date1")
if _buy.empty or _d1.empty:
    raise SystemExit("subs_btl_auto or targeting_dataiku_auto has no readable date: check the exports")
D_DATE, _campaigns = _buy.max(), sorted(set(_d1))
if len(_campaigns) != 1:
    raise SystemExit(f"targeting_dataiku_auto holds {len(_campaigns)} campaign dates ({_campaigns[0]} .. {_campaigns[-1]}): "
                     "T must be ONE campaign day. Export the last campaign only (one date1), then run notebook 02 again")
T_DATE = _campaigns[0]
if T_DATE > D_DATE:
    raise SystemExit(f"the campaign day T = {T_DATE} is after the last purchase day D = {D_DATE}: check the exports")
_share = float(((_buy >= T_DATE) & (_buy <= D_DATE)).mean())
if _share < MIN_PURCHASES_IN_WINDOW:
    raise SystemExit(f"only {_share:.1%} of the {len(_buy):,} purchases fall between T = {T_DATE} and D = {D_DATE} "
                     f"(minimum {MIN_PURCHASES_IN_WINDOW:.0%}): export subs_btl for the purchases from T to D")
if SRC_WEEK.name != D_DATE.isoformat():
    raise SystemExit(f"data/weekly/{SRC_WEEK.name} was prepared for {SRC_WEEK.name}, but its last purchase day is {D_DATE}: "
                     "run notebook 02 with RUN_DATE = None")
RUN_DATE, CAMPAIGN_DATE = D_DATE.isoformat(), T_DATE.isoformat()
PURCHASE_DAYS = (D_DATE - T_DATE).days + 1
print(f"dates: D = {D_DATE} (most recent purchase, subs_btl_auto) | T = {T_DATE} (campaign date1, targeting_dataiku_auto) | "
      f"purchases counted T .. D = {PURCHASE_DAYS} days ({_share:.1%} of {len(_buy):,} purchases inside) | "
      "training *_auto at T, scoring *_score at D")

# --- the real copy: the whole base to train, RUN_SIZE lines to score --------------------------------------------------
if not (RUN_SIZE == "all" or (isinstance(RUN_SIZE, int) and RUN_SIZE > 0)):
    raise SystemExit(f'RUN_SIZE = {RUN_SIZE!r}: 1_000, 10_000, 25_000, 50_000 or "all"')
if not (TRAIN_LINES == "all" or (isinstance(TRAIN_LINES, int) and TRAIN_LINES > 0)):
    raise SystemExit(f'TRAIN_LINES = {TRAIN_LINES!r}: "all" or a number of lines')
LIST_KEYS = {"t2c": "target2/a_constraint/target2_a", "t2f": "target2/b_free/target2_b"}
SIZE_ENV = {"t2c": "FSDR_T2_CONSTRAINT_SIZE", "t2f": "FSDR_T2_FREE_SIZE"}
DST = REAL / "weekly" / RUN_DATE


def _canon_id(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def _canon_val(s):
    s = _canon_id(s)
    return s.where(~s.str.fullmatch(r"\d{9,}"), s.str[-8:])


def _find(cols, *names):
    up = {str(c).upper(): c for c in cols}
    return [up[n] for n in names if n in up]


def _headerless(p):
    opener = gzip.open if p.name.endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8-sig", errors="replace") as f:
        first = next(csv.reader(io.StringIO(f.readline())), [])
    vals = [h.strip() for h in first if h.strip()]
    return bool(vals) and sum(bool(pd.Series([v]).str.fullmatch(r"-?\d+([.,]\d+)?|\d{4}-\d{2}-\d{2}.*").iloc[0]) for v in vals) / len(vals) >= 0.5


_lists = {}
for _k in SCORE_GROUPS:
    _p = SRC_WEEK / f"{LIST_KEYS[_k]}.csv.gz"
    if not _p.exists():
        raise SystemExit(f"no list {LIST_KEYS[_k]} in {SRC_WEEK}: SCORE_GROUPS = {SCORE_GROUPS} needs it (notebook 02 writes it)")
    _df = _rd(_p)
    _c = _find(_df.columns, "ACCESS_METHOD_VAL", "MSISDN")
    if not _c:
        raise SystemExit(f"{_p}: no access_method_val column")
    _lists[_k] = _df.assign(access_method_val=_canon_val(_df[_c[0]]))
_rng_s = np.random.default_rng(SEED + 1)                           # the scored lines: RUN_SIZE, split between the groups
_full = sum(len(v) for v in _lists.values())
if RUN_SIZE != "all":
    _quota = {k: RUN_SIZE // len(_lists) + (i < RUN_SIZE % len(_lists)) for i, k in enumerate(_lists)}
    for _k in _lists:
        if _quota[_k] < len(_lists[_k]):
            _lists[_k] = _lists[_k].iloc[np.sort(_rng_s.choice(len(_lists[_k]), _quota[_k], replace=False))]
        else:
            print(f"note: RUN_SIZE asks {_quota[_k]:,} lines of {_k}, its list holds {len(_lists[_k]):,}: all of them")
print(f"scored: {sum(len(v) for v in _lists.values()):,} of the {_full:,} lines of the lists "
      f"(RUN_SIZE = {RUN_SIZE!r}: " + ", ".join(f"{k} {len(v):,}" for k, v in _lists.items()) + ")")
_want = {"source_created": SRC_MAN.get("created"), "campaign_date": CAMPAIGN_DATE, "train_lines_setting": TRAIN_LINES,
         "run_size": RUN_SIZE, "score_groups": list(SCORE_GROUPS), "seed": SEED}
_info_p = DST / "_sample_info.json"
_old = json.loads(_info_p.read_text(encoding="utf-8")) if _info_p.exists() else {}
if not REBUILD_COPY and all(_old.get(k) == v for k, v in _want.items()):
    print(f"copy reused: {DST} ({_old['train_lines']:,} training lines + {_old['score_lines']:,} target lines)")
else:
    _t0 = time.time()
    rng = np.random.default_rng(SEED)
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    _base = _rd(next(SRC_WEEK.rglob("base_prepaid.csv.gz")))
    _VALc, _IDc, _ACCc, _PARTYc = (_find(_base.columns, n)[0] for n in ("ACCESS_METHOD_VAL", "ACCESS_METHOD_ID", "ACCOUNT_ID", "PARTY_ID"))
    _bval = _canon_val(_base[_VALc])
    _pool = pd.Index(_bval[_bval.str.fullmatch(r"\d{8}") & (_bval > "69999999")].unique())     # BR-02 eligible MSISDNs
    _listed = pd.Index([])
    for _k in LIST_KEYS:                                     # outside BOTH lists, whatever is scored
        _p = SRC_WEEK / f"{LIST_KEYS[_k]}.csv.gz"
        if _p.exists():
            _d = _rd(_p)
            _c = _find(_d.columns, "ACCESS_METHOD_VAL", "MSISDN")
            if _c:
                _listed = _listed.append(pd.Index(_canon_val(_d[_c[0]])))
    _train_pool = _pool.difference(_listed)
    if TRAIN_LINES == "all" or TRAIN_LINES >= len(_train_pool):
        train_vals = set(_train_pool)                            # the whole eligible base, outside the lists
    else:
        train_vals = set(rng.choice(_train_pool, TRAIN_LINES, replace=False))
    WHOLE = TRAIN_LINES == "all"                                 # the tables are linked whole: no filter
    score_vals = set().union(*[set(v["access_method_val"]) for v in _lists.values()])
    VALS = train_vals | score_vals
    _rows_b = _base[_bval.isin(VALS)]
    IDS, ACCOUNTS, PARTIES = (set(_canon_id(_rows_b[c])) - {""} for c in (_IDc, _ACCc, _PARTYc))

    def _keep_mask(df):
        ids, vals = _find(df.columns, "ACCESS_METHOD_ID"), _find(df.columns, "ACCESS_METHOD_VAL", "MSISDN")
        if ids or vals:
            m = pd.Series(False, index=df.index)
            for c in ids:
                m |= _canon_id(df[c]).isin(IDS)
            for c in vals:
                m |= _canon_val(df[c]).isin(VALS)
            return m, "+".join(ids + vals)
        acc, party = _find(df.columns, "ACCOUNT_ID"), _find(df.columns, "PARTY_ID")
        if acc or party:
            m = pd.Series(False, index=df.index)
            for c in acc:
                m |= _canon_id(df[c]).isin(ACCOUNTS)
            for c in party:
                m |= _canon_id(df[c]).isin(PARTIES)
            return m, "+".join(acc + party)
        return None, "copied whole"

    _files = sorted([p for p in SRC_WEEK.rglob("*.csv.gz") if p.relative_to(SRC_WEEK).parts[0] not in ("target1", "target2", "control")],
                    key=lambda p: (not p.name.startswith("base_prepaid"), not p.name.startswith("consumer_aact_account"), str(p)))
    for p in _files:
        out = DST / p.relative_to(SRC_WEEK)
        out.parent.mkdir(parents=True, exist_ok=True)
        if WHOLE:                                                # a hard link (no copy, no disk), else a copy
            try:
                os.link(p, out)
            except OSError:
                shutil.copy2(p, out)
            continue
        if _headerless(p):
            shutil.copy2(p, out)
            print(f"{p.name}: no header, copied whole")
            continue
        first, n_in, n_out, how = True, 0, 0, ""
        for chunk in _rd(p, chunksize=500_000):
            n_in += len(chunk)
            m, how = _keep_mask(chunk)
            part = chunk if m is None else chunk[m]
            n_out += len(part)
            if p.name.startswith("consumer_aact_account"):
                pc = _find(part.columns, "PARTY_ID")
                if pc:
                    PARTIES |= set(_canon_id(part[pc[0]])) - {""}
            part.to_csv(out, mode="w" if first else "a", header=first, index=False,
                        compression={"method": "gzip", "compresslevel": 6})
            first = False
        if first:
            shutil.copy2(p, out)                                     # header-only file
        print(f"{str(p.relative_to(SRC_WEEK)):72s} {n_in:>12,} -> {n_out:>9,}  [{how}]")
    for _k, rel in LIST_KEYS.items():                            # the lists of SCORE_GROUPS in full, the others empty
        dst = DST / f"{rel}.csv.gz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        (_lists[_k] if _k in _lists else pd.DataFrame(columns=["access_method_val"])).to_csv(
            dst, index=False, compression={"method": "gzip", "compresslevel": 6})
    pd.DataFrame({"access_method_val": sorted(train_vals)}).to_csv(DST / "_train_lines.csv", index=False)
    _man = dict(SRC_MAN, run_date=RUN_DATE, anchor_date=CAMPAIGN_DATE, source=f"real25k copy of data/weekly/{SRC_WEEK.name}",
                train_lines=len(train_vals), score_lines=len(score_vals), score_groups=list(SCORE_GROUPS))
    (DST / "_manifest.json").write_text(json.dumps(_man, indent=2, default=str), encoding="utf-8")
    _old = dict(_want, train_lines=len(train_vals), score_lines=len(score_vals),
                env_sizes={SIZE_ENV[k]: (len(_lists[k]) if k in _lists else 0) for k in LIST_KEYS},
                score_group=SCORE_GROUPS[0], run_date=RUN_DATE, anchor_date=CAMPAIGN_DATE)
    _info_p.write_text(json.dumps(_old, indent=2, default=str), encoding="utf-8")
    print(f"copy written: {DST} -- {len(train_vals):,} training lines" + (" (the whole eligible base)" if WHOLE else "")
          + f" + {len(score_vals):,} target lines, {len(_files)} tables "
          + ("linked whole" if WHOLE else "filtered") + f" in {time.time() - _t0:,.0f} s")

_info = _old
os.environ["FSDR_WEEKLY_DIR"] = str(REAL / "weekly")
os.environ["FSDR_RUNS_DIR"] = str(REAL / "runs")
os.environ["FSDR_CACHE_DIR"] = str(REAL / "cache")
os.environ["FSDR_OUTPUT_DIR"] = str(REAL / "output")
os.environ["FSDR_CACHE_PARTITIONS"] = "8"
for _k, _v in _info["env_sizes"].items():
    os.environ[_k] = str(_v)                                       # a group that is not scored has size 0
TRAIN_VALS = set(_rd(DST / "_train_lines.csv")["access_method_val"])
GENERATE = GENERATE_TARGETS = False
print(f"RUN_SIZE {RUN_SIZE!r} | training lines {len(TRAIN_VALS):,} | scored: " + ", ".join(f"{k} {_info['env_sizes'][SIZE_ENV[k]]:,}" for k in SCORE_GROUPS)
      + f" | outputs in {REAL}")''')]

REAL_PATCH = r'''
# --- the dates of the real run: T = the campaign day, purchases from T to D (before any engine step) -------------------
from datetime import date as _date
_cal = use("nbx.fsdr", "calendar")
_T = _date.fromisoformat(CAMPAIGN_DATE)
_cal.anchor_date = lambda d: _T                                  # training (*_auto) anchored on T
_n = (D - _T).days
config.REWARD_WINDOW = (-_n, 0)                                   # takers: purchases from T to D
config.TARGETING_WINDOW = (-_n, -_n)                              # last week's campaign: the day T
_ends = {**{f"activities_*_auto {p}": w.end for p, w in _cal.activity_windows(_T).items()},
         **{f"subs_amt_auto {p}": w.end for p, w in _cal.subs_windows(_T).items()},
         **{f"paygo_activity_auto {p}": w.end for p, w in _cal.payg_windows(_T).items()}}
_bad = {k: v for k, v in _ends.items() if (v >= _T if STRICT_AUTO_WINDOW else v > _T)}
if _bad:
    raise SystemExit(f"*_auto windows reach T = {_T}: {_bad} -- the training data must describe the customer before the campaign")
print(f"engine dates: D = {D}, T = {_T}; takers counted {_cal.campaign_windows(D)['reward'].start} .. "
      f"{_cal.campaign_windows(D)['reward'].end} ({_n + 1} days); the *_auto windows end on or before {max(_ends.values())}")'''

RESILIENCE = [("md", r'''## The LLM does not stop: retries, Ollama restart, cache, progress, Windows awake

- **Retries:** each request is retried `LLM_RETRIES` (6) times on a timeout, a connection error or an invalid JSON
  answer (one that does not match the schema), with a back-off of 2, 4, 8, 16, 32, 60 s. A retry changes the seed, so
  an invalid answer is not repeated.
- **Ollama restart:** when Ollama does not answer, `ollama serve` is started and the request waits for it (up to 60 s).
- **Cache:** every answer is appended to the cache file at once (`agents/llm_cache.jsonl`, and the shared
  `data/real25k/runs/_llm_cache.jsonl` from step 5b): a re-run resumes where it stopped.
- **Progress:** every 100 answers, one line with the speed and the ETA, printed and written to `agents/llm_progress.log`.
- **Windows awake:** sleep is blocked while the notebook runs (released by the final report).'''),
              ("code", r'''import ctypes, http.client, random as _rnd, shutil as _sh, subprocess as _sp, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

LLM_LOG = AGENTS_DIR / "llm_progress.log"
_RESTART = threading.Lock()


def llm_log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with LLM_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _valid(ans, schema):
    """The answer has the schema's shape: required keys, array lengths, enums (else it is retried)."""
    if schema.get("type") == "object":
        return isinstance(ans, dict) and all(k in ans for k in schema.get("required", [])) and all(
            _valid(ans[k], s) for k, s in schema.get("properties", {}).items() if k in ans)
    if schema.get("type") == "array":
        return (isinstance(ans, list) and schema.get("minItems", 0) <= len(ans) <= schema.get("maxItems", 10**9)
                and all(_valid(a, schema.get("items", {})) for a in ans))
    if "enum" in schema:
        return ans in schema["enum"]
    if schema.get("type") == "integer":
        return isinstance(ans, (int, float)) and schema.get("minimum", -1e18) <= ans <= schema.get("maximum", 1e18)
    return True


def ollama_alive():
    try:
        _http("/api/version", timeout=5)
        return True
    except Exception:
        return False


def restart_ollama():
    with _RESTART:
        if ollama_alive():
            return True
        exe = _sh.which("ollama")
        if not exe:
            llm_log("Ollama does not answer and the ollama command is not found: start the Ollama app")
            return False
        llm_log("Ollama does not answer: starting 'ollama serve'")
        _sp.Popen([exe, "serve"], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                  creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0))
        for _ in range(30):
            time.sleep(2)
            if ollama_alive():
                llm_log("Ollama is back")
                return True
        llm_log("Ollama did not come back within 60 s")
        return False


def ask(agent, system, user, schema, num_predict=256, use_cache=True):
    """One JSON answer of the local LLM, retried up to LLM_RETRIES times; None only after every retry failed."""
    key = hashlib.sha256(json.dumps([MODEL, system, user, schema], sort_keys=True).encode()).hexdigest()
    st = _stat(agent)
    with _lock:
        st.setdefault("retried", 0)
        if use_cache and key in _cache:
            st["cached"] += 1
            return _cache[key]
    last = ""
    for attempt in range(LLM_RETRIES + 1):
        body = {"model": MODEL, "stream": False, "format": schema, "keep_alive": "60m",
                "options": {"temperature": 0, "seed": SEED + attempt, "num_ctx": 4096, "num_predict": num_predict},
                "messages": [{"role": "system", "content": system + " Reply with compact JSON only."},
                             {"role": "user", "content": user}]}
        if THINK_OFF:
            body["think"] = False
        t = time.time()
        try:
            r = _http("/api/chat", body, timeout=LLM_TIMEOUT_S)
            answer = json.loads(r["message"]["content"])
            if not _valid(answer, schema):
                raise ValueError("the answer does not match the schema")
            with _lock:
                st["calls"] += 1
                st["seconds"] += time.time() - t
                st["prompt_tokens"] += r.get("prompt_eval_count", 0)
                st["output_tokens"] += r.get("eval_count", 0)
                if use_cache:
                    _cache[key] = answer
                    with LLM_CACHE.open("a", encoding="utf-8") as f:       # at once: a re-run resumes from here
                        f.write(json.dumps({"key": key, "agent": agent, "answer": answer}) + "\n")
            return answer
        except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException, urllib.error.URLError) as e:
            last = f"{type(e).__name__}: {str(e)[:160]}"
            if attempt == LLM_RETRIES:
                break
            with _lock:
                st["retried"] += 1
            if isinstance(e, (OSError, urllib.error.URLError, http.client.HTTPException)) and not ollama_alive():
                restart_ollama()
            time.sleep(min(60, 2 ** (attempt + 1)) * (0.8 + 0.4 * _rnd.random()))
    with _lock:
        st["errors"] += 1
        st["last_error"] = last
    llm_log(f"[llm] {agent}: an answer FAILED after {LLM_RETRIES} retries ({last})")
    return None


def ask_many(agent, jobs):
    """jobs: (system, user, schema, num_predict[, use_cache]); PARALLEL at a time; answers in order; progress every 100."""
    t, n, done = time.time(), len(jobs), 0
    out = [None] * n
    with ThreadPoolExecutor(max_workers=max(1, PARALLEL)) as ex:
        futs = {ex.submit(ask, agent, *j): i for i, j in enumerate(jobs)}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if n >= 100 and (done % 100 == 0 or done == n):
                el = time.time() - t
                st = STATS.get(agent, {})
                llm_log(f"[llm] {agent}: {done:,}/{n:,} answers | {done / max(el, 1e-9) * 60:,.0f}/min | "
                        f"ETA {_mm((n - done) * el / max(done, 1))} | cached {st.get('cached', 0):,} | "
                        f"retried {st.get('retried', 0):,} | failed {st.get('errors', 0):,}")
    _stat(agent)["wall_seconds"] = _stat(agent).get("wall_seconds", 0.0) + time.time() - t
    return out


_AWAKE = 0x80000000 | 0x00000001                                   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
if sys.platform == "win32":
    ctypes.windll.kernel32.SetThreadExecutionState(_AWAKE)
    print("Windows kept awake until the final report")
print(f"LLM client: {LLM_RETRIES} retries, {LLM_TIMEOUT_S} s per request, progress in {LLM_LOG.relative_to(ROOT)}")''')]

PREFLIGHT = [("md", r'''## Pre-flight check

Before the long run: Ollama answers, `MODEL` is installed, the model is **100 % in GPU memory**, and
`OLLAMA_NUM_PARALLEL` (seen by this process) equals `PARALLEL`. With `PREFLIGHT_STRICT = True` a failed check stops
the notebook here, with what to fix.'''),
             ("code", r'''_pf = []
if not ollama_alive() and not restart_ollama():
    _pf.append("Ollama does not answer on " + OLLAMA_URL + ": open the Ollama app (or ollama serve)")
else:
    _tags = [m["name"] for m in _http("/api/tags", timeout=30).get("models", [])]
    if MODEL not in _tags:
        _pf.append(f"model {MODEL} is not installed (ollama pull {MODEL})")
    _ps = next((m for m in _http("/api/ps", timeout=30).get("models", []) if m["name"] == MODEL), None)
    _gpu = None if _ps is None else _ps.get("size_vram", 0) / max(_ps.get("size", 1), 1)
    if _gpu is None or _gpu < 0.99:
        _pf.append(f"the model is {'not loaded' if _gpu is None else f'only {_gpu:.0%} in GPU memory'}: choose a smaller "
                   "MODEL or close the other GPU programs, then run from G1 again")
_np = os.environ.get("OLLAMA_NUM_PARALLEL")
if _np is None or str(_np).strip() != str(PARALLEL):
    _pf.append(f"OLLAMA_NUM_PARALLEL = {_np} in this process, PARALLEL = {PARALLEL}: run setx OLLAMA_NUM_PARALLEL {PARALLEL}, "
               "then restart Ollama AND Jupyter (or set PARALLEL to the server's value)")
print("pre-flight:", "OK" if not _pf else "FAILED")
for _x in _pf:
    print(" -", _x)
if _pf and PREFLIGHT_STRICT:
    raise SystemExit("pre-flight check failed (PREFLIGHT_STRICT = True): " + " | ".join(_pf))''')]

G2_SWAP = [('if TEST_10K:\n    print(f"TEST_10K: {TIME_BUDGET_MIN} min of budget',
            'if TEST_10K and TIME_BUDGET_MIN != float("inf"):\n    print(f"TEST_10K: {TIME_BUDGET_MIN} min of budget')]

STEP7_SWAP = [('AGENTS = {"actions": NBA.drop(columns=["PROFILE", "REASON"]), "nbo": nbo, ',
               '''CAND_STORE = []


def _nbo_rec(group_key, cands):
    """Agent 4 (nbo above), and the scored candidates kept for the top-up of step 7b (the short list first)."""
    out = nbo(group_key, cands)
    CAND_STORE.append(pd.DataFrame({VAL: cands[VAL].astype(str).to_numpy(), "BTL_NAME": cands["BTL_NAME"].astype(str).to_numpy(),
                                    "VALIDITY": cands["VALIDITY"].to_numpy(), "PRICE": cands["PRICE"].to_numpy(),
                                    "RULE_SCORE": cands["SCORE"].to_numpy(),
                                    "SCORE_TOPUP": out.where(out.notna(), cands["SCORE"] * 0.01).to_numpy()}))
    return out


AGENTS = {"actions": NBA.drop(columns=["PROFILE", "REASON"]), "nbo": _nbo_rec, ''')]

TOPUP = [("md", r'''## Step 7b · 6 offers for every line: the top-up with the engine's own rules

Code 5 (step 8) can leave a slot empty, for these reasons (step 8b counts them on your data):
- **slots 1–4:** the challenger price passes found no offer of that type in the price band (1.0–1.5× the anchor,
  widened once to 2.0×), and the fallback passes found none either;
- **slots 5–6:** no third offer of the validity above the anchor (EXT-6), and no favourite or forced bundle;
- **F-09 / F-11 / F-12:** an offer was cleared as a duplicate (same type and price, or same offer) or a conflict;
- **F-04:** a line with nothing in slots 1–4 is dropped from the file.

The top-up fills each empty slot (monthly slots 1, 2, 5 with a 30-day offer and weekly slots 3, 4, 6 with a 7-day offer
first, any validity after) from, in order:
1. **the short list**: the best `NBO_SHORTLIST` candidates per validity agent 4 scored for the line;
2. **the same action's offers**: the line's other allowed candidates (its universe: CC-v1 for T2C, CF-v1 for T2F);
3. **the favourites**: the line's own purchases of the last 270 days (code 5 F-06);
4. **the segment's other allowed actions**: the second choice of the cell first, then the others by their historical
   share; an at-risk segment gets no upsell (H-06); T2F keeps its free universe (L2 data and combo only).

Inside each source an offer priced at or above the line's usual spend (F-05) comes first. Every added offer is in the
catalogue, has both scripts (English and Arabic), and is not the same offer, nor the same type and price, nor in conflict
with another slot. `TOPUP_DROPPED = True` also tops up the lines F-04 drops, so they stay in the file. The auditor adds
**A-25: every row of the final file has 6 distinct, valid offers** (blocking).'''),
         ("code", r'''fo = use("nbx.fsdr", "final_output")
uni = use("nbx.fsdr.offers", "universe")
SLOTS6 = range(1, 7)
MONTHLY_SLOT = {1, 2, 5}
_CAT = load_catalog(cache)
INFO = _CAT.assign(BTL_NAME=_CAT["BTL_NAME"].astype(str).str.strip()).drop_duplicates("BTL_NAME").set_index("BTL_NAME")
_scr = cache.read("BTL_script_english_arabic_ok") if cache.has("BTL_script_english_arabic_ok") else pd.DataFrame()
_scr.columns = [str(c).upper() for c in _scr.columns]
SCRIPTED = set() if _scr.empty else set(_scr.loc[(_scr["ENGLISH SCRIPT"].fillna("").astype(str).str.strip() != "")
                                                 & (_scr["ARABIC SCRIPT"].fillna("").astype(str).str.strip() != ""),
                                                 "DESTINATION_OFFER_NAME"].astype(str).str.strip())
VALID_OFFERS = set(INFO.index) & SCRIPTED
KEYTP = (INFO["TYPE"].astype(str) + "|" + INFO["PRICE"].astype(str)).to_dict()           # F-09: same type + price
VALIDITY_OF = pd.to_numeric(INFO["VALIDITY"], errors="coerce").to_dict()
SERVICE_OF = INFO["SERVICE"].astype(str).str.upper().to_dict()
PRICE_OF = pd.to_numeric(INFO["PRICE"], errors="coerce").to_dict()
CONFLICT = set(fo._conflict_pairs(cache).index)                                           # F-12 pairs "a|b"
_bought = uni.purchases(cache)
_bought = _bought[_bought["BTL_NAME"].isin(INFO.index)]
POP = _bought.groupby("BTL_NAME")["NB_SUB"].sum()                                         # E-O1: buyers in the base
REC = uni.recommendable(_CAT, _bought)
LINE = panel.drop_duplicates(VAL).set_index(VAL)
PRIOR_ORDER = sorted(config.ACTIONS, key=lambda a: -config.ACTION_PRIOR.get(a, 0))
_UNI = {}


def _ordered(names):
    names = pd.Index(pd.unique(pd.Series(list(names), dtype=object)))
    return sorted(names, key=lambda o: (-float(POP.get(o, 0)), -float(PRICE_OF.get(o, 0) or 0)))


def universe_offers(action, free_only):
    """The recommendable offers of an action's universe, both validities, by buyers then price."""
    k = (action, free_only)
    if k not in _UNI:
        names = [n for v in (30, 7) for n in uni.constraint_universe(REC, action, v)["BTL_NAME"].astype(str)]
        if free_only:
            names = [n for n in names if SERVICE_OF.get(n) in config.FREE_UNIVERSE_SERVICES
                     and str(INFO.at[n, "GENEROSITY"]) == config.FREE_UNIVERSE_GENEROSITY]
        _UNI[k] = _ordered(names)
    return _UNI[k]


def _cand_lists():
    """Per line: (short list, the action's other offers) from the candidates agent 4 scored (step 7)."""
    if not CAND_STORE:
        return {}, {}
    c = pd.concat(CAND_STORE, ignore_index=True).drop_duplicates([VAL, "BTL_NAME"], keep="last")
    c["IN_SHORT"] = c.groupby([VAL, "VALIDITY"])["RULE_SCORE"].rank(method="first", ascending=False) <= NBO_SHORTLIST
    c = c.sort_values([VAL, "SCORE_TOPUP", "PRICE"], ascending=[True, False, False], kind="mergesort")
    short = c[c["IN_SHORT"]].groupby(VAL, sort=False)["BTL_NAME"].agg(list).to_dict()
    rest = c[~c["IN_SHORT"]].groupby(VAL, sort=False)["BTL_NAME"].agg(list).to_dict()
    return short, rest


TOPUP_STATS = {"before": {}, "after": {}, "why_missing": {}, "added_by_source": {}, "brought_back_F04": 0,
               "still_short": 0, "rows": 0}
_CANDS = {}


def _bump(d, k, n=1):
    d[k] = d.get(k, 0) + n


def _anchor(s, v, o):
    per = "WEEKLY" if VALIDITY_OF.get(o) == 7 else "MONTHLY"
    col = f"F05_DATA_{per}" if SERVICE_OF.get(o) == "DATA" else f"F05_{per}"
    a = s.at[v, col] if col in s.columns else np.nan
    return np.nan if a is None else float(a)


def topup(s, scored, f04=False):
    """Fills the empty slots of s (index = line, S1..S6, REASON_i, REMOVED_i, F05_* anchors) in place."""
    if not _CANDS:
        _CANDS["short"], _CANDS["rest"] = _cand_lists()
    short, rest = _CANDS["short"], _CANDS["rest"]
    act = scored.assign(access_method_val=scored["access_method_val"].astype(str)).drop_duplicates(
        "access_method_val").set_index("access_method_val")["action"]
    fav = fo.favourites(cache, s.index, _CAT)
    fav = fav.assign(BTL_NAME=fav["BTL_NAME"].astype(str)).groupby(VAL, sort=False)["BTL_NAME"].agg(list).to_dict()
    cols = [f"S{i}" for i in SLOTS6]
    n_before = s[cols].notna().sum(axis=1)
    for k, n in n_before.value_counts().items():
        _bump(TOPUP_STATS["before"], int(k), int(n))
    for i in SLOTS6:                                                        # why each slot is empty
        empty = s[f"S{i}"].isna()
        if f04:
            _bump(TOPUP_STATS["why_missing"], "F-04: no offer in slots 1-4 (line dropped by code 5)", int(empty.sum()))
            continue
        rem = s.get(f"REMOVED_{i}", pd.Series(None, index=s.index))
        by_rule = rem[empty & rem.notna()].astype(str).str.split(":").str[0]
        for r, n in by_rule.value_counts().items():
            _bump(TOPUP_STATS["why_missing"], f"{r}: cleared by code 5", int(n))
        n_none = int((empty & rem.isna()).sum())
        if n_none:
            _bump(TOPUP_STATS["why_missing"], "slots 1-4: no offer in the price band (passes)" if i <= 4 else
                  "slots 5-6: no 3rd offer above the anchor (EXT-6), no favourite / forced bundle", n_none)
    for v in s.index[n_before < 6]:
        names = [x for x in (s.at[v, c] for c in cols) if isinstance(x, str) and x]
        keys = {KEYTP.get(x) for x in names}
        line = LINE.loc[v] if v in LINE.index else None
        group = str(line["GROUP_KEY"]) if line is not None else "t2c"
        action = str(act.get(v, line["ACTION_AGENT"] if line is not None else ""))
        segment = str(line["SEGMENT"]) if line is not None else ""
        rank2 = line.get("ACTION_AGENT_RANK2") if line is not None else None
        others = [a for a in ([rank2] if isinstance(rank2, str) else []) + PRIOR_ORDER if a and a != action]
        others = [a for a in dict.fromkeys(others) if not (segment in AT_RISK_SEGMENTS and a in UPSELL_ACTIONS
                                                            and a != (line["ACTION_AGENT"] if line is not None else None))]
        sources = [("short list", short.get(v, [])), ("action offers", rest.get(v, [])), ("favourite", fav.get(v, []))]
        sources += [(f"other action {a}", universe_offers(a, group == "t2f")) for a in others]
        for i in [i for i in SLOTS6 if not isinstance(s.at[v, f"S{i}"], str) or not s.at[v, f"S{i}"]]:
            want, pick = (30 if i in MONTHLY_SLOT else 7), None
            for relax in (False, True):
                for src_name, offers in sources:
                    second = None
                    for o in offers:
                        if o not in VALID_OFFERS or o in names or KEYTP.get(o) in keys:
                            continue
                        if not relax and VALIDITY_OF.get(o) != want:
                            continue
                        if any(f"{o}|{x}" in CONFLICT for x in names):
                            continue
                        a = _anchor(s, v, o)
                        if np.isnan(a) or (PRICE_OF.get(o) or 0) >= a:
                            pick = (o, src_name)
                            break
                        second = second or (o, src_name)
                    pick = pick or second
                    if pick:
                        break
                if pick:
                    break
            if pick:
                s.at[v, f"S{i}"], s.at[v, f"REASON_{i}"] = pick[0], f"top-up: {pick[1]}"
                names.append(pick[0])
                keys.add(KEYTP.get(pick[0]))
                _bump(TOPUP_STATS["added_by_source"], pick[1].split(" ")[0] if pick[1].startswith("other") else pick[1])
    n_after = s[cols].notna().sum(axis=1)
    for k, n in n_after.value_counts().items():
        _bump(TOPUP_STATS["after"], int(k), int(n))
    TOPUP_STATS["rows"] += len(s)
    TOPUP_STATS["still_short"] += int((n_after < 6).sum())
    if f04:
        TOPUP_STATS["brought_back_F04"] += int((n_after > 0).sum())


# --- code 5 with the top-up: inside build_input_flytxt (after F-09 / F-11 / F-12), and the F-04 lines after it --------
_format_orig = getattr(fo, "_format_orig", fo._format)
fo._format_orig = _format_orig
_build_orig = getattr(fo, "_build_orig", fo.build_input_flytxt)
fo._build_orig = _build_orig


def _format_topup(s, scored, src, catalog, run_date):
    topup(s, scored)
    return _format_orig(s, scored, src, catalog, run_date)


def _build_topup(monthly, weekly, scored, src, run_date):
    flytxt, slots = _build_orig(monthly, weekly, scored, src, run_date)
    if TOPUP_DROPPED:
        dropped = slots.loc[~slots["IN_INPUT_FLYTXT"].astype(bool), VAL].astype(str)
        if len(dropped):
            s2 = pd.DataFrame(index=pd.Index(dropped.unique(), name=VAL))
            for i in SLOTS6:
                s2[f"S{i}"], s2[f"REASON_{i}"], s2[f"REMOVED_{i}"] = None, None, None
            s2 = s2.join(fo.f05_anchors(src, pd.Series(s2.index)))
            topup(s2, scored, f04=True)
            s2 = s2[s2[[f"S{i}" for i in SLOTS6]].notna().any(axis=1)]
            if len(s2):
                flytxt = pd.concat([flytxt, _format_orig(s2, scored, src, load_catalog(src), run_date)],
                                   ignore_index=True).drop_duplicates("msisdn")
                slots = pd.concat([slots[~slots[VAL].astype(str).isin(s2.index)], s2.reset_index().assign(IN_INPUT_FLYTXT=True)],
                                  ignore_index=True)
    return flytxt, slots


fo._format = _format_topup
fo.build_input_flytxt = _build_topup

# --- the auditor: A-25, every row of the final file has 6 distinct, valid offers (blocking) ----------------------------
_oc_orig = getattr(scoring_groups, "_oc_orig", scoring_groups.output_checks)
scoring_groups._oc_orig = _oc_orig


def _checks_6(flytxt, control, catalog_names, run_date):
    out = _oc_orig(flytxt, control, catalog_names, run_date)
    names = flytxt[[f"destination_offer_name_{i}" for i in SLOTS6]].astype(str).apply(lambda c: c.str.strip())
    scripts = flytxt[[f"offer_script_{l}_{i}" for l in ("en", "ar") for i in SLOTS6]].astype(str).apply(lambda c: c.str.strip())
    ok = ((names != "").all(axis=1) & (names.nunique(axis=1) == 6) & names.isin(VALID_OFFERS).all(axis=1)
          & (scripts != "").all(axis=1))
    out.append({"id": "A-25", "check": f"6 distinct, valid offers (catalogue + both scripts) in every row "
                                       f"({int((~ok).sum()):,} of {len(flytxt):,} rows not)",
                "status": "PASS" if len(flytxt) and bool(ok.all()) else "FAIL"})
    return out


scoring_groups.output_checks = _checks_6
print(f"top-up ready: {len(VALID_OFFERS):,} valid offers (catalogue + both scripts) of {len(INFO):,}, {len(CONFLICT) // 2:,} "
      f"conflict pairs, {len(REC):,} recommendable; A-25 added to the auditor")''')]

TOPUP_REPORT = [("md", r'''## Step 8b · The top-up in figures: why lines had fewer than 6 offers, and what was added'''),
                ("code", r'''_dist = pd.DataFrame({"lines before": pd.Series(TOPUP_STATS["before"]), "lines after": pd.Series(TOPUP_STATS["after"])})
_dist = _dist.reindex(range(0, 7)).fillna(0).astype(int).rename_axis("offers per line")
show(_dist.reset_index(), n=7, title="offers per line, before and after the top-up")
show(pd.Series(TOPUP_STATS["why_missing"], name="empty slots").sort_values(ascending=False).rename_axis("why the slot was empty")
     .reset_index(), n=10, title="why slots were empty before the top-up")
show(pd.Series(TOPUP_STATS["added_by_source"], name="offers added").rename_axis("source").reset_index(), n=10,
     title="offers added, by source")
_b = sum(k * n for k, n in TOPUP_STATS["before"].items()) / max(sum(TOPUP_STATS["before"].values()), 1)
_a = sum(k * n for k, n in TOPUP_STATS["after"].items()) / max(sum(TOPUP_STATS["after"].values()), 1)
print(f"offers per line: {_b:.2f} before -> {_a:.2f} after | F-04 lines brought back: {TOPUP_STATS['brought_back_F04']:,} | "
      f"lines still under 6: {TOPUP_STATS['still_short']:,}")
if TOPUP_STATS["still_short"]:
    print("WARNING: some lines could not reach 6 offers (no valid offer left in any source): A-25 blocks the final file. "
          "Check the catalogue scripts (BTL_script_english_arabic_ok) and the conflict pairs")''')]

STEP12_SWAPS = [
    ("fb_below = pd.Series(0, index=v360.index)",
     "fb_below = pd.Series(0, index=v360.index)\ntu_below = pd.Series(0, index=v360.index)"),
    ('    fb_below += (b & sel_by.astype(str).str.startswith("fallback")).astype(int)',
     ('    fb_below += (b & sel_by.astype(str).str.startswith("fallback")).astype(int)\n'
     '    tu_below += (b & reason.astype(str).str.startswith("top-up")).astype(int)')),
    ('v360["CHK_FALLBACK_PASS_BELOW_ANCHOR"] = fb_below',
     'v360["CHK_FALLBACK_PASS_BELOW_ANCHOR"] = fb_below\nv360["CHK_TOPUP_BELOW_ANCHOR"] = tu_below'),
    ('"CHK_FAVOURITE_BELOW_ANCHOR", "CHK_FALLBACK_PASS_BELOW_ANCHOR",\n',
     '"CHK_FAVOURITE_BELOW_ANCHOR", "CHK_FALLBACK_PASS_BELOW_ANCHOR",\n         "CHK_TOPUP_BELOW_ANCHOR",\n'),
    ('    of_which_fallback_pass=("CHK_FALLBACK_PASS_BELOW_ANCHOR", lambda s: int((s > 0).sum())),',
     ('    of_which_fallback_pass=("CHK_FALLBACK_PASS_BELOW_ANCHOR", lambda s: int((s > 0).sum())),\n'
     '    of_which_topup=("CHK_TOPUP_BELOW_ANCHOR", lambda s: int((s > 0).sum())),')),
    ('v360["CHK_FALLBACK_PASS_BELOW_ANCHOR"]) > 0).sum())',
     'v360["CHK_FALLBACK_PASS_BELOW_ANCHOR"] - v360["CHK_TOPUP_BELOW_ANCHOR"]) > 0).sum())'),
]

FINAL_REPORT = [("md", r'''## Final report (real run)

The dates used, the takers of last week's campaign, the propensities of agent 3, the offers per line, the LLM answers
and the speed. Written to `data/real25k/output/<D>/real_report_<RUN_SIZE>.json`. It also releases the "keep awake" of Windows.'''),
                ("code", r'''_seg = pd.concat([pd.read_parquet(p, columns=[VAL, "W0_TARGETED", "W0_REWARD"])
                  for p in sorted((train_dir / T3_TRAIN / "segments").glob("*.parquet"))])
_tr = _seg[_seg[VAL].isin(TRAIN_VALS)]
_tg = int((_tr["W0_TARGETED"] == 1).sum())
_tk = int(((_tr["W0_TARGETED"] == 1) & (_tr["W0_REWARD"] == 1)).sum())
_spread = []
for _u, (_al, _ans) in ANS3.items():
    if ok3(_ans, _al):
        _s = np.clip(np.asarray(_ans["scores"], float), 0, None)
        _p = (1 - EPSILON) * (_s / _s.sum() if _s.sum() > 0 else np.full(len(_s), 1 / len(_s))) + EPSILON / len(_s)
        _spread.append(float(_p.max() - _p.min()))
_flat = float(np.mean(np.asarray(_spread) < 0.05)) if _spread else None
_ag = {a: s for a, s in STATS.items() if a not in ("G2 speed test", "G1 first request")}
_llm = {k: int(sum(s.get(k, 0) for s in _ag.values())) for k in ("calls", "cached", "retried", "errors")}
_no_llm = n_cust - len(LLM_LINES)
_fb3 = int((NBA["DECIDED_BY"] == "fallback_cell_policy").sum()) if len(NBA) else 0
_wall34 = sum(STATS.get(a, {}).get("wall_seconds", 0.0) for a in ("Agent 3 NBA", "Agent 4 NBO"))
_b = sum(k * n for k, n in TOPUP_STATS["before"].items()) / max(sum(TOPUP_STATS["before"].values()), 1)
_a = sum(k * n for k, n in TOPUP_STATS["after"].items()) / max(sum(TOPUP_STATS["after"].values()), 1)
REPORT = {
    "sizes": {"RUN_SIZE": RUN_SIZE, "TRAIN_LINES": TRAIN_LINES, "score_groups": list(SCORE_GROUPS)},
    "dates": {"D_last_purchase": RUN_DATE, "T_campaign": CAMPAIGN_DATE, "purchase_window_days": PURCHASE_DAYS},
    "training": {"lines": len(TRAIN_VALS), "targeted_last_week": _tg, "takers": _tk,
                 "taker_rate": round(_tk / _tg, 4) if _tg else None},
    "agent3_propensity": {"customers": int(len(NBA)),
                          **({k: round(float(v), 4) for k, v in NBA["PROBA"].describe()[["min", "25%", "50%", "75%", "max"]].items()}
                             if len(NBA) else {}),
                          "profiles_almost_flat_share": None if _flat is None else round(_flat, 3)},
    "offers_per_line": {"before_topup": round(_b, 2), "after_topup": round(_a, 2),
                        "brought_back_F04": TOPUP_STATS["brought_back_F04"], "still_under_6": TOPUP_STATS["still_short"]},
    "llm_answers": {"fresh": _llm["calls"], "cached": _llm["cached"], "retried": _llm["retried"], "failed": _llm["errors"]},
    "cell_policy": {"customers_not_sent_to_llm": int(_no_llm), "agent3_fallbacks": _fb3,
                    "share": round((_no_llm + _fb3) / max(n_cust, 1), 4)},
    "speed": {"customers": int(len(LLM_LINES)), "agents_3_4_minutes": round(_wall34 / 60, 1),
              "customers_per_minute": round(len(LLM_LINES) / (_wall34 / 60), 1) if _llm["calls"] and _wall34 > 30 else None,
              "notebook_minutes": round((time.time() - NB_START) / 60, 1)},
    "final_file": {"ready": bool(fin.get("ready")), "rows": int(fin.get("input_flytxt_rows", 0))},
}
(OUTPUT / f"real_report_{RUN_SIZE}.json").write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
for _k, _v in REPORT.items():
    print(f"{_k:18s} {_v}")
if _flat is not None and _flat > 0.5:
    print(f"WARNING: agent 3 propensities are almost flat for {_flat:.0%} of the profiles (all actions scored alike): "
          "the LLM barely separates the actions -- try a stronger MODEL or a finer PROFILE_LEVEL")
if REPORT["cell_policy"]["share"] > 0:
    print(f"WARNING: {REPORT['cell_policy']['share']:.2%} of the customers got the cell policy (must be 0): see the failed answers in "
          f"{LLM_LOG.relative_to(ROOT)}")
if _llm["errors"]:
    print(f"WARNING: {_llm['errors']:,} LLM answers failed after {LLM_RETRIES} retries")
if sys.platform == "win32":
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print("Windows may sleep again")''')]

PROOF = r'''# --- proof: everything ran from this notebook; vowpalwabbit may be preloaded by the environment, never used --------
import types as _types


def _from_vw(v):
    try:
        name = v.__name__ if isinstance(v, _types.ModuleType) else (getattr(v, "__module__", "") or "")
    except Exception:
        return False
    return str(name).split(".")[0] == "vowpalwabbit"


mods = sorted(m for m in sys.modules if m == "nbx" or m.startswith("nbx."))
from_disk = [m for m in mods if not getattr(sys.modules[m], "__fsdr_notebook__", False)]
vw_users = [m for m in mods if any(_from_vw(v) for v in list(vars(sys.modules[m]).values()))]
print(f"{len(mods)} engine modules, all defined in this notebook: {not from_disk}")
print("vowpalwabbit:", "not loaded" if "vowpalwabbit" not in sys.modules else
      ("preloaded by the environment (a .pth file), used by no engine module" if not vw_users else f"USED by {vw_users}"))
assert not from_disk, f"loaded from the project folder: {from_disk}"
assert not vw_users, f"an engine module uses vowpalwabbit: {vw_users}"
print("api / gradio loaded:", any(m in sys.modules for m in ("gradio", "fastapi", "uvicorn")))'''

DEFAULTS = {"Step 0": (3, 3), "Part A": (0.3, 0.3), "Part B": (0.2, 0.2),
            "The LLM does not stop": (0.1, 0.1), "G1": (0.5, 0.5), "Pre-flight check": (0.1, 0.1), "G2": (1, 1),
            "The groups to score": (0.1, 0.1), "Step 1": (0.5, 0.5), "Step 2": (15, 15), "Step 3": (25, 25), "Step 4": (2, 2),
            "Step 4b": (20, 20), "Step 5": (1, 1), "Step 5b": (2, 2), "Step 6": (60, 60), "Step 7": (60, 60),
            "Step 7 (CPU)": (8, 8), "Step 7b": (0.5, 0.5), "Step 8": (8, 8), "Step 8b": (0.1, 0.1), "Step 9": (0.5, 0.5),
            "Step 10": (0.5, 0.5), "Step 11": (0.5, 0.5), "Step 12": (5, 5), "Time per agent on your GPU": (0.1, 0.1),
            "FAST_MODE in figures": (0.1, 0.1), "G3": (0.3, 0.3), "Final report": (0.3, 0.3), "Proof": (0.1, 0.1)}

TIMER_SAVE_OLD = '        allh.setdefault(_NB, {})[_SIZE] = {**_HIST, **{k: round(v, 1) for k, v in STEP["done"].items()}}'
TIMER_SAVE_NEW = ('        keep = {k: round(v, 1) for k, v in STEP["done"].items()          # not the report cells, not an outlier\n'
                  '                if k not in ("Proof", "Time per step") and not (_estimate(k) > 0 and v > 10 * _estimate(k))}\n'
                  '        allh.setdefault(_NB, {})[_SIZE] = {**_HIST, **keep}')
