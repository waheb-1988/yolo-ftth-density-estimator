"""notebooks3 additions shared by 02_prepare_week and the agents notebooks 03 / 04:

- the new export names of 2026-11-18 (subs_btl_test, dv_segment, SEGMENT_Conflicting_Offers_h, the 11 model files
  without "_TD"), and the exports that are not used (target_new_april26, tenure, the filtered reactivation file);
- the dates: D = t1 = the most recent date of the data (scoring, *_score), T = t2 = t1 - 7 (training, *_auto);
  last week's purchases are counted from t2 to t1;
- targeting_dataiku_auto holds the whole history: only the latest campaign is kept (step 4c of 02);
- RUN_SIZE: 1_000 / 5_000 / 10_000 target lines (a test copy) or "all" (the full week);
- a step timer: the time of every step, its estimate and the time left.
Everything is applied inside the notebooks; the engine (nbx/) is not modified.
"""
import re

NEW_NAMES = {   # the export name (without _hdfs) -> the contract name
    "subs_btl_test": "subs_btl_auto",
    "dv_segment": "DEVICEBEHAVIORSEGMENTATION.dv_segment",
    "SEGMENT_Conflicting_Offers_h": "conflicting_offers_h",
    "REV_DECREASE_204_SCORING_predictions": "REV_DECREASE_204_SCORING_TD.predictions",
    "PROPENSITY_REACTIVATION_15D_401_Scored": "PROPENSITY_REACTIVATION_15D_401_SCORING.Scored",
    "DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_Scored": "DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_TD.Scored",
    "DAIM_FRAGILITY_VOICE_SCORING_prediction": "DAIM_FRAGILITY_VOICE_SCORING_TD.prediction",
    "DAIM_FRAGILITY_DATA_SCORING_prediction": "DAIM_FRAGILITY_DATA_SCORING_TD.prediction",
    "DAIM_EARLY_CHURN_30_60_SCORING_scoring_data_scored": "DAIM_EARLY_CHURN_30_60_SCORING_TD.scoring_data_scored",
    "DAIM_EARLY_CHURN_0_30_SCORING_Early_Churn_0_30_Prediction": "DAIM_EARLY_CHURN_0_30_SCORING_TD.Early_Churn_0_30_Prediction",
    "DAIM_CHURN_GLOBAL_SCORING_prediction": "DAIM_CHURN_GLOBAL_SCORING_TD.prediction",
    "CHURN_ONE_DAY_ACTIVE_107_Churn_One_Day_Scoring": "CHURN_ONE_DAY_ACTIVE_107_SCORING.Churn_One_Day_Scoring",
    "APPETENCY_VOICE_204_SCORING_Scored": "APPETENCY_VOICE_204_SCORING_TD.Scored",
    "APPETENCY_DATA_204_SCORING_prediction": "APPETENCY_DATA_204_SCORING_TD.prediction",
}
NOT_USED = {   # exported, but not read by the flux
    "target_new_april26": "the old target list (the lists are target2_a / target2_b)",
    "tenure": "tenure comes from customer_profile, else access_method_id_auto.ACTIVATION_DATE",
    "DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_filterd_nationality": "a filtered copy: the _Scored file is used",
}

# ---------------------------------------------------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------------------------------------------------
DATES_SETTINGS = r'''
# --- the dates (notebooks3): t1 = D = the most recent date of the data, t2 = T = t1 - 7 ------------------------------
ANCHOR_LAG_DAYS = 7               # T = D - 7: the *_auto tables (training) are one week before the *_score tables (scoring)
REWARD_WINDOW = (-7, 0)           # last week's purchases are counted from t2 = D - 7 to t1 = D (engine default: D-7 .. D-1)
TARGETING_WINDOW = (-7, -1)       # last week's campaign (date1) is expected from t2 to t1 - 1 (engine default: D-8 .. D-2)
'''

PREP_SETTINGS_SWAPS = [
    ('RUN_DATE = "2026-11-17"           # the run date D (the SAME in notebook 03 / 04); None = read it from subs_btl_auto / targeting_dataiku_auto',
     ('RUN_DATE = None                   # D = t1 (scoring); None = the most recent day of the data (the last purchase of subs_btl);\n'
     '                                  #   or a date "2026-11-19". 03 / 04 then take the latest week prepared here')),
    ("FIX_HEADERS = False               # a file without header: True = add the contract header when the sanity check passes",
     "FIX_HEADERS = True                # a file without header: add the contract header when the column count and the IDs match"),
    ("TEST_10K = True                   # True: also build data/weekly_sample/<D>/ for notebook 03 / 04 (TEST_10K = True there too)",
     'RUN_SIZE = "all"                  # "all" = the full week only (~50k per list): for 04_..._real25k (whole base to\n'
     '                                  #   train, its own RUN_SIZE scored) and for 03 / 04 with RUN_SIZE = "all"\n'
     '                                  #   | 1_000 | 5_000 | 10_000 = also a test copy for 03 / 04 (the SAME RUN_SIZE there)\n'
     'TEST_10K = RUN_SIZE != "all"      # (from RUN_SIZE)\n'
     'LAST_CAMPAIGN_DAYS = 7            # targeting_dataiku_auto holds the history: keep the latest campaign only (its last 7 days)'
     + DATES_SETTINGS.rstrip("\n")),
    ("TRAIN_LINES = 10_000              # lines of your real base_prepaid used to train (agent 2b learns from them only)",
     "TRAIN_LINES = 10_000              # test copy: lines of your real base_prepaid used to train (agent 2b learns from them)"),
    ("SCORE_LINES = 10_000              # lines scored, all from ONE of your real lists (None = the whole list)",
     'SCORE_LINES = None if RUN_SIZE == "all" else int(RUN_SIZE)   # test copy: the target lines scored (from RUN_SIZE)'),
]

AGENT_SETTINGS_SWAPS = [
    ('RUN_DATE = "2026-11-17"         # the run date D (the SAME as in notebook 02): data/weekly/<D>/',
     'RUN_DATE = None                 # D = t1; None = the latest week prepared by notebook 02 (or "2026-11-19")'),
    ("TEST_10K = True                 # True = data/weekly_sample/<D> written by notebook 02 with TEST_10K = True",
     'TEST_10K = RUN_SIZE != "all"     # (from RUN_SIZE): the test copy of notebook 02, data/weekly_sample/<D>'),
]

AGENT_SETTINGS_ADD = r'''
# --- the size of the run (the SAME as RUN_SIZE in notebook 02) ------------------------------------------------------------
RUN_SIZE = 10_000                 # 1_000 | 5_000 | 10_000 = the test copy of notebook 02 | "all" = the full week (~50k per list)
''' + DATES_SETTINGS + r'''
if RUN_DATE is None:              # the latest week notebook 02 prepared, for this RUN_SIZE
    from pathlib import Path as _P
    _r = _P.cwd()
    while not (_r / "pyproject.toml").exists() and _r != _r.parent:
        _r = _r.parent
    _base = _r / "data" / ("weekly" if RUN_SIZE == "all" else "weekly_sample")
    _weeks = sorted(p.name for p in _base.glob("20??-??-??") if (p / "_manifest.json").exists())
    if not _weeks:
        raise SystemExit(f"no week prepared in {_base}: run notebook 02 first (RUN_SIZE = {RUN_SIZE!r})")
    RUN_DATE = _weeks[-1]
    print(f"RUN_DATE = {RUN_DATE}: the latest week prepared by notebook 02 (data/{_base.name}/)")
'''

AGENT_PATCH = r'''
# --- the dates of notebooks3: T = D - ANCHOR_LAG_DAYS, last week's purchases from t2 to t1 ---------------------------------
from datetime import timedelta as _tdl
_cal = use("nbx.fsdr", "calendar")
_cal.anchor_date = lambda d: d - _tdl(days=ANCHOR_LAG_DAYS)
config.REWARD_WINDOW, config.TARGETING_WINDOW = tuple(REWARD_WINDOW), tuple(TARGETING_WINDOW)
print(f"dates: D = t1 = {D}, T = t2 = {_cal.anchor_date(D)} (D - {ANCHOR_LAG_DAYS}); last week's purchases counted "
      f"{_cal.campaign_windows(D)['reward'].start} .. {_cal.campaign_windows(D)['reward'].end}")
if TEST_10K and _info.get("score_lines") != (None if RUN_SIZE == "all" else int(RUN_SIZE)):
    print(f"note: the test copy of notebook 02 holds {_info.get('score_lines'):,} target lines, RUN_SIZE here is {RUN_SIZE}: "
          "run notebook 02 with the same RUN_SIZE to test this size")'''

# ---------------------------------------------------------------------------------------------------------------------
# notebook 02: the patch cell (after the imports), step 3, step 4c
# ---------------------------------------------------------------------------------------------------------------------
PREP_TITLE_OLD = "# 02 · Prepare the week from `data_raw/` (NBA tables, auto and scoring dates, cleaning)"
PREP_TITLE_NEW = r'''# 02 · Prepare the week from `data_raw/` (notebooks3: new export names, dates t1 / t2, the latest campaign, run size)

**What notebooks3 changes here** (the rest is the notebooks2 notebook):
- **The new export names:** `subs_btl_test` → `subs_btl_auto`, `dv_segment` → `DEVICEBEHAVIORSEGMENTATION.dv_segment`,
  `SEGMENT_Conflicting_Offers_h` → `conflicting_offers_h`, and the 11 model files without `_TD`. The exports that are
  not used are ignored: `target_new_april26`, `tenure`, `DAIM_PROPENSITY_REACTIVATION_30D_404_SCORING_filterd_nationality`.
- **The dates:** **D = t1** = the most recent day of the data (the last purchase day of `subs_btl`), the anchor of
  the `*_score` tables; **T = t2 = t1 − 7**, the anchor of the `*_auto` tables, where the model trains. Last week's
  purchases are counted from t2 to t1.
- **`targeting_dataiku_auto` holds the whole history:** step 4c keeps, per line, its most recent row, and only if it
  belongs to the latest campaign (`LAST_CAMPAIGN_DAYS`). Without this, a customer targeted weeks ago would count as
  "targeted last week".
- **`RUN_SIZE`:** `1_000`, `5_000` or `10_000` = also a test copy of that many target lines; `"all"` = the full week.
- **`FIX_HEADERS = True`:** a file without a header gets the contract header (when its columns and IDs match).
- **The time of every step**, its estimate and the time left are printed as `[time]` lines.'''

PREP_PATCH = r'''# --- notebooks3: the new export names and the dates t1 / t2 (applied here, the engine is not modified) -----------------
NEW_NAMES = ''' + repr(NEW_NAMES) + r'''
NOT_USED = ''' + repr(NOT_USED) + r'''
for _k, _v in NEW_NAMES.items():
    schemas.TABLE_ALIASES[_k] = _v
    schemas.TABLE_ALIASES[_k.lower()] = _v
for _k in NOT_USED:
    schemas.TABLE_ALIASES[_k] = schemas.TABLE_ALIASES[_k.lower()] = "__not_used__"
cal.anchor_date = lambda d: d - timedelta(days=ANCHOR_LAG_DAYS)            # T = t2 = D - 7
config.REWARD_WINDOW = tuple(REWARD_WINDOW)                                 # purchases from t2 to t1
config.TARGETING_WINDOW = tuple(TARGETING_WINDOW)                           # last week's campaign from t2 to t1 - 1
print("new export names:", len(NEW_NAMES), "| not used:", ", ".join(NOT_USED))
print(f"dates: T = D - {ANCHOR_LAG_DAYS}; last week's purchases counted D{REWARD_WINDOW[0]:+d} .. D{REWARD_WINDOW[1]:+d}")'''

STEP3_SWAPS = [
    ('''D is read from the data: the last fulfilment of `subs_btl_auto` is D−1 (BR-23), the last `date1` of
`targeting_dataiku_auto` is D−2 (BR-21). T = the last Thursday on or before D.''',
     '''**D = t1** is read from the data: the most recent purchase day of `subs_btl_auto` (export `subs_btl_test`), unless
`RUN_DATE` is set. **T = t2 = t1 − 7** (`ANCHOR_LAG_DAYS`). `access_method_id_auto` (the newest activation) and the
latest campaign of `targeting_dataiku_auto` are shown for information.'''),
    ('''dated = {"subs_btl_auto": ("FULFILMENTSUCCESSDATE_TIME", 1), "targeting_dataiku_auto": ("date1", 2),
         "targeting_auto": ("START_DATE", 0)}''',
     '''dated = {"subs_btl_auto": ("FULFILMENTSUCCESSDATE_TIME", 0),            # t1 = the most recent purchase day
         "access_method_id_auto": ("ACTIVATION_DATE", None),             # information: the newest activation
         "targeting_dataiku_auto": ("date1", None),                      # information: the history of the campaigns
         "targeting_auto": ("START_DATE", None)}'''),
    ('''"implies D": s.max() + timedelta(days=lag) if t != "targeting_auto" else None})''',
     '''"implies D": s.max() + timedelta(days=lag) if lag is not None else None})'''),
    ('''cw = cal.campaign_windows(D)
chk = []''',
     '''_s = DATES.get("targeting_dataiku_auto")
if _s is not None and len(_s):        # the table holds the history: the window check reads the latest campaign only
    _start = _s.max() - timedelta(days=LAST_CAMPAIGN_DAYS - 1)
    DATES["targeting_dataiku_auto"] = _s[_s >= _start]
    print(f"targeting_dataiku_auto: {len(_s):,} rows of history, {_s.min()} .. {_s.max()}; the latest campaign = "
          f"{len(DATES['targeting_dataiku_auto']):,} rows from {_start} (step 4c keeps only those)")
_b = DATES.get("subs_btl_auto")
if _b is not None and len(_b):        # the purchase history: the window check reads the reward window only
    _w = cal.campaign_windows(D)["reward"]
    print(f"subs_btl_auto: {len(_b):,} purchases, {_b.min()} .. {_b.max()}; {int(((_b >= _w.start) & (_b <= _w.end)).sum()):,} "
          f"in the reward window {_w.start} .. {_w.end} (the others are history, not used for the reward)")
    DATES["subs_btl_auto"] = _b[_b >= _w.start]
cw = cal.campaign_windows(D)
chk = []'''),
]

STEP_4C = [("md", r'''## Step 4c · Last week's campaign: the latest row per line, latest campaign only

`targeting_dataiku_auto` holds **every campaign** (the history). The engine reads, per line, its most recent row as
"targeted last week" and counts its purchases from t2 to t1 as the reward. So this step keeps, per line
(`access_method_id`), **its most recent row**, and only if that row belongs to **the latest campaign** (`date1` within
`LAST_CAMPAIGN_DAYS` days of the most recent `date1`). A line targeted weeks ago is then not "targeted last week".
`data_raw/` is not changed; the counts go to `_manifest.json`.'''),
                   ("code", r'''_t = "targeting_dataiku_auto"
_p = DEST / schemas.TABLES[_t].folder / f"{_t}.csv.gz"
if not _p.exists():
    print("no targeting_dataiku_auto: no last campaign (agent 2b keeps its prior)")
else:
    _df = read_csv(_p)
    _dc, _ic = find(_df.columns, "DATE1")[0], find(_df.columns, "ACCESS_METHOD_ID")[0]
    _d1 = pd.to_datetime(_df[_dc], errors="coerce")
    _last = _d1.max()
    _start = _last - pd.to_timedelta(int(LAST_CAMPAIGN_DAYS) - 1, unit="D")
    _df = _df.assign(_D=_d1, _K=canon_id(_df[_ic])).sort_values("_D", kind="stable")
    _latest = _df.drop_duplicates("_K", keep="last")
    _keep = _latest[_latest["_D"] >= _start]
    print(f"targeting_dataiku_auto: {len(_df):,} rows, {_df['_K'].nunique():,} lines, date1 {_d1.min():%Y-%m-%d} .. {_last:%Y-%m-%d}")
    print(f"latest campaign {_start:%Y-%m-%d} .. {_last:%Y-%m-%d}: {len(_keep):,} lines kept "
          f"({len(_latest) - len(_keep):,} lines whose last targeting is older are not 'last week')")
    display(_keep["_D"].dt.date.value_counts().sort_index().rename_axis("date1").to_frame("lines kept"))
    _keep.drop(columns=["_D", "_K"]).to_csv(_p, index=False, compression={"method": "gzip", "compresslevel": 6})
    _mp = DEST / config.MANIFEST_FILE
    _m = json.loads(_mp.read_text(encoding="utf-8"))
    _m["last_campaign"] = {"from": f"{_start:%Y-%m-%d}", "to": f"{_last:%Y-%m-%d}", "rows_in_export": int(len(_df)),
                           "lines_kept": int(len(_keep)), "lines_older": int(len(_latest) - len(_keep))}
    _mp.write_text(json.dumps(_m, indent=2, default=str), encoding="utf-8")'''),
]

# ---------------------------------------------------------------------------------------------------------------------
# the step timer
# ---------------------------------------------------------------------------------------------------------------------
TIMER = r'''# --- the step timer: the time of every step, its estimate and the time left (do not edit) ------------------------------
import json as _json, time as _time2
from datetime import datetime as _dt, timedelta as _td
from pathlib import Path as _Path

_root = _Path.cwd()
while not (_root / "pyproject.toml").exists() and _root != _root.parent:
    _root = _root.parent
STEP_FILE = _root / "data" / "runs" / "_step_times.json"        # the times of the earlier runs (step names only, no MSISDN)
STEP_ORDER = __ORDER__
STEP_DEFAULT = __DEFAULT__    # minutes (full week, test copy): only for the first run, then your own measured times
_NB, _SIZE = "__NB__", str(RUN_SIZE) + __SIZE_SUFFIX__
try:
    _HIST = _json.loads(STEP_FILE.read_text(encoding="utf-8")).get(_NB, {}).get(_SIZE, {})
except (OSError, ValueError):
    _HIST = {}
STEP = {"cur": None, "t0": None, "done": {}, "start": _time2.time()}


def _mm(s):
    s = int(round(s or 0))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _estimate(key):
    """Seconds: agents 3 / 4 from step 5b once it ran, else the time of your last run of the same size, else the default."""
    g = globals()
    if key in ("Step 6", "Step 7") and "U" in g and "keep" in g and "s_nba" in g:
        kept = g["U"][g["keep"]]
        nba = float((kept["group"] == "t2c").sum()) * g["s_nba"]
        if key == "Step 6":
            return nba
        cpu = _HIST.get("Step 7 (CPU)", 60.0 * STEP_DEFAULT["Step 7 (CPU)"][0 if RUN_SIZE == "all" else 1])
        return float(kept["sec"].sum()) - nba + cpu
    if key in _HIST:
        return float(_HIST[key])
    d = STEP_DEFAULT.get(key)
    return 60.0 * (d[0] if RUN_SIZE == "all" else d[1]) if d else 0.0


def _save():
    try:
        allh = _json.loads(STEP_FILE.read_text(encoding="utf-8")) if STEP_FILE.exists() else {}
        allh.setdefault(_NB, {})[_SIZE] = {**_HIST, **{k: round(v, 1) for k, v in STEP["done"].items()}}
        STEP_FILE.parent.mkdir(parents=True, exist_ok=True)
        STEP_FILE.write_text(_json.dumps(allh, indent=1), encoding="utf-8")
    except (OSError, ValueError):
        pass


def _close():
    if STEP["cur"]:
        dur = _time2.time() - STEP["t0"]
        STEP["done"][STEP["cur"]] = dur
        print(f"[time] {STEP['cur']} done in {_mm(dur)} (estimate {_mm(_estimate(STEP['cur']))})")
        _save()
    STEP["cur"] = None


def step(key):
    """At the top of each step: closes the previous step, prints this step's estimate and the time left."""
    _close()
    STEP["cur"], STEP["t0"] = key, _time2.time()
    rest = STEP_ORDER[STEP_ORDER.index(key):] if key in STEP_ORDER else [key]
    left = sum(_estimate(k) for k in rest)
    print(f"[time] {key}: estimate {_mm(_estimate(key))} | elapsed {_mm(_time2.time() - STEP['start'])} | "
          f"left about {_mm(left)} (end about {_dt.now() + _td(seconds=left):%H:%M})")


def step_report():
    """The time of every step against its estimate (the last cell)."""
    _close()
    rows = [{"step": k, "minutes": round(STEP["done"][k] / 60, 1) if k in STEP["done"] else None,
             "estimate (min)": round(_estimate(k) / 60, 1)} for k in STEP_ORDER]
    import pandas as _pd
    t = _pd.DataFrame(rows)
    total = (_time2.time() - STEP["start"]) / 60
    print(f"total {total:.1f} min (RUN_SIZE {RUN_SIZE}); the next run of the same size uses these times as its estimates "
          f"({STEP_FILE.name})")
    return t


print(f"step timer ready: {len(STEP_ORDER)} steps, estimates from "
      + ("your last run of this size" if _HIST else "the defaults (first run of this size)"))
'''

PREP_DEFAULT = {"Step 1": (2, 2), "Step 2": (0.5, 0.5), "Step 3": (5, 5), "Step 4": (15, 15), "Step 4b": (10, 10),
                "Step 4c": (1, 1), "Step 5": (0.1, 5)}
AGENT_DEFAULT = {"Part A": (1, 1), "Part B": (0.2, 0.2), "G1": (0.3, 0.3), "G2": (1, 1), "The groups to score": (0.1, 0.1),
                 "Step 1": (1, 0.2), "Step 2": (15, 0.5), "Step 3": (25, 0.5), "Step 4": (3, 2), "Step 4b": (20, 0.5),
                 "Step 5": (2, 0.2), "Step 5b": (2, 0.2), "Step 6": (60, 20), "Step 7": (30, 10), "Step 7 (CPU)": (5, 0.5),
                 "Step 8": (4, 0.3), "Step 9": (1, 0.3), "Step 10": (1, 0.3), "Step 11": (1, 0.2), "Step 12": (3, 0.3),
                 "Time per agent on your GPU": (0.1, 0.1), "FAST_MODE in figures": (0.1, 0.1), "G3": (0.3, 0.3),
                 "Proof": (0.1, 0.1)}


def heading_key(src: str) -> str:
    h = src.splitlines()[0].lstrip("#").strip()
    return h.split(" · ")[0].strip() if " · " in h else re.sub(r"\s*[`(:].*$", "", h).strip()


def add_timer(pairs, after: int, nb: str, defaults: dict, size_suffix: str, first_key: str | None = None,
              timer_src: str | None = None):
    """pairs: [(kind, src)] of a notebook. Inserts the timer cell at `after`, puts step("<key>") at the top of the first
    code cell after each "## " heading that follows it, and adds the time report at the end."""
    out, order, pending = list(pairs[:after]), [first_key] if first_key else [], None
    for kind, src in pairs[after:]:
        if kind == "md" and src.startswith("## "):
            pending = heading_key(src)
        elif kind == "code" and pending and src.lstrip().startswith("%%"):
            pending = None                       # an engine cell of Part A (cell magic first): counted in "Part A"
        elif kind == "code" and pending:
            order.append(pending)
            src = f'step("{pending}")\n' + src
            pending = None
        out.append((kind, src))
    order.append("Time per step")
    timer = ((timer_src or TIMER).replace("__ORDER__", repr(order)).replace("__DEFAULT__", repr(defaults))
             .replace("__NB__", nb).replace("__SIZE_SUFFIX__", size_suffix))
    if first_key:
        timer += f'step("{first_key}")\n'
    out.insert(after, ("code", timer))
    out += [("md", ("## Time per step\n\nThe time of every step of this run against its estimate. The next run of the same "
                   "`RUN_SIZE` uses these times as its estimates (`data/runs/_step_times.json`, step names and seconds only).")),
            ("code", 'step("Time per step")\nstep_report()')]
    return out


def swap(src: str, pairs) -> str:
    for a, b in pairs:
        assert src.count(a) == 1, f"not found once: {a[:70]}"
        src = src.replace(a, b)
    return src


def build_prep(prep):
    """The notebooks2 PREP cells with the notebooks3 changes."""
    cells = []
    for kind, src in prep:
        if kind == "md" and src.startswith(PREP_TITLE_OLD):
            src = src.replace(PREP_TITLE_OLD, PREP_TITLE_NEW, 1)
        if kind == "code" and src.lstrip().startswith("# --- settings"):
            src = swap(src, PREP_SETTINGS_SWAPS)
        if kind in ("md", "code") and any(a in src for a, _ in STEP3_SWAPS):
            src = swap(src, [p for p in STEP3_SWAPS if p[0] in src])
        if kind == "md" and src.startswith("## Step 5 ·"):
            cells += STEP_4C
            src = src.replace("## Step 5 · `TEST_10K`: 10,000 lines to train, 10,000 lines of one target group to score",
                              "## Step 5 · The test copy (`RUN_SIZE` = 1k / 5k / 10k): training lines + target lines of one group")
        cells.append((kind, src))
        if kind == "code" and "def contract_name(stem: str)" in src:
            cells.append(("code", PREP_PATCH))
    settings_at = next(i for i, (k, s) in enumerate(cells) if k == "code" and s.lstrip().startswith("# --- settings"))
    return add_timer(cells, settings_at + 1, "02_prepare_week", PREP_DEFAULT, '""')
