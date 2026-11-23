"""The FINAL weekly pair: 02_prepare_week_final.ipynb + 04_nbx_llm_final.ipynb.

Every week: the raw exports + the target file nbx_llm_target.csv.gz in data/data_raw/, then 02 final, then 04 final.
- 02 final = 02 with: the target list is TARGET_FILE (nbx_llm_target) only, scored as group a (constraint, T2C);
  target2_a / target2_b / the NBA sample in data_raw are ignored; the run stops when the target file is missing, or
  when an export is older than the newest one by more than MAX_FILE_AGE_DAYS (last week's file left in data_raw).
- 04 final = 04 real with: RUN_SIZE = "all" (every line of the target file), SCORE_GROUPS = ["t2c"], training on the
  whole base, outputs in data/nbx_llm/ (apart from the tests in data/real25k/).
The engine (nbx/) is not modified.
"""

PREP_TITLE = r'''# 02 final · Prepare the week: the raw exports + the target file `nbx_llm_target`

**Every week:** copy this week's exports **and** the target file `nbx_llm_target.csv.gz` into `data/data_raw/`
(remove last week's files), then **Restart & Run All**. Then run `04_nbx_llm_final` (`python run_notebook04.py --final`).

What this notebook does (the same steps as `02_prepare_week`, with the final-run rules):
- **The target:** `nbx_llm_target` is this week's target list. **All its lines** are scored by 04 final, with the
  **constraint logic** (group a, T2C: the action's offer families). It is written as
  `data/weekly/<D>/target2/a_constraint/target2_a.csv.gz`. Any `target2_a` / `target2_b` / NBA sample file in
  `data_raw` is **ignored**. No target file = the notebook stops.
- **The right files:** step 1 lists every export with its date; an export older than the newest one by more than
  `MAX_FILE_AGE_DAYS` (a file of last week left in `data_raw`) **stops** the run (`STRICT_FRESH = True`). The dates
  D and T are then read from the data (step 3), and 04 final checks them again.
- **`RUN_SIZE = "all"`:** the full week, no test copy.
- The new export names, the NBA dedup (step 4b), the latest campaign (step 4c) and the `[time]` lines are unchanged.'''

PREP_SETTINGS_SWAPS = [
    ('GROUP_MAP = {"cstr": "t2c", "free": "t2f", "free_w_nat": "t2f"}   # its group -> T2C (a, constraint) / T2F (b, free)',
     ('GROUP_MAP = {"cstr": "t2c", "free": "t2f", "free_w_nat": "t2f"}   # (not used by the final run)\n'
     '\n'
     '# --- the final weekly run ---------------------------------------------------------------------------------------------\n'
     'TARGET_FILE = "nbx_llm_target"    # this week\'s target list in data_raw (nbx_llm_target.csv.gz): ALL its lines are scored\n'
     '                                  #   with the constraint logic (group a, T2C); target2_a / target2_b / the NBA sample: ignored\n'
     'MAX_FILE_AGE_DAYS = 5             # an export older than the newest one by more than this = probably last week\'s file\n'
     'STRICT_FRESH = True               # True = stop on such a file (replace it, or set False once you checked it)')),
]

STEP1_SWAPS = [
    ('''    kind = ("contract" if table else
            "target list" if re.fullmatch(r"target2_[ab]", s0) else
            "NBA target sample" if s0 == NBA_SAMPLE_TABLE.lower() else "unknown (ignored)")
    rows.append({"file": p.name, "table": table or SUFFIX.sub("", stem),''',
     '''    is_target = s0 == TARGET_FILE.lower()                       # the final run: TARGET_FILE is the only target list
    if is_target:
        table, note = None, ""
    kind = ("contract" if table else
            "target list" if is_target else
            "ignored (not TARGET_FILE)" if re.fullmatch(r"target2_[ab]", s0) or s0 == NBA_SAMPLE_TABLE.lower() else
            "unknown (ignored)")
    rows.append({"file": p.name, "table": table or ("target2_a" if is_target else SUFFIX.sub("", stem)),'''),
]

STEP1_EXTRA = r'''

# --- the final run: the target file, and every export of THIS week ----------------------------------------------------
_tf = inv[inv["kind"] == "target list"]
if _tf.empty:
    raise SystemExit(f"no {TARGET_FILE}.csv.gz (or .csv) in {RAW}: copy this week's target file there "
                     "(one MSISDN per line, or a column msisdn / access_method_val)")
if len(_tf) > 1:
    raise SystemExit(f"{len(_tf)} files for {TARGET_FILE} in {RAW}: keep one")
_ign = inv[inv["kind"].astype(str).str.startswith("ignored (not TARGET_FILE)")]
if len(_ign):
    print(f"ignored (the final run scores {TARGET_FILE} only): " + ", ".join(_ign["file"]))
_m = inv[inv["kind"].isin(["contract", "target list"])].copy()
_m["modified"] = pd.to_datetime([Path(x).stat().st_mtime for x in _m["path"]], unit="s").round("min")
_m["days_older_than_newest"] = ((_m["modified"].max() - _m["modified"]).dt.total_seconds() / 86400).round(1)
display(Markdown(f"**The exports of this week, oldest first** (newest {_m['modified'].max():%Y-%m-%d %H:%M})"))
display(_m.sort_values("modified")[["file", "table", "kind", "modified", "days_older_than_newest", "MB"]].head(12))
_old = _m[_m["days_older_than_newest"] > MAX_FILE_AGE_DAYS]
if len(_old):
    _msg = (f"{len(_old)} export(s) older than the newest one by more than {MAX_FILE_AGE_DAYS} days (last week's file?): "
            + ", ".join(f"{r.file} ({r.days_older_than_newest} d)" for r in _old.itertuples()))
    if STRICT_FRESH:
        raise SystemExit(_msg + " -- replace them with this week's export, or set STRICT_FRESH = False once checked")
    print("WARNING:", _msg)
else:
    print(f"OK: every export is within {MAX_FILE_AGE_DAYS} days of the newest one; target file: {_tf.iloc[0]['file']} "
          f"({_tf.iloc[0]['MB']} MB) -> target2_a (group a, constraint)")'''

FINAL_TITLE = r'''# 04 final · The weekly run: every line of `nbx_llm_target`, the whole base to train, the constraint logic, on the GPU

The real run of `04_agents_nba_fast_ollama_real25k` with the final settings. Run it after `02_prepare_week_final`,
headless: `python run_notebook04.py --final` (or Restart & Run All).

| | |
|---|---|
| **Target** | every line of `nbx_llm_target` (written by 02 final as the group-a list): `RUN_SIZE = "all"`, `SCORE_GROUPS = ["t2c"]` |
| **Logic** | group a, constraint universe: the challenger rules (CC) with the LLM agents 3 (NBA) and 4 (NBO) |
| **Training** | the whole eligible prepaid base (`TRAIN_LINES = "all"`), on the `*_auto` tables at T |
| **Dates** | read from the data: D = the last purchase day, T = the campaign day (`date1`); the run stops if they do not hold |
| **LLM** | no budget, no stop: retries, Ollama restart, cache; FAST_MODE `minimal`; 6 offers per line (top-up + A-25) |
| **Outputs** | `data/nbx_llm/output/<D>/final/input_flytxt.csv.gz` (the file to load), the 360 views in `target2/`, the report |

Then check the results with `05_check_flytxt_results` (it finds this run by itself).'''

FINAL_SETTINGS_SWAPS = [
    ('RUN_SIZE = 25_000                 # target lines SCORED: 1_000 | 10_000 | 25_000 | 50_000 | "all" (the full lists, ~50k each)',
     'RUN_SIZE = "all"                  # the final run: EVERY line of nbx_llm_target (a number = a test on that many lines)'),
]

FINAL_PATH_SWAPS = [('"data" / "real25k"', '"data" / "nbx_llm"'), ("data/real25k", "data/nbx_llm"), ("data\\real25k", "data\\nbx_llm")]
