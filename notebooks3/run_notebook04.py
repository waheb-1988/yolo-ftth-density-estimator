"""Runs notebook 04 headless, with NO cell timeout.

    python notebooks3/run_notebook04.py --final       THE WEEKLY RUN: 04_nbx_llm_final (every line of nbx_llm_target)
    python notebooks3/run_notebook04.py               04_agents_nba_fast_ollama_real25k (tests: its own RUN_SIZE)
    python notebooks3/run_notebook04.py --size 50000  (or 1000, 25000, all: RUN_SIZE for this run only, either notebook)

- Every cell runs in order, as "Restart & Run All". The LLM progress is printed here and written to
  data/<nbx_llm | real25k>/runs/<D>/agents/llm_progress.log (follow it in another window: Get-Content <log> -Wait).
- The executed copy, with its outputs, is saved to data/<nbx_llm | real25k>/executed/ even when a cell fails. It holds MSISDNs:
  keep it on this laptop, never commit it.
- A re-run resumes the LLM part where it stopped (every answer is cached at once).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NB = HERE / "04_agents_nba_fast_ollama_real25k.ipynb"
NB_FINAL = HERE / "04_nbx_llm_final.ipynb"
SHOW = ("[time]", "[llm]", "dates:", "engine dates:", "pre-flight", "copy ", "VERDICT", "final file", "A-25", "WARNING",
        "offers per line", "top-up ready")


def _echo(cell, cell_index, **_):
    """After each cell: its [time] / [llm] / verdict lines."""
    for out in cell.get("outputs", []):
        text = out.get("text", "") if out.get("output_type") == "stream" else ""
        for line in str(text).splitlines():
            if line.strip().startswith(SHOW) or any(k in line for k in ("[llm]", "VERDICT", "WARNING", "BLOCKED", "READY")):
                print(f"  [cell {cell_index}] {line}", flush=True)


def _no_quick_edit():
    """Windows: a click in this console window selects text and FREEZES the runner (so the notebook stops between two
    cells until Esc is pressed). QuickEdit is switched off for this window while the run lasts."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-10)                                 # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, (mode.value & ~0x0040) | 0x0080)  # - ENABLE_QUICK_EDIT_MODE, + ENABLE_EXTENDED_FLAGS
            print("QuickEdit is off in this window: a click no longer pauses the run", flush=True)
    except (OSError, AttributeError, ValueError) as e:
        print(f"QuickEdit not changed ({e}): do not click inside this window while it runs", flush=True)


def _set_size(nb, size: str | None) -> None:
    """--size 1000 | 25000 | 50000 | all: replaces RUN_SIZE in the settings cell (in memory; the file is not changed)."""
    if size is None:
        return
    value = '"all"' if size.lower() == "all" else str(int(size.replace("_", "").replace(",", "")))
    for c in nb.cells:
        if c.cell_type == "code" and "TRAIN_LINES =" in c.source and re.search(r"(?m)^RUN_SIZE = ", c.source):
            c.source = re.sub(r"(?m)^RUN_SIZE = \S+", f"RUN_SIZE = {value}", c.source, count=1)
            print(f"RUN_SIZE = {value} for this run (the notebook file keeps its own value)", flush=True)
            return
    raise SystemExit("--size: no RUN_SIZE line found in the settings cell")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run notebook 04 real headless, with no cell timeout.")
    ap.add_argument("--final", action="store_true", help="the weekly run: 04_nbx_llm_final (outputs in data/nbx_llm/)")
    ap.add_argument("--size", help='target lines scored: 1000, 10000, 25000, 50000 or all (default: the notebook\'s RUN_SIZE)')
    args = ap.parse_args()
    _no_quick_edit()
    nb_path = NB_FINAL if args.final else NB
    out_dir = ROOT / "data" / ("nbx_llm" if args.final else "real25k") / "executed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"04_{'final' if args.final else 'real25k'}_{datetime.now():%Y%m%d_%H%M%S}.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    _set_size(nb, args.size)
    client = NotebookClient(nb, timeout=None, kernel_name="python3", allow_errors=False,
                            resources={"metadata": {"path": str(HERE)}})
    try:
        client.on_cell_executed = _echo                      # nbclient >= 0.6
    except AttributeError:
        pass
    t0, code = time.time(), 0
    print(f"running {nb_path.name} (no cell timeout) -> {out.relative_to(ROOT)}", flush=True)
    try:
        client.execute()
        print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    except CellExecutionError as e:
        code = 1
        print(f"STOPPED after {(time.time() - t0) / 60:.1f} min: {str(e).strip().splitlines()[-1] if str(e).strip() else e}",
              flush=True)
    except KeyboardInterrupt:
        code = 130
        print("interrupted: a re-run resumes the LLM part from the cache", flush=True)
    finally:
        nbformat.write(nb, out)
        print(f"executed notebook (with outputs, MSISDNs): {out.relative_to(ROOT)}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
