"""Runs notebooks3/04_agents_nba_fast_ollama_real25k.ipynb headless, with NO cell timeout.

    python notebooks3/run_notebook04.py            (in the activated .venv, from the project folder)

- Every cell runs in order, as "Restart & Run All". The LLM progress is printed here and written to
  data/real25k/runs/<D>/agents/llm_progress.log (follow it in another window: Get-Content <log> -Wait).
- The executed copy, with its outputs, is saved to data/real25k/executed/ even when a cell fails. It holds MSISDNs:
  keep it on this laptop, never commit it.
- A re-run resumes the LLM part where it stopped (every answer is cached at once).
"""
from __future__ import annotations

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
SHOW = ("[time]", "[llm]", "dates:", "engine dates:", "pre-flight", "copy ", "VERDICT", "final file", "A-25", "WARNING",
        "offers per line", "top-up ready")


def _echo(cell, cell_index, **_):
    """After each cell: its [time] / [llm] / verdict lines."""
    for out in cell.get("outputs", []):
        text = out.get("text", "") if out.get("output_type") == "stream" else ""
        for line in str(text).splitlines():
            if line.strip().startswith(SHOW) or any(k in line for k in ("[llm]", "VERDICT", "WARNING", "BLOCKED", "READY")):
                print(f"  [cell {cell_index}] {line}", flush=True)


def main() -> int:
    out_dir = ROOT / "data" / "real25k" / "executed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"04_real25k_{datetime.now():%Y%m%d_%H%M%S}.ipynb"
    nb = nbformat.read(NB, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name="python3", allow_errors=False,
                            resources={"metadata": {"path": str(HERE)}})
    try:
        client.on_cell_executed = _echo                      # nbclient >= 0.6
    except AttributeError:
        pass
    t0, code = time.time(), 0
    print(f"running {NB.name} (no cell timeout) -> {out.relative_to(ROOT)}", flush=True)
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
