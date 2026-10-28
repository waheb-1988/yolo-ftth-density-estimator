"""V0 vs V1.0 closed-loop comparison: same hidden worlds (seed), same scenarios, same arms, offline backend.

    python scripts/compare_v0_v1.py <out_dir> [--workers 6]   (V0 runs with this Python, V1.0 with its .venv)
    python scripts/compare_analyze.py <out_dir> <summary.json>
Each run: <out>/<config>/<scenario>/seed<k>/ (simulation root) + run.log. Skips runs that already finished.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # folder holding nbx-python-upgrade and nbx-python-upgrade-V1.0
V0, V1 = ROOT / "nbx-python-upgrade", ROOT / "nbx-python-upgrade-V1.0"
CONFIGS = {
    "V0": (V0, [sys.executable], {}),
    "V1.0": (V1, [str(V1 / ".venv" / "Scripts" / "python.exe")], {}),      # .venv/bin/python on Linux
    "V1.0_no_floor": (V1, [str(V1 / ".venv" / "Scripts" / "python.exe")], {"FSDR_OFFER_PRICE_FLOOR": "none"}),
    "V1.0_price_band": (V1, [str(V1 / ".venv" / "Scripts" / "python.exe")], {"FSDR_OFFER_POLICY": "price_band"}),
}
SCENARIOS = ["baseline", "price_sensitive", "churn_wave"]
SEEDS = [1, 2, 3, 4, 5]
N, WEEKS, START = 5000, 3, "2026-10-21"


def jobs():
    for cfg in CONFIGS:
        for scen in SCENARIOS:
            if cfg == "V1.0_price_band" and scen != "baseline":
                continue
            for seed in SEEDS:
                yield cfg, scen, seed


def run(out: Path, cfg: str, scen: str, seed: int) -> str:
    folder = out / cfg / scen / f"seed{seed}"
    if (folder / "simulation_report.json").exists():
        return f"skip {cfg} {scen} {seed}"
    folder.mkdir(parents=True, exist_ok=True)
    cwd, py, extra = CONFIGS[cfg]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "LLM_BACKEND": "offline", **extra}
    cmd = [*py, "run.py", "--backend", "offline", "fsdr-simulate", "--weeks", str(WEEKS), "--n", str(N), "--date", START,
           "--scenario", scen, "--seed", str(seed), "--arms", "free,cstr,llm", "--out", str(folder)]
    t0 = time.time()
    with (folder / "run.log").open("w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return f"{'ok ' if rc == 0 else 'FAIL'} {cfg} {scen} seed{seed} rc={rc} {time.time() - t0:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda j: run(out, *j), list(jobs())):
            print(msg, flush=True)
