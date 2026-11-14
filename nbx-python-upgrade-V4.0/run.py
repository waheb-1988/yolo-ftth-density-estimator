"""FSDR launcher: one entry point for every task, on Windows, macOS and Linux. Run it through uv:

    uv sync                               create .venv from uv.lock (once, and after a dependency change)
    uv run python run.py check            verify Python, packages and the data folders
    uv run python run.py start            FastAPI + Gradio together (Ctrl+C stops both)
    uv run python run.py api [--reload]   FastAPI only   -> http://localhost:8000/docs
    uv run python run.py ui               Gradio only    -> http://localhost:7860
    uv run python run.py test [args]      run the test suite (args go to pytest)

FSDR V4.0 (step by step: docs/SETUP_NEW_LAPTOP.md and notebooks/01_training_step_by_step.ipynb, 02_scoring_...):
    fsdr-generate --date 2026-10-22 --mode small|medium|full   data/training/<date>/ + data/scoring/<date>/ (5 folders each)
    fsdr-load --date 2026-10-22           step T0: validate both twins, completeness, Parquet cache (data/runs/<date>/train)
    fsdr-features --date 2026-10-22       steps T2 + T3: features, segments, the two actions (training)
    fsdr-targets --generate               dummy lists in data/scoring/<date>/target1|2/a_constraint|b_free/ + control/
    fsdr-run-targets                      panel -> target 1 -> target 2 (scored separately) -> control -> push
    fsdr-score-target --target target1    one target only -> data/output/<date>/target1/
    fsdr-push                             data/output/<date>/push/input_flytxt.csv.gz (both targets PASS)
    fsdr-validate --data-dir data/training/2026-10-22
    fsdr-schemas     export one JSON Schema per table to docs/fsdr_json_schemas/
    fsdr-train       [--data-dir DIR] train + register the bandit (latest run folder by default)
    fsdr-score       score the test sample (arms free / cstr / llm) with the champion
    fsdr-recommend --arm cstr|free|llm   offers (llm = action and offer agents)
    fsdr-decide --msisdn N --trace       one agentic decision with its full trace
    fsdr-batch [--arm llm] [--limit 100]  action -> top 4 offers -> top 2 roaming -> international
    fsdr-feedback    rewards of last week's campaign (run on the next week's folder)
    fsdr-simulate --weeks 3 --n 2000      closed loop on the hidden world, report per arm
    fsdr-report [--week D]                take rate / revenue per arm, uplift vs control
    fsdr-agent --agent training --trace   data / training / scoring / monitor agent
    fsdr-weekly [--llm-limit 500]         the whole weekly cycle in one command (for the scheduler)

Notebooks (see docs/SETUP_NEW_LAPTOP.md §8):
    notebook --name training [--lab]      notebooks/01_training_step_by_step.ipynb (every training step, explained)
    notebook --name scoring  [--lab]      notebooks/02_scoring_step_by_step.ipynb  (target 1, target 2, push)
    notebook [--lab]                      notebooks/fsdr_results.ipynb (read-only view of a run)
    notebook-html [--date D]              run the notebook headless -> data/runs/<D>/fsdr_results_<D>.html

vLLM (see docs/GPU_SETUP.md):
    vllm-install     install vLLM in WSL2 (once)
    vllm-serve       serve the model on http://127.0.0.1:8001, then set LLM_BACKEND=vllm

Add --backend offline|vllm to any command to override LLM_BACKEND.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 10)

# (import name, package name, needed for) -- the versions are pinned in uv.lock
REQUIRED = [
    ("fastapi", "fastapi", "API"),
    ("uvicorn", "uvicorn", "API server"),
    ("multipart", "python-multipart", "file upload"),
    ("gradio", "gradio", "web UI"),
    ("httpx", "httpx", "UI -> API calls"),
    ("pydantic", "pydantic", "schemas"),
    ("dotenv", "python-dotenv", ".env loading"),
    ("numpy", "numpy", "scoring"),
    ("pandas", "pandas", "FSDR tables"),
    ("scipy", "scipy", "collaborative filtering"),
    ("networkx", "networkx", "cohort graph"),
    ("implicit", "implicit", "collaborative filtering (BPR)"),
    ("vowpalwabbit", "vowpalwabbit", "NBA contextual bandit"),
]
OPTIONAL = [
    ("pytest", "pytest", "tests (uv sync installs the dev group)"),
    ("pytest_asyncio", "pytest-asyncio", "async tests"),
    ("jsonschema", "jsonschema", "schema tests"),
]


def _env() -> dict:
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _py(*args: str, **kw) -> int:
    return subprocess.call([sys.executable, *args], cwd=ROOT, env=_env(), **kw)


def cmd_check(_args) -> int:
    ok = True
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < MIN_PYTHON:
        print(f"  [FAIL] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required")
        ok = False

    for group, rows, fatal in (("Required packages", REQUIRED, True), ("Optional packages", OPTIONAL, False)):
        print(f"\n{group}:")
        for mod, pip_name, why in rows:
            try:
                importlib.import_module(mod)
                print(f"  [ok]   {pip_name:24s} {why}")
            except Exception as e:  # ImportError, or a broken native build
                print(f"  [{'FAIL' if fatal else 'skip'}] {pip_name:24s} {why}  ({type(e).__name__})")
                ok = ok and not fatal

    sys.path.insert(0, str(ROOT))
    import nbx  # noqa: F401  (switches the usage analytics off before anything else)
    from nbx.core import config

    print(f"\nData directory: {config.DATA_DIR}")
    for name, hint in (("training", "fsdr-generate (or copy the training exports)"),
                       ("scoring", "fsdr-generate (or copy the scoring exports + target lists)"),
                       ("output", "fsdr-run-targets")):
        runs = sorted(p.name for p in (config.DATA_DIR / name).glob("20??-??-??") if p.is_dir())
        print(f"  [{'ok' if runs else 'todo'}] data/{name + '/':10s} " + (f"{len(runs)} run date(s), latest {runs[-1]}"
              if runs else f"empty (run: uv run python run.py {hint})"))

    print(f"\nLLM backend: {config.LLM_BACKEND}"
          + ("  (pure Python, deterministic agents)" if config.LLM_BACKEND == "offline"
             else f"  {config.llm_url()}  model {config.DEFAULT_LLM}"))
    print(f"\nLocal only (FSDR_LOCAL_ONLY): {'on' if config.LOCAL_ONLY else 'OFF'}")
    bad = config.local_violations()
    print(f"  [{'ok' if not bad else 'WARN'}]   no prompt, trace or data leaves this machine" if not bad else "")
    for b in bad:
        print(f"  [{'refused' if config.LOCAL_ONLY else 'WARN'}] {b}")
    off = [k for k in ("GRADIO_ANALYTICS_ENABLED", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK", "VLLM_NO_USAGE_STATS")
           if os.environ.get(k) in ("False", "1")]
    print(f"  [{'ok' if len(off) == 4 else 'WARN'}]   usage analytics off: {', '.join(off)}")
    print("  [info] no cloud LLM API (OpenAI, Anthropic / Claude, ...) is used; the LLM runs on this machine "
          "(offline, vLLM, Ollama, llama.cpp or LM Studio)")
    print(f"API key: {'required (FSDR_API_KEY)' if config.API_KEY else 'not set (open API, local dev only)'}")
    print("\nAll checks passed." if ok else "\nSome checks failed -- see above. Install with: uv sync")
    return 0 if ok else 1


def _api_cmd(args) -> list[str]:
    from nbx.core import config

    cmd = [sys.executable, "-m", "uvicorn", "nbx.api.main:app", "--host", args.host, "--port", str(config.FASTAPI_PORT)]
    return cmd + (["--reload", "--reload-dir", "nbx"] if getattr(args, "reload", False) else [])


def _warn_if_exposed(args) -> None:
    """Review #5: an API bound to the network without FSDR_API_KEY is open to anyone who reaches the port."""
    from nbx.core import config

    if args.host not in ("127.0.0.1", "localhost") and not config.API_KEY:
        print(f"WARNING: the API listens on {args.host} without FSDR_API_KEY: every /fsdr/* route is open. "
              "Set FSDR_API_KEY (and FSDR_ADMIN_KEY for forced promotions) in .env.")


def cmd_api(args) -> int:
    sys.path.insert(0, str(ROOT))
    _warn_if_exposed(args)
    return subprocess.call(_api_cmd(args), cwd=ROOT, env=_env())


def cmd_ui(_args) -> int:
    return _py("-m", "nbx.ui.gradio_app")


def _wait_for_api(port: int, proc: subprocess.Popen, timeout_s: float = 120) -> bool:
    import httpx

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1)
    return False


def cmd_start(args) -> int:
    sys.path.insert(0, str(ROOT))
    from nbx.core import config

    _warn_if_exposed(args)
    print(f"Starting API on http://localhost:{config.FASTAPI_PORT} (LLM_BACKEND={config.LLM_BACKEND}) ...")
    api = subprocess.Popen(_api_cmd(args), cwd=ROOT, env=_env())
    ui = None
    try:
        if not _wait_for_api(config.FASTAPI_PORT, api):
            print("ERROR: the API did not become healthy -- see the log above.")
            return 1
        print(f"\n  API      http://localhost:{config.FASTAPI_PORT}\n  API docs http://localhost:{config.FASTAPI_PORT}/docs\n"
              f"  Web UI   http://localhost:{config.GRADIO_PORT}\n\nPress Ctrl+C to stop.\n")
        ui = subprocess.Popen([sys.executable, "-m", "nbx.ui.gradio_app"], cwd=ROOT, env=_env())
        while api.poll() is None and ui.poll() is None:
            time.sleep(1)
        return api.returncode or (ui.returncode or 0)
    except KeyboardInterrupt:
        print("\nStopping ...")
        return 0
    finally:
        for p in (ui, api):
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()


def cmd_test(args) -> int:
    return _py("-m", "pytest", *args.pytest_args)


FSDR_COMMANDS = {
    "fsdr-generate": "write dummy data: data/training/<D>/ + data/scoring/<D>/ (5 folders each, .csv.gz)",
    "fsdr-load": "step T0: validate + completeness + Parquet cache (data agent)",
    "fsdr-features": "steps T2 + T3: features, segments, micro-segments, action_original / action_agent",
    "fsdr-panel": "step P1: panel of the 4 groups (T1C, T1F, T2C, T2F) + control",
    "fsdr-offers": "step P6: code-2 tables and the 8 reco tables of the 4 groups (.csv.gz)",
    "fsdr-targets": "target lists of data/scoring/<D>/target1|2/a_constraint|b_free/: check, or --generate",
    "fsdr-final": "final step: input_flytxt to push + per-target files + details",
    "fsdr-run-groups": "V3.0 flow: the 4 groups together -> data/runs/<D>/final/",
    "fsdr-score-target": "V4.0: one target (--target target1|target2) -> data/output/<D>/<target>/",
    "fsdr-push": "V4.0: both targets merged -> data/output/<D>/push/input_flytxt.csv.gz",
    "fsdr-run-targets": "V4.0: panel -> target 1 -> target 2 -> control -> push, in one go",
    "fsdr-validate": "validate a run folder against the data contract",
    "fsdr-schemas": "export one JSON Schema per table",
    "fsdr-train": "train the NBA bandit on a run folder",
    "fsdr-score": "score the target sample with the champion model",
    "fsdr-recommend": "offers for an arm: --arm free|cstr|llm",
    "fsdr-decide": "one llm-arm decision with the agent trace",
    "fsdr-batch": "batch: action -> top 4 offers -> top 2 roaming -> international",
    "fsdr-feedback": "rewards of last week's campaign at D+7 (campaign log -> nba_rewards)",
    "fsdr-simulate": "closed-loop simulation: --weeks N --n 2000 --arms free,cstr,llm",
    "fsdr-report": "weekly report per arm: take rate, revenue, uplift vs control",
    "fsdr-agent": "operations agent on a run: --agent data|training|scoring|monitor",
    "fsdr-weekly": "the whole weekly cycle: validate, feedback, train, score, offers, report",
}


def cmd_fsdr(args) -> int:
    return _py("-m", "nbx.fsdr.cli", args.command, *args.fsdr_args)


VLLM_VARS = ("VLLM_API_KEY", "VLLM_HF_MODEL", "VLLM_SERVED_NAME", "VLLM_PORT", "VLLM_MAX_MODEL_LEN",
             "VLLM_GPU_UTIL", "VLLM_MAX_SEQS", "VLLM_OFFLINE")


def cmd_vllm(args) -> int:
    """Runs vllm/install_wsl.sh or vllm/serve_wsl.sh: inside WSL2 on Windows, directly on Linux."""
    script = ROOT / "vllm" / ("install_wsl.sh" if args.command == "vllm-install" else "serve_wsl.sh")
    try:
        from dotenv import dotenv_values
        dotenv = {k: v for k, v in dotenv_values(ROOT / ".env").items() if k in VLLM_VARS and v}
    except ImportError:
        dotenv = {}
    env = {**os.environ, **dotenv}
    if os.name != "nt":
        return subprocess.call(["bash", str(script)], env=env)
    # WSLENV lists the Windows variables that cross into WSL.
    env["WSLENV"] = ":".join([*filter(None, [os.environ.get("WSLENV")]), *[v for v in VLLM_VARS if v in env]])
    drive, rest = script.as_posix().split(":", 1)
    cmd = ["wsl.exe", *(["-d", args.distro] if args.distro else []), "--", "bash", f"/mnt/{drive.lower()}{rest}"]
    return subprocess.call(cmd, env=env)


NOTEBOOK = ROOT / "notebooks" / "fsdr_results.ipynb"
NOTEBOOKS = {"results": (NOTEBOOK, "notebooks/build_results_notebook.py"),
             "training": (ROOT / "notebooks" / "01_training_step_by_step.ipynb", "notebooks/build_step_notebooks.py"),
             "scoring": (ROOT / "notebooks" / "02_scoring_step_by_step.ipynb", "notebooks/build_step_notebooks.py")}


def cmd_notebook(args) -> int:
    """Opens a notebook: in VS Code (pick the .venv kernel) or, with --lab, in JupyterLab."""
    path, builder = NOTEBOOKS[args.name]
    if not path.exists():
        _py(builder)
    if args.lab:
        return _py("-m", "jupyterlab", str(path))
    code = shutil.which("code")
    if code:
        return subprocess.call([code, str(path)])
    print(f"VS Code not on PATH: open {path} yourself, or run: uv run python run.py notebook --name {args.name} --lab")
    return 0


def cmd_notebook_html(args) -> int:
    """Runs the notebook on one run date without a browser and saves it as HTML next to the run's steps."""
    runs = ROOT / "data" / "runs"
    dates = sorted(p.name for p in runs.glob("20??-??-??") if p.is_dir()) if runs.is_dir() else []
    run = args.date or (dates[-1] if dates else None)
    if not run:
        print("No run in data/runs: run fsdr-load first.")
        return 1
    env = {**_env(), "FSDR_NOTEBOOK_RUN_DATE": run}
    out = runs / run
    rc = subprocess.call([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", "--execute", str(NOTEBOOK),
                          "--output-dir", str(out), "--output", f"fsdr_results_{run}"], cwd=ROOT, env=env)
    if rc == 0:
        print(f"Report: {out / f'fsdr_results_{run}.html'}")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="FSDR launcher", formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("--backend", choices=["offline", "vllm"], help="override LLM_BACKEND")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify Python, packages and data files").set_defaults(fn=cmd_check)
    for name, fn, text in (("start", cmd_start, "API + web UI together"), ("api", cmd_api, "API only")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to expose on the network)")
        p.add_argument("--reload", action="store_true", help="auto-reload the API on code changes")
        p.set_defaults(fn=fn)
    sub.add_parser("ui", help="web UI only (needs the API running)").set_defaults(fn=cmd_ui)
    p = sub.add_parser("test", help="run the test suite")
    p.add_argument("pytest_args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_test)
    p = sub.add_parser("notebook", help="open a notebook (VS Code, or --lab for JupyterLab)")
    p.add_argument("--name", choices=list(NOTEBOOKS), default="results",
                   help="training / scoring: the step-by-step notebooks; results: the read-only view (default)")
    p.add_argument("--lab", action="store_true")
    p.set_defaults(fn=cmd_notebook)
    p = sub.add_parser("notebook-html", help="run the results notebook headless and save it as HTML")
    p.add_argument("--date", help="run date (default: the latest in data/runs)")
    p.set_defaults(fn=cmd_notebook_html)
    for name, text in FSDR_COMMANDS.items():
        p = sub.add_parser(name, help=text, add_help=False)
        p.add_argument("fsdr_args", nargs=argparse.REMAINDER)
        p.set_defaults(fn=cmd_fsdr)
    for name, text in (("vllm-install", "install vLLM (WSL2 on Windows)"),
                       ("vllm-serve", "serve the agents' model with vLLM on port 8001")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--distro", default=os.environ.get("WSL_DISTRO", "Ubuntu"), help="WSL distribution")
        p.set_defaults(fn=cmd_vllm)

    # pytest options such as -v or -k reach REMAINDER only via parse_known_args.
    args, extra = parser.parse_known_args()
    if extra and args.command != "test" and args.command not in FSDR_COMMANDS:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if args.command == "test":
        args.pytest_args = extra + args.pytest_args
    if args.command in FSDR_COMMANDS:
        args.fsdr_args = extra + args.fsdr_args
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
