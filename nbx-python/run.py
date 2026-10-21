"""NBX launcher: one entry point for every task, on Windows, macOS and Linux.

    python run.py check            verify Python, packages and data files
    python run.py setup            build the search indexes from the shipped data
    python run.py seed             regenerate synthetic data, then build indexes
    python run.py start            FastAPI + Gradio together (Ctrl+C stops both)
    python run.py api [--reload]   FastAPI only   -> http://localhost:8000/docs
    python run.py ui               Gradio only    -> http://localhost:7860
    python run.py test [args]      run the test suite (args go to pytest)

Add --backend offline|ollama|vllm to any command to override LLM_BACKEND.
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 10)

# (import name, pip name, needed for)
REQUIRED = [
    ("fastapi", "fastapi", "API"),
    ("uvicorn", "uvicorn", "API server"),
    ("gradio", "gradio", "web UI"),
    ("httpx", "httpx", "UI -> API calls"),
    ("pydantic", "pydantic", "schemas"),
    ("dotenv", "python-dotenv", ".env loading"),
    ("numpy", "numpy", "scoring"),
    ("scipy", "scipy", "collaborative filtering"),
    ("networkx", "networkx", "knowledge graph"),
    ("implicit", "implicit", "collaborative filtering (BPR)"),
    ("vowpalwabbit", "vowpalwabbit", "NBA contextual bandit"),
    ("bs4", "beautifulsoup4", "competitor scraping"),
    ("sentence_transformers", "sentence-transformers", "MiniLM embeddings"),
    ("chromadb", "chromadb", "vector store + competitor search"),
]
OPTIONAL = [
    ("faiss", "faiss-cpu", "FAISS vector store"),
    ("qdrant_client", "qdrant-client", "Qdrant vector store"),
    ("pymilvus", "pymilvus", "Milvus Lite vector store"),
    ("lancedb", "lancedb", "LanceDB vector store"),
    ("pytest", "pytest", "tests"),
    ("pytest_asyncio", "pytest-asyncio", "async tests"),
]
DATA_FILES = [
    "customers.json", "offers.json", "purchases.json", "segments.json", "revenue_history.json",
    "nba_scores_churn.json", "nba_scores_reactivation.json", "nba_scores_appetency.json",
    "nba_scores_fragility.json", "nba_scores_revenue_decrease.json", "nba_activity_usage.json",
    "nba_targeting.json", "competitors/competitor_offers.json", "competitors/competitor_passages.json",
]


def _env() -> dict:
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _py(*args: str, **kw) -> int:
    return subprocess.call([sys.executable, *args], cwd=ROOT, env=_env(), **kw)


def _indexed_docs(config) -> int:
    """Documents in the default Chroma collection (0 if not built). Only
    needed for retrieval_mode=vector and competitor search."""
    try:
        import chromadb
        from nbx.vectorstores.base import sanitize_embedding_key

        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        return client.get_collection(f"nbo_docs__{sanitize_embedding_key(config.DEFAULT_EMBEDDING_MODEL)}").count()
    except Exception:
        return 0


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
    from nbx.core import config

    print(f"\nData directory: {config.DATA_DIR}")
    missing = [f for f in DATA_FILES if not (config.DATA_DIR / f).exists()]
    for f in missing:
        print(f"  [FAIL] missing {f}  (run: python run.py seed)")
    if not missing:
        print(f"  [ok]   all {len(DATA_FILES)} data files present")
    ok = ok and not missing
    indexed = _indexed_docs(config)
    print(f"  [{'ok' if indexed else 'todo'}] search index "
          + (f"built ({indexed} documents)" if indexed else "not built yet (run: python run.py setup)"))

    print(f"\nLLM backend: {config.LLM_BACKEND}"
          + ("  (pure Python, deterministic agents)" if config.LLM_BACKEND == "offline" else ""))
    print("\nAll checks passed." if ok else "\nSome checks failed -- see above. Install with: pip install -r requirements.txt")
    return 0 if ok else 1


def cmd_setup(_args) -> int:
    rc = _py("-m", "nbx.scripts.seed_data", "--index-only")
    return rc or _py("-m", "nbx.scripts.scrape_competitors", "--index-only")


def cmd_seed(_args) -> int:
    rc = _py("-m", "nbx.scripts.seed_data")
    return rc or _py("-m", "nbx.scripts.scrape_competitors", "--index-only")


def _api_cmd(args) -> list[str]:
    from nbx.core import config

    cmd = [sys.executable, "-m", "uvicorn", "nbx.api.main:app", "--host", args.host, "--port", str(config.FASTAPI_PORT)]
    return cmd + (["--reload", "--reload-dir", "nbx"] if getattr(args, "reload", False) else [])


def cmd_api(args) -> int:
    sys.path.insert(0, str(ROOT))
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


def main() -> int:
    parser = argparse.ArgumentParser(description="NBX launcher", formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("--backend", choices=["offline", "ollama", "vllm"], help="override LLM_BACKEND")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify Python, packages and data files").set_defaults(fn=cmd_check)
    sub.add_parser("setup", help="build search indexes from the shipped data").set_defaults(fn=cmd_setup)
    sub.add_parser("seed", help="regenerate synthetic data and build indexes").set_defaults(fn=cmd_seed)
    for name, fn, text in (("start", cmd_start, "API + web UI together"), ("api", cmd_api, "API only")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to expose on the network)")
        p.add_argument("--reload", action="store_true", help="auto-reload the API on code changes")
        p.set_defaults(fn=fn)
    sub.add_parser("ui", help="web UI only (needs the API running)").set_defaults(fn=cmd_ui)
    p = sub.add_parser("test", help="run the test suite")
    p.add_argument("pytest_args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_test)

    args = parser.parse_args()
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
