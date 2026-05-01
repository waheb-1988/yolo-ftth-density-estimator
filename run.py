#!/usr/bin/env python3
"""FALCON launcher — python run.py --demo"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def setup_directories():
    for d in ["data", "data/sample_images", "data/results"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def pre_generate_demo_data():
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from app.data_loader import initialize_demo_data
        initialize_demo_data()
    except Exception as e:
        print(f"[FALCON] Warning: demo pre-generation failed ({e})")
        print("         Data will be generated on first dashboard load.")


def main():
    parser = argparse.ArgumentParser(
        description="FALCON — Flavia AI Coverage & Location Opportunity Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --demo            Run with simulated AI vehicle density data
  python run.py                   Run in production mode (real YOLO detection)
  python run.py --demo --port 8502  Use a custom port
        """,
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run in demo mode with synthetic aerial imagery")
    parser.add_argument(
        "--port", type=int, default=8501,
        help="Streamlit server port (default: 8501)")
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not open the browser automatically")
    args = parser.parse_args()

    print("=" * 60)
    print("  FALCON -- FTTH Coverage Optimizer")
    print("  Flavia AI Coverage & Location Opportunity Network")
    print("=" * 60)

    setup_directories()
    os.environ["FALCON_DEMO_MODE"] = "1" if args.demo else "0"

    if args.demo:
        print("\n[1/2] Pre-generating synthetic aerial imagery & zone scores ...")
        pre_generate_demo_data()
        print("      Done.\n")

    print(f"[2/2] Launching Streamlit dashboard on port {args.port} ...")
    print(f"      URL : http://localhost:{args.port}")
    print(f"      Mode: {'DEMO' if args.demo else 'PRODUCTION'}")
    print("\n      Press Ctrl+C to stop FALCON.\n")

    dashboard_path = Path(__file__).parent / "app" / "dashboard.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(args.port),
        "--server.headless", "true" if args.no_browser else "false",
        "--browser.gatherUsageStats", "false",
        "--theme.primaryColor", "#1a1a2e",
        "--theme.backgroundColor", "#ffffff",
        "--theme.secondaryBackgroundColor", "#f0f2f6",
        "--theme.textColor", "#1a1a2e",
    ]

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n[FALCON] Stopped.")


if __name__ == "__main__":
    main()
