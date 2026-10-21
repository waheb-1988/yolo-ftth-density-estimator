"""Scrapes Omantel and Vodafone Oman public plan pages into the competitor
vector store (see core/competitors.py).

Run directly:  python -m nbx.scripts.scrape_competitors
Offline:       python -m nbx.scripts.scrape_competitors --index-only
               (re-index the JSON snapshot in data/competitors/, no scraping)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nbx.core import competitors, config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-only", action="store_true", help="Index the existing snapshot instead of scraping.")
    args = parser.parse_args()

    if args.index_only:
        summary = competitors.index_snapshot()
        print(f"{summary['indexed']} competitor documents indexed from {config.COMPETITOR_OFFERS_FILE}")
        sys.exit(0)

    print(f"Scraping {len(config.COMPETITOR_SOURCES)} pages ({config.COMPETITOR_REQUEST_DELAY_S}s between requests)...")
    summary = competitors.refresh()
    for p in summary["pages"]:
        print(f"  {p['operator']:14s} offers={p['offers']:3d} passages={p['passages']:3d}  {p['url']}")
    for e in summary["errors"]:
        print(f"  ERROR {e['url']}: {e['error']}")
    print(f"\n{summary['offers']} structured offers, {summary['passages']} passages, {summary['indexed']} documents indexed "
          f"-> {config.COMPETITOR_OFFERS_FILE}")
