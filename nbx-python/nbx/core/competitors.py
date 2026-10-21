"""Competitor intelligence: Omantel and Vodafone Oman public plan pages.

Pipeline (scripts/scrape_competitors.py, POST /competitors/refresh):
  1. fetch     -- the public pages in config.COMPETITOR_SOURCES, one request
                  every COMPETITOR_REQUEST_DELAY_S seconds, identified UA;
  2. parse     -- structured offers where the page has a recognizable shape
                  (Vodafone `plan-revamp` cards, Omantel data-add-on tables),
                  plus every page split into text passages;
  3. normalize -- price in OMR (baiza / 1000), data in GB, validity in days,
                  monthly-equivalent price and price per GB;
  4. index     -- passages + one passage per structured offer go into their
                  own Chroma collection (same embedding model as the rest of
                  the app); the structured offers are kept as JSON for exact
                  numeric comparison.

The agents never see raw HTML: they call `compare_offer` (numbers) and
`search` (passages) through the competitor_benchmark tool.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone

import httpx

from nbx.core import config

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
_DATA_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(GB|MB)\b", re.I)
_PRICE_RE = re.compile(r"(?:OMR\s*(\d+(?:\.\d+)?))|(?:(\d+(?:\.\d+)?)\s*(OMR|R\.O\.?|BZ|Baiza|BAIZA)\b)", re.I)
_DAYS_RE = re.compile(r"(\d+)\s*(?:Days?|days?)\b")


def parse_data_gb(text: str) -> float | None:
    """'6 GB' -> 6.0, '500 MB' -> 0.488, '40 GB + 20 GB' -> 60.0 (plan + bonus)."""
    parts = _DATA_RE.findall(text or "")
    if not parts:
        return None
    total = sum(float(v) / 1024 if unit.upper() == "MB" else float(v) for v, unit in parts)
    return round(total, 3)


def parse_price_omr(text: str) -> float | None:
    m = _PRICE_RE.search(text or "")
    if not m:
        return None
    if m.group(1):
        return float(m.group(1))
    v, unit = float(m.group(2)), m.group(3).upper()
    return round(v / 1000, 3) if unit in ("BZ", "BAIZA") else v


_WEEKS_RE = re.compile(r"(\d+)\s*weeks?\b", re.I)


def parse_days(text: str) -> int | None:
    t = (text or "").lower()
    m = _DAYS_RE.search(text or "")
    if m:
        return int(m.group(1))
    m = _WEEKS_RE.search(text or "")
    if m:
        return int(m.group(1)) * 7
    if "commitment" in t or "months" in t or "bill cycle" in t or t.strip() == "12":
        return 30  # postpaid: billed monthly
    for word, days in (("3 hours", 1), ("hour", 1), ("daily", 1), ("1 day", 1), ("weekly", 7), ("week", 7), ("monthly", 30), ("month", 30)):
        if word in t:
            return days
    return None


def enrich(offer: dict) -> dict:
    price, days, gb = offer.get("price_omr"), offer.get("validity_days"), offer.get("data_gb")
    monthly = round(price * 30 / days, 2) if price is not None and days else None
    per_gb = round(price / gb, 3) if price is not None and gb else None
    return {**offer, "monthly_equiv_omr": monthly, "price_per_gb_omr": per_gb}


def _category(page_type: str, text: str) -> str:
    t = text.lower()
    if "roam" in t or "jawazak" in t:
        return "Roaming"
    if page_type == "international" or "international" in t:
        return "International"
    if page_type == "data" or "data only" in t or "data plans" in t:
        return "Data"
    if page_type == "voice":
        return "Voice"
    return "Bundle"  # full plans: data + minutes


def _offer_id(operator: str, name: str, price, days) -> str:
    return hashlib.sha1(f"{operator}|{name}|{price}|{days}".encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _soup(html: str):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form"]):
        t.decompose()
    return soup


def parse_vodafone_cards(html: str, url: str, page_type: str) -> list[dict]:
    """Vodafone `div.plan-revamp` cards: [badge] | 6GB | RED 5 | 5 | / | 28 Days | 6 GB Local Data | ..."""
    soup = _soup(html)
    out = []
    for card in soup.select("div.plan-revamp"):
        parts = [p for p in card.stripped_strings]
        if "/" not in parts:
            continue
        slash = parts.index("/")
        if slash < 2:
            continue
        name, price_txt = parts[slash - 2], parts[slash - 1]
        try:
            price = float(price_txt.replace(",", ""))
        except ValueError:
            continue
        validity = parse_days(parts[slash + 1]) if slash + 1 < len(parts) else None
        body = " | ".join(parts[slash + 2:])
        headline_gb = parse_data_gb(parts[slash - 3]) if slash >= 3 else None
        minutes = next((p for p in parts if re.search(r"minutes", p, re.I)), "")
        features = [p for p in parts[slash + 2:] if len(p) < 60 and not p.startswith("*") and p not in ("Plan Benefits",)]
        out.append(enrich({
            "operator": "Vodafone Oman",
            "plan": name,
            # Full plans; roaming/international passes inside a card are features, not its category.
            "category": "Bundle" if minutes else "Data",
            "segment": page_type if page_type in ("students", "visitors") else "mass",
            "price_omr": price,
            "validity_days": validity,
            "data_gb": headline_gb or parse_data_gb(body),
            "minutes": minutes,
            "features": features[:8],
            "source_url": url,
            "page_type": page_type,
        }))
    return out


def parse_omantel_data_tables(html: str, url: str, page_type: str) -> list[dict]:
    """Omantel add-on tables: header 'Data / Subscription Fee / Activation [/ Validity]'
    followed by rows 'data, price, Dial:..., [validity]'. Tables appear in the
    order of the page's period tabs (Daily, Weekly, Monthly, PAYG)."""
    lines = _soup(html).get_text("\n", strip=True).split("\n")
    tabs = [l for l in lines[:40] if l in ("Daily", "Weekly", "Monthly", "PAYG")]
    tab_days = {"Daily": 1, "Weekly": 7, "Monthly": 30, "PAYG": None}
    out, block = [], -1
    i = 0
    while i < len(lines):
        if lines[i] == "Data" and i + 2 < len(lines) and lines[i + 1] == "Subscription Fee":
            block += 1
            i += 3
            continue
        if block >= 0 and (_DATA_RE.fullmatch(lines[i]) or lines[i].startswith("Non-Stop") or re.fullmatch(r"\d+(\.\d+)? ?(GB|MB).*", lines[i])):
            name = lines[i]
            window = lines[i + 1:i + 6]
            price_line = next((w for w in window if _PRICE_RE.search(w)), None)
            price = parse_price_omr(price_line) if price_line else None
            explicit = next((w for w in window if re.fullmatch(r"\d+ (Days?|Hours?)", w) or w in ("1 Day", "3 Hours")), None)
            period = tabs[block] if block < len(tabs) else None
            validity = parse_days(explicit) if explicit else tab_days.get(period)
            if price is not None:
                out.append(enrich({
                    "operator": "Omantel",
                    "plan": f"{name} ({period or 'add-on'})",
                    "category": "Data",
                    "segment": "mass",
                    "price_omr": price,
                    "validity_days": validity,
                    "data_gb": None if name.lower().startswith("non-stop") else parse_data_gb(name),
                    "minutes": "",
                    "features": ["Non-stop data (fair usage applies)"] if name.lower().startswith("non-stop") else
                                (["Night data 9PM-8AM"] if "night" in name.lower() else []),
                    "source_url": url,
                    "page_type": page_type,
                }))
        i += 1
    return out


def parse_omantel_api(payload: list, url: str) -> list[dict]:
    """Omantel portal-plans JSON: [{categoryCode, categoryEntries: [{shortDescription,
    price: [...], attributes: [{name, value}]}]}]."""
    out = []
    for cat in payload or []:
        code = cat.get("categoryCode", "")
        mapping = next((v for k, v in config.OMANTEL_API_CATEGORIES.items() if code.startswith(k)), None)
        if mapping is None:
            continue
        category, page_type, segment = mapping
        for e in cat.get("categoryEntries", []):
            attrs = {a["name"]: str(a["value"]).strip() for a in e.get("attributes", []) if a.get("value") not in (None, "null", "")}
            prices = [p.get("value") if isinstance(p, dict) else p for p in (e.get("price") or [])]
            try:
                price = float(prices[0])
            except (IndexError, TypeError, ValueError):
                continue
            data_txt = attrs.get("INCLUDED_DATA") or attrs.get("LOC_DATA_PACKAGE_SIZE") or ""
            minutes = attrs.get("INCLUDED_MINUTES") or attrs.get("LOC_MIN_PACKAGE_SIZE") or ""
            features = [f"{k.replace('INCLUDED_', '').replace('_', ' ').lower()}: {v}" for k, v in attrs.items()
                        if k in ("INCLUDED_INT_MINUTES", "INCLUDED_ROAMING_DATA", "CONTRACT_DURATION")]
            if "+" in data_txt:
                features.append(f"data includes bonus: {data_txt}")
            out.append(enrich({
                "operator": "Omantel",
                "plan": (e.get("shortDescription") or "").strip(),
                "category": category,
                "segment": segment,
                "price_omr": price,
                "validity_days": parse_days(attrs.get("DURATION_LABEL", "")) or 30,
                "data_gb": None if "unlimited" in data_txt.lower() else parse_data_gb(data_txt),
                "minutes": minutes,
                "features": features,
                "source_url": url,
                "page_type": page_type,
            }))
    return out


def page_passages(html: str, url: str, operator: str, page_type: str, max_chars: int = 700) -> list[dict]:
    """Visible text split into passages at section-sized boundaries, with
    boilerplate (app download / top-up banners) dropped."""
    lines = _soup(html).get_text("\n", strip=True).split("\n")
    noise = ("Skip to main content", "Download the app", "NEW MOBILE APP", "Top up now!", "Get the app", "Next", "العربية")
    lines = [l for l in lines if not any(l.startswith(n) for n in noise)]
    passages, buf = [], []
    for l in lines:
        buf.append(l)
        if sum(len(b) + 1 for b in buf) >= max_chars:
            passages.append(" ".join(buf))
            buf = []
    if buf:
        passages.append(" ".join(buf))
    title = lines[0] if lines else page_type
    return [
        {"text": f"{operator} - {title}: {p}", "operator": operator, "page_type": page_type, "source_url": url}
        for p in passages
        if len(p) > 80 and re.search(r"\d", p)  # keep passages that carry facts
    ]


# ---------------------------------------------------------------------------
# Scrape + store
# ---------------------------------------------------------------------------


def scrape(sources=None, fetch=None, delay_s: float | None = None, api_sources=None, fetch_json=None) -> dict:
    """Fetches and parses every source. `fetch(url) -> html` is injectable
    for tests. Returns {"offers", "passages", "pages", "errors", "scraped_at"}."""
    sources = sources or config.COMPETITOR_SOURCES
    delay_s = config.COMPETITOR_REQUEST_DELAY_S if delay_s is None else delay_s
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    client = None
    api_sources = config.COMPETITOR_API_SOURCES if api_sources is None else api_sources
    if fetch is None or fetch_json is None:
        client = httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": config.COMPETITOR_USER_AGENT})
        fetch = fetch or (lambda u: client.get(u).raise_for_status().text)  # noqa: E731
        fetch_json = fetch_json or (lambda u: client.get(u, headers={"Accept": "application/json"}).raise_for_status().json())  # noqa: E731
    offers, passages, pages, errors = [], [], [], []
    try:
        for operator, url in api_sources:
            try:
                found = parse_omantel_api(fetch_json(url), url)
            except Exception as e:  # the HTML pages still run
                errors.append({"url": url, "error": f"{type(e).__name__}: {e}"})
                continue
            offers += found
            pages.append({"url": url, "operator": operator, "offers": len(found), "passages": 0})
            if delay_s:
                time.sleep(delay_s)
        for n, (operator, page_type, url) in enumerate(sources):
            if n and delay_s:
                time.sleep(delay_s)
            try:
                html = fetch(url)
            except Exception as e:  # one bad page must not lose the others
                errors.append({"url": url, "error": f"{type(e).__name__}: {e}"})
                continue
            found = parse_vodafone_cards(html, url, page_type) if operator == "Vodafone Oman" else parse_omantel_data_tables(html, url, page_type)
            pp = page_passages(html, url, operator, page_type)
            offers += found
            passages += pp
            pages.append({"url": url, "operator": operator, "offers": len(found), "passages": len(pp)})
    finally:
        if client:
            client.close()
    # De-duplicate offers repeated across pages (e.g. a plan on two tabs).
    uniq = {}
    for o in offers:
        o["id"] = _offer_id(o["operator"], o["plan"], o["price_omr"], o["validity_days"])
        o["scraped_at"] = scraped_at
        uniq.setdefault(o["id"], o)
    for p in passages:
        p["scraped_at"] = scraped_at
    return {"offers": list(uniq.values()), "passages": passages, "pages": pages, "errors": errors, "scraped_at": scraped_at}


def offer_text(o: dict) -> str:
    bits = [f"{o['operator']} {o['plan']} ({o['category']})", f"{o['price_omr']} OMR"]
    if o.get("validity_days"):
        bits.append(f"for {o['validity_days']} days")
    if o.get("data_gb"):
        bits.append(f"{o['data_gb']} GB data")
    if o.get("minutes"):
        bits.append(o["minutes"])
    if o.get("features"):
        bits.append("; ".join(o["features"][:4]))
    return ", ".join(bits) + "."


def _collection():
    import chromadb
    from nbx.vectorstores.base import sanitize_embedding_key

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    name = f"competitor_offers__{sanitize_embedding_key(config.DEFAULT_EMBEDDING_MODEL)}"
    return client, name


def save_and_index(result: dict) -> dict:
    """Writes the JSON snapshots and rebuilds the competitor collection."""
    from nbx.llms.embeddings import embed_texts

    config.COMPETITOR_DIR.mkdir(parents=True, exist_ok=True)
    config.COMPETITOR_OFFERS_FILE.write_text(json.dumps(result["offers"], indent=2, ensure_ascii=False), encoding="utf-8")
    config.COMPETITOR_PASSAGES_FILE.write_text(json.dumps(result["passages"], indent=2, ensure_ascii=False), encoding="utf-8")

    docs = [{"id": f"offer-{o['id']}", "text": offer_text(o), "meta": {"kind": "offer", "operator": o["operator"], "category": o["category"], "source_url": o["source_url"]}}
            for o in result["offers"]]
    docs += [{"id": f"passage-{hashlib.sha1(p['text'].encode()).hexdigest()[:12]}", "text": p["text"],
              "meta": {"kind": "passage", "operator": p["operator"], "category": p["page_type"], "source_url": p["source_url"]}}
             for p in result["passages"]]
    docs = list({d["id"]: d for d in docs}.values())

    client, name = _collection()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    if docs:
        emb = embed_texts([d["text"] for d in docs], config.DEFAULT_EMBEDDING_MODEL)
        col.upsert(ids=[d["id"] for d in docs], embeddings=emb.tolist(), documents=[d["text"] for d in docs],
                   metadatas=[{**d["meta"], "scraped_at": result["scraped_at"]} for d in docs])
    return {"offers": len(result["offers"]), "passages": len(result["passages"]), "indexed": len(docs),
            "pages": result["pages"], "errors": result["errors"], "scraped_at": result["scraped_at"]}


def refresh() -> dict:
    return save_and_index(scrape())


def index_snapshot() -> dict:
    """Rebuilds the competitor collection from the JSON snapshot already on
    disk, without scraping -- for a fresh checkout or an offline machine."""
    offers = load_offers()
    passages = (json.loads(config.COMPETITOR_PASSAGES_FILE.read_text(encoding="utf-8"))
                if config.COMPETITOR_PASSAGES_FILE.exists() else [])
    scraped_at = next((r["scraped_at"] for r in offers + passages if r.get("scraped_at")), "")
    return save_and_index({"offers": offers, "passages": passages, "pages": [], "errors": [], "scraped_at": scraped_at})


# ---------------------------------------------------------------------------
# Read side (used by the agents' competitor_benchmark tool and the APIs)
# ---------------------------------------------------------------------------


def load_offers() -> list[dict]:
    p = config.COMPETITOR_OFFERS_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def search(query: str, top_k: int = 4, operator: str | None = None) -> list[dict]:
    from nbx.llms.embeddings import embed_texts

    try:
        client, name = _collection()
        col = client.get_collection(name)
    except Exception:
        return []
    if col.count() == 0:
        return []
    emb = embed_texts([query], config.DEFAULT_EMBEDDING_MODEL)[0]
    res = col.query(query_embeddings=[emb.tolist()], n_results=min(top_k, col.count()),
                    where={"operator": operator} if operator else None)
    out = []
    for i, doc_id in enumerate(res["ids"][0]):
        meta = res["metadatas"][0][i]
        out.append({"id": doc_id, "text": res["documents"][0][i], "score": round(1 - res["distances"][0][i], 3),
                    "operator": meta.get("operator"), "kind": meta.get("kind"), "source_url": meta.get("source_url")})
    return out


def _gb(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return parse_data_gb(str(value))


def validity_band(days: int | None) -> str:
    """Customers compare like with like: a daily pack against daily packs."""
    d = days or 30
    return "daily" if d <= 3 else "weekly" if d <= 14 else "monthly"


def compare_offer(our: dict, competitors: list[dict] | None = None, per_operator: int = 1, segment: str = "mass") -> dict:
    """Numeric comparison of one of our catalog offers against the closest
    competitor plan of each operator: same broad category, then nearest
    monthly-equivalent price. Returns verdicts the agents can quote."""
    competitors = competitors if competitors is not None else load_offers()
    our_monthly = round(our["price_omr"] * 30 / max(our["validity_days"], 1), 2)
    our_gb = _gb(our.get("data_gb"))
    our_unlimited = str(our.get("data_gb", "")).lower() == "unlimited"
    our_per_gb = round(our["price_omr"] / our_gb, 3) if our_gb else None
    want = {"Data": {"Data", "Bundle"}, "Bundle": {"Bundle", "Data"}, "NBO": {"Bundle", "Data"}, "Voice": {"Voice", "Bundle"},
            "International": {"International", "Bundle"}, "Roaming": {"Roaming"}}.get(our["category"], {"Bundle", "Data"})
    band = validity_band(our["validity_days"])
    rows = []
    for operator in sorted({c["operator"] for c in competitors}):
        pool = [c for c in competitors if c["operator"] == operator and c["category"] in want and c.get("monthly_equiv_omr")
                and c.get("segment", "mass") == segment]
        # Only like-for-like validity: a daily pack scaled to a "monthly price"
        # says nothing about what a customer would actually switch to.
        pool = [c for c in pool if validity_band(c.get("validity_days")) == band]
        pool.sort(key=lambda c: abs(c["monthly_equiv_omr"] - our_monthly))
        for c in pool[:per_operator]:
            verdicts = []
            delta = round(our_monthly - c["monthly_equiv_omr"], 2)
            verdicts.append("cheaper per month" if delta < -0.25 else "more expensive per month" if delta > 0.25 else "same monthly price")
            if our_unlimited and c.get("data_gb"):
                verdicts.append("more data (unlimited)")
            elif our_gb and c.get("data_gb"):
                our_gb_month = our_gb * 30 / max(our["validity_days"], 1)
                their_gb_month = c["data_gb"] * 30 / max(c["validity_days"] or 30, 1)
                verdicts.append("more data per month" if our_gb_month > their_gb_month * 1.05 else
                                "less data per month" if our_gb_month < their_gb_month * 0.95 else "similar data")
            rows.append({
                "same_validity_band": validity_band(c.get("validity_days")) == band,
                "operator": operator, "plan": c["plan"], "price_omr": c["price_omr"], "validity_days": c["validity_days"],
                "data_gb": c.get("data_gb"), "monthly_equiv_omr": c["monthly_equiv_omr"], "price_per_gb_omr": c.get("price_per_gb_omr"),
                "monthly_delta_omr": delta, "verdict": ", ".join(verdicts), "source_url": c["source_url"], "scraped_at": c.get("scraped_at"),
            })
    if not rows:
        position = f"no comparable {band} competitor plan in the snapshot"
    elif all("cheaper" in r["verdict"] or "more data" in r["verdict"] for r in rows):
        position = "strong: beats every compared competitor plan on price or data"
    elif any("more expensive" in r["verdict"] and "less data" in r["verdict"] for r in rows):
        position = "weak: a competitor offers more data for less money"
    else:
        position = "mixed"
    return {"our_monthly_equiv_omr": our_monthly, "our_price_per_gb_omr": our_per_gb, "position": position, "closest": rows}
