"""Ooredoo-style theme, NBX logo, landing page and HTML renderers for the
Gradio app (gradio_app.py). Pure presentation: everything rendered here
comes from the backend's decision / event JSON.
"""
from __future__ import annotations

import html

import gradio as gr

RED, RED_DIM, RED_SOFT = "#D3163B", "#8A0F28", "#FF5470"

# Original NBX mark (hexagonal AI core + ascending signal bars) -- the same
# mark as the Next.js Ooredoo edition, not Ooredoo's own logotype.
LOGO_SVG = """
<svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="NBX" class="nbx-mark">
  <defs>
    <linearGradient id="nbxg" x1="4" y1="36" x2="36" y2="4" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#8A0F28"/><stop offset="55%" stop-color="#D3163B"/><stop offset="100%" stop-color="#FF5470"/>
    </linearGradient>
    <radialGradient id="nbxglow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#D3163B" stop-opacity="0.85"/><stop offset="100%" stop-color="#D3163B" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <path d="M20 2.5L35.5 11.25V28.75L20 37.5L4.5 28.75V11.25L20 2.5Z" fill="#0A0A0C" stroke="url(#nbxg)" stroke-width="1.4"/>
  <circle cx="20" cy="20" r="12" fill="url(#nbxglow)" opacity="0.5"/>
  <rect x="13" y="21" width="3.2" height="7" rx="1" fill="#5A5A60"/>
  <rect x="18.4" y="16.5" width="3.2" height="11.5" rx="1" fill="url(#nbxg)"/>
  <rect x="23.8" y="11.5" width="3.2" height="16.5" rx="1" fill="url(#nbxg)"/>
  <circle cx="25.4" cy="8.4" r="2.1" fill="url(#nbxg)"/>
</svg>"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#fff1f3", c100="#ffe0e5", c200="#ffc2cc", c300="#ff94a6", c400="#ff5470",
        c500="#d3163b", c600="#b8102f", c700="#8a0f28", c800="#6b0c20", c900="#4a0816", c950="#2b040c",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f5f5f7", c100="#e4e4e8", c200="#b9bbc1", c300="#9c9ca3", c400="#6b6b72",
        c500="#4a4a50", c600="#2a2a2e", c700="#1e1e22", c800="#17171a", c900="#131315", c950="#0a0a0c",
    ),
    font=[gr.themes.GoogleFont("Manrope"), gr.themes.Font("system-ui"), gr.themes.Font("sans-serif")],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#0a0a0c", body_background_fill_dark="#0a0a0c",
    body_text_color="#f5f5f7", body_text_color_dark="#f5f5f7",
    body_text_color_subdued="#9c9ca3", body_text_color_subdued_dark="#9c9ca3",
    background_fill_primary="#131315", background_fill_primary_dark="#131315",
    background_fill_secondary="#17171a", background_fill_secondary_dark="#17171a",
    block_background_fill="#17171a", block_background_fill_dark="#17171a",
    block_border_color="#2a2a2e", block_border_color_dark="#2a2a2e",
    block_label_text_color="#9c9ca3", block_label_text_color_dark="#9c9ca3",
    block_title_text_color="#f5f5f7", block_title_text_color_dark="#f5f5f7",
    border_color_primary="#2a2a2e", border_color_primary_dark="#2a2a2e",
    input_background_fill="#131315", input_background_fill_dark="#131315",
    input_border_color="#2a2a2e", input_border_color_dark="#2a2a2e",
    button_primary_background_fill=f"linear-gradient(135deg, {RED}, {RED_DIM})",
    button_primary_background_fill_dark=f"linear-gradient(135deg, {RED}, {RED_DIM})",
    button_primary_background_fill_hover=RED_SOFT, button_primary_background_fill_hover_dark=RED_SOFT,
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#1e1e22", button_secondary_background_fill_dark="#1e1e22",
    button_secondary_text_color="#f5f5f7", button_secondary_text_color_dark="#f5f5f7",
    table_even_background_fill="#17171a", table_even_background_fill_dark="#17171a",
    table_odd_background_fill="#131315", table_odd_background_fill_dark="#131315",
    color_accent_soft="#2b040c", color_accent_soft_dark="#2b040c",
)

# Gradio switches to its dark palette when <body> has .dark; force it so the
# Ooredoo graphite look doesn't depend on the viewer's OS setting.
FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"

CSS = f"""
.gradio-container {{ max-width: 1440px !important; }}
footer {{ display: none !important; }}
.nbx-mark {{ width: 44px; height: 44px; }}
.nbx-header {{ display:flex; align-items:center; gap:14px; padding: 6px 4px 2px; }}
.nbx-header .word {{ font-weight: 800; font-size: 22px; letter-spacing: .5px; color:#f5f5f7; }}
.nbx-header .tag {{ font-size: 10px; letter-spacing: .22em; text-transform: uppercase; color:#9c9ca3; }}
.nbx-header .status {{ margin-left:auto; font-size:12px; color:#9c9ca3; }}
.nbx-hero {{ position:relative; overflow:hidden; border-radius:22px; padding:56px 44px; border:1px solid #2a2a2e;
            background: radial-gradient(1000px 400px at 85% -10%, rgba(211,22,59,.35), transparent 60%),
                        radial-gradient(700px 300px at 0% 120%, rgba(138,15,40,.35), transparent 60%), #0f0f12; }}
.nbx-hero .mark .nbx-mark {{ width: 88px; height: 88px; filter: drop-shadow(0 0 28px rgba(211,22,59,.55)); }}
.nbx-hero .eyebrow {{ color:{RED_SOFT}; font-size:12px; font-weight:700; letter-spacing:.22em; text-transform:uppercase; margin-top:18px; }}
.nbx-hero h1 {{ font-size: 64px; line-height:1; margin: 10px 0 6px; font-weight: 800;
               background: linear-gradient(135deg, #fff 30%, {RED_SOFT}); -webkit-background-clip:text; color: transparent; }}
.nbx-hero h2 {{ font-size: 22px; font-weight: 600; color:#f5f5f7; margin: 0 0 10px; }}
.nbx-hero p {{ max-width: 720px; color:#b9bbc1; font-size:16px; line-height:1.6; }}
.nbx-cards {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:14px; margin-top: 18px; }}
.nbx-card {{ border:1px solid #2a2a2e; background:#17171a; border-radius:18px; padding:20px; }}
.nbx-card .icon {{ width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center;
                  background: rgba(211,22,59,.14); color:{RED_SOFT}; font-size:18px; font-weight:800; }}
.nbx-card h3 {{ margin: 12px 0 6px; font-size: 16px; color:#f5f5f7; }}
.nbx-card p {{ margin:0; color:#9c9ca3; font-size: 13px; line-height:1.55; }}
.nbx-kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin-top:18px; }}
.nbx-kpi {{ border:1px solid #2a2a2e; background:#131315; border-radius:16px; padding:16px 18px; }}
.nbx-kpi .v {{ font-size: 26px; font-weight: 800; color:#f5f5f7; }}
.nbx-kpi .l {{ font-size: 12px; color:#9c9ca3; margin-top: 2px; }}
.nbx-kpi .v.red {{ color:{RED_SOFT}; }}
.nbx-section {{ border:1px solid #2a2a2e; background:#17171a; border-radius:18px; padding:18px 20px; }}
.nbx-section h3 {{ margin:0 0 10px; font-size:15px; color:#f5f5f7; }}
.nbx-muted {{ color:#9c9ca3; font-size:13px; }}
.nbx-badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:11px; font-weight:600; border:1px solid #3a3a40; color:#b9bbc1; margin-right:6px; }}
.nbx-badge.red {{ border-color: rgba(211,22,59,.45); color:{RED_SOFT}; background: rgba(211,22,59,.12); }}
.nbx-badge.green {{ border-color: rgba(47,191,113,.4); color:#6ee7a8; background: rgba(47,191,113,.1); }}
.nbx-badge.amber {{ border-color: rgba(224,165,48,.4); color:#f5c96a; background: rgba(224,165,48,.1); }}
.nbx-action {{ border:1px solid rgba(211,22,59,.45); border-radius:18px; padding:20px 22px;
              background: radial-gradient(600px 200px at 100% 0%, rgba(211,22,59,.25), transparent 70%), #17171a; }}
.nbx-action .label {{ color:{RED_SOFT}; font-size:11px; font-weight:700; letter-spacing:.2em; text-transform:uppercase; }}
.nbx-action .name {{ font-size: 30px; font-weight: 800; margin: 4px 0 6px; color:#fff; }}
.nbx-pipe {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:10px; }}
.nbx-pipe .lane {{ border:1px solid #2a2a2e; border-radius:14px; padding:12px 14px; background:#131315; }}
.nbx-pipe .lane.running {{ border-color: rgba(211,22,59,.6); box-shadow: 0 0 22px -6px rgba(211,22,59,.6); }}
.nbx-pipe .lane.waiting {{ opacity:.55; }}
.nbx-pipe .lane b {{ color:#f5f5f7; font-size:13px; }}
.nbx-pipe .lane div {{ color:#9c9ca3; font-size:11px; margin-top:4px; }}
.nbx-trace {{ max-height: 560px; overflow-y:auto; }}
.nbx-ev {{ display:flex; gap:10px; padding:8px 0; border-bottom:1px solid #1f1f23; }}
.nbx-ev .who {{ min-width: 112px; font-size:11px; font-weight:700; color:#b9bbc1; text-transform:uppercase; letter-spacing:.06em; }}
.nbx-ev .what {{ font-size:13px; color:#e4e4e8; line-height:1.5; word-break: break-word; }}
.nbx-ev .what code {{ color:#f5c96a; background:transparent; font-size:12px; }}
.nbx-ev.policy .what {{ color:#f5c96a; }} .nbx-ev.fix .what {{ color:{RED_SOFT}; }} .nbx-ev.final .what {{ color:#6ee7a8; }}
.nbx-table {{ width:100%; border-collapse: collapse; font-size:13px; }}
.nbx-table th {{ text-align:left; color:#9c9ca3; font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.06em; padding:8px 10px; border-bottom:1px solid #2a2a2e; }}
.nbx-table td {{ padding:9px 10px; border-bottom:1px solid #1f1f23; color:#e4e4e8; vertical-align: top; }}
.nbx-why li {{ margin: 6px 0; color:#e4e4e8; font-size:13.5px; line-height:1.55; }}
.nbx-why .kind {{ display:inline-block; min-width: 96px; color:{RED_SOFT}; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
.nbx-narrative {{ font-size:15px; line-height:1.7; color:#f5f5f7; border-left: 0; background: rgba(211,22,59,.07);
                 border:1px solid rgba(211,22,59,.25); border-radius:14px; padding:14px 16px; }}
.nbx-sms {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:10px; }}
.nbx-sms div {{ border:1px solid #2a2a2e; background:#131315; border-radius:14px; padding:12px 14px; font-size:14px; line-height:1.6; }}
.nbx-sms .rtl {{ direction: rtl; text-align: right; }}
.nbx-sms .h {{ font-size:11px; color:#9c9ca3; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; direction:ltr; text-align:left; }}
"""


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def header_html(status: str = "") -> str:
    return (f'<div class="nbx-header">{LOGO_SVG}<div><div class="word">NBX</div>'
            f'<div class="tag">Next Best Experience · powered for Ooredoo Oman</div></div>'
            f'<div class="status">{esc(status)}</div></div>')


FEATURES = [
    ("A", "Agentic Best Action Recommender", "A strategist agent investigates churn, usage trends, the contextual bandit and contact policy, then chooses one action and explains why it agrees with or overrules the bandit."),
    ("O", "Agentic Offer Recommender", "An offer specialist searches the catalog, collaborative filtering and the cohort graph, checks eligibility, and ranks offers by simulated expected value."),
    ("C", "Competitor Intelligence", "Omantel and Vodafone Oman public plans, scraped into a vector store. Every offer is benchmarked on monthly price and data before it is recommended."),
    ("G", "Guardrails + Critic", "Hard business rules are enforced in code; a critic agent reviews every decision and can send it back once. Nothing unsafe ships."),
]


def landing_html(summary: dict | None, competitor_count: int, llm: str) -> str:
    cards = "".join(f'<div class="nbx-card"><div class="icon">{a}</div><h3>{esc(t)}</h3><p>{esc(d)}</p></div>' for a, t, d in FEATURES)
    kpis = ""
    if summary:
        items = [
            (summary["customers"], "subscribers analysed", ""),
            (summary["high_risk"], "high churn risk", "red"),
            (f"{summary['revenue_at_risk']:.0f} OMR", "monthly revenue at risk", "red"),
            (f"{summary['avg_arpu']:.2f} OMR", "average ARPU", ""),
            (competitor_count, "competitor plans tracked", ""),
        ]
        kpis = '<div class="nbx-kpis">' + "".join(
            f'<div class="nbx-kpi"><div class="v {c}">{esc(v)}</div><div class="l">{esc(l)}</div></div>' for v, l, c in items) + "</div>"
    return f"""
<div class="nbx-hero">
  <div class="mark">{LOGO_SVG}</div>
  <div class="eyebrow">Agentic AI decisioning · Ooredoo Oman</div>
  <h1>NBX</h1>
  <h2>Next Best Experience</h2>
  <p>AI agents that decide the next best action and the next best offer for every subscriber, reason with real
  tools and competitor prices, check themselves against business guardrails, and explain every decision in plain language.</p>
  <div><span class="nbx-badge red">Local LLM: {esc(llm)}</span><span class="nbx-badge">Live data</span><span class="nbx-badge">Arabic + English</span></div>
  {kpis}
</div>
<div class="nbx-cards">{cards}</div>"""


# ---------------------------------------------------------------------------
# Agentic run rendering
# ---------------------------------------------------------------------------
AGENT_LABEL = {"orchestrator": "Orchestrator", "nba": "Step 1 · Best Action", "nbo": "Step 2 · Offer", "critic": "Critic", "messenger": "Messenger"}
AGENT_ROLE = {"orchestrator": "bandit + risk policy", "nba": "agent chooses the action", "nbo": "agent picks offers for that action",
              "critic": "guardrails + review", "messenger": "SMS + explanation"}


def render_pipeline(events: list[dict], running: bool) -> str:
    start = next((e for e in events if e["type"] == "run_start"), None)
    if not start:
        return ""
    state = {a: "waiting" for a in start["agents"]}
    tools = {a: 0 for a in start["agents"]}
    for e in events:
        a = e.get("agent")
        if a not in state:
            continue
        if e["type"] == "agent_start":
            state[a] = "running"
        elif e["type"] == "tool_call":
            tools[a] += 1
        elif e["type"] == "agent_final":
            state[a] = "done"
    lanes = []
    for a in start["agents"]:
        s = state[a] if running or state[a] != "running" else "done"
        status = {"waiting": "waiting", "running": "reasoning…", "done": "done"}[s]
        lanes.append(f'<div class="lane {s}"><b>{AGENT_LABEL.get(a, a)}</b><div>{AGENT_ROLE.get(a, "")}</div>'
                     f'<div>{status}{f" · {tools[a]} tool calls" if tools[a] else ""}</div></div>')
    mode = "LLM · " + esc(start.get("model")) if start.get("mode") == "agentic" else "deterministic fallback"
    return f'<div class="nbx-muted" style="margin-bottom:8px">Mode: {mode}</div><div class="nbx-pipe">{"".join(lanes)}</div>'


def _ev_line(e: dict) -> tuple[str, str] | None:
    t = e["type"]
    if t == "thought":
        return "", esc(e.get("text"))
    if t == "tool_call":
        args = html.escape(str(e.get("args") or {}))
        return "", f"calls <code>{esc(e.get('tool'))}</code> {args[:160]}"
    if t == "tool_result":
        r = str(e.get("result"))
        return "", f'<span class="nbx-muted">→ {esc(r[:220])}{"…" if len(r) > 220 else ""}</span>'
    if t == "policy":
        return "policy", "⚑ " + esc(e.get("text"))
    if t == "revision":
        return "policy", "↺ " + esc(e.get("text"))
    if t == "auto_fix":
        return "fix", "✎ " + esc(e.get("text"))
    if t == "guardrails":
        f = e.get("findings") or []
        return ("policy" if f else "final"), ("Guardrails: " + "; ".join(esc(x["code"]) for x in f) if f else "All hard guardrails passed.")
    if t == "agent_final":
        o = e.get("output") or {}
        if "action" in o:
            return "final", f"Decision: <b>{esc(o['action'])}</b> — {esc(o.get('rationale', ''))[:260]}"
        if "offers" in o:
            return "final", "Offers: " + ", ".join(esc(x["offer"]) for x in o["offers"])
        if "verdict" in o:
            return "final", f"Verdict: {esc(o['verdict'])}{' (revised once)' if o.get('revised') else ''}"
        return "final", "Message and explanation drafted."
    return None


def render_trace(events: list[dict]) -> str:
    rows = []
    for e in events:
        line = _ev_line(e)
        if line:
            cls, txt = line
            rows.append(f'<div class="nbx-ev {cls}"><div class="who">{AGENT_LABEL.get(e.get("agent"), "")}</div><div class="what">{txt}</div></div>')
    return f'<div class="nbx-section"><h3>Agent reasoning, live</h3><div class="nbx-trace">{"".join(rows)}</div></div>'


def render_action(d: dict) -> str:
    risk_cls = {"HIGH": "red", "MEDIUM": "amber", "LOW": "green"}.get(d["risk_level"], "")
    return f"""<div class="nbx-action"><div class="label">Recommended action</div><div class="name">{esc(d['action'])}</div>
<div class="nbx-muted" style="font-size:14px;color:#d4d4d8">{esc(d['nba_rationale'])}</div>
<div style="margin-top:10px"><span class="nbx-badge {risk_cls}">{esc(d['risk_level'].title())} risk</span>
<span class="nbx-badge">Channel: {esc(d['channel'])}</span><span class="nbx-badge">Send: {esc(d['send_window'])}</span>
<span class="nbx-badge">Confidence: {round((d.get('confidence') or 0) * 100)}%</span>
<span class="nbx-badge {'red' if d['is_push_action'] else ''}">{'Sales push' if d['is_push_action'] else 'Non-push'}</span>
<span class="nbx-badge {'green' if d['critic']['verdict'] == 'approve' and not d['critic']['revised'] else 'amber'}">Critic: {esc(d['critic']['verdict'])}{' · revised' if d['critic']['revised'] else ''}</span></div></div>"""


def render_offers(d: dict) -> str:
    if not d["offers"]:
        return '<div class="nbx-section"><h3>Offers</h3><div class="nbx-muted">No offer for this decision.</div></div>'
    comp = {o["offer"]: o["competition"] for o in d.get("explanation", {}).get("offers", [])}
    rows = "".join(
        f"<tr><td><b>{i}</b></td><td><b>{esc(o['offer'])}</b><div class='nbx-muted'>{esc(o['category'])} · {esc(o['data_gb'])} data · "
        f"{esc(o['voice_mins'])} min · {o['validity_days']} days</div></td><td>{o['price_omr']} OMR</td>"
        f"<td>{round((o['p_accept'] or 0) * 100)}%</td><td>{o['expected_value_omr']} OMR/mo</td>"
        f"<td>{esc(comp.get(o['offer'], {}).get('position', '-'))}</td></tr>"
        for i, o in enumerate(d["offers"], 1)
    )
    return (f'<div class="nbx-section"><h3>Recommended offers, ranked by expected value</h3><table class="nbx-table">'
            f"<tr><th>#</th><th>Offer</th><th>Price</th><th>P(accept)</th><th>Expected value</th><th>Vs competitors</th></tr>{rows}</table>"
            f'<div class="nbx-muted" style="margin-top:8px">{esc(d["nbo_rationale"])}</div></div>')


def render_why(d: dict) -> str:
    e = d.get("explanation")
    if not e:
        return ""
    a = e["action"]
    drivers = "".join(f"<tr><td>{esc(x['signal'])}</td><td><b>{esc(x['value'])}</b></td><td>{esc(x['effect'])}</td></tr>" for x in a["drivers"])
    rejected = "".join(f"<li>{esc(r)}</li>" for r in a["alternatives_rejected"]) or "<li>No alternative was close.</li>"
    parts = [
        f'<div class="nbx-section"><h3>Why this decision</h3><div class="nbx-narrative">{esc(e["narrative"])}</div></div>',
        f'<div class="nbx-section"><h3>Why {esc(d["action"])}</h3>'
        f'<table class="nbx-table"><tr><th>Signal</th><th>Value</th><th>Effect on the decision</th></tr>{drivers}</table>'
        f'<ul class="nbx-why"><li><span class="kind">Bandit</span>{esc(a["bandit_comparison"])}</li>'
        f'<li><span class="kind">Channel</span>{esc(a["channel_and_timing"])}</li>'
        + (f'<li><span class="kind">Language</span>{esc(a["language"])}</li>' if a.get("language") else "")
        + (f'<li><span class="kind">Market</span>{esc(a["competitive_pressure"])}</li>' if a.get("competitive_pressure") else "")
        + f'</ul><div class="nbx-muted" style="margin-top:6px"><b>Alternatives rejected</b></div><ul class="nbx-why">{rejected}</ul></div>',
    ]
    for o in e["offers"]:
        reasons = "".join(f'<li><span class="kind">{esc(r["kind"])}</span>{esc(r["text"])}</li>' for r in o["reasons"])
        watch = "".join(f"<li>{esc(w)}</li>" for w in o["watch_outs"])
        comp_rows = "".join(
            f"<tr><td>{esc(c['operator'])}</td><td>{esc(c['plan'])}</td><td>{c['price_omr']} OMR / {c['validity_days']} d</td>"
            f"<td>{esc(c['data_gb'] if c['data_gb'] is not None else 'unlimited')} GB</td><td>{c['monthly_equiv_omr']} OMR</td><td><b>{esc(c['verdict'])}</b></td></tr>"
            for c in o["competition"]["closest"])
        parts.append(
            f'<div class="nbx-section"><h3>Why {esc(o["offer"])}</h3><div class="nbx-muted">{esc(o["headline"])}</div>'
            f'<ul class="nbx-why">{reasons}</ul>'
            + (f'<div class="nbx-muted"><b>Watch-outs</b></div><ul class="nbx-why">{watch}</ul>' if watch else "")
            + (f'<div class="nbx-muted" style="margin:8px 0 4px"><b>Closest competitor plans</b> · ours: '
               f'{o["competition"]["our_monthly_equiv_omr"]} OMR/month</div><table class="nbx-table"><tr><th>Operator</th><th>Plan</th>'
               f'<th>Price</th><th>Data</th><th>Monthly equiv.</th><th>Ours is</th></tr>{comp_rows}</table>' if comp_rows else "")
            + "</div>")
    parts.append(f'<div class="nbx-muted">Evidence used: {esc(", ".join(e["data_sources"]))}.</div>')
    return "".join(parts)


def render_message(d: dict) -> str:
    m = d["message"]
    if not (m.get("sms_en") or m.get("agent_script")):
        return ""
    langs = " + ".join(m.get("languages") or ["Arabic", "English"])
    head = (f'<h3>Personalized outreach</h3><div class="nbx-muted" style="margin:-4px 0 10px">{esc(m.get("nationality_segment", ""))} customer '
            f'· languages: {esc(langs)} · lead: <b>{esc(m.get("lead_language", ""))}</b></div>')
    return (f'<div class="nbx-section">{head}<div class="nbx-sms">'
            f'<div><div class="h">SMS · English</div>{esc(m["sms_en"])}</div>'
            f'<div class="rtl" lang="ar"><div class="h">SMS · العربية (Arabic)</div>{esc(m["sms_ar"])}</div>'
            + (f'<div class="{"rtl" if m.get("native_rtl") else ""}" lang="{esc((m.get("native_language") or "")[:2].lower())}">'
               f'<div class="h">SMS · {esc(m["native_language"])}</div>{esc(m["sms_native"])}</div>' if m.get("sms_native") else "")
            + f'<div><div class="h">Call-center script</div>{esc(m["agent_script"])}</div></div></div>')


def render_stats(d: dict) -> str:
    s = d["stats"]
    return (f'<div class="nbx-muted">{s["llm_calls"]} LLM calls · {s["tool_calls"]} tool calls · {s["elapsed_ms"] / 1000:.1f}s · '
            f'mode {esc(d["mode"])}{" · " + esc(d["model"]) if d.get("model") else ""}</div>')


# ---------------------------------------------------------------------------
# Customer card + batch results
# ---------------------------------------------------------------------------
def pretty(s) -> str:
    return str(s or "").replace("_", " ").lower()


def describe(c: dict) -> str:
    """One-line description: '#15 · High Value · high risk (58% churn) · international'."""
    return (f"#{c['msisdn']} · {c['value_segment']} · {pretty(c['risk_level'])} risk ({round(c['churn'] * 100)}% churn) · "
            f"{pretty(c['need_signal'])} · {c.get('nationality_segment', c.get('nationality', ''))}")


def render_profile(c: dict) -> str:
    """Full customer card from GET /customers/{msisdn}/360."""
    risk_cls = {"HIGH": "red", "MEDIUM": "amber", "LOW": "green"}.get(c["risk_level"], "")
    p = c.get("profile", {})
    rev = [r["omr"] for r in c.get("revenue_monthly", []) if r.get("omr") is not None]
    trend = f"{rev[0]:.2f} → {rev[-1]:.2f} OMR" if len(rev) >= 2 else "n/a"
    facts = [
        ("ARPU", f"{c['arpu']:.2f} OMR"), ("Avg monthly revenue", f"{c['avg_monthly_revenue']:.2f} OMR"),
        ("Revenue (3 months)", trend), ("Tenure", f"{c['tenure_months']} months"),
        ("Device segment", c["device_segment"]), ("Usage", pretty(c["usage_type"])),
        ("Data tier", pretty(p.get("data_usage_tier"))), ("Activity", pretty(c["activity_level"])),
        ("Preferred channel", pretty(c["preferred_channel"])), ("Favourite category", c["favorite_category"]),
        ("Last product", p.get("last_product_name", "-")), ("Days since purchase", c["days_since_last_purchase"]),
        ("Days inactive", c["days_inactive"]), ("Network", c["network"]), ("Nationality segment", f"{c.get('nationality_segment', '-')} ({c['nationality']})"),
        ("Message languages", " + ".join(c.get("message_languages", []))), ("Lead language", c.get("lead_language", "-")),
        ("International caller", "yes" if p.get("is_intl_caller") else "no"), ("Roamer", "yes" if p.get("is_roamer") else "no"),
        ("NBA segment", pretty(c.get("nba_segment")) or "-"), ("Segment policy", c["policy_action"]),
    ]
    grid = "".join(f'<div class="nbx-kpi"><div class="l">{esc(k)}</div><div style="font-weight:700;color:#f5f5f7;margin-top:2px">'
                   f"{esc(v)}</div></div>" for k, v in facts)
    scores = "".join(
        f'<span class="nbx-badge {"red" if v >= 0.5 else "amber" if v >= 0.3 else ""}">{esc(k.replace("proba_", "").replace("_", " "))}: '
        f"{round(v * 100)}%</span>" for k, v in sorted(c.get("scores", {}).items(), key=lambda kv: -kv[1]))
    buys = ", ".join(f"{b['program_rule']} ({b['purchase_date'][:10]})" for b in c.get("purchases", [])[:4]) or "none"
    return (f'<div class="nbx-section"><div class="label" style="color:#ff5470;font-size:11px;font-weight:700;letter-spacing:.2em">CUSTOMER</div>'
            f'<div style="font-size:22px;font-weight:800;color:#fff;margin:4px 0 8px">{esc(describe(c))}</div>'
            f'<div><span class="nbx-badge {risk_cls}">{esc(c["risk_level"].title())} risk</span>'
            f'<span class="nbx-badge">revenue-decrease {round(c["revenue_decrease"] * 100)}%</span>'
            f'<span class="nbx-badge">data appetency {round(c["data_appetency"] * 100)}%</span></div>'
            f'<div class="nbx-kpis" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr))">{grid}</div>'
            f'<div class="nbx-muted" style="margin-top:12px"><b>Propensity scores:</b></div><div style="margin-top:6px">{scores}</div>'
            f'<div class="nbx-muted" style="margin-top:10px"><b>Recent purchases:</b> {esc(buys)}</div></div>')


def render_batch(rows: list[dict], scope: str) -> str:
    if not rows:
        return ""
    head = "<tr><th>MSISDN</th><th>Customer</th><th>Action</th><th>Channel</th>"
    head += "<th>Offers</th><th>Lead offer EV</th><th>Vs competitors</th>" if scope != "nba" else ""
    head += "<th>Critic</th><th>Why</th><th>Time</th></tr>"
    body = []
    for r in rows:
        if r.get("error"):
            body.append(f"<tr><td><b>{esc(r['msisdn'])}</b></td><td colspan='9' style='color:#ff5470'>{esc(r['error'])}</td></tr>")
            continue
        if r.get("pending"):
            body.append(f"<tr><td><b>{esc(r['msisdn'])}</b></td><td>{esc(r.get('description', ''))}</td>"
                        f"<td colspan='8' class='nbx-muted'>{esc(r['pending'])}</td></tr>")
            continue
        cells = [f"<td><b>{esc(r['msisdn'])}</b></td>", f"<td>{esc(r['description'])}</td>",
                 f"<td><span class='nbx-badge red'>{esc(r['action'])}</span></td>", f"<td>{esc(r['channel'])}</td>"]
        if scope != "nba":
            cells += [f"<td>{esc(', '.join(r['offers']) or '-')}</td>", f"<td>{esc(r['ev'])}</td>", f"<td>{esc(r['position'])}</td>"]
        cells += [f"<td>{esc(r['critic'])}</td>", f"<td style='min-width:320px'>{esc(r['why'])}</td>", f"<td>{esc(r['time'])}</td>"]
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="nbx-section" style="overflow-x:auto"><table class="nbx-table">{head}{"".join(body)}</table></div>'
