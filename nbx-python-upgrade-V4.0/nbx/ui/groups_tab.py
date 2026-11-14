"""FSDR Targets & Output tab (V4.0): the whole flow from the interface, each target scored separately.

1 generate / prepare (training) the run date  ->  2 upload (or generate) the target lists, check them
->  3 panel, then Target 1 and Target 2 scored separately, then the push of both  ->  4 download the files.
Every button calls the FastAPI /fsdr endpoints -- no business logic here.

Folders (docs/SETUP_NEW_LAPTOP.md §5):
    data/training/<D>/   data/scoring/<D>/   (5 folders each)   data/scoring/<D>/target1|2/a_constraint|b_free/, control/
    data/output/<D>/target1/  target2/  control/  push/
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import gradio as gr
import httpx

from nbx.ui import fsdr_tabs as ft

GROUP_LABELS = {"t1c": "Target 1 · a constraint (T1C, code 3)", "t1f": "Target 1 · b free (T1F, code 4)",
                "t2c": "Target 2 · a challenger constraint (T2C, CC-v1)", "t2f": "Target 2 · b challenger free (T2F, CF-v1)",
                "control": "Control (never contacted)"}
GROUP_COLS = ["group", "folder", "rules", "lines", "in_input_flytxt", "offers_per_line", "dropped_F04", "favourites",
              "forced_F08", "cleared_F09", "cleared_F11", "cleared_F12"]


def _run(d: str) -> str:
    d = (d or "").strip()
    try:
        date.fromisoformat(d)
    except ValueError:
        raise gr.Error(f"Run date must be YYYY-MM-DD, got '{d}'")
    return d


async def do_generate(d, n, force):
    r = await ft._post(f"/fsdr/v4/{_run(d)}/generate", {"n": int(n), "force": bool(force)})
    return ft._section("Dummy input generated (training + scoring)", ft._kv(r))


async def do_prepare(d, force):
    r = await ft._post(f"/fsdr/v4/{_run(d)}/prepare", force=str(bool(force)).lower())
    html = ft._section("T0 · validation of both folders and Parquet cache", ft._kv(r["t0"]))
    if r["features"]:
        f = r["features"]
        html += ft._section(f"Training T2 + T3 · {f['lines']:,} lines, {f['micro_segments']} micro-segments",
                            ft._table([{"segment": k, "lines": v} for k, v in f["segments"].items()], ["segment", "lines"]))
    return html


def _check_html(r: dict) -> str:
    files = ft._table([{"group": GROUP_LABELS[k], "file": v or "— missing —"} for k, v in r["files"].items()],
                      ["group", "file"])
    html = ft._section(f"Lists in {r['folder']}", files)
    c = r.get("check")
    if not c:
        return html + ft._section("Check", r.get("note") or "No list yet: upload one per group, or generate dummy lists.")
    head = (f"<b>{'BLOCKED' if c['blocking'] else 'OK'}</b> · {c['read']:,} lines read, "
            f"{c['dropped']:,} dropped · kept per group: {c['kept']}")
    issues = ft._table([{"level": i["level"], "check": i["check"], "message": i["message"], "count": i.get("count", 0)}
                        for i in c["issues"]], ["level", "check", "message", "count"]) if c["issues"] else "No issue."
    return html + ft._section("Check", head + issues)


async def do_upload(d, group, path):
    if not path:
        raise gr.Error("Choose the list file (.csv.gz, or .csv / .json for tests).")
    async with httpx.AsyncClient(timeout=ft.TIMEOUT, headers=ft.HEADERS) as c:
        r = await c.post(f"{ft.API_BASE}/fsdr/targets/{_run(d)}/{group}",
                         files={"file": (Path(path).name, Path(path).read_bytes(), "application/octet-stream")})
    if r.status_code >= 400:
        raise gr.Error(r.json().get("detail", r.text))
    return _check_html(r.json())


async def do_check(d):
    return _check_html(await ft._get(f"/fsdr/targets/{_run(d)}"))


async def do_dummy_targets(d):
    r = await ft._post(f"/fsdr/v4/{_run(d)}/targets/generate")
    return _check_html(await ft._get(f"/fsdr/targets/{_run(d)}")) + ft._section(
        "Dummy lists written", ft._kv({"folder": r["folder"], "sizes": r["sizes"], "scale_factor": r["scale_factor"]}))


def _target_html(f: dict) -> str:
    state = "READY" if f["push_ready"] else "BLOCKED — see BLOCKED.txt"
    html = ft._section(f"{f['target']} · {state} · input_flytxt_{f['target']} {f['input_flytxt_rows']:,} rows · "
                       f"up to {f['reco_offers']} offers · details {f['details_rows']:,} lines × {f['details_columns']} "
                       f"columns · group audit {f['p6_group_audit']}",
                       ft._table(f["output_checks"], ["id", "check", "status"]))
    return html + ft._section("Groups a / b", ft._table([{c: g.get(c, "") for c in GROUP_COLS} for g in f["per_group"]],
                                                        GROUP_COLS))


def _push_html(p: dict | None) -> str:
    if not p:
        return ft._section("Push", "Not built yet: score both targets, then 3d.")
    if p["push_ready"]:
        return ft._section(f"Push READY · input_flytxt {p['input_flytxt_rows']:,} rows",
                           ft._kv({"rows per target": p["rows_per_target"], "folder": p["step_dir"],
                                   "next": "human approval (H-08) before sending"}))
    return ft._section("Push BLOCKED", ft._kv({"reasons": "; ".join(p["reasons"]), "folder": p["step_dir"]}))


def _files(out_dir: str) -> list[str]:
    root = Path(out_dir)
    found = sorted(root.glob("push/*.csv.gz"))
    for t in ("target1", "target2"):
        found += sorted((root / t).glob("*.csv.gz")) + sorted((root / t).glob("*/*.csv.gz"))
        found += [root / t / "final_rules_report.csv"]
    found += sorted(root.glob("control/*.csv.gz"))
    return [str(p) for p in found if p.exists() and not p.name.endswith(".schema.json")]


async def do_panel(d):
    p = await ft._post(f"/fsdr/v4/{_run(d)}/panel")
    issues = [i for i in p.get("target_issues") or [] if i["level"] == "blocking"]
    return ft._section(f"P1 panel · {p['mode']} · {p['status']} · balance gap {p.get('max_balance_gap_points')} pts",
                       ft._kv(p["sizes"]) + ("".join(f"<br>BLOCKING {i['message']}" for i in issues)))


async def do_target(d, target):
    r = await ft._post(f"/fsdr/v4/{_run(d)}/score/{target}")
    o = r["offers"]
    html = ft._section(f"P6 {target} · group audit {o['audit']}",
                       ft._table(o["groups"], ["group", "rules", "contacted", "with_offers", "offers", "consistency_retry"]))
    return html + _target_html(r["final"])


async def do_target1(d):
    return await do_target(d, "target1")


async def do_target2(d):
    return await do_target(d, "target2")


async def do_push(d):
    r = await ft._post(f"/fsdr/v4/{_run(d)}/push")
    out = str(Path(r["step_dir"]).parent)
    return _push_html(r) + ft._section("Control", ft._kv(r["control"])), _files(out)


async def do_run_all(d):
    r = await ft._post(f"/fsdr/v4/{_run(d)}/run")
    p = r["panel"]
    html = ft._section(f"P1 panel · {p['mode']} · {p['status']}", ft._kv(p["sizes"]))
    html += "".join(_target_html(r[t]) for t in ("target1", "target2") if t in r)
    html += _push_html(r["push"]) + ft._section("Control", ft._kv(r["control"]))
    return html, _files(str(Path(r["push"]["step_dir"]).parent))


async def do_show(d):
    o = await ft._get(f"/fsdr/v4/{_run(d)}/output")
    html = "".join(_target_html(o[t]) for t in ("target1", "target2") if o.get(t))
    return html + _push_html(o.get("push")), _files(o["folder"])


def build() -> None:
    with gr.Tab("FSDR Targets & Output", id="fsdr-groups"):
        gr.HTML('<div class="nbx-muted">V4.0: <b>Target 1</b> = a constraint (T1C, code 3) + b free (T1F, code 4) · '
                '<b>Target 2</b> = a challenger constraint (T2C, CC-v1) + b challenger free (T2F, CF-v1) · '
                '<b>control</b>. Each target is scored on its own into <code>data/output/&lt;D&gt;/&lt;target&gt;/</code>, '
                'up to 6 offers per line. Decisions use the LLM agents\' fallbacks until Phases 5-6.</div>')
        run_date = gr.Textbox(label="Run date D (YYYY-MM-DD)", value=date.today().isoformat())
        with gr.Group():
            gr.Markdown("**1 · Data and training** — `data/training/<D>/` + `data/scoring/<D>/` (5 folders each, `.csv.gz`)")
            with gr.Row():
                gen_n = gr.Number(label="Dummy lines", value=2000, precision=0, scale=1)
                gen_force = gr.Checkbox(label="Replace existing data", value=False, scale=1)
                gen_btn = gr.Button("1a · Generate dummy data", scale=1)
                prep_force = gr.Checkbox(label="Re-run even if done", value=False, scale=1)
                prep_btn = gr.Button("1b · Training (validate + features + segments)", variant="primary", scale=1)
            out1 = gr.HTML()
        with gr.Group():
            gr.Markdown("**2 · Target lists** — `data/scoring/<D>/target1|target2/a_constraint|b_free/` and `control/`")
            with gr.Row():
                grp = gr.Dropdown([(v, k) for k, v in GROUP_LABELS.items()], value="t1c", label="Group", scale=2)
                up = gr.File(label="List (.csv.gz with access_method_val)", file_types=[".gz", ".csv", ".json", ".jsonl"],
                             scale=3)
            with gr.Row():
                up_btn = gr.Button("2a · Upload this list", variant="primary")
                dummy_btn = gr.Button("2b · Generate dummy lists (all groups)")
                check_btn = gr.Button("2c · Check the lists")
            out2 = gr.HTML()
        with gr.Group():
            gr.Markdown("**3 · Scoring, one target at a time** — `data/output/<D>/target1/`, `target2/`, `push/`")
            with gr.Row():
                panel_btn = gr.Button("3a · Panel (from the lists)")
                t1_btn = gr.Button("3b · Score Target 1", variant="primary")
                t2_btn = gr.Button("3c · Score Target 2", variant="primary")
                push_btn = gr.Button("3d · Build the push file")
            with gr.Row():
                all_btn = gr.Button("3a → 3d in one go")
                show_btn = gr.Button("Show the last output")
            out3 = gr.HTML()
            files = gr.File(label="4 · Download: push file, per-target and per-group files, details, rules reports, control",
                            file_count="multiple", interactive=False)
        gen_btn.click(do_generate, inputs=[run_date, gen_n, gen_force], outputs=out1)
        prep_btn.click(do_prepare, inputs=[run_date, prep_force], outputs=out1)
        up_btn.click(do_upload, inputs=[run_date, grp, up], outputs=out2)
        dummy_btn.click(do_dummy_targets, inputs=run_date, outputs=out2)
        check_btn.click(do_check, inputs=run_date, outputs=out2)
        panel_btn.click(do_panel, inputs=run_date, outputs=out3)
        t1_btn.click(do_target1, inputs=run_date, outputs=out3)
        t2_btn.click(do_target2, inputs=run_date, outputs=out3)
        push_btn.click(do_push, inputs=run_date, outputs=[out3, files])
        all_btn.click(do_run_all, inputs=run_date, outputs=[out3, files])
        show_btn.click(do_show, inputs=run_date, outputs=[out3, files])
