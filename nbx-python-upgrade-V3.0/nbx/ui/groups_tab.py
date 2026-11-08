"""FSDR Targets & Final tab (V3.0): the whole four-group flow from the interface.

1 generate / prepare the input of a run date  ->  2 upload (or generate) the target lists, check them
->  3 run panel + scoring per group + final files  ->  4 download the file to push, the per-target files, the details.
Every button calls the FastAPI /fsdr endpoints -- no business logic here.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import gradio as gr
import httpx

from nbx.ui import fsdr_tabs as ft

GROUP_LABELS = {"t1c": "Target 1 · constraint (T1C, code 3)", "t1f": "Target 1 · free (T1F, code 4)",
                "t2c": "Target 2 · challenger constraint (T2C, CC-v1)", "t2f": "Target 2 · challenger free (T2F, CF-v1)",
                "control": "Control (never contacted)"}


def _run(d: str) -> str:
    d = (d or "").strip()
    try:
        date.fromisoformat(d)
    except ValueError:
        raise gr.Error(f"Run date must be YYYY-MM-DD, got '{d}'")
    return d


async def do_generate(d, n, force):
    r = await ft._post(f"/fsdr/v3/{_run(d)}/generate", {"n": int(n), "force": bool(force)})
    return ft._section("Input generated", ft._kv(r))


async def do_prepare(d, force):
    r = await ft._post(f"/fsdr/v3/{_run(d)}/prepare", force=str(bool(force)).lower())
    html = ft._section("T0 · validation and Parquet cache", ft._kv(r["t0"]))
    if r["features"]:
        f = r["features"]
        html += ft._section(f"T2 + T3 · {f['lines']:,} lines, {f['micro_segments']} micro-segments",
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
    r = await ft._post(f"/fsdr/v3/{_run(d)}/targets/generate")
    return _check_html(await ft._get(f"/fsdr/targets/{_run(d)}")) + ft._section(
        "Dummy lists written", ft._kv({"folder": r["folder"], "sizes": r["sizes"], "scale_factor": r["scale_factor"]}))


def _final_html(f: dict) -> str:
    state = "READY TO PUSH" if f["push_ready"] else "PUSH BLOCKED — see push/BLOCKED.txt"
    html = ft._section(f"{state} · input_flytxt {f['input_flytxt_rows']:,} rows · details {f['details_rows']:,} lines "
                       f"× {f['details_columns']} columns",
                       ft._table(f["output_checks"], ["id", "check", "status"]))
    cols = ["group", "rules", "lines", "in_input_flytxt", "dropped_F04", "favourites", "forced_F08", "cleared_F09",
            "cleared_F11", "cleared_F12"]
    return html + ft._section("Per target", ft._table([{c: t.get(c, "") for c in cols} for t in f["per_target"]], cols))


def _files(step_dir: str) -> list[str]:
    root = Path(step_dir)
    order = ["push", "target1", "target2", "control"]
    found = [p for sub in order for p in sorted((root / sub).glob("*.csv.gz"))] + [root / "final_rules_report.csv"]
    return [str(p) for p in found if p.exists()]


async def do_run(d):
    r = await ft._post(f"/fsdr/groups/{_run(d)}/run")
    p, o, f = r["panel"], r["offers"], r["final"]
    html = ft._section(f"P1 panel · {p['mode']} · {p['status']}", ft._kv(p["sizes"]))
    html += ft._section(f"P6 scoring per group · group audit {o['audit']}",
                        ft._table(o["groups"], ["group", "rules", "contacted", "with_offers", "offers", "consistency_retry"]))
    return html + _final_html(f), _files(f["step_dir"])


async def do_show(d):
    f = await ft._get(f"/fsdr/final/{_run(d)}")
    return _final_html(f), _files(f["step_dir"])


def build() -> None:
    with gr.Tab("FSDR Targets & Final", id="fsdr-groups"):
        gr.HTML('<div class="nbx-muted">V3.0 four groups: <b>T1C</b> production code 3 · <b>T1F</b> production code 4 · '
                '<b>T2C</b> challenger CC-v1 · <b>T2F</b> challenger CF-v1 · <b>control</b>. Decisions use the LLM '
                'agents\' fallbacks until Phases 5-6 (marked on every row).</div>')
        run_date = gr.Textbox(label="Run date D (YYYY-MM-DD)", value=date.today().isoformat())
        with gr.Group():
            gr.Markdown("**1 · Input of the run date** — `data/input/<D>/` (5 folders, `.csv.gz`)")
            with gr.Row():
                gen_n = gr.Number(label="Dummy lines", value=2000, precision=0, scale=1)
                gen_force = gr.Checkbox(label="Replace existing data", value=False, scale=1)
                gen_btn = gr.Button("1a · Generate dummy input", scale=1)
                prep_force = gr.Checkbox(label="Re-run even if done", value=False, scale=1)
                prep_btn = gr.Button("1b · Prepare (validate + features)", variant="primary", scale=1)
            out1 = gr.HTML()
        with gr.Group():
            gr.Markdown("**2 · Target lists** — `data/input/<D>/targets/`, one list per group")
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
            gr.Markdown("**3 · Run** — panel → scoring per group → final files (`data/runs/<D>/final/`)")
            with gr.Row():
                run_btn = gr.Button("3 · Run panel + scoring + final", variant="primary")
                show_btn = gr.Button("Show the last final output")
            out3 = gr.HTML()
            files = gr.File(label="4 · Download: push file, per-target files, details, rules report",
                            file_count="multiple", interactive=False)
        gen_btn.click(do_generate, inputs=[run_date, gen_n, gen_force], outputs=out1)
        prep_btn.click(do_prepare, inputs=[run_date, prep_force], outputs=out1)
        up_btn.click(do_upload, inputs=[run_date, grp, up], outputs=out2)
        dummy_btn.click(do_dummy_targets, inputs=run_date, outputs=out2)
        check_btn.click(do_check, inputs=run_date, outputs=out2)
        run_btn.click(do_run, inputs=run_date, outputs=[out3, files])
        show_btn.click(do_show, inputs=run_date, outputs=[out3, files])
