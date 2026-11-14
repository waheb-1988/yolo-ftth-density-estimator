"""V4.0 local only: the API docs page without any CDN.

FastAPI's Swagger UI loads its JavaScript and CSS from cdn.jsdelivr.net. With FSDR_LOCAL_ONLY=true (default) the
page at /docs is this one instead: plain HTML + a few lines of JavaScript, served by the API itself. It lists every
route of /openapi.json (method, path, summary, parameters) and has a small "Try it" form (JSON body or a file).
FSDR_SWAGGER_CDN=true brings the standard Swagger UI back (it needs internet in the browser).
"""
from __future__ import annotations

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FSDR API (local docs)</title>
<style>
 :root { --bg:#0a0a0c; --card:#131315; --line:#2a2a2e; --text:#f5f5f7; --muted:#9c9ca3; --accent:#d3163b; }
 body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 system-ui, sans-serif; }
 main { max-width:1100px; margin:0 auto; padding:16px; }
 h1 { font-size:20px; margin:8px 0; } .muted { color:var(--muted); }
 details { background:var(--card); border:1px solid var(--line); border-radius:8px; margin:6px 0; }
 summary { cursor:pointer; padding:8px 12px; display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }
 .m { font:600 12px ui-monospace, Consolas, monospace; padding:2px 6px; border-radius:4px; min-width:48px; text-align:center; }
 .GET { background:#1f3a5f; } .POST { background:#5f1f2e; } .DELETE { background:#5f4a1f; } .PUT { background:#3a5f1f; }
 code, pre, textarea, input { font:12px ui-monospace, Consolas, monospace; }
 .body { padding:0 12px 12px; } pre { background:#0d0d10; border:1px solid var(--line); padding:8px; overflow:auto;
 max-height:420px; white-space:pre-wrap; word-break:break-word; } textarea, input { width:100%; box-sizing:border-box;
 background:#0d0d10; color:var(--text); border:1px solid var(--line); border-radius:4px; padding:6px; }
 button { background:var(--accent); color:#fff; border:0; border-radius:4px; padding:6px 14px; cursor:pointer; margin-top:6px; }
 table { border-collapse:collapse; width:100%; } td { border-top:1px solid var(--line); padding:3px 6px; vertical-align:top; }
</style></head><body><main>
<h1>FSDR Recommender API</h1>
<p class="muted">Local docs (no CDN, nothing loaded from outside this machine). Raw schema: <a href="/openapi.json" style="color:#9cf">/openapi.json</a>.
 API key (only if FSDR_API_KEY is set): <input id="key" style="width:260px" placeholder="X-API-Key"></p>
<input id="filter" placeholder="filter routes, e.g. v4 or targets">
<div id="routes"></div>
</main><script>
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
fetch("/openapi.json").then(r => r.json()).then(spec => {
  const box = document.getElementById("routes"); let html = "";
  for (const [path, ops] of Object.entries(spec.paths)) for (const [m, op] of Object.entries(ops)) {
    const M = m.toUpperCase(), params = op.parameters || [], rb = op.requestBody;
    const isFile = rb && rb.content && rb.content["multipart/form-data"];
    const rows = params.map(p => `<tr><td><code>${esc(p.name)}</code></td><td>${esc(p.in)}</td><td>${p.required ? "required" : ""}</td></tr>`).join("");
    html += `<details data-k="${esc((M + " " + path + " " + (op.summary || "")).toLowerCase())}"><summary><span class="m ${M}">${M}</span>
      <code>${esc(path)}</code><span class="muted">${esc(op.summary || "")}</span></summary><div class="body">
      ${op.description ? `<p class="muted">${esc(op.description)}</p>` : ""}
      ${rows ? `<table>${rows}</table>` : ""}
      <p>Path (fill the {placeholders}, add ?query=…):</p><input class="u" value="${esc(path)}">
      ${M !== "GET" ? (isFile ? `<p>File:</p><input type="file" class="f">` : `<p>JSON body:</p><textarea class="b" rows="4">{}</textarea>`) : ""}
      <button data-m="${M}">Try it</button><pre class="o"></pre></div></details>`;
  }
  box.innerHTML = html;
  box.addEventListener("click", async e => {
    if (e.target.tagName !== "BUTTON") return;
    const d = e.target.closest("details"), out = d.querySelector(".o"), m = e.target.dataset.m;
    const headers = {}; const key = document.getElementById("key").value; if (key) headers["X-API-Key"] = key;
    let body; const f = d.querySelector(".f"), b = d.querySelector(".b");
    if (f && f.files[0]) { body = new FormData(); body.append("file", f.files[0]); }
    else if (b) { headers["Content-Type"] = "application/json"; body = b.value || "{}"; }
    out.textContent = "...";
    try { const r = await fetch(d.querySelector(".u").value, {method: m, headers, body});
      const t = await r.text(); let pretty = t; try { pretty = JSON.stringify(JSON.parse(t), null, 2); } catch (_) {}
      out.textContent = r.status + " " + r.statusText + "\\n" + pretty; } catch (err) { out.textContent = String(err); }
  });
  document.getElementById("filter").addEventListener("input", e => {
    const q = e.target.value.toLowerCase();
    for (const d of box.querySelectorAll("details")) d.style.display = d.dataset.k.includes(q) ? "" : "none";
  });
});
</script></body></html>"""
