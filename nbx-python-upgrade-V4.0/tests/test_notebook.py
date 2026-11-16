"""The results notebook is built from notebooks/build_results_notebook.py, is valid, and is stored without outputs."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_is_built_from_its_builder_and_has_no_outputs(tmp_path):
    spec = importlib.util.spec_from_file_location("nb_builder", ROOT / "notebooks" / "build_results_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    built = nbformat.read(mod.build(tmp_path / "nb.ipynb"), as_version=4)
    nbformat.validate(built)
    stored = nbformat.read(ROOT / "notebooks" / "fsdr_results.ipynb", as_version=4)
    assert [c.source for c in stored.cells] == [c.source for c in built.cells]      # rebuild after editing the builder
    assert all(not c.get("outputs") for c in stored.cells if c.cell_type == "code")  # outputs hold MSISDNs
    for c in built.cells:
        if c.cell_type == "code":
            compile(c.source, "<cell>", "exec")


def test_step_notebooks_are_built_from_their_builder_and_have_no_outputs(tmp_path):
    spec = importlib.util.spec_from_file_location("nb_steps", ROOT / "notebooks" / "build_step_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    built = mod.build(tmp_path)
    assert [p.name for p in built] == ["01_training_step_by_step.ipynb", "02_scoring_step_by_step.ipynb"]
    for p in built:
        nb = nbformat.read(p, as_version=4)
        nbformat.validate(nb)
        stored = nbformat.read(ROOT / "notebooks" / p.name, as_version=4)
        assert [c.source for c in stored.cells] == [c.source for c in nb.cells]        # rebuild after editing the builder
        assert all(not c.get("outputs") for c in stored.cells if c.cell_type == "code")
        for c in nb.cells:
            if c.cell_type == "code":
                body = c.source.split("\n", 1)[1] if c.source.startswith("%%engine") else c.source
                compile(body, "<cell>", "exec")
        steps = [c.source.split("\n")[0] for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("## Step")]
        assert len(steps) >= 6, steps


def test_gpu_notebook_is_built_standalone_and_has_no_outputs(tmp_path):
    spec = importlib.util.spec_from_file_location("nb_gpu", ROOT / "notebooks" / "build_gpu_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nb = nbformat.read(mod.build(tmp_path), as_version=4)
    nbformat.validate(nb)
    stored = nbformat.read(ROOT / "notebooks" / mod.NAME, as_version=4)
    assert [c.source for c in stored.cells] == [c.source for c in nb.cells]            # rebuild after editing the builder
    assert all(not c.get("outputs") for c in stored.cells if c.cell_type == "code")
    code_cells = [c.source for c in nb.cells if c.cell_type == "code"]
    for s in code_cells:
        compile(s.split("\n", 1)[1] if s.startswith("%%engine") else s, "<cell>", "exec")
        if not s.startswith("%%engine"):
            assert "sys.path.insert" not in s
            assert not any(l.lstrip().startswith(("from nbx", "import nbx")) for l in s.splitlines())
            assert not any(w in s for w in ("import gradio", "import fastapi", "import uvicorn", "import httpx"))
    titles = " ".join(c.source.split("\n")[0] for c in nb.cells if c.cell_type == "markdown")
    for agent in ("Agent 1", "Agent 2 ", "Agent 2b", "Agent 3", "Agent 4", "Agent 6B", "Agent 7", "Agent 9", "Agent 8"):
        assert agent in titles, agent
    assert 'OLLAMA_URL = "http://127.0.0.1:11434"' in code_cells[0]


STEP_NOTEBOOKS = ("01_training_step_by_step.ipynb", "02_scoring_step_by_step.ipynb")


def test_step_notebooks_are_standalone():
    """V4.0: the engine code is IN the notebook (Part A), identical to the project files; nothing imports nbx from the
    project folder; vowpalwabbit (the Windows msvcp140.dll kernel crash) is never imported; T2 / T3 in-process."""
    for name in STEP_NOTEBOOKS:
        nb = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        code = [c.source for c in nb.cells if c.cell_type == "code"]
        engine = [s for s in code if s.startswith("%%engine")]
        steps = [s for s in code if not s.startswith("%%engine")]
        assert len(engine) >= 25 and any("register_cell_magic" in s for s in steps)
        for s in engine:
            head, body = s.split("\n", 1)
            mod = head.split()[1]
            assert "import vowpalwabbit" not in body, mod
            if mod == "nbx":
                continue
            f = ROOT / mod.replace(".", "/")
            f = f / "__init__.py" if f.is_dir() else f.with_suffix(".py")
            assert body.rstrip() == f.read_text(encoding="utf-8").rstrip(), f"{mod}: rebuild the notebooks"
        for s in steps:
            assert "sys.path.insert" not in s
            for line in s.splitlines():
                assert not line.lstrip().startswith(("from nbx", "import nbx")), line
    settings = [s for s in (c.source for c in nbformat.read(ROOT / "notebooks" / STEP_NOTEBOOKS[0], as_version=4).cells)
                if s.lstrip().startswith("# --- settings")]
    assert "WORKERS = 1" in settings[0]


def _test_settings(src: str, run_date: str) -> str:
    """The notebooks default to the real run (2026-11-19, no dummy data): the tests use dummy data on a test date."""
    for old, new in (('RUN_DATE = "2026-11-19"', f'RUN_DATE = "{run_date}"'), ("GENERATE = False ", "GENERATE = True "),
                     ("GENERATE_TARGETS = False", "GENERATE_TARGETS = True")):
        src = src.replace(old, new)
    return src


def _deployment_default(monkeypatch):
    """The notebooks run as deployed: Target 2 only, 50k + 50k (the suite itself tests both targets at 25k)."""
    monkeypatch.setenv("FSDR_TARGETS", "target2")
    monkeypatch.delenv("FSDR_T2_CONSTRAINT_SIZE", raising=False)
    monkeypatch.delenv("FSDR_T2_FREE_SIZE", raising=False)
    monkeypatch.delenv("FSDR_CONTROL_SIZE", raising=False)                   # default 0: no control group


def test_step_notebooks_run_in_a_folder_without_the_project_code(tmp_path, monkeypatch):
    """Executes both notebooks in an empty project folder (no nbx/): they must run end to end."""
    import shutil

    import pytest

    _deployment_default(monkeypatch)

    nbclient = pytest.importorskip("nbclient")
    (tmp_path / "pyproject.toml").write_text("# marker\n", encoding="utf-8")
    (tmp_path / "notebooks").mkdir()
    for name in STEP_NOTEBOOKS:
        nb = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        for c in nb.cells:                                           # a smaller run for the test
            if c.cell_type == "code" and c.source.lstrip().startswith("# --- settings"):
                c.source = _test_settings(c.source, "2026-10-22").replace("N_LINES = 3000", "N_LINES = 600")
        nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                resources={"metadata": {"path": str(tmp_path / "notebooks")}}).execute()
        last = [c for c in nb.cells if c.cell_type == "code"][-1]
        text = "".join(o.get("text", "") for o in last.outputs)
        assert "all defined in this notebook: True" in text and "vowpalwabbit loaded: False" in text
    assert (tmp_path / "data" / "output" / "2026-10-22" / "final" / "summary.json").exists()


def _fake_answer(schema):
    """A valid reply for a JSON schema (what Ollama's structured output guarantees)."""
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object":
        return {k: _fake_answer(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        n = schema.get("minItems", 1)
        return [(_fake_answer(schema["items"]) if schema["items"].get("type") != "integer" else 10 + 7 * i % 90)
                for i in range(n)]
    if t == "integer":
        return 50
    if t == "boolean":
        return True
    return "fake answer"


def test_gpu_notebook_runs_end_to_end_with_a_local_llm(tmp_path, monkeypatch):
    """The whole notebook against a fake Ollama on 127.0.0.1 (schema-valid answers): every agent runs, the agents'
    decisions reach Target 2, Target 1 stays the rules, and the final file is written."""
    import http.server
    import json
    import threading

    import pytest

    nbclient = pytest.importorskip("nbclient")
    _deployment_default(monkeypatch)

    class Fake(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send({"/api/version": {"version": "fake"}, "/api/tags": {"models": [{"name": "fake:1b"}]},
                        "/api/ps": {"models": [{"name": "fake:1b", "size": 1, "size_vram": 1}]}}.get(self.path, {}))

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            content = json.dumps(_fake_answer(body["format"])) if "format" in body else "ok"
            self._send({"message": {"content": content}, "response": "ok", "eval_count": 20, "prompt_eval_count": 100})

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        (tmp_path / "pyproject.toml").write_text("# marker\n", encoding="utf-8")
        (tmp_path / "notebooks").mkdir()
        nb = nbformat.read(ROOT / "notebooks" / "03_gpu_agents_step_by_step.ipynb", as_version=4)
        for c in nb.cells:
            if c.cell_type == "code" and c.source.lstrip().startswith("# --- settings"):
                c.source = (_test_settings(c.source, "2026-10-29").replace("N_LINES = 3000", "N_LINES = 800")
                            .replace('"http://127.0.0.1:11434"', f'"http://127.0.0.1:{srv.server_port}"')
                            .replace("MAX_LLM_LINES = None", "MAX_LLM_LINES = 25"))
        nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                resources={"metadata": {"path": str(tmp_path / "notebooks")}}).execute()
    finally:
        srv.shutdown()
    text = "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert "all defined in this notebook: True" in text and "api / gradio loaded: False" in text
    assert "T0 DONE -- pipeline train_week0" in text
    out = tmp_path / "data" / "output" / "2026-10-29"
    assert json.loads((out / "final" / "summary.json").read_text())["ready"]
    p6 = tmp_path / "data" / "runs" / "2026-10-29" / "score"
    import pandas as pd
    t2 = pd.read_csv(p6 / "P6_offers_target2" / "group_report.csv")
    assert t2["llm_scored_offers"].sum() > 0 and "llm_nba" in "".join(t2["decided_by"])
    assert not (p6 / "P6_offers_target1").exists() and not (out / "target1").exists()   # Target 1 is not scored
    assert (tmp_path / "data" / "runs" / "2026-10-29" / "train" / "T3_segments" / "dict_segm_actions_agent_v1.json").exists()


def test_vllm_notebook_is_built_standalone_and_has_no_outputs(tmp_path):
    spec = importlib.util.spec_from_file_location("nb_gpu2", ROOT / "notebooks" / "build_gpu_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nb = nbformat.read(mod.build(tmp_path, backend="vllm"), as_version=4)
    nbformat.validate(nb)
    stored = nbformat.read(ROOT / "notebooks" / mod.VLLM_NAME, as_version=4)
    assert [c.source for c in stored.cells] == [c.source for c in nb.cells]            # rebuild after editing the builder
    assert all(not c.get("outputs") for c in stored.cells if c.cell_type == "code")
    code_cells = [c.source for c in nb.cells if c.cell_type == "code"]
    for s in code_cells:
        compile(s.split("\n", 1)[1] if s.startswith("%%engine") else s, "<cell>", "exec")
        if not s.startswith("%%engine"):
            assert "11434" not in s and "/api/chat" not in s                         # no Ollama call left
    assert 'LLM_URL = "http://127.0.0.1:8001"' in code_cells[0] and "/v1/chat/completions" in "".join(code_cells)
    # the agent steps are the same cells as in the Ollama notebook
    ollama = [c.source for c in nbformat.read(ROOT / "notebooks" / mod.NAME, as_version=4).cells]
    shared = [c.source for c in nb.cells if c.source.startswith(("## Step", "## Time per agent"))]
    assert shared and all(s in ollama for s in shared)


def test_vllm_notebook_runs_end_to_end_with_a_local_openai_server(tmp_path, monkeypatch):
    """Notebook 04 against a fake vLLM (OpenAI API, json_schema response_format, a Bearer key)."""
    import http.server
    import json
    import threading

    import pytest

    nbclient = pytest.importorskip("nbclient")
    seen = {"auth": set(), "thinking_off": 0}

    class Fake(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _authorised(self):
            seen["auth"].add(self.headers.get("Authorization"))
            if self.headers.get("Authorization") != "Bearer test-key":
                self._send({"error": "Unauthorized"}, 401)
                return False
            return True

        def do_GET(self):
            if self._authorised():
                self._send({"data": [{"id": "qwen3:8b", "max_model_len": 8192}]})

        def do_POST(self):
            if not self._authorised():
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["thinking_off"] += body.get("chat_template_kwargs", {}).get("enable_thinking") is False
            schema = body["response_format"]["json_schema"]["schema"]
            self._send({"choices": [{"message": {"content": json.dumps(_fake_answer(schema))}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20}})

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("VLLM_API_KEY", "test-key")                       # the kernel inherits it
    _deployment_default(monkeypatch)
    try:
        (tmp_path / "pyproject.toml").write_text("# marker\n", encoding="utf-8")
        (tmp_path / "notebooks").mkdir()
        nb = nbformat.read(ROOT / "notebooks" / "04_vllm_agents_step_by_step.ipynb", as_version=4)
        for c in nb.cells:
            if c.cell_type == "code" and c.source.lstrip().startswith("# --- settings"):
                c.source = (_test_settings(c.source, "2026-10-29").replace("N_LINES = 3000", "N_LINES = 800")
                            .replace('"http://127.0.0.1:8001"', f'"http://127.0.0.1:{srv.server_port}"')
                            .replace("MAX_LLM_LINES = None", "MAX_LLM_LINES = 25"))
        nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                resources={"metadata": {"path": str(tmp_path / "notebooks")}}).execute()
    finally:
        srv.shutdown()
    text = "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert "MODEL  : qwen3:8b" in text and "all defined in this notebook: True" in text
    assert seen["auth"] == {"Bearer test-key"} and seen["thinking_off"] > 0
    assert "test-key" not in text                                          # the key is never printed
    out = tmp_path / "data" / "output" / "2026-10-29"
    assert json.loads((out / "final" / "summary.json").read_text())["ready"]
    import pandas as pd
    t2 = pd.read_csv(tmp_path / "data" / "runs" / "2026-10-29" / "score" / "P6_offers_target2" / "group_report.csv")
    assert t2["llm_scored_offers"].sum() > 0 and "llm_nba" in "".join(t2["decided_by"])
