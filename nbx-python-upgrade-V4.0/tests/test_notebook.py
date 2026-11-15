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


def test_step_notebooks_run_in_a_folder_without_the_project_code(tmp_path):
    """Executes both notebooks in an empty project folder (no nbx/): they must run end to end."""
    import shutil

    import pytest

    nbclient = pytest.importorskip("nbclient")
    (tmp_path / "pyproject.toml").write_text("# marker\n", encoding="utf-8")
    (tmp_path / "notebooks").mkdir()
    for name in STEP_NOTEBOOKS:
        nb = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        for c in nb.cells:                                           # a smaller run for the test
            if c.cell_type == "code" and c.source.lstrip().startswith("# --- settings"):
                c.source = c.source.replace("N_LINES = 3000", "N_LINES = 600")
        nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                resources={"metadata": {"path": str(tmp_path / "notebooks")}}).execute()
        last = [c for c in nb.cells if c.cell_type == "code"][-1]
        text = "".join(o.get("text", "") for o in last.outputs)
        assert "all defined in this notebook: True" in text and "vowpalwabbit loaded: False" in text
    assert (tmp_path / "data" / "output" / "2026-10-22" / "final" / "summary.json").exists()
