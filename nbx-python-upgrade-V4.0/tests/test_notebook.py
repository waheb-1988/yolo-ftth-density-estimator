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
                compile(c.source, "<cell>", "exec")
        steps = [c.source.split("\n")[0] for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("## Step")]
        assert len(steps) >= 6, steps
