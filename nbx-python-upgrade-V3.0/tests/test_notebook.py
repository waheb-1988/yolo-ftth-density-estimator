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
