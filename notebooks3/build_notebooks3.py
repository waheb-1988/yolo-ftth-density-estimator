"""Builds the notebooks of notebooks3/ (V4.0):

    02_prepare_week.ipynb                 data_raw/ -> data/weekly/<D>/: the new export names, D = t1 / T = t1 - 7, the
                                          latest campaign of targeting_dataiku_auto, RUN_SIZE (1k / 5k / 10k / all)
    03_agents_nba_fast_ollama_gpu.ipynb   every agent on the GPU (Ollama); agents 3 / 4 once per customer profile
    04_agents_nba_fast_vllm_gpu.ipynb     the same with vLLM (WSL2)
    04_agents_nba_fast_ollama_real25k.ipynb   03 on the real week: dates from the data (D = last purchase, T = campaign
                                          day), 25k training lines, no LLM budget + retries, 6 offers per line, report

    uv run python notebooks3/build_notebooks3.py      (or: python notebooks3/build_notebooks3.py in the activated .venv)

Every notebook prints the time of each step, its estimate and the time left. The cells come from notebooks2/ plus
nb3_fast.py (FAST_MODE, the groups, the 360 view) and nb3_data.py (names, dates, run size, timer). The engine (nbx/)
is not modified. The notebooks are written WITHOUT outputs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b2 = _load("build_notebooks2", HERE.parent / "notebooks2" / "build_notebooks2.py")   # puts notebooks2/ on sys.path
gb, sb, ag = b2.gb, b2.sb, b2.ag
md, code = sb.md, sb.code
sys.path.insert(0, str(HERE))
import nb3_data as nd
import nb3_fast as fm
import nb3_real as nr

NAMES = {"prep": "02_prepare_week.ipynb", "ollama": "03_agents_nba_fast_ollama_gpu.ipynb",
         "vllm": "04_agents_nba_fast_vllm_gpu.ipynb", "real25k": "04_agents_nba_fast_ollama_real25k.ipynb"}


def _write(cells, name, out_dir):
    """As notebooks2's writer, with deterministic cell ids: the same cells give the same file, byte for byte."""
    for i, c in enumerate(cells):
        c["id"] = hashlib.sha1(f"{name}|{i}|{c['cell_type']}|{c['source']}".encode()).hexdigest()[:8]
    return b2._write(cells, name, out_dir)


def _settings(base: str) -> str:
    s = nd.swap(gb._swap(base, [fm.TIME_SWAP]), nd.AGENT_SETTINGS_SWAPS)
    anchor = "import time as _time\n"
    assert s.count(anchor) == 1, "settings: anchor not found"
    s = s.replace(anchor, fm.FAST_SETTINGS.lstrip("\n") + nd.AGENT_SETTINGS_ADD.lstrip("\n") + anchor)
    return s.rstrip("\n") + "\n" + fm.GROUPS_ENV


def _at(steps, heading):
    i = [k for k, (kind, src) in enumerate(steps) if kind == "md" and src.startswith(heading)]
    assert len(i) == 1, f"step not found once: {heading}"
    return i[0]


def steps() -> list:
    s = list(ag.STEPS)
    i = _at(s, "## Step 1 ·")
    s[i:i] = [fm.GROUPS_CELL[0], ("code", fm.GROUPS_CELL[1][1] + "\n" + nd.AGENT_PATCH)]
    i = _at(s, "## Step 6 ·")
    s[i:i + 2] = fm.STEP_6
    i = _at(s, "## Step 7 ·")
    s[i:i + 2] = fm.STEP_7
    i = _at(s, "## Step 5 ·")
    s[i + 2:i + 2] = fm.STEP_5B
    i = _at(s, "## Step 11 ·")
    s[i + 2:i + 2] = fm.STEP_12
    i = _at(s, "## Time per agent")
    s[i + 2:i + 2] = fm.FAST_TIME
    i = _at(s, "## G3 ·")
    s.insert(i + 2, ("code", fm.G3_FAST))
    return s


def _pairs(cells) -> list:
    return [("code" if c.cell_type == "code" else "md", c.source) for c in cells]


def _cells(pairs) -> list:
    return [code(s) if k == "code" else md(s) for k, s in pairs]


def agents(backend: str) -> list:
    vllm = backend == "vllm"
    title = fm.TITLE.format(num="04" if vllm else "03", server="vLLM (WSL2)" if vllm else "Ollama",
                            url="http://127.0.0.1:8001" if vllm else "http://127.0.0.1:11434",
                            nb2=b2.NAMES[backend])
    g1 = [(c.cell_type if c.cell_type == "code" else "md", c.source) for c in gb.VLLM_G1] if vllm else ag.G1_OLLAMA
    pairs = [("md", title), ("md", ag.VLLM_SETUP if vllm else ag.OLLAMA_SETUP),
             ("code", _settings(b2.VLLM_SETTINGS if vllm else ag.SETTINGS)),
             *_pairs(sb._engine_cells(sb.engine_modules(b2.ROOTS))),
             ("md", "## Part B · The steps\n\nThe engine is loaded. Each step below calls it and shows what it wrote."),
             ("code", ag.SETUP), ("code", gb.VLLM_HELPERS if vllm else gb.LLM_HELPERS)]
    pairs += [(k, sb._use_imports(s) if k == "code" else s) for k, s in g1 + steps()]
    pairs += [("md", ("## Proof: nothing came from the project folder\n\nEvery engine module was defined by a cell of this "
                     "notebook, vowpalwabbit was never loaded, and no API / Gradio package was imported.")),
              ("code", sb.PROOF + '\nprint("api / gradio loaded:", any(m in sys.modules for m in ("gradio", "fastapi", "uvicorn")))')]
    pairs = nd.add_timer(pairs, 3, NAMES[backend][:-6], nd.AGENT_DEFAULT, '"|" + ",".join(SCORE_GROUPS)', first_key="Part A")
    return _cells(pairs)


def _swap_any(pairs, swaps):
    """Applies each (old, new) to the one code cell that holds old."""
    out = list(pairs)
    for a, b in swaps:
        hits = [i for i, (k, src) in enumerate(out) if k == "code" and a in src]
        assert len(hits) == 1, f"not found in one cell: {a[:70]}"
        out[hits[0]] = ("code", out[hits[0]][1].replace(a, b))
    return out


def steps_real25k() -> list:
    s = _swap_any(list(ag.STEPS), nr.G2_SWAP)
    i = _at(s, "## Step 1 ·")
    s[i:i] = [fm.GROUPS_CELL[0], ("code", fm.GROUPS_CELL[1][1] + "\n" + nr.REAL_PATCH)]
    i = _at(s, "## Step 6 ·")
    s[i:i + 2] = fm.STEP_6
    i = _at(s, "## Step 7 ·")
    s[i:i + 2] = _swap_any(fm.STEP_7, nr.STEP7_SWAP)
    i = _at(s, "## Step 5 ·")
    s[i + 2:i + 2] = fm.STEP_5B
    i = _at(s, "## Step 8 ·")
    s[i:i] = nr.TOPUP
    i = _at(s, "## Step 8 ·")
    s[i + 2:i + 2] = nr.TOPUP_REPORT
    i = _at(s, "## Step 11 ·")
    s[i + 2:i + 2] = _swap_any(fm.STEP_12, nr.STEP12_SWAPS)
    i = _at(s, "## Time per agent")
    s[i + 2:i + 2] = fm.FAST_TIME
    i = _at(s, "## G3 ·")
    s[i + 2:i + 2] = [("code", fm.G3_FAST), *nr.FINAL_REPORT]
    return s


def agents_real25k() -> list:
    base = gb._swap(ag.SETTINGS, nr.SETTINGS_SWAPS)
    cut = base.index("import time as _time\n")
    settings = (base[:cut] + fm.FAST_SETTINGS.lstrip("\n") + nr.REAL_SETTINGS.lstrip("\n")
                + "import time as _time\nNB_START = _time.time()\n")
    pairs = [("md", nr.TITLE), ("md", ag.OLLAMA_SETUP), ("code", settings), *nr.STEP0,
             *_pairs(sb._engine_cells(sb.engine_modules(b2.ROOTS))),
             ("md", "## Part B · The steps\n\nThe engine is loaded. Each step below calls it and shows what it wrote."),
             ("code", ag.SETUP), ("code", gb.LLM_HELPERS), *nr.RESILIENCE]
    pairs += [(k, sb._use_imports(s) if k == "code" else s) for k, s in ag.G1_OLLAMA + nr.PREFLIGHT + steps_real25k()]
    pairs += [("md", ("## Proof: nothing came from the project folder\n\nEvery engine module was defined by a cell of this "
                      "notebook, and no engine module uses vowpalwabbit (it may be preloaded by the environment).")),
              ("code", nr.PROOF)]
    timer = nd.TIMER.replace(nr.TIMER_SAVE_OLD, nr.TIMER_SAVE_NEW)
    assert timer != nd.TIMER, "timer: the save line was not found"
    pairs = nd.add_timer(pairs, 3, NAMES["real25k"][:-6], nr.DEFAULTS, '"|" + ",".join(SCORE_GROUPS) + "|train=" + str(TRAIN_LINES)',
                         timer_src=timer)
    return _cells(pairs)


def build(out_dir: Path = HERE) -> list[Path]:
    return [_write(_cells(nd.build_prep(b2.dt.PREP)), NAMES["prep"], out_dir),
            *[_write(agents(k), NAMES[k], out_dir) for k in ("ollama", "vllm")],
            _write(agents_real25k(), NAMES["real25k"], out_dir)]


if __name__ == "__main__":
    for p in build():
        print("written", p)
