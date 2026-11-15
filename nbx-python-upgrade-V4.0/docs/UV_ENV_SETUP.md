# Creating the uv environment from scratch on a new laptop

This guide takes you from a copied `nbx-python-upgrade-V4.0` folder, **without `.venv`**, to a working environment and a running notebook. It is written for Windows and PowerShell. The examples use `D:\production\nbx-python-upgrade-V4.0`; change the path if yours is different.

Time: about 10 minutes, most of it downloading packages.

---

## 1. Copy the folder (without `.venv`)

Copy `nbx-python-upgrade-V4.0` to the new laptop, for example to `D:\production\`.

**Must be in the copy** (uv builds the environment from these):

| File | Role |
|---|---|
| `pyproject.toml` | the list of dependencies and the dependency groups |
| `uv.lock` | the exact version of every package (the same versions on every machine) |
| `.python-version` | `3.12`: the Python version uv uses |
| `nbx/`, `notebooks/`, `run.py`, `tests/`, `docs/`, `.env.example` | the code |

**Do not copy:**

| Do not copy | Why |
|---|---|
| `.venv\` | it holds the old machine's paths and DLLs. A copied `.venv` gives "access denied", a dead kernel, or the wrong Python. You create it below. |
| `__pycache__\`, `.ipynb_checkpoints\`, `.pytest_cache\` | caches, rebuilt by themselves |
| `data\` | optional: copy it only if you want the old runs |
| `.env` | optional: you can create it again from `.env.example` (§6) |

**Place:** avoid folders synced by OneDrive, and very long paths. `D:\production\` or `C:\work\` are fine.

If a `.venv` came with the copy by mistake, delete it before §4:

```powershell
cd D:\production\nbx-python-upgrade-V4.0
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
```

---

## 2. Install uv (once per laptop)

Choose one of the two ways.

**A. With pip** (needs a Python already installed):

```powershell
pip install uv
```

If PowerShell then says `uv : The term 'uv' is not recognized`, the Python `Scripts` folder is not on the PATH. Either use `python -m uv` instead of `uv` in every command below, or add the folder that pip printed in its warning to the PATH.

**B. With the official installer** (no Python needed):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Then close and reopen PowerShell**, and check:

```powershell
uv --version          # prints the version, e.g. uv 0.13.x
```

---

## 3. Install Python 3.12 (managed by uv)

```powershell
cd D:\production\nbx-python-upgrade-V4.0
uv python install 3.12
uv python list --only-installed       # shows the 3.12 that uv will use
```

uv keeps its own Python, separate from any other Python on the laptop, so other installs are not touched. If a 3.12 is already installed, this step does nothing.

---

## 4. Create the environment

```powershell
cd D:\production\nbx-python-upgrade-V4.0
uv sync --locked
```

What this does:
- creates `.venv\` inside the project folder, with Python 3.12;
- installs **exactly** the versions of `uv.lock`;
- installs the three default groups:
  - `dev`: pytest, for the tests;
  - `tracing`: langsmith (it sends nothing unless you switch it on);
  - `notebook`: JupyterLab, ipykernel, nbconvert, matplotlib.

`--locked` stops with an error instead of changing `uv.lock` if `pyproject.toml` and the lock do not match. That way the laptop gets the same versions as the development machine. If it stops with "the lockfile needs to be updated", the copy is incomplete: copy `pyproject.toml` and `uv.lock` again, from the same version.

**Behind a company proxy or SSL inspection** (download or certificate errors), run these before `uv sync`:

```powershell
$env:HTTPS_PROXY = "http://proxy.company:8080"     # your proxy
$env:UV_NATIVE_TLS = "true"                        # use the Windows certificate store
```

---

## 5. Check the environment

```powershell
uv run python -c "import sys; print(sys.version); print(sys.executable)"
```

It must print `3.12.x` and `D:\production\nbx-python-upgrade-V4.0\.venv\Scripts\python.exe`.

```powershell
uv run python -c "import pandas, numpy, pyarrow, yaml, dotenv, matplotlib, jupyterlab; print('packages ok')"
uv run python run.py check
```

`run.py check` lists every package and folder; each line should say `[ok]`. Until §6, it may warn that `.env` is missing.

To see what was installed:

```powershell
uv pip list
```

---

## 6. Create `.env`

```powershell
copy .env.example .env
notepad .env
```

Set at least:

| Setting | Value |
|---|---|
| `FSDR_LOCAL_ONLY` | `true` (the default: nothing leaves the laptop) |
| `FSDR_PII_SALT` | a long random secret (it pseudonymises the customers) |
| `LLM_BACKEND` | `offline`, until you start a local LLM server (`ollama`, `vllm`, `llamacpp`, `lmstudio`) |

Never send `.env` to anyone, and never put it in a zip or an e-mail.

---

## 7. Run the notebooks

### JupyterLab (the browser)

```powershell
cd D:\production\nbx-python-upgrade-V4.0
uv run jupyter lab
```

1. The browser opens at `http://localhost:8888`.
2. Open `notebooks/01_training_step_by_step.ipynb`.
3. Click **Kernel → Restart Kernel and Run All Cells**.
4. The first cell ("Your machine") must print:
   ```
   kernel  : OK, the project's .venv
   GPU 0   : NVIDIA ...            (if the laptop has an NVIDIA GPU)
   ```
5. The last cell must print `all defined in this notebook: True` and `vowpalwabbit loaded: False`.
6. Then do the same with `02_scoring_step_by_step.ipynb`.

Stop JupyterLab with **Ctrl+C** in PowerShell.

### VS Code

1. **File → Open Folder** → `D:\production\nbx-python-upgrade-V4.0`.
2. Open the notebook, then **Select Kernel** (top right) → **Python Environments** → `.venv (Python 3.12)`.
3. If `.venv` is not in the list, reload VS Code: Ctrl+Shift+P → **Developer: Reload Window**.

### Terminal, without opening the notebook

```powershell
uv run jupyter nbconvert --to html --execute notebooks/01_training_step_by_step.ipynb --output-dir data/runs
```

---

## 8. Activating the environment (optional)

`uv run …` always uses `.venv`, so you do not need to activate it. If you prefer an activated shell:

```powershell
.venv\Scripts\Activate.ps1          # the prompt starts with (nbx-python-upgrade-V4.0)
python run.py check
jupyter lab
deactivate                           # leave
```

If PowerShell refuses ("running scripts is disabled"), run this once, then try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

In `cmd.exe`, use `.venv\Scripts\activate.bat` instead.

---

## 9. Rebuilding the environment

Do this when the environment is broken, or after you copy a new version of the code:

```powershell
cd D:\production\nbx-python-upgrade-V4.0
# close JupyterLab and VS Code first: they lock files in .venv
Remove-Item -Recurse -Force .venv
uv sync --locked
```

uv keeps the downloaded packages in its cache, so a rebuild takes less than a minute.

**Updating the code from a new copy of the folder:** copy the new `nbx/`, `notebooks/`, `docs/`, `tests/`, `run.py`, `pyproject.toml` and `uv.lock` over the old ones. Keep the laptop's `.env` and `data\`. Then run `uv sync --locked`. It installs only what changed.

---

## 10. Without uv (fallback with pip)

If uv cannot be installed, use a plain Python 3.12:

```powershell
cd D:\production\nbx-python-upgrade-V4.0
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-notebook.txt     # runtime + Jupyter
pip install -r requirements-dev.txt          # + tests (optional)
python run.py check
jupyter lab
```

The three files have the same versions as `uv.lock`:

| File | Installs |
|---|---|
| `requirements.txt` | the runtime only |
| `requirements-dev.txt` | the runtime and the tests |
| `requirements-notebook.txt` | the runtime and Jupyter |

---

## 11. Problems

| What you see | Fix |
|---|---|
| `uv` is not recognized | Reopen PowerShell. With `pip install uv`, use `python -m uv`, or add Python's `Scripts` folder to the PATH (§2). |
| `uv sync --locked`: "the lockfile needs to be updated" | `pyproject.toml` and `uv.lock` come from different versions. Copy both again, from the same version. |
| Download, SSL or certificate errors | Set the proxy and `UV_NATIVE_TLS=true` (§4). |
| "Access is denied" when deleting `.venv` | JupyterLab or VS Code still uses it. Close them, then delete it again. |
| The notebook kernel dies, "access denied", or a crash in `msvcp140.dll` | Delete `.venv\Lib\site-packages\00_vw_preload.pth` if you created it. Better still, rebuild `.venv` (§9). Restart the kernel and run from the top. V4.0 never loads `vowpalwabbit` in the notebooks. |
| First cell: `kernel : NOT the project's .venv` | Select the `.venv` kernel (§7), or start JupyterLab with `uv run jupyter lab` from the project folder. |
| `ModuleNotFoundError: No module named 'pandas'` | Same cause: the kernel is not the project's `.venv`. |
| `ModuleNotFoundError: No module named 'nbx…'` in a step | Part A of the notebook was not run, or the kernel was restarted. Run Part A again. |
| `warning: Failed to inspect Python interpreter … Access is denied (os error 5)` (in French: `Accès refusé`) | Harmless. uv looked at another Python on the laptop that it may not read, and it does not use that Python. It is **not** the notebook crash. |
| The wrong Python version (not 3.12) | `uv python install 3.12`, delete `.venv`, then `uv sync --locked`. |

---

## Short version

```powershell
pip install uv                                   # once per laptop (or the official installer)
cd D:\production\nbx-python-upgrade-V4.0         # the copied folder, without .venv
uv python install 3.12
uv sync --locked                                 # creates .venv from uv.lock
uv run python run.py check
copy .env.example .env                           # then set FSDR_PII_SALT
uv run jupyter lab                               # notebooks/01_..., then 02_...
```

Next step: put the real data in place, as [DATA_UPDATE.md](DATA_UPDATE.md) explains.
