#!/usr/bin/env bash
# Installs vLLM in WSL2 Ubuntu for the FSDR agents -- no sudo needed.
#   python run.py vllm-install      (from Windows)   or   bash vllm/install_wsl.sh   (inside WSL)
#
# vllm 0.6.6.post1 is pinned on purpose: it still supports compute capability 7.x GPUs
# (e.g. GTX 1650, compute 7.5) through its V0 engine; newer releases target 8.0+.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null; then
  echo "No GPU visible in WSL. Install the NVIDIA Windows driver (it ships the WSL CUDA driver)." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$HOME/.local/bin/uv"

# uv-managed Python: it ships Python.h, which Triton needs to build its CUDA helper at the first
# request (Ubuntu's python3.12 has no headers without `sudo apt install python3-dev`).
[ -d "$HOME/vllm-env" ] || "$UV" venv "$HOME/vllm-env" --python 3.12 --python-preference only-managed
# shellcheck disable=SC1091
source "$HOME/vllm-env/bin/activate"
# Pins for vllm 0.6.6 (torch 2.5.1): transformers >= 4.48 needs torch 2.6, and xgrammar
# (guided JSON for response_format) dropped TokenizerInfo.from_huggingface after 0.1.10.
# ziglang is a C compiler from pip, for Triton when gcc is not installed (no sudo needed).
"$UV" pip install "vllm==0.6.6.post1" "transformers==4.47.1" "xgrammar==0.1.10" "ziglang==0.13.0"
mkdir -p "$HOME/vllm-env/cc"
printf '#!/bin/sh\nexec "%s" -m ziglang cc "$@"\n' "$(command -v python)" > "$HOME/vllm-env/cc/cc"
chmod +x "$HOME/vllm-env/cc/cc"
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "Done. Start the server with: python run.py vllm-serve   (or bash vllm/serve_wsl.sh)"
