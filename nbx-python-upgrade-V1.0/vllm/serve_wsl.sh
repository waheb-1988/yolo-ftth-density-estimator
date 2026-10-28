#!/usr/bin/env bash
# Serves one model with vLLM in WSL2 for the FSDR agents (OpenAI-compatible API on port 8001).
#   python run.py vllm-serve        (from Windows)   or   bash vllm/serve_wsl.sh   (inside WSL)
#
# Defaults are sized for a 4 GB GPU (GTX 1650): Qwen2.5-3B-Instruct in 4-bit GPTQ (~2 GB of
# weights; measured 13-14 tokens/s, vs 4 tokens/s for the AWQ build on Turing), fp16 (Turing has
# no bf16), 8k context, no CUDA graphs (--enforce-eager) to save memory.
# vLLM reads the quantization method from the model's config. On a bigger GPU set
# VLLM_HF_MODEL / VLLM_SERVED_NAME, e.g.
#   VLLM_HF_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ VLLM_SERVED_NAME=qwen2.5:7b bash vllm/serve_wsl.sh
# The served name must match DEFAULT_LLM in .env (the app falls back to the served model anyway).
set -euo pipefail

MODEL="${VLLM_HF_MODEL:-Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4}"
NAME="${VLLM_SERVED_NAME:-qwen2.5:3b}"
PORT="${VLLM_PORT:-8001}"
# Review #19: the key goes through the environment (vLLM reads VLLM_API_KEY), never on the command line, so it
# is not printed with the arguments in the startup log.
export VLLM_API_KEY="${VLLM_API_KEY:-}"
[ -z "$VLLM_API_KEY" ] && unset VLLM_API_KEY

# shellcheck disable=SC1091
source "$HOME/vllm-env/bin/activate"
export VLLM_USE_V1=0
# Triton compiles a small C helper at the first request: use gcc if present, else the zig compiler.
command -v gcc >/dev/null || export CC="${CC:-$HOME/vllm-env/cc/cc}"
exec vllm serve "$MODEL" --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT" \
  --dtype half \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}" \
  --max-num-seqs "${VLLM_MAX_SEQS:-8}" \
  --enforce-eager
