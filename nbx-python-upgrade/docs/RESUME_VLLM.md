# vLLM for the FSDR agents: status and how to start

Status on 2026-10-23: **done.** The FSDR agents run on vLLM on this laptop:
- vLLM 0.6.6.post1 in WSL2;
- GTX 1650, 4 GB;
- model `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4`, served as `qwen2.5:3b`.

Full details: [FSDR_AGENTIC_RECOMMENDER.md §1](FSDR_AGENTIC_RECOMMENDER.md#1-is-an-llm-used-offline-ollama-vllm) and
[GPU_SETUP.md §14](GPU_SETUP.md#14-small-gpu-on-windows-wsl2-gtx-1650-4-gb).

## Results (measured)

| Step | Result |
|---|---|
| One decision (`fsdr-decide --msisdn 70094659`) | `mode: agentic`, `policy_source: llm`, 26 s |
| 3 customers | all agentic, 4–5 LLM calls, 80–105 s each; offer overrides with reasons; critic approved |
| Batch of 10 (llm arm) | 10/10 agentic in 800 s; actions RETAIN ×3, UPSELL VOICE ×2, UPSELL, UPSELL DATA, CROSS SELL VOICE, MAINTAIN, NO_CONTACT |
| App | `/health`: `"llm_backend": "vllm"`, `"loaded_models": ["qwen2.5:3b"]`; the Agentic Recommender stream works (7 LLM calls, 153 s) |

Changes made along the way:
1. **GPTQ instead of AWQ.** It is 3× faster on Turing.
2. **Evidence prefetch** (`FSDR_AGENT_PREFETCH=1`). One decision went from 310 s to 25–150 s.
3. **A plain `verdict` sentence in `contact_policy`.** NO_CONTACT went from 6/10 to 1/10.

## Commands to start (Windows, PowerShell, in `nbx-python-upgrade`)

```powershell
# 1. vLLM server in WSL (about 3 min before "Application startup complete"; leave this window open)
python run.py vllm-serve

# 2. In a second window: is it up?  (expects the model list "qwen2.5:3b")
curl.exe -s http://127.0.0.1:8001/v1/models -H "Authorization: Bearer <VLLM_API_KEY from .env>"

# 3. The app on vLLM
python run.py start              # http://localhost:7860  /  http://localhost:8000/docs

# One decision with the agent trace
$env:PYTHONIOENCODING="utf-8"; python run.py fsdr-decide --msisdn 70094659 --trace

# Without the GPU
python run.py --backend offline start
```

Stop: press Ctrl+C in each window, or run `wsl.exe -d Ubuntu -- pkill -f "vllm serve"`.

On a new machine, run `python run.py vllm-install` once first. Then create `.env` from `.env.example` with
`LLM_BACKEND=vllm`, `DEFAULT_LLM=qwen2.5:3b` and a random `VLLM_API_KEY`.

## Possible next steps

- A larger model on a server GPU (7B) for better reasoning and whole-arm batches.
- An LLM critic that judges the quality of each justification (today the critic is code only).
- Phase 5b (data / training / scoring / monitor agents) and Phase 6 (campaign log, feedback, simulation), from
  `fsdr_plan.md`.
