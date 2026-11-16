# vLLM vs Ollama on the analytics laptop, and why vLLM needs WSL2

**Audience:** the security team, and the business owner of the campaign engine.
**Laptop:** NVIDIA RTX PRO 5000 Blackwell Laptop GPU (24 GB), driver 597.13 (CUDA 13.2), Windows 11.
**Context:** the FSDR campaign engine (V4.0) runs entirely on this laptop. Its LLM agents need a **local** LLM server. There are two options:
- **Ollama:** native on Windows; already installed and working.
- **vLLM:** faster for the weekly batch, but it runs on Linux only, so on Windows it needs **WSL2**.

This document compares the two and lists what WSL2 changes on the laptop. An email for the request is in §6.

---

## 1. What the LLM server is used for

- **Every week**, the engine scores the customers of Target 2: 100,000 customers in production (group a 50,000 + group b 50,000). For each customer:
  - the LLM scores the allowed **actions** and the allowed **offers**;
  - the rules then decide, and the audit can block the output.
- **What the LLM receives:** a short text of facts about each customer: segment, revenue trend, usage, churn score, and the allowed actions and offers.
  - **No MSISDN, no name, no identifier** is ever sent.
  - The LLM never sees a customer file.
- **What the LLM returns:** a few numbers in a fixed JSON schema (scores from 0 to 100).
- **Where it runs:** the server is on the laptop itself (`127.0.0.1`). The engine refuses any LLM address that is not local (`FSDR_LOCAL_ONLY=true`, enforced in code).
- **The output** is a file saved on the laptop. Nothing is sent anywhere: a person checks the file and hands it over.

---

## 2. Comparison

| | **Ollama** | **vLLM** |
|---|---|---|
| Runs on | Windows natively (a desktop app and a local service) | **Linux only**; on Windows, inside **WSL2** (a Microsoft Linux VM) |
| Installation | the Ollama installer (already done) | enable WSL2 (admin, once), Ubuntu, then vLLM in a Python venv inside Ubuntu (no root needed) |
| Speed, weekly batch of 100k customers (estimate on this GPU) | about **8–14 h** (`qwen3:8b`), 16–28 h (`qwen3:14b`) | about **3–6 h** (`Qwen3-8B-AWQ`), 6–12 h (`Qwen3-14B-AWQ`) |
| Why vLLM is faster | — | it batches up to 32 customers at once on the GPU (continuous batching, paged KV cache) |
| Answer quality | the same model family (Qwen3), 4-bit | the same model family (Qwen3), 4-bit |
| JSON output (a fixed schema) | yes | yes |
| GPU memory | loads the model only (5–9 GB) | reserves about 85% of the GPU memory at start (its KV cache) |
| Model format | GGUF, from `registry.ollama.ai` | safetensors (data only, no executable code), from `huggingface.co`; served **without** `--trust-remote-code` |
| Listens on | `127.0.0.1:11434`, no authentication by default (local only) | `127.0.0.1:8001` inside WSL2, reached from Windows through WSL's localhost forwarding; **Bearer API key** (`VLLM_API_KEY`) |
| Telemetry | prompts are not sent; the app checks `ollama.com` for updates | usage statistics switched off (`VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`); Hugging Face telemetry off; `VLLM_OFFLINE=1` forbids any download after the first one |
| Admin rights | not needed after the install | **once**, to enable WSL2 |
| Uninstall | uninstall the app | `wsl --unregister Ubuntu` (deletes the Linux disk), then optionally disable the WSL features |
| Notebook | `03_gpu_agents_step_by_step.ipynb` | `04_vllm_agents_step_by_step.ipynb` (the same agents and steps) |

**Measured on the development laptop** (GTX 1650, 4 GB):
- Both notebooks ran end to end, and every agent answered.
- On that old GPU, vLLM's legacy engine was slower than Ollama: 14.3 s per customer, against 3.7 s.
- The vLLM speed-up needs the modern engine (vLLM 0.32.0) on a modern GPU, such as this Blackwell laptop.
- The first run of the notebook on the new laptop measures the real speed (step G2) before the long steps.

**Both options give the same decisions and the same controls.** The only difference is speed: one weekly run of 3–6 h with vLLM, against 8–14 h with Ollama (100k customers).

---

## 3. Why vLLM needs WSL2

- vLLM is published for Linux only; there is no native Windows build.
- On Windows, the supported way to run Linux software is **WSL2** (Windows Subsystem for Linux, version 2), a Windows feature made by Microsoft.
- WSL2 runs a real Linux kernel in a lightweight **utility virtual machine**, built on Hyper-V technology.
- The **Windows NVIDIA driver** (597.13) gives WSL2 access to the GPU, so no driver is installed inside Linux.
- Docker Desktop would also run vLLM, but it uses WSL2 itself and adds more software. A remote Linux GPU server would avoid WSL2, but then the data would leave the laptop.

---

## 4. What WSL2 changes on the laptop, and the controls

| Topic | What happens | Control in this set-up |
|---|---|---|
| **Admin rights** | enabling WSL2 (`wsl --install -d Ubuntu`) needs administrator rights once, and a restart | done once by IT; nothing else needs admin; the vLLM install in Ubuntu needs no `sudo` (a user-level Python venv) |
| **A second OS** | an Ubuntu file system lives in a virtual disk (`ext4.vhdx`) in the user's profile | used only for vLLM and the model cache; `wsl --unregister Ubuntu` deletes it completely |
| **Access to Windows files** | by default WSL2 mounts the Windows drives (`/mnt/c`, `/mnt/d`) with the user's own permissions | WSL2 runs as the logged-in user, with no more rights than that user. vLLM reads only its model cache; the customer data is read by the Windows notebook, not by WSL. The automount can be switched off in `/etc/wsl.conf` if required |
| **Network, inbound** | default NAT mode: WSL2 has a private virtual network; other machines on the LAN cannot reach it without an explicit port proxy | vLLM listens on `127.0.0.1` only, and requires the API key; no port proxy and no firewall rule are created |
| **Network, outbound** | WSL2 shares the laptop's connection | needed **once**: Ubuntu (Microsoft / Canonical), `astral.sh` and `pypi.org` for the Python packages, and `huggingface.co` for the model weights (about 6 GB). Then `VLLM_OFFLINE=1`: no download is possible. Outbound access can be restricted to these domains, or closed after the first run |
| **Telemetry** | vLLM and Hugging Face send usage statistics by default | switched off in `vllm/serve_wsl.sh` (`VLLM_NO_USAGE_STATS`, `DO_NOT_TRACK`, `HF_HUB_DISABLE_TELEMETRY`); offline mode after the first download |
| **Monitoring** | Linux processes inside WSL2 | Microsoft documents a Defender for Endpoint plug-in for WSL, and Intune / Group Policy settings to allow, restrict or block WSL |
| **Software pinned** | | vLLM `0.32.0` (released 2026-10-18), pinned in `vllm/install_wsl.sh`, a release at least two weeks old as the project's supply-chain rule requires; the model `Qwen/Qwen3-8B-AWQ` (Apache-2.0 licence) |
| **Secrets** | | `VLLM_API_KEY` is in `.env`, which is never committed. It is passed through the environment, never on a command line, and the notebook never prints it |
| **Customer data** | | stays in `D:\production\…\data\` on the Windows disk; it is never copied into WSL and never sent to the LLM as identifiers |

**What does not change:**
- no service is opened to the network;
- no customer data leaves the laptop;
- the engine sends nothing anywhere, and the final file is handed over by a person;
- the rules and the audit still decide.

---

## 5. Recommendation

1. **Now:** run the weekly batch with **Ollama** (notebook 03). It needs no new component, and it is already approved and installed.
2. **If the 8–14 h weekly run is too long:** approve **WSL2 + vLLM** on this one laptop, with the controls of §4. Then:
   - IT enables WSL2;
   - the first start downloads the packages and the model;
   - `VLLM_OFFLINE=1` is set from then on;
   - the speed is measured (step G2 of notebook 04) and compared with Ollama.
3. **To roll back:** `wsl --unregister Ubuntu` and disable the WSL features. Notebook 03 (Ollama) keeps working unchanged.

---

## 6. Email to the security team

> **Subject:** Request: enable WSL2 on one analytics laptop to run a local LLM server (vLLM)
>
> Hello,
>
> For the campaign recommendation engine (FSDR), we run a local LLM on the analytics laptop (NVIDIA RTX PRO 5000, 24 GB). It helps choose the action and the offers for about 100,000 customers each week. Everything runs on the laptop: the LLM server listens on 127.0.0.1 only, the engine refuses any non-local address, no MSISDN or other identifier is sent to the model, and the output is a file that we check and hand over ourselves.
>
> Today we use Ollama, which runs natively on Windows. A weekly run takes an estimated 8–14 hours. vLLM, an open-source LLM server, does the same work in an estimated 3–6 hours, because it processes many customers at once on the GPU. vLLM is only available for Linux, so on Windows it needs WSL2 (Windows Subsystem for Linux), Microsoft's supported way to run Linux software in a lightweight virtual machine.
>
> **What we would install**
> - The WSL2 feature and an Ubuntu distribution (`wsl --install -d Ubuntu`). This needs admin rights once, and a restart.
> - Inside Ubuntu, without root: a Python virtual environment with vLLM 0.32.0 (pinned version), installed from PyPI.
> - The model `Qwen/Qwen3-8B-AWQ` (Apache-2.0 licence, about 6 GB) from Hugging Face, once.
>
> **Controls**
> - The vLLM server listens on 127.0.0.1:8001 only and requires an API key. No port proxy, firewall rule or network exposure.
> - Usage statistics are off. After the first download, offline mode forbids any download.
> - Outbound access is needed once, to: Microsoft / Canonical (Ubuntu), astral.sh and pypi.org (packages), huggingface.co (model). It can be restricted to these domains, or closed afterwards.
> - The customer data stays on the Windows disk and is not copied into WSL. WSL2 runs with the user's own permissions, and the drive automount can be switched off if required.
> - Microsoft provides Defender for Endpoint coverage for WSL, and Intune / Group Policy settings to control it.
> - Rollback: `wsl --unregister Ubuntu` removes everything. The Ollama set-up keeps working.
>
> **Request:** approval to enable WSL2 on this one laptop (asset tag: [asset tag]) for this use, with the controls above, or the conditions you would require. A full comparison of Ollama and vLLM, with the details of every point above, is attached (`VLLM_VS_OLLAMA_SECURITY.md`).
>
> Thank you,
> [Your name]
> [Team]
