# vllm-multi-gpu-launcher

[English](README.md) · [简体中文](README.zh-CN.md)

> **One vLLM server per GPU — CPU-pinned and thread-capped — to run a
> high-concurrency model fleet without the load-average explosion.**

**vllm-multi-gpu-launcher** is a zero-dependency Bash launcher for serving large
language models with [vLLM](https://github.com/vllm-project/vllm) across
multiple GPUs on a single machine. It starts one OpenAI-compatible
`vllm serve` instance per GPU, pins each instance to its own slice of CPU cores,
and caps the low-level math-library and tokenizer threads — collapsing the
system load average from hundreds to a low double-digit number while keeping
(and usually improving) throughput.

> **TL;DR** — One machine, six GPUs, six `vllm serve` processes, and a system
> load average of **260** that wouldn't go away no matter how many cores you
> threw at it. The cause was not real computation: every process spawned
> OpenMP / MKL / tokenizer thread pools sized to the *whole* machine, and those
> threads spin-waited and thrashed across every core. After pinning each
> instance to a dedicated slice of cores and capping the low-level threads, the
> same workload settled at a load average of **~13** with *better* throughput.

---

## What it does

`launch_vllm.sh`:

1. **Auto-detects GPUs** via `nvidia-smi` and starts one instance per GPU.
2. **Pins each instance to a contiguous slice of CPU cores** (`taskset -c`),
   so instances never fight over the same cores or hop across NUMA nodes.
3. **Caps low-level math library threads** (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`)
   and **disables the tokenizer's implicit thread pool**
   (`TOKENIZERS_PARALLELISM=false`) — the three things that actually caused the
   load explosion.
4. **Checks ports before launch** and **polls `/health` after launch**, failing
   loudly instead of silently leaving a half-started fleet.
5. **Persists PIDs** so `stop_vllm.sh` can tear the fleet down cleanly.

## Requirements

- NVIDIA driver + `nvidia-smi` on `PATH`.
- A working `vllm` executable in the current environment (or a venv in
  `VENV_DIR`).
- `taskset` (from `util-linux`) and `ss` or `netstat` (optional but
  recommended).
- `curl` for the health check.

## Quick start

```bash
git clone <this-repo>
cd vllm-multi-gpu-launcher

cp .env.example .env
# edit .env: set MODEL_PATH and VLLM_API_KEY

./launch_vllm.sh
```

`MODEL_PATH` and `VLLM_API_KEY` are **required** — the launcher refuses to run
without them, so you can never accidentally expose an unauthenticated server.
Everything else has a sane default (see below).

```bash
./stop_vllm.sh        # stop everything
./stop_vllm.sh 0 2    # stop only GPU 0 and GPU 2
```

## How it works (why the CPU goes high)

Inference itself is GPU-bound, but the *glue* around it is not. On a box with
many cores, three things multiply into thousands of busy threads:

| Cause | What happens | Fix in this script |
|---|---|---|
| **OpenMP / MKL spin-wait** | PyTorch and MKL create a thread pool sized to the *system* core count, and those threads `while(true)`-spin while the GPU is busy. | Cap `OMP_NUM_THREADS` / `MKL_NUM_THREADS` to 4–8. |
| **Tokenizer thread pool** | FastTokenizer spawns a Rayon pool per process, again sized to the whole machine. | `TOKENIZERS_PARALLELISM=false`. |
| **Core over-subscription** | Linux freely migrates every thread across all cores: cache thrash + context-switch storm. | `taskset -c` to give each instance its own contiguous slice. |

The combination — **pin the cores + cap the threads** — is what collapses load
from hundreds to a healthy low number. See [`docs/tuning-guide.md`](docs/tuning-guide.md)
for the full investigation and the trade-offs.

## Configuration reference

Every knob is an environment variable. Defaults are shown in **bold**; values
can be set via `export` or in a `.env` file (shell `export` wins).

### Required

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_PATH` | Model weights path or HF repo id, passed to `--model`. | *(required — script exits if empty)* |
| `VLLM_API_KEY` | API key for `--api-key`. Must not contain whitespace. | *(required — script exits if empty)* |

### Server

| Variable | vLLM flag / effect | Default | Why |
|---|---|---|---|
| `VLLM_HOST` | `--host` | **`127.0.0.1`** | Safe default; set `0.0.0.0` only for remote access (key still enforced). |
| `VLLM_BASE_PORT` | `--port` (instance *i* → +*i*) | **`38000`** | One port per GPU so a load balancer and the health check can address each instance. |
| `VLLM_SERVED_MODEL_NAMES` | `--served-model-name` (space-separated aliases) | *(model dir basename)* | Removes the hard-coded `gpt-4o`/`gpt-5` masquerade; default tells the truth. |

### Memory & batching

| Variable | vLLM flag | Default | Why |
|---|---|---|---|
| `VLLM_GPU_MEMORY_UTILIZATION` | `--gpu-memory-utilization` | **`0.90`** | Leave headroom so long contexts don't OOM. |
| `VLLM_MAX_NUM_SEQS` | `--max-num-seqs` | **`16`** | Per-GPU concurrent-sequence cap; bounds KV cache and scheduler/CPU churn. |
| `VLLM_MAX_MODEL_LEN` | `--max-model-len` | **`40960`** | Set to your real longest-in+out; avoids `auto` over-reserving memory. |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` | **`4096`** | Tokens per forward step. With chunked prefill, 2048–4096 avoids activation-memory spikes and decode jitter. |
| `VLLM_QUANTIZATION` | `--quantization` | *(empty = auto-detect)* | Auto picks the fast kernel (`gptq_marlin`). Setting plain `gptq` can force the slow legacy kernel — see the guide. |

### CPU thread control (the high-load fix)

| Variable | Effect | Default | Why |
|---|---|---|---|
| `VLLM_OMP_NUM_THREADS` | `export OMP_NUM_THREADS` | **`8`** | Caps OpenMP operator threads per process; stops whole-machine over-subscription. |
| `VLLM_MKL_NUM_THREADS` | `export MKL_NUM_THREADS` | **`8`** | Caps MKL threads per process for the same reason. |
| `VLLM_TOKENIZERS_PARALLELISM` | `export TOKENIZERS_PARALLELISM` | **`false`** | Disables the tokenizer's implicit thread pool. |

> ⚠️ Do **not** add `VLLM_CPU_OMP_THREADS_BOUND` for GPU serving — that variable
> only affects the vLLM *CPU* backend and is a no-op here.

### Features

| Variable | Effect | Default |
|---|---|---|
| `VLLM_ENABLE_PREFIX_CACHING` | `--enable-prefix-caching` | **`1`** (on) |
| `VLLM_ENABLE_CHUNKED_PREFILL` | `--enable-chunked-prefill` | **`1`** (on) |

### Model-specific flags (off by default)

| Variable | Effect | Default |
|---|---|---|
| `VLLM_REASONING_PARSER` | `--reasoning-parser` | *(unset = omitted)* |
| `VLLM_TOOL_CALL_PARSER` | `--tool-call-parser` | *(unset = omitted)* |
| `VLLM_ENABLE_AUTO_TOOL_CHOICE` | `--enable-auto-tool-choice` | **`0`** (off) |

For a Qwen3 tool-calling deployment, set `VLLM_REASONING_PARSER=qwen3`,
`VLLM_TOOL_CALL_PARSER=hermes`, `VLLM_ENABLE_AUTO_TOOL_CHOICE=1`.

### Paths & health

| Variable | Purpose | Default |
|---|---|---|
| `VLLM_LOG_DIR` | Per-GPU log files (`vllm_gpu_<id>.log`) | **`./vllm_logs`** |
| `VLLM_RUN_DIR` | Per-GPU PID files (for `stop_vllm.sh`) | **`./vllm_run`** |
| `VLLM_HEALTH_TIMEOUT` | `/health` polling timeout (s) | **`180`** |
| `VENV_DIR` | Optional virtualenv to activate | **`./.venv`** |

## Operations

```bash
./launch_vllm.sh          # start one instance per GPU, wait for every /health
./stop_vllm.sh            # stop all instances (graceful → SIGKILL)
tail -f vllm_logs/vllm_gpu_0.log
curl http://127.0.0.1:38000/health
```

The launcher exits non-zero if any instance fails to become healthy within
`VLLM_HEALTH_TIMEOUT`, so it is safe to chain into your deploy pipeline.

## Security

- `VLLM_API_KEY` is **mandatory**; the script never starts unauthenticated.
- `VLLM_HOST` defaults to `127.0.0.1`. Set it to `0.0.0.0` only when you really
  need remote access — an API key is still required in that case.
- The real `.env` (containing your key) is git-ignored; only `.env.example` is
  committed.

## License

[MIT](LICENSE).

## Further reading

- [docs/tuning-guide.md](docs/tuning-guide.md) — (English) the full `260 → 13`
  load investigation, parameter tuning tables, NUMA / Docker / K8s appendices,
  and a terminology glossary.
- [docs/tuning-guide.zh-CN.md](docs/tuning-guide.zh-CN.md) — 中文版调优指南。