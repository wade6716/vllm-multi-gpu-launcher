# Tuning guide: vLLM high CPU load, diagnosis & optimization

This document is the deep background behind `launch_vllm.sh`. Read the
[README](../README.md) for usage; this guide explains *why* the knobs exist and
how to reason about them when your workload changes.

---

## 1. The symptom: load average 50+ (or 260) with idle-looking GPUs

A machine serving a quantized LLM showed a load average of **260** with six
`vllm serve` instances — far beyond any real computation. The same workload, at
under 10 concurrent requests per GPU, has no business pinning every core.

The load is **not** computation. It is scheduling pressure: thousands of threads
in the *runnable* state, spinning and being migrated between cores.

## 2. The three root causes, in order of impact

### 2.1 OpenMP / MKL spin-wait (thread pools sized to the whole machine)

PyTorch and MKL initialize their CPU thread pools from the **system** core
count. On a 192-core box, *each* of six processes tries to create ~192 threads.
Their default wait policy is **active spin** (`OMP_WAIT_POLICY=ACTIVE`): when a
thread has no work (e.g. the GPU is busy forwarding), it does not sleep — it
spins in a `while(true)` poll of a status flag.

Because inference alternates "CPU prepares batch → GPU forwards → CPU prepares"
hundreds of times per second, those threads spin during every microsecond the
GPU is busy. Linux then sees every core permanently runnable.

**Fix in the launcher:**

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

`VLLM_OMP_NUM_THREADS` / `VLLM_MKL_NUM_THREADS` in `.env`. Why *not* larger
even if you have 128 cores? See §4.

> `VLLM_CPU_OMP_THREADS_BOUND` is **not** the right knob here — it belongs to
> the vLLM *CPU* backend (`device="cpu"`) and is a no-op for GPU inference.

### 2.2 Tokenizer thread pool explosion

HuggingFace's Rust `tokenizers` spawns a Rayon thread pool per process, again
sized to the machine. Six instances → on the order of a thousand idle tokenizer
threads.

**Fix in the launcher:**

```bash
export TOKENIZERS_PARALLELISM=false
```

This disables *intra-request* parallelism (splitting one prompt across threads).
For online serving, a single prompt tokenizes in well under a millisecond, so
single-threaded tokenization costs nothing — but the thread churn it removes is
huge. It also avoids the classic "The current process just got forked" deadlock
warnings in multiprocess setups.

> Only when you batch-tokenize millions of *offline* texts in one process does
> `TOKENIZERS_PARALLELISM=true` buy you anything.

### 2.3 Core over-subscription (no affinity)

Without `taskset`, the kernel is free to migrate every thread of every instance
across *all* cores. Each migration flushes per-core caches and adds context
switches. Six instances on one die thrash each other's L3.

**Fix in the launcher:** pin each instance to a contiguous slice, computed as
`cores_per_gpu = nproc --all / gpu_count`.

## 3. Contribution breakdown (why "all three", not just one)

| Measure | Contribution | Why |
|---|---|---|
| `taskset -c` affinity | ★★★★★ | Divides physical territory; stops cross-core / cross-NUMA churn. |
| `TOKENIZERS_PARALLELISM=false` | ★★★★ | Removes a whole class of idle Rayon pools. |
| `OMP/MKL_NUM_THREADS=8` | ★★ | Caps math-library threads; matters less because heavy math runs on GPU, but stops the last over-subscription source. |

Each alone is insufficient:

- Only `taskset`, no thread caps → a single instance can still spin *inside*
  its own slice.
- Only thread caps, no `taskset` → processes still migrate across all cores and
  the tokenizer still spawns pools sized to the whole machine.

## 4. Why cap at 4–8 threads even when you have 128 cores

GPU-bound serving does almost no *large* CPU tensor work: the CPU-side ops are
small logits/sampling/padding ops. Splitting a tiny problem across 64 threads
costs more in wake-ups and barrier synchronisation than just computing it on 4.

Worse, more threads = more active-spin waiters = more power and memory-bus
contention. This is a "negative speedup" regime. **More cores does not mean
raise this number.**

The one real cost of `PASSIVE` / small thread count: a few *microseconds* of
thread-wake latency for CPU ops. For a GPU-bound model whose forward pass takes
milliseconds, that is invisible. You might measure a ~1–3% TTFT wobble only in a
single-request, zero-load microbenchmark — irrelevant in production.

## 5. Verification when the problem persists

If CPU stays pegged after these fixes, the remaining hot threads are doing *real*
work. Inspect per-thread CPU to see which category:

```bash
pidstat -u -t -p <vllm_pid> 1
```

- Many `omp*` / `torch*` threads at 100% → thread caps not applied (check the
  running process env, not just the shell).
- One main-thread / uvicorn thread pegged → Python GIL / event-loop / JSON
  serialisation contention; look at the frontend proxy and tokenizer paths.
- `python` workers pegged → true CPU-bound work (unusual for this setup).

## 6. Parameter tuning tables

### `VLLM_MAX_NUM_BATCHED_TOKENS`

With chunked prefill enabled, do **not** size this to your full context length.
The value is the per-*step* token budget.

| Scenario | Value | Notes |
|---|---|---|
| Balanced, default | **2048–4096** | Stable memory, smooth decode. |
| Low-latency interactive | 1024–2048 | Shortest step latency. |
| High-throughput offline | up to 8192 | Only with plenty of headroom. |
| (Anti-pattern) | 16384+ | Large activation buffers → OOM risk and decode jitter. |

Constraint: `max_num_batched_tokens >= max_num_seqs`.

### `VLLM_MAX_NUM_SEQS`

Per-GPU concurrent-sequence cap. Bump only while watching KV-cache memory. On a
24 GB card holding a large quantized model, 16–24 is a sensible band; FP8 KV
quantization (§8.3) lets you roughly double it.

## 7. Deployment appendices

### 7.1 Multi-socket / NUMA (bare metal)

On dual-socket machines the launcher's flat `taskset` split is not optimal —
you want each instance's cores on the *same* NUMA node as its GPU. Find the
topology:

```bash
nvidia-smi topo -m
lscpu
```

Then bind manually instead of letting the launcher split:

```bash
# GPU 0 on NUMA node 0
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 vllm serve ...
```

On a **single** socket (one CPU), skip all NUMA tuning entirely — it buys you
nothing; the flat core split is already correct.

### 7.2 Containers (Docker / K8s)

Container runtimes add a *core-visibility trap*: the guest still sees the
host's core count, while `--cpus=N` applies a CFS quota. CFS throttling stalls
the process for hundreds of milliseconds when it bursts past quota, causing TTFT
spikes. Use hard affinity, not soft quota:

```yaml
# ❌ soft quota — can throttle
# --cpus=16
# ✅ hard affinity — no CFS stalls
--cpuset-cpus="0-7"
```

Plus:

```bash
--ipc=host                  # vLLM multiprocess IPC needs lots of shared memory
--ulimit memlock=-1:-1      # allow pinned (page-locked) host memory for DMA
-e TOKENIZERS_PARALLELISM=false
-e OMP_NUM_THREADS=4
-e MKL_NUM_THREADS=4
```

Prefer **one container per GPU** over six processes in one big container:
better fault isolation, rolling restarts, and native per-container CPU sets.

Startup is faster in containers when you persist the compile caches (otherwise
Triton/Torch re-JITs every launch):

```bash
-v /root/.cache/vllm:/root/.cache/vllm \
-v /root/.cache/triton:/root/.cache/triton \
-v /root/.cache/torch:/root/.cache/torch \
-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
```

## 8. Advanced topics (when you outgrow one-instance-per-GPU)

### 8.1 Prefix-aware routing (cheapest big win)

A naive `least_conn`/round-robin LB destroys your `--enable-prefix-caching`
hit rate: turn 2 of a conversation may land on a different GPU and redo the
whole prefix. Route by prefix hash (SGLang Router / radix-tree router) to pin
same-prefix requests to the same instance — Prefill cost can collapse toward
zero.

### 8.2 Speculative decoding

A tiny draft model (or EAGLE/Medusa heads) guesses several tokens that the main
model verifies in one forward pass. No accuracy loss; ~1.8–2.5x tokens/s on
decode-bound models, which shortens per-request occupancy and lifts concurrency.

### 8.3 KV-cache quantization

`--kv-cache-dtype fp8` halves KV memory, letting you double `max_num_seqs`.
Caveats:

- **RTX 3090 (Ampere, CC 8.6) does NOT support native FP8** — it will error or
  fall back to an on-GPU dequant that can *slow* decoding. Prefer the default
  FP16 KV cache or `int8` on that generation.
- FP8 KV cache is a *memory* win; on hardware without FP8 Tensor Cores there is
  no compute speedup.

### 8.4 Prefill/Decode disaggregation — only with NVLink

Splitting Prefill and Decode into separate GPU pools works by shipping KV cache
between cards. A 40k-token KV cache is on the order of **1.5–2.5 GB**:

| Interconnect | KV transfer | Verdict |
|---|---|---|
| NVLink (300–900 GB/s) | ~2–5 ms | ✅ PD separation pays off (1.5–3x). |
| PCIe 4.0 x16 (~20–25 GB/s real) | ~100 ms+ and bus contention | ❌ Transfer eats the Prefill savings; keep six independent instances. |

**On a no-NVLink single box, do not do PD separation.** Stay with the
one-instance-per-GPU design and invest in §8.1–§8.3 instead.

## 9. Quantization gotcha: `--quantization gptq`

Explicitly setting `--quantization gptq` can force the legacy AutoGPTQ kernel
and cut throughput ~6x (measured ~600 → ~100 tokens/s) versus letting vLLM
auto-detect `gptq_marlin`. The launcher therefore defaults `VLLM_QUANTIZATION`
to **empty** (auto-detect). If you must set it, prefer `gptq_marlin`.

Check the startup log for the effective kernel:

```
Using Marlin linear kernel          # fast path
Loading format: gptq_marlin         # fast path
```

## 10. Terminology

| Term | Meaning |
|---|---|
| **Instance** | One `vllm serve` process bound to one GPU. |
| **Core pinning / affinity** | Constraining a process to a fixed set of CPU cores (`taskset -c`, `--cpuset-cpus`). |
| **Spin-wait** | A thread polling a flag in a tight loop instead of sleeping; low latency but burns CPU. |
| **NUMA** | Non-Uniform Memory Access; memory local to one CPU socket is faster than remote. |
| **KV cache** | GPU memory holding keys/values so tokens are not recomputed each step. |
| **Prefix caching** | Reusing KV for a shared prompt prefix across requests. |
| **Chunked prefill** | Breaking a long prompt into chunks interleaved with decode steps. |
| **Continuous batching** | vLLM's scheduling that slots new sequence into any finished slot each iteration. |
| **TTFT / ITL** | Time-to-first-token / inter-token latency. |
| **CFS quota** | The kernel's soft CPU-time limiter (`--cpus=N`); can throttle sudden bursts. |

---

## 中文要点

- 负载飙高（260/50+）本质是**线程空转与跨核争抢**，不是真计算；三件套缺一不可：
  `taskset` 绑核（★★★★★）+ `TOKENIZERS_PARALLELISM=false`（★★★★）+ `OMP/MKL=8`
  （★★）。
- 核心数再多，`OMP/MKL` 也保持在 4–8，属于「负加速」区间，越大越慢。
- `--max-num-batched-tokens` 在开启 chunked-prefill 后取 2048–4096 即可，无需等于
  上下文长度；16384+ 有 OOM 与延迟抖动风险。
- 显式 `--quantization gptq` 会锁死慢内核（吞吐 ~600→100），应留空自动走
  `gptq_marlin`。
- 单 CPU 机器无需 NUMA 调优；容器用 `--cpuset-cpus` 硬绑核而不要 `--cpus` 软限流，
  并加 `--ipc=host`、`--ulimit memlock=-1:-1`。
- RTX 3090（Ampere）不支持原生 FP8 KV；无 NVLink 的单机不建议做 PD 分离，坚持
  「每卡一实例」并转向前缀路由/投机采样/FP8(受支持时) 提效。