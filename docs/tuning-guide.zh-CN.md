# 调优指南：vLLM 高 CPU 负载的排查与优化

[English](tuning-guide.md) · [简体中文](tuning-guide.zh-CN.md)

本文是 `launch_vllm.sh` 背后的深层原理。使用方法见 [README](../README.md)；本指南
解释这些开关*为什么*存在，以及当你的负载变化时该如何推理权衡。

---

## 1. 症状：GPU 看着闲，负载却 50+（甚至 260）

一台部署量化 LLM 的机器，跑着六个 `vllm serve` 实例，系统负载高达 **260**——远超
任何真实计算量。单卡不到 10 并发请求的工作量，根本不该把每个核心都打满。

这个负载**不是**计算造成的，而是调度压力：成千上万个 *runnable* 状态的线程在空转
并被内核在核心间反复迁移。

## 2. 三大根因（按影响排序）

### 2.1 OpenMP / MKL 自旋等待（线程池按整机核心数开）

PyTorch 和 MKL 按**系统**核心数初始化 CPU 线程池。在 192 核的机器上，六个进程每个
都会试图创建约 192 个线程。它们的默认等待策略是**主动自旋**（`OMP_WAIT_POLICY=ACTIVE`）：
线程没活干时（例如 GPU 正在 forward）也不休眠，而是 `while(true)` 轮询状态标志。

因为推理在「CPU 准备 batch → GPU forward → CPU 准备」之间每秒切换上百次，这些线程
就在 GPU 忙碌的每一个微秒里空转。Linux 于是看到所有核心永远处于可运行状态。

**启动器里的修法：**

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

对应 `.env` 里的 `VLLM_OMP_NUM_THREADS` / `VLLM_MKL_NUM_THREADS`。为什么即使有 128
核也不调大？见 §4。

> `VLLM_CPU_OMP_THREADS_BOUND` 在这里是**错误**的开关——它属于 vLLM *CPU* 后端
> （`device="cpu"`），对 GPU 推理无效。

### 2.2 Tokenizer 线程池爆炸

HuggingFace 的 Rust `tokenizers` 每个进程开一套 Rayon 线程池，同样按整机核心数。
六个实例 → 上千个空闲的分词线程。

**启动器里的修法：**

```bash
export TOKENIZERS_PARALLELISM=false
```

它关闭的是*单请求内*并行（把一条 prompt 拆到多线程处理）。在线服务里，单条 prompt
分词不到 1 毫秒，单线程分词几乎没有成本——但去掉的线程抖动却是巨大的。它还能避免
多进程场景下经典的 “The current process just got forked” 死锁告警。

> 只有在一个进程里批量分词几百万条*离线*文本时，`TOKENIZERS_PARALLELISM=true` 才有
> 意义。

### 2.3 核心过度订阅（无亲和性）

没有 `taskset` 时，内核可以随意把每个实例的每个线程迁移到*所有*核心。每次迁移都会
冲掉每核缓存并增加上下文切换。六个实例在同一颗 die 上互相践踏彼此的 L3。

**启动器里的修法：** 把每个实例钉在连续的一段核心上，按
`cores_per_gpu = nproc --all / gpu_count` 计算。

## 3. 贡献度拆解（为什么「三件套」缺一不可）

| 手段 | 贡献 | 原因 |
|---|---|---|
| `taskset -c` 亲和性 | ★★★★★ | 划分物理地盘，消除跨核/跨 NUMA 抖动。 |
| `TOKENIZERS_PARALLELISM=false` | ★★★★ | 去掉一整类空闲的 Rayon 池。 |
| `OMP/MKL_NUM_THREADS=8` | ★★ | 限数学库线程；因重算力在 GPU 上所以贡献次之，但堵住最后一处过度订阅。 |

单独任何一个都不够：

- 只 `taskset` 不限线程 → 单实例仍会在自己那段核心里**内部**自旋。
- 只限线程不 `taskset` → 进程仍在所有核心间迁移，tokenizer 仍按整机核心数开池。

## 4. 为什么有 128 核也只能设 4–8 线程

GPU 为主的推理几乎没有*大规模* CPU 张量计算：CPU 侧只是些 logits/采样/填充的小算子。
把一个小问题拆给 64 个线程，其唤醒与 barrier 同步开销反而比 4 个线程直接算完还大。

更糟的是，线程越多 = 主动自旋的等待者越多 = 功耗与内存总线争用越多。这是「负加速」
区间。**核心越多，越不能调大这个数。**

`PASSIVE` / 小线程数的唯一真实代价：CPU 算子几*微秒*的线程唤醒延迟。对于 forward
耗时毫秒级的 GPU-bound 模型，这完全不可感知。只有在单请求、零负载的微基准里，才可能
测出约 1–3% 的 TTFT 抖动——生产环境无关紧要。

## 5. 修完仍打满时如何验证

如果这些修法之后 CPU 仍被打满，剩下的热线程就是在做*真实*工作。看每线程 CPU 判断
类别：

```bash
pidstat -u -t -p <vllm_pid> 1
```

- 大量 `omp*` / `torch*` 线程 100% → 线程上限没生效（检查运行中进程的环境变量，而非
  只看 shell）。
- 单个主线程 / uvicorn 线程打满 → Python GIL / 事件循环 / JSON 序列化争用；去查前端
  代理与分词路径。
- `python` worker 打满 → 真正的 CPU-bound 计算（此场景少见）。

## 6. 参数调优表

### `VLLM_MAX_NUM_BATCHED_TOKENS`

开启 chunked prefill 后，**不要**把它设成完整上下文长度。这个值是每*步*的 token
预算。

| 场景 | 取值 | 说明 |
|---|---|---|
| 通用默认 | **2048–4096** | 显存稳定、decode 平滑。 |
| 低延迟交互 | 1024–2048 | 单步延迟最短。 |
| 高吞吐离线 | 最高 8192 | 仅在显存余量充足时。 |
| （反模式） | 16384+ | 大激活值缓冲区 → OOM 风险与 decode 抖动。 |

约束：`max_num_batched_tokens >= max_num_seqs`。

### `VLLM_MAX_NUM_SEQS`

单卡并发序列上限。只盯着 KV 缓存显存往上调。在承载大模型的 24 GB 卡上，16–24 是
合理区间；FP8 KV 量化（§8.3）可让你大约翻倍。

## 7. 部署附录

### 7.1 双路 / NUMA（裸机）

双路机器上启动器的平均 `taskset` 切分不是最优——你希望每个实例的核心与它的 GPU 在
**同一个** NUMA 节点上。先看拓扑：

```bash
nvidia-smi topo -m
lscpu
```

然后手动绑定，而不是让启动器平均切：

```bash
# GPU 0 在 NUMA 节点 0 上
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 vllm serve ...
```

**单** CPU（单 socket）时完全跳过 NUMA 调优——它带不来任何收益，平均核心切分本就
正确。

### 7.2 容器（Docker / K8s）

容器运行时额外带来一个*核心可见性陷阱*：guest 仍看到宿主机的核心数，而 `--cpus=N`
施加的是 CFS 配额。突发请求打满配额时 CFS 节流会把进程挂起几百毫秒，导致 TTFT 毛刺。
要用硬亲和，不要软配额：

```yaml
# ❌ 软配额——可能被节流
# --cpus=16
# ✅ 硬亲和——无 CFS 停顿
--cpuset-cpus="0-7"
```

再加上：

```bash
--ipc=host                  # vLLM 多进程 IPC 需要大量共享内存
--ulimit memlock=-1:-1      # 允许锁页（page-locked）宿主机内存用于 DMA
-e TOKENIZERS_PARALLELISM=false
-e OMP_NUM_THREADS=4
-e MKL_NUM_THREADS=4
```

优先**每卡一个容器**，而不是一个大容器里跑六个进程：故障隔离更好、可滚动重启、还能
用容器原生的 CPU 集合。

容器里启动更快的做法是持久化编译缓存（否则 Triton/Torch 每次启动都重新 JIT）：

```bash
-v /root/.cache/vllm:/root/.cache/vllm \
-v /root/.cache/triton:/root/.cache/triton \
-v /root/.cache/torch:/root/.cache/torch \
-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
```

## 8. 进阶话题（当「每卡一实例」不够用时）

### 8.1 前缀感知路由（最便宜的收益）

朴素的 `least_conn`/轮询 LB 会毁掉你的 `--enable-prefix-caching` 命中率：对话第二轮
可能落到另一张 GPU，于是重算整个前缀。按前缀哈希路由（SGLang Router / radix-tree
router），把相同前缀的请求钉在同一实例上——Prefill 成本可趋向于零。

### 8.2 投机采样 / 投机解码

一个小 draft 模型（或 EAGLE/Medusa 头）先猜几个 token，主模型一次 forward 批量验证。
无精度损失；decode-bound 模型约 1.8–2.5 倍 tokens/s，缩短单请求占卡时间、抬升并发。

### 8.3 KV 缓存量化

`--kv-cache-dtype fp8` 让 KV 显存减半，从而把 `max_num_seqs` 翻倍。注意事项：

- **RTX 3090（Ampere，CC 8.6）不支持原生 FP8**——会报错或退化为片上反量化，反而
  *拖慢* decode。这一代建议保持默认 FP16 KV 缓存或用 `int8`。
- FP8 KV 缓存是*显存*收益；在无 FP8 Tensor Core 的硬件上没有任何算力加速。

### 8.4 Prefill/Decode 分离——只在有 NVLink 时

把 Prefill 和 Decode 拆到不同 GPU 池，靠的是在卡间搬运 KV 缓存。40k token 的 KV 缓存
约 **1.5–2.5 GB**：

| 互联 | KV 搬运 | 结论 |
|---|---|---|
| NVLink（300–900 GB/s） | ~2–5 ms | ✅ PD 分离划算（1.5–3x）。 |
| PCIe 4.0 x16（实际 ~20–25 GB/s） | ~100 ms+ 且总线争用 | ❌ 搬运吃掉了 Prefill 省下的时间；保持六个独立实例。 |

**没有 NVLink 的单机，不要做 PD 分离。** 坚持「每卡一实例」，把精力投入到 §8.1–§8.3。

## 9. 量化陷阱：`--quantization gptq`

显式设置 `--quantization gptq` 可能强制启用旧的 AutoGPTQ 内核，吞吐约降 6 倍（实测
~600 → ~100 tokens/s），而让 vLLM 自动探测 `gptq_marlin` 则很快。因此启动器把
`VLLM_QUANTIZATION` 默认设为**空**（自动探测）。若必须显式指定，请优先 `gptq_marlin`。

看启动日志确认生效的内核：

```
Using Marlin linear kernel          # 快速路径
Loading format: gptq_marlin         # 快速路径
```

## 10. 术语表

| 术语 | 含义 |
|---|---|
| **实例 Instance** | 绑定到一张 GPU 的单个 `vllm serve` 进程。 |
| **核心绑定 / 亲和性** | 把进程约束到固定的一组 CPU 核心（`taskset -c`、`--cpuset-cpus`）。 |
| **自旋等待 Spin-wait** | 线程不睡眠而在紧密循环里轮询标志；延迟低但烧 CPU。 |
| **NUMA** | 非统一内存访问；靠近某个 CPU socket 的内存比远端更快。 |
| **KV 缓存** | 存放 key/value 的显存，避免每步重算 token。 |
| **前缀缓存** | 跨请求复用共享 prompt 前缀的 KV。 |
| **分块预填 Chunked prefill** | 把长 prompt 切成块，与 decode 步骤交替执行。 |
| **连续批处理 Continuous batching** | vLLM 的调度方式：每次迭代把新序列塞进任意空出的槽位。 |
| **TTFT / ITL** | 首字延迟 / 字间延迟。 |
| **CFS 配额** | 内核的软 CPU 时间片限制（`--cpus=N`）；可能对突发请求节流。 |