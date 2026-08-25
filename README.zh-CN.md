# vllm-multi-gpu-launcher

[English](README.md) · [简体中文](README.zh-CN.md)

> **每张 GPU 一个 vLLM 服务 —— CPU 绑核 + 线程收敛 —— 让高并发模型集群不再被
> 系统负载暴涨拖垮。**

**vllm-multi-gpu-launcher** 是一个零依赖的 Bash 启动器，用于在**单机多卡**环境下
用 [vLLM](https://github.com/vllm-project/vllm) 部署大语言模型。它为每张 GPU 启动
一个 OpenAI 兼容的 `vllm serve` 实例，把每个实例钉在独立的一段 CPU 核心上，并限制
底层数学库与分词器线程——将系统负载从几百降到低两位数，同时吞吐不降反升。

> **TL;DR** — 一台机器、六张 GPU、六个 `vllm serve` 进程，系统负载高达 **260**，
> 再怎么加核心也压不下去。原因不是真计算：每个进程都按**整机核心数**开
> OpenMP / MKL / tokenizer 线程池，这些线程在 GPU 忙碌时空转、跨核乱窜。把每个
> 实例钉在专属核心段并收敛底层线程后，同样的负载降到 **~13**，吞吐反而更好。

---

## 作用

`launch_vllm.sh`：

1. 通过 `nvidia-smi` **自动探测 GPU 数量**，每卡起一个实例；
2. 用 `taskset -c` 把每个实例**钉在独立的一段 CPU 核心上**，避免跨核抢跑 / 跨 NUMA
   抖动；
3. 限制 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 并关闭 `TOKENIZERS_PARALLELISM`——
   这才是负载暴涨的真正根源；
4. 启动前**端口预检**、启动后**逐实例轮询 `/health`**，失败即非零退出，绝不留下
   半启动的集群；
5. **持久化 PID**，配合 `stop_vllm.sh` 干净停机。

## 环境要求

- NVIDIA 驱动 + `nvidia-smi` 在 `PATH` 中。
- 当前环境有可用的 `vllm` 命令（或 `VENV_DIR` 指定的虚拟环境）。
- `taskset`（来自 `util-linux`）以及 `ss` 或 `netstat`（可选但推荐）。
- `curl`（用于健康检查）。

## 快速开始

```bash
git clone <this-repo>
cd vllm-multi-gpu-launcher

cp .env.example .env
# 编辑 .env：填 MODEL_PATH 与 VLLM_API_KEY

./launch_vllm.sh
```

`MODEL_PATH` 与 `VLLM_API_KEY` 为**必填**——启动器缺失其一即拒绝运行，因此绝不会
意外暴露一个无鉴权的服务。其余参数都有合理默认值（见下文）。

```bash
./stop_vllm.sh        # 停止全部实例
./stop_vllm.sh 0 2    # 只停 GPU 0 与 GPU 2
```

## 为什么 CPU 会飙高

推理主干在 GPU 上，但周边「胶水」逻辑不是。在多核机器上，三件事相乘出上千个忙碌
线程：

| 原因 | 具体表现 | 脚本里的修法 |
|---|---|---|
| **OpenMP / MKL 自旋等待** | PyTorch 和 MKL 按**整机核心数**开线程池，GPU 忙碌时这些线程 `while(true)` 空转。 | 把 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 压到 4–8。 |
| **Tokenizer 线程池** | FastTokenizer 每个进程再开一套 Rayon 池，同样按整机核心数。 | `TOKENIZERS_PARALLELISM=false`。 |
| **核心过度订阅** | Linux 让每个线程在所有核心间自由迁移：缓存失效 + 上下文切换风暴。 | `taskset -c` 给每个实例分一段专属连续核心。 |

**绑核 + 收敛线程**这套组合拳，就是把负载从几百降到健康低值的关键。完整的排查与
权衡分析见 [`docs/tuning-guide.zh-CN.md`](docs/tuning-guide.zh-CN.md)。

## 配置参考

所有开关都是环境变量。默认值以**加粗**标出；可以通过 `export` 或在 `.env` 文件中
设置（shell `export` 优先级更高）。

### 必填

| 变量 | 作用 | 默认值 |
|---|---|---|
| `MODEL_PATH` | 模型权重路径或 HF 仓库 id，传给 `--model`。 | *(必填——为空即退出)* |
| `VLLM_API_KEY` | 用作 `--api-key`。不能包含空白字符。 | *(必填——为空即退出)* |

### 服务

| 变量 | vLLM flag / 行为 | 默认值 | 为什么 |
|---|---|---|---|
| `VLLM_HOST` | `--host` | **`127.0.0.1`** | 安全默认；只有需要远程访问时才改 `0.0.0.0`（仍强制 key）。 |
| `VLLM_BASE_PORT` | `--port`（实例 *i* → +*i*） | **`38000`** | 每卡一个端口，便于 LB 与健康检查定位每个实例。 |
| `VLLM_SERVED_MODEL_NAMES` | `--served-model-name`（空格分隔的多别名） | *(模型目录 basename)* | 去掉写死的 `gpt-4o`/`gpt-5` 伪装，默认即真实模型名。 |

### 显存与批处理

| 变量 | vLLM flag | 默认值 | 为什么 |
|---|---|---|---|
| `VLLM_GPU_MEMORY_UTILIZATION` | `--gpu-memory-utilization` | **`0.90`** | 留显存余量，防长上下文 OOM。 |
| `VLLM_MAX_NUM_SEQS` | `--max-num-seqs` | **`16`** | 单卡并发序列上限，约束 KV 缓存与调度/CPU 争用。 |
| `VLLM_MAX_MODEL_LEN` | `--max-model-len` | **`40960`** | 设为实际最长输入+输出，避免 `auto` 过度预留。 |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` | **`4096`** | 单步 token 预算；配合 chunked prefill 取 2048–4096 防激活值尖峰与 decode 抖动。 |
| `VLLM_QUANTIZATION` | `--quantization` | *(空 = 自动探测)* | 自动走快内核（`gptq_marlin`）；显式 `gptq` 可能锁死慢内核——见调优指南。 |

### CPU 线程控制（高负载的核心修法）

| 变量 | 行为 | 默认值 | 为什么 |
|---|---|---|---|
| `VLLM_OMP_NUM_THREADS` | `export OMP_NUM_THREADS` | **`8`** | 限制每进程 OpenMP 算子线程，杜绝整机过度订阅。 |
| `VLLM_MKL_NUM_THREADS` | `export MKL_NUM_THREADS` | **`8`** | 同理限制每进程 MKL 线程。 |
| `VLLM_TOKENIZERS_PARALLELISM` | `export TOKENIZERS_PARALLELISM` | **`false`** | 关闭分词器隐式线程池。 |

> ⚠️ GPU 推理**不要**用 `VLLM_CPU_OMP_THREADS_BOUND`——那是 vLLM *CPU* 后端的
> 变量，在这里是无效配置。

### 功能开关

| 变量 | 行为 | 默认值 |
|---|---|---|
| `VLLM_ENABLE_PREFIX_CACHING` | `--enable-prefix-caching` | **`1`**（开） |
| `VLLM_ENABLE_CHUNKED_PREFILL` | `--enable-chunked-prefill` | **`1`**（开） |

### 模型专属参数（默认关闭）

| 变量 | 行为 | 默认值 |
|---|---|---|
| `VLLM_REASONING_PARSER` | `--reasoning-parser` | *(未设置 = 不拼参数)* |
| `VLLM_TOOL_CALL_PARSER` | `--tool-call-parser` | *(未设置 = 不拼参数)* |
| `VLLM_ENABLE_AUTO_TOOL_CHOICE` | `--enable-auto-tool-choice` | **`0`**（关） |

Qwen3 工具调用部署可设：`VLLM_REASONING_PARSER=qwen3`、`VLLM_TOOL_CALL_PARSER=hermes`、
`VLLM_ENABLE_AUTO_TOOL_CHOICE=1`。

### 路径与健康检查

| 变量 | 作用 | 默认值 |
|---|---|---|
| `VLLM_LOG_DIR` | 每卡日志文件（`vllm_gpu_<id>.log`） | **`./vllm_logs`** |
| `VLLM_RUN_DIR` | 每卡 PID 文件（供 `stop_vllm.sh`） | **`./vllm_run`** |
| `VLLM_HEALTH_TIMEOUT` | `/health` 轮询超时（秒） | **`180`** |
| `VENV_DIR` | 可选的虚拟环境激活路径 | **`./.venv`** |

## 运维

```bash
./launch_vllm.sh          # 每卡起一个实例，等待全部 /health 通过
./stop_vllm.sh            # 停止全部实例（优雅 → SIGKILL）
tail -f vllm_logs/vllm_gpu_0.log
curl http://127.0.0.1:38000/health
```

任一实例未能在 `VLLM_HEALTH_TIMEOUT` 内通过健康检查时，启动器以非零码退出，因此
可以安全地接入你的部署流水线。

## 安全

- `VLLM_API_KEY` **必填**；脚本绝不无鉴权启动。
- `VLLM_HOST` 默认 `127.0.0.1`。只有确实需要远程访问时才设 `0.0.0.0`——此时依然
  强制要求 API key。
- 真实的 `.env`（含你的 key）已 git 忽略，只提交 `.env.example`。

## 许可证

[MIT](LICENSE)。

## 延伸阅读

- [docs/tuning-guide.zh-CN.md](docs/tuning-guide.zh-CN.md) — 中文版调优指南：完整的
  `260 → 13` 排查、参数调优表、NUMA / Docker / K8s 附录与术语表。
- [docs/tuning-guide.md](docs/tuning-guide.md) — English version.