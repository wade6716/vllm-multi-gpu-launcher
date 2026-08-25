#!/usr/bin/env bash
# =============================================================================
# launch_vllm.sh — Launch one vLLM instance per GPU, with CPU core pinning
#
# What it solves:
#   Running several `vllm serve` processes on one machine without isolation
#   makes the CPU load average explode (often 200+) because every process
#   spawns OpenMP / MKL / tokenizer threads sized to the *whole* machine and
#   they all spin-wait and thrash across cores. This launcher pins each
#   instance to a dedicated slice of CPU cores, caps the low-level math
#   library threads, and disables the tokenizer's implicit thread pool.
#
#   The result: load average drops from hundreds to a small double digit
#   number while inference throughput actually improves.
#
# Configuration:
#   All knobs are environment variables with sane defaults (see .env.example).
#   `MODEL_PATH` and `VLLM_API_KEY` are REQUIRED — the script refuses to run
#   without them.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# 0. Load .env if present (KEY=VALUE lines; quote values that contain specials)
# -----------------------------------------------------------------------------
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "$SCRIPT_DIR/.env"
    set +a
    echo "[INFO] Loaded configuration from $SCRIPT_DIR/.env"
fi

# -----------------------------------------------------------------------------
# 1. Required configuration (fail fast, never start unauthenticated)
# -----------------------------------------------------------------------------
if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "[ERROR] MODEL_PATH is required. Export it or add it to .env." >&2
    exit 1
fi

if [[ -z "${VLLM_API_KEY:-}" ]]; then
    echo "[ERROR] VLLM_API_KEY is required. The launcher refuses to start an" >&2
    echo "        unauthenticated server. Export VLLM_API_KEY or add it to .env." >&2
    exit 1
fi

if [[ "$VLLM_API_KEY" =~ [[:space:]] ]]; then
    echo "[ERROR] VLLM_API_KEY must not contain whitespace." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 2. Tuning parameters (environment-overridable defaults)
# -----------------------------------------------------------------------------
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-38000}"
VLLM_SERVED_MODEL_NAMES="${VLLM_SERVED_MODEL_NAMES:-}"

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-40960}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"

VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"

VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"

# Model-specific flags (disabled by default to stay generic)
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-}"
VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-0}"

# Paths
VLLM_LOG_DIR="${VLLM_LOG_DIR:-$SCRIPT_DIR/vllm_logs}"
VLLM_RUN_DIR="${VLLM_RUN_DIR:-$SCRIPT_DIR/vllm_run}"
VLLM_HEALTH_TIMEOUT="${VLLM_HEALTH_TIMEOUT:-180}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"

# -----------------------------------------------------------------------------
# 3. CPU thread control (the core of the "high load" fix)
# -----------------------------------------------------------------------------
# OpenMP / MKL were spawning one thread pool per *system core* in every
# instance; cap them so a single process never over-subscribes its own cores.
# NOTE: do NOT use VLLM_CPU_OMP_THREADS_BOUND here — that variable belongs to
# the vLLM *CPU* backend and has no effect for GPU inference.
export OMP_NUM_THREADS="${VLLM_OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${VLLM_MKL_NUM_THREADS:-8}"
# Disable HuggingFace/tokenizers' implicit Rayon thread pool.
export TOKENIZERS_PARALLELISM="${VLLM_TOKENIZERS_PARALLELISM:-false}"

# -----------------------------------------------------------------------------
# 4. Activate a venv if present
# -----------------------------------------------------------------------------
if [[ -d "$VENV_DIR" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    echo "[INFO] Activated venv: $VENV_DIR"
else
    echo "[WARN] $VENV_DIR not found, using the current environment."
fi

mkdir -p "$VLLM_LOG_DIR" "$VLLM_RUN_DIR"

# -----------------------------------------------------------------------------
# 5. Detect GPUs
# -----------------------------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi not found. Install NVIDIA drivers first." >&2
    exit 1
fi

GPU_COUNT=$(nvidia-smi -L | wc -l)
if [[ "$GPU_COUNT" -le 0 ]]; then
    echo "[ERROR] No GPUs detected by nvidia-smi." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 6. CPU core count and per-GPU allocation
# -----------------------------------------------------------------------------
# Use `nproc --all` (NOT `nproc`): plain `nproc` reports OMP_NUM_THREADS once
# that env var is exported, which would miscalculate the core split.
TOTAL_CORES=$(nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 0)

if [[ "$TOTAL_CORES" -ge "$GPU_COUNT" ]]; then
    CORES_PER_GPU=$((TOTAL_CORES / GPU_COUNT))
else
    CORES_PER_GPU=0
fi

echo "[INFO] Detected $GPU_COUNT GPU(s) and $TOTAL_CORES CPU core(s)."
echo "[INFO] Allocating ~$CORES_PER_GPU CPU core(s) per vLLM instance."

# -----------------------------------------------------------------------------
# 7. Port availability pre-check
# -----------------------------------------------------------------------------
port_in_use() {
    local port="$1"
    if command -v ss &>/dev/null; then
        ss -tuln | grep -q ":$port "
    elif command -v netstat &>/dev/null; then
        netstat -tuln | grep -q ":$port "
    else
        return 1
    fi
}

for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
    port=$((VLLM_BASE_PORT + gpu_id))
    if port_in_use "$port"; then
        echo "[ERROR] Port $port is already in use. Clear old processes first." >&2
        exit 1
    fi
done

enabled() {
    case "${1:-0}" in
        1 | true | yes | on) return 0 ;;
        *) return 1 ;;
    esac
}

# -----------------------------------------------------------------------------
# 8. Launch one instance per GPU
# -----------------------------------------------------------------------------
echo "[INFO] Launching $GPU_COUNT vLLM instance(s)..."
PIDS=()

for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
    port=$((VLLM_BASE_PORT + gpu_id))
    log_file="$VLLM_LOG_DIR/vllm_gpu_${gpu_id}.log"

    # --- CPU affinity: isolate each instance to a contiguous core slice ---
    TASKSET_CMD=()
    if command -v taskset &>/dev/null && [[ "$CORES_PER_GPU" -gt 0 ]]; then
        core_start=$((gpu_id * CORES_PER_GPU))
        core_end=$((core_start + CORES_PER_GPU - 1))
        TASKSET_CMD=(taskset -c "${core_start}-${core_end}")
        echo "[GPU $gpu_id] Binding CPU cores ${core_start}-${core_end} | port $port | log $log_file"
    else
        echo "[WARN] taskset unavailable or too few cores; [GPU $gpu_id] runs un-pinned."
    fi

    # --- served model name(s): default to the model directory's base name ---
    if [[ -n "$VLLM_SERVED_MODEL_NAMES" ]]; then
        read -r -a served_names <<< "$VLLM_SERVED_MODEL_NAMES"
    else
        served_names=("$(basename "$MODEL_PATH")")
    fi

    # --- Build the vLLM serve argument list ---
    serve_args=(
        "$MODEL_PATH"
        --served-model-name "${served_names[@]}"
        --host "$VLLM_HOST"
        --port "$port"
        --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
        --max-num-seqs "$VLLM_MAX_NUM_SEQS"
        --max-model-len "$VLLM_MAX_MODEL_LEN"
        --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
        --api-key "$VLLM_API_KEY"
    )

    [[ -n "$VLLM_QUANTIZATION" ]] && serve_args+=(--quantization "$VLLM_QUANTIZATION")

    enabled "$VLLM_ENABLE_PREFIX_CACHING"   && serve_args+=(--enable-prefix-caching)
    enabled "$VLLM_ENABLE_CHUNKED_PREFILL"  && serve_args+=(--enable-chunked-prefill)

    [[ -n "$VLLM_REASONING_PARSER" ]] && serve_args+=(--reasoning-parser "$VLLM_REASONING_PARSER")
    [[ -n "$VLLM_TOOL_CALL_PARSER" ]] && serve_args+=(--tool-call-parser "$VLLM_TOOL_CALL_PARSER")
    enabled "$VLLM_ENABLE_AUTO_TOOL_CHOICE" && serve_args+=(--enable-auto-tool-choice)

    # --- Launch ---
    CUDA_VISIBLE_DEVICES=$gpu_id "${TASKSET_CMD[@]}" nohup vllm serve "${serve_args[@]}" \
        > "$log_file" 2>&1 &

    pid=$!
    PIDS+=("$pid")
    echo "$pid" > "$VLLM_RUN_DIR/vllm_gpu_${gpu_id}.pid"
    echo "[GPU $gpu_id] Started with PID $pid"
done

# -----------------------------------------------------------------------------
# 9. Health-check polling for every instance
# -----------------------------------------------------------------------------
echo ""
echo "[INFO] Waiting for all instances to pass health checks (timeout: ${VLLM_HEALTH_TIMEOUT}s)..."

START_TIME=$(date +%s)
FAILED=0

for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
    port=$((VLLM_BASE_PORT + gpu_id))
    pid="${PIDS[$gpu_id]}"
    log_file="$VLLM_LOG_DIR/vllm_gpu_${gpu_id}.log"

    echo -n "[GPU $gpu_id] Waiting for /health on port $port ... "

    while true; do
        elapsed=$(( $(date +%s) - START_TIME ))

        if ! kill -0 "$pid" 2>/dev/null; then
            echo -e "\n[ERROR] GPU $gpu_id (PID $pid) exited unexpectedly. See $log_file" >&2
            FAILED=1
            break
        fi

        if curl -s -f "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo "READY (${elapsed}s)"
            break
        fi

        if [[ "$elapsed" -ge "$VLLM_HEALTH_TIMEOUT" ]]; then
            echo -e "\n[TIMEOUT] GPU $gpu_id on port $port not healthy in ${VLLM_HEALTH_TIMEOUT}s. See $log_file" >&2
            FAILED=1
            break
        fi

        sleep 3
    done
done

echo ""
if [[ "$FAILED" -eq 0 ]]; then
    echo "=========================================================="
    echo "[SUCCESS] All $GPU_COUNT instance(s) are healthy and ready!"
    echo "Ports: $VLLM_BASE_PORT ~ $((VLLM_BASE_PORT + GPU_COUNT - 1))"
    echo "Stop them with: $SCRIPT_DIR/stop_vllm.sh"
    echo "=========================================================="
else
    echo "=========================================================="
    echo "[WARN] One or more instances failed to start cleanly."
    echo "Check the logs under '$VLLM_LOG_DIR/' for details."
    echo "=========================================================="
    exit 1
fi