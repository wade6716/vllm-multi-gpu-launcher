#!/usr/bin/env bash
# =============================================================================
# stop_vllm.sh — Stop vLLM instances started by launch_vllm.sh
#
# Usage:
#   ./stop_vllm.sh          # stop every instance (all GPU ids)
#   ./stop_vllm.sh 0 2      # stop only GPU 0 and GPU 2
#
# It reads the PID files that launch_vllm.sh writes into VLLM_RUN_DIR
# (default ./vllm_run) and terminates each process gracefully (SIGTERM),
# escalating to SIGKILL if it does not exit within VLLM_STOP_TIMEOUT seconds.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "$SCRIPT_DIR/.env"
    set +a
fi

VLLM_RUN_DIR="${VLLM_RUN_DIR:-$SCRIPT_DIR/vllm_run}"
VLLM_STOP_TIMEOUT="${VLLM_STOP_TIMEOUT:-10}"

stop_one() {
    local gpu_id="$1"
    local pid_file="$VLLM_RUN_DIR/vllm_gpu_${gpu_id}.pid"

    if [[ ! -f "$pid_file" ]]; then
        echo "[WARN] No PID file for GPU $gpu_id ($pid_file). Nothing to stop."
        return 0
    fi

    local pid
    pid="$(cat "$pid_file")"

    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[INFO] GPU $gpu_id (PID $pid) already stopped."
        rm -f "$pid_file"
        return 0
    fi

    echo -n "[GPU $gpu_id] Stopping PID $pid ... "
    kill "$pid" 2>/dev/null || true

    local waited=0
    while kill -0 "$pid" 2>/dev/null && [[ "$waited" -lt "$VLLM_STOP_TIMEOUT" ]]; do
        sleep 1
        waited=$((waited + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo -n "still alive, sending SIGKILL ... "
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi

    rm -f "$pid_file"
    echo "stopped."
}

if [[ "$#" -eq 0 ]]; then
    # Stop every PID file found under VLLM_RUN_DIR.
    shopt -s nullglob
    pid_files=("$VLLM_RUN_DIR"/vllm_gpu_*.pid)
    shopt -u nullglob

    if [[ "${#pid_files[@]}" -eq 0 ]]; then
        echo "[WARN] No PID files found under '$VLLM_RUN_DIR' — nothing to stop."
        exit 0
    fi

    for pid_file in "${pid_files[@]}"; do
        gpu_id="$(basename "$pid_file" .pid)"
        gpu_id="${gpu_id#vllm_gpu_}"
        stop_one "$gpu_id"
    done
else
    for gpu_id in "$@"; do
        stop_one "$gpu_id"
    done
fi

echo "[INFO] Done."