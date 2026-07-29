#!/usr/bin/env bash
#SBATCH --job-name=carla-spawn-smoke
#SBATCH --partition=gpuidle
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=carla-spawn-smoke-%j.log
#SBATCH --error=carla-spawn-smoke-%j.log

set -euo pipefail

CARLAMAYO_REPO_ROOT="/home/aqiu/Desktop/Carlamayo"
CARLAMAYO_VENV="$CARLAMAYO_REPO_ROOT/a1_5_carla_venv"
CARLAMAYO_CARLA_ROOT="/home/aqiu/carla"
CARLAMAYO_JOB_ID="${SLURM_JOB_ID:-manual}"
CARLAMAYO_RUN_ROOT="/home/aqiu/carlamayo-runs/${CARLAMAYO_JOB_ID}/spawn-control-smoke"

cd "$CARLAMAYO_REPO_ROOT"
mkdir -p "$CARLAMAYO_RUN_ROOT"

IFS=, read -r CARLAMAYO_CARLA_GPU _ <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${CARLAMAYO_CARLA_GPU:-}" ]]; then
    echo "Expected one assigned GPU, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
fi

# A per-job port avoids the common failure where multiple CARLA jobs on one
# node all attempt to bind the default port 2000.
if [[ -n "${CARLAMAYO_CARLA_PORT:-}" ]]; then
    CARLAMAYO_SMOKE_PORT="$CARLAMAYO_CARLA_PORT"
elif [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    CARLAMAYO_SMOKE_PORT=$((20000 + SLURM_JOB_ID % 30000))
else
    CARLAMAYO_SMOKE_PORT=2000
fi

if ! [[ "$CARLAMAYO_SMOKE_PORT" =~ ^[0-9]+$ ]] \
    || ((CARLAMAYO_SMOKE_PORT < 1024 || CARLAMAYO_SMOKE_PORT > 65533)); then
    echo "Invalid CARLAMAYO_CARLA_PORT: ${CARLAMAYO_SMOKE_PORT}" >&2
    exit 1
fi

echo "Node: $(hostname)"
echo "Assigned GPU: ${CARLAMAYO_CARLA_GPU}"
echo "CARLA RPC port: ${CARLAMAYO_SMOKE_PORT}"
echo "Artifacts: ${CARLAMAYO_RUN_ROOT}"
nvidia-smi

CARLAMAYO_CARLA_LOG="$CARLAMAYO_RUN_ROOT/carla-server.log"
CARLAMAYO_CARLA_PID=""

cleanup() {
    if [[ -n "$CARLAMAYO_CARLA_PID" ]] && kill -0 "$CARLAMAYO_CARLA_PID" 2>/dev/null; then
        echo "Stopping CARLA server PID ${CARLAMAYO_CARLA_PID}"
        kill "$CARLAMAYO_CARLA_PID" 2>/dev/null || true
        wait "$CARLAMAYO_CARLA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting CARLA; log: ${CARLAMAYO_CARLA_LOG}"
CUDA_VISIBLE_DEVICES="$CARLAMAYO_CARLA_GPU" \
    "$CARLAMAYO_CARLA_ROOT/CarlaUE4.sh" \
    -RenderOffScreen \
    -nosound \
    "-carla-rpc-port=${CARLAMAYO_SMOKE_PORT}" \
    "-ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=${CARLAMAYO_CARLA_GPU}" \
    >"$CARLAMAYO_CARLA_LOG" 2>&1 &
CARLAMAYO_CARLA_PID=$!

CARLAMAYO_CARLA_READY=0
for _ in $(seq 1 90); do
    if ! kill -0 "$CARLAMAYO_CARLA_PID" 2>/dev/null; then
        echo "CARLA exited before opening port ${CARLAMAYO_SMOKE_PORT}." >&2
        tail -100 "$CARLAMAYO_CARLA_LOG" >&2 || true
        exit 1
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/${CARLAMAYO_SMOKE_PORT}") 2>/dev/null; then
        exec 3>&- 3<&-
        CARLAMAYO_CARLA_READY=1
        break
    fi
    sleep 1
done

if [[ "$CARLAMAYO_CARLA_READY" -ne 1 ]]; then
    echo "CARLA did not open port ${CARLAMAYO_SMOKE_PORT} within 90 seconds." >&2
    tail -100 "$CARLAMAYO_CARLA_LOG" >&2 || true
    exit 1
fi

echo "CARLA is ready. Running spawn/control matrix."
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$CARLAMAYO_CARLA_GPU" \
    "$CARLAMAYO_VENV/bin/python" scripts/carla_spawn_control_smoke.py \
    --port "$CARLAMAYO_SMOKE_PORT" \
    --output-jsonl "$CARLAMAYO_RUN_ROOT/results.jsonl" \
    "$@" \
    | tee "$CARLAMAYO_RUN_ROOT/diagnostic.log"
