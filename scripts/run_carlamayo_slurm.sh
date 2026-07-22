#!/usr/bin/env bash
#SBATCH --job-name=carlamayo
#SBATCH --partition=mv-ltd
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=carlamayo-slurm-%j.log
#SBATCH --error=carlamayo-slurm-%j.log

set -euo pipefail

CARLAMAYO_REPO_ROOT="/home/aqiu/Desktop/Carlamayo"
CARLAMAYO_VENV="$CARLAMAYO_REPO_ROOT/a1_5_carla_venv"
CARLAMAYO_CARLA_ROOT="/home/aqiu/carla"

cd "$CARLAMAYO_REPO_ROOT"

IFS=, read -r CARLAMAYO_CARLA_GPU CARLAMAYO_MODEL_GPU _ <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${CARLAMAYO_CARLA_GPU:-}" || -z "${CARLAMAYO_MODEL_GPU:-}" ]]; then
    echo "Expected two assigned GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
fi

echo "Node: $(hostname)"
echo "Assigned GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "CARLA GPU: ${CARLAMAYO_CARLA_GPU}"
echo "Alpamayo GPU: ${CARLAMAYO_MODEL_GPU}"
nvidia-smi

CARLAMAYO_CARLA_LOG="$CARLAMAYO_REPO_ROOT/carla-server-${SLURM_JOB_ID}.log"
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
    "-ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=${CARLAMAYO_CARLA_GPU}" \
    >"$CARLAMAYO_CARLA_LOG" 2>&1 &
CARLAMAYO_CARLA_PID=$!

CARLAMAYO_CARLA_READY=0
for _ in $(seq 1 90); do
    if ! kill -0 "$CARLAMAYO_CARLA_PID" 2>/dev/null; then
        echo "CARLA exited before opening port 2000." >&2
        tail -100 "$CARLAMAYO_CARLA_LOG" >&2 || true
        exit 1
    fi
    if (exec 3<>/dev/tcp/127.0.0.1/2000) 2>/dev/null; then
        exec 3>&- 3<&-
        CARLAMAYO_CARLA_READY=1
        break
    fi
    sleep 1
done

if [[ "$CARLAMAYO_CARLA_READY" -ne 1 ]]; then
    echo "CARLA did not open port 2000 within 90 seconds." >&2
    tail -100 "$CARLAMAYO_CARLA_LOG" >&2 || true
    exit 1
fi

echo "CARLA is ready. Verifying the Alpamayo GPU."
CUDA_VISIBLE_DEVICES="$CARLAMAYO_MODEL_GPU" \
    "$CARLAMAYO_VENV/bin/python" - <<'PY'
from module.inference import require_cuda_runtime

print(f"Alpamayo CUDA device: {require_cuda_runtime()}")
PY

echo "Starting CarlaMayo closed-loop async run."
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$CARLAMAYO_MODEL_GPU" \
    "$CARLAMAYO_VENV/bin/python" carlamayo_closed_loop.py --async
