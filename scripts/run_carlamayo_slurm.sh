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
export CARLAMAYO_CARLA_ROOT
export CARLAMAYO_CARLA_PYTHONAPI="$CARLAMAYO_CARLA_ROOT/PythonAPI/carla"

cd "$CARLAMAYO_REPO_ROOT"

# Per-job ports allow multiple seed runs to share the six-GPU node without
# racing on CARLA's RPC/streaming or Traffic Manager ports.  The job-ID slot
# is only a starting point: a node-local flock and live probe resolve collisions
# with jobs whose IDs differ by 10,000 or which use manually selected ports.
source "$CARLAMAYO_REPO_ROOT/scripts/slurm_port_reservation.sh"
carlamayo_reserve_port_slot \
    "${CARLAMAYO_CARLA_PORT:-}" \
    "${SLURM_JOB_ID:-0}"
CARLAMAYO_RPC_PORT="$CARLAMAYO_RESERVED_PORT"
export CARLAMAYO_CARLA_PORT="$CARLAMAYO_RPC_PORT"
export CARLAMAYO_TRAFFIC_MANAGER_PORT=$((CARLAMAYO_RPC_PORT + 3))

IFS=, read -r CARLAMAYO_CARLA_GPU CARLAMAYO_MODEL_GPU _ <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${CARLAMAYO_CARLA_GPU:-}" ]]; then
    echo "Expected at least one assigned GPU, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
fi
if [[ -z "${CARLAMAYO_MODEL_GPU:-}" ]]; then
    if [[ "${CARLAMAYO_ALLOW_SHARED_GPU:-0}" == "1" ]]; then
        CARLAMAYO_MODEL_GPU="$CARLAMAYO_CARLA_GPU"
        echo "Validation mode: CARLA and Alpamayo share one explicitly authorized GPU."
    else
        echo "Expected two assigned GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
        exit 1
    fi
fi

echo "Node: $(hostname)"
echo "Assigned GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "CARLA GPU: ${CARLAMAYO_CARLA_GPU}"
echo "Alpamayo GPU: ${CARLAMAYO_MODEL_GPU}"
echo "CARLA RPC port: ${CARLAMAYO_CARLA_PORT}"
echo "Traffic Manager port: ${CARLAMAYO_TRAFFIC_MANAGER_PORT}"
nvidia-smi

if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    CARLAMAYO_RUN_KEY="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    CARLAMAYO_RUN_KEY="${SLURM_JOB_ID}"
fi
CARLAMAYO_RUN_ROOT="${CARLAMAYO_RUN_ROOT:-/home/aqiu/carlamayo-runs/${CARLAMAYO_RUN_KEY}}"
mkdir -p "$CARLAMAYO_RUN_ROOT"
CARLAMAYO_CARLA_LOG="$CARLAMAYO_RUN_ROOT/carla-server.log"
CARLAMAYO_CARLA_PID=""

cleanup() {
    if [[ -n "$CARLAMAYO_CARLA_PID" ]] && kill -0 "$CARLAMAYO_CARLA_PID" 2>/dev/null; then
        echo "Stopping CARLA server PID ${CARLAMAYO_CARLA_PID}"
        kill "$CARLAMAYO_CARLA_PID" 2>/dev/null || true
        wait "$CARLAMAYO_CARLA_PID" 2>/dev/null || true
    fi
    carlamayo_release_port_slot
}
trap cleanup EXIT INT TERM

echo "Starting CARLA; log: ${CARLAMAYO_CARLA_LOG}"
CUDA_VISIBLE_DEVICES="$CARLAMAYO_CARLA_GPU" \
    "$CARLAMAYO_CARLA_ROOT/CarlaUE4.sh" \
    -RenderOffScreen \
    "-carla-rpc-port=${CARLAMAYO_CARLA_PORT}" \
    "-ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=${CARLAMAYO_CARLA_GPU}" \
    >"$CARLAMAYO_CARLA_LOG" 2>&1 &
CARLAMAYO_CARLA_PID=$!

CARLAMAYO_CARLA_READY=0
for _ in $(seq 1 90); do
    if ! kill -0 "$CARLAMAYO_CARLA_PID" 2>/dev/null; then
        echo "CARLA exited before opening port ${CARLAMAYO_CARLA_PORT}." >&2
        tail -100 "$CARLAMAYO_CARLA_LOG" >&2 || true
        exit 1
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/${CARLAMAYO_CARLA_PORT}") 2>/dev/null; then
        exec 3>&- 3<&-
        CARLAMAYO_CARLA_READY=1
        break
    fi
    sleep 1
done

if [[ "$CARLAMAYO_CARLA_READY" -ne 1 ]]; then
    echo "CARLA did not open port ${CARLAMAYO_CARLA_PORT} within 90 seconds." >&2
    tail -100 "$CARLAMAYO_CARLA_LOG" >&2 || true
    exit 1
fi

echo "CARLA is ready. Verifying the Alpamayo GPU."
CUDA_VISIBLE_DEVICES="$CARLAMAYO_MODEL_GPU" \
    "$CARLAMAYO_VENV/bin/python" - <<'PY'
from module.inference import require_cuda_runtime

print(f"Alpamayo CUDA device: {require_cuda_runtime()}")
PY

export CARLAMAYO_OUTPUT_VIDEO="$CARLAMAYO_RUN_ROOT/carla_alpamayo_closed_loop_result.mp4"
export CARLAMAYO_LIVE_PREVIEW_IMAGE="$CARLAMAYO_RUN_ROOT/carla_alpamayo_closed_loop_latest.jpg"

{
    echo "experiment_id=${CARLAMAYO_EXPERIMENT_ID:-}"
    echo "experiment_arm=${CARLAMAYO_EXPERIMENT_ARM:-}"
    echo "scenario_seed=${CARLAMAYO_EXPERIMENT_SEED:-}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_dirty=$(test -n "$(git status --porcelain)" && echo true || echo false)"
    echo "camera_profile=${CARLAMAYO_CAMERA_PROFILE:-}"
    if [[ -n "${CARLAMAYO_CAMERA_PROFILE:-}" ]]; then
        echo "camera_profile_sha256=$(sha256sum "$CARLAMAYO_CAMERA_PROFILE" | awk '{print $1}')"
    fi
    echo "node=$(hostname)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    printf "arguments="
    printf "%q " "$@"
    printf "\n"
} >"$CARLAMAYO_RUN_ROOT/experiment-manifest.txt"

echo "Starting CarlaMayo closed-loop synchronized safety run."
echo "Artifacts: ${CARLAMAYO_RUN_ROOT}"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$CARLAMAYO_MODEL_GPU" \
    "$CARLAMAYO_VENV/bin/python" carlamayo_closed_loop.py \
    --telemetry-jsonl "$CARLAMAYO_RUN_ROOT/runtime.jsonl" \
    --max-episode-seconds "${CARLAMAYO_EPISODE_SECONDS:-60}" \
    "$@"
