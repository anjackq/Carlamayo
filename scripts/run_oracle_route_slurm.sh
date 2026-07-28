#!/usr/bin/env bash
#SBATCH --job-name=cm-oracle
#SBATCH --partition=mv-ltd
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=oracle-route-%j.log
#SBATCH --error=oracle-route-%j.log

set -euo pipefail

CARLAMAYO_REPO_ROOT="/home/aqiu/Desktop/Carlamayo"
CARLAMAYO_VENV="$CARLAMAYO_REPO_ROOT/a1_5_carla_venv"
CARLAMAYO_CARLA_ROOT="/home/aqiu/carla"
export CARLAMAYO_CARLA_ROOT
export CARLAMAYO_CARLA_PYTHONAPI="$CARLAMAYO_CARLA_ROOT/PythonAPI/carla"

cd "$CARLAMAYO_REPO_ROOT"

if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    CARLAMAYO_RPC_PORT=$((20000 + 4 * (SLURM_JOB_ID % 10000)))
else
    CARLAMAYO_RPC_PORT=2000
fi
export CARLAMAYO_CARLA_PORT="$CARLAMAYO_RPC_PORT"
export CARLAMAYO_TRAFFIC_MANAGER_PORT=$((CARLAMAYO_RPC_PORT + 3))

IFS=, read -r CARLAMAYO_CARLA_GPU _ <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${CARLAMAYO_CARLA_GPU:-}" ]]; then
    echo "Expected one assigned CARLA GPU." >&2
    exit 1
fi
if [[ ! "$CARLAMAYO_CARLA_GPU" =~ ^[0-9]+$ ]] \
    || ! nvidia-smi -i "$CARLAMAYO_CARLA_GPU" \
        --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
    echo \
        "CARLA requires a full graphics-capable GPU ordinal; allocation" \
        "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} is not usable." \
        >&2
    exit 1
fi

CARLAMAYO_RUN_ROOT="/home/aqiu/carlamayo-runs/${SLURM_JOB_ID}"
mkdir -p "$CARLAMAYO_RUN_ROOT"
CARLAMAYO_CARLA_LOG="$CARLAMAYO_RUN_ROOT/carla-server.log"
CARLAMAYO_CARLA_PID=""

cleanup() {
    if [[ -n "$CARLAMAYO_CARLA_PID" ]] && kill -0 "$CARLAMAYO_CARLA_PID" 2>/dev/null; then
        kill "$CARLAMAYO_CARLA_PID" 2>/dev/null || true
        wait "$CARLAMAYO_CARLA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

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
    echo "CARLA did not open port ${CARLAMAYO_CARLA_PORT}." >&2
    exit 1
fi

CARLAMAYO_WORLD_READY=0
for _ in $(seq 1 18); do
    if CARLAMAYO_PROBE_PORT="$CARLAMAYO_CARLA_PORT" \
        "$CARLAMAYO_VENV/bin/python" - <<'PY'
import os
import carla

client = carla.Client("localhost", int(os.environ["CARLAMAYO_PROBE_PORT"]))
client.set_timeout(5.0)
client.get_world()
PY
    then
        CARLAMAYO_WORLD_READY=1
        break
    fi
    sleep 5
done
if [[ "$CARLAMAYO_WORLD_READY" -ne 1 ]]; then
    echo "CARLA opened its port but did not expose a world within 90 seconds." >&2
    tail -100 "$CARLAMAYO_CARLA_LOG" >&2 || true
    exit 1
fi

PYTHONUNBUFFERED=1 "$CARLAMAYO_VENV/bin/python" \
    scripts/run_oracle_route_experiment.py \
    --telemetry-jsonl "$CARLAMAYO_RUN_ROOT/runtime.jsonl" \
    "$@"
