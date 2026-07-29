#!/usr/bin/env bash
#SBATCH --job-name=cm-reach-gate
#SBATCH --partition=eit-wicon
#SBATCH --gres=gpu:b200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/aqiu/carlamayo-runs/reachability-gate-%A_%a.log
#SBATCH --error=/home/aqiu/carlamayo-runs/reachability-gate-%A_%a.log

set -euo pipefail

CARLAMAYO_REPO_ROOT="/home/aqiu/Desktop/Carlamayo"
: "${CARLAMAYO_CAMERA_PROFILE:?Set CARLAMAYO_CAMERA_PROFILE to the local gated profile}"
if [[ ! -r "$CARLAMAYO_CAMERA_PROFILE" ]]; then
    echo "Camera profile is not readable: $CARLAMAYO_CAMERA_PROFILE" >&2
    exit 1
fi

case "${SLURM_ARRAY_TASK_ID:?}" in
    0) CARLAMAYO_EXPERIMENT_ARM="current"; CARLAMAYO_EXPERIMENT_SEED="0" ;;
    1) CARLAMAYO_EXPERIMENT_ARM="reachability-first"; CARLAMAYO_EXPERIMENT_SEED="0" ;;
    2) CARLAMAYO_EXPERIMENT_ARM="current"; CARLAMAYO_EXPERIMENT_SEED="1" ;;
    3) CARLAMAYO_EXPERIMENT_ARM="reachability-first"; CARLAMAYO_EXPERIMENT_SEED="1" ;;
    4) CARLAMAYO_EXPERIMENT_ARM="current"; CARLAMAYO_EXPERIMENT_SEED="2" ;;
    5) CARLAMAYO_EXPERIMENT_ARM="reachability-first"; CARLAMAYO_EXPERIMENT_SEED="2" ;;
    *)
        echo "Unexpected array task: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 1
        ;;
esac

export CARLAMAYO_EXPERIMENT_ID="synchronous-reachability-go-no-go-v1"
export CARLAMAYO_EXPERIMENT_ARM
export CARLAMAYO_EXPERIMENT_SEED
export CARLAMAYO_EPISODE_SECONDS="60"

exec bash "$CARLAMAYO_REPO_ROOT/scripts/run_carlamayo_slurm.sh" \
    --empty-road \
    --scenario-seed "$CARLAMAYO_EXPERIMENT_SEED" \
    --ego-spawn-index 0 \
    --mode navigation \
    --navigation-source route \
    --route-destination=-43.350975036621094,-2.8402605056762695,0 \
    --camera-alignment projection-only \
    --camera-profile "$CARLAMAYO_CAMERA_PROFILE" \
    --num-traj-samples 3 \
    --diffusion-temperature 1.0 \
    --road-assessment-backend serial \
    --low-speed-longitudinal-governor \
    --trajectory-reachability-audit \
    --candidate-ranking-policy "$CARLAMAYO_EXPERIMENT_ARM"
