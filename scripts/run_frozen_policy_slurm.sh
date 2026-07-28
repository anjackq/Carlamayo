#!/usr/bin/env bash
#SBATCH --job-name=cm-frozen
#SBATCH --partition=eit-wicon
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=frozen-policy-%j.log
#SBATCH --error=frozen-policy-%j.log

set -euo pipefail

CARLAMAYO_REPO_ROOT="/home/aqiu/Desktop/Carlamayo"
CARLAMAYO_VENV="$CARLAMAYO_REPO_ROOT/a1_5_carla_venv"
cd "$CARLAMAYO_REPO_ROOT"

CARLAMAYO_RUN_ROOT="/home/aqiu/carlamayo-runs/${SLURM_JOB_ID}"
mkdir -p "$CARLAMAYO_RUN_ROOT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
    "$CARLAMAYO_VENV/bin/python" scripts/run_frozen_camera_ablation.py \
    --conditioning-source per-fixture \
    --output "$CARLAMAYO_RUN_ROOT/frozen-policy.jsonl" \
    "$@"
