#!/bin/bash
#SBATCH --job-name=causal_rl_phase6
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Phase 6: vectorized multi-task PPO on live GPT-2 ablations.
# Usage (single):
#   sbatch --export=ALL,K=1,N=8,STEPS=200000,TAG=mt_k1 scripts/train.sbatch
# Usage (array — K=1 and K=5):
#   sbatch --array=0-1 scripts/train.sbatch

set -euo pipefail

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

PROJECT="${HOME}/causality-RL"
source "${PROJECT}/.venv/bin/activate"

cd "${PROJECT}/phase 6"
mkdir -p logs results

# Array mode: index 0 -> K=1, index 1 -> K=5
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    CASES_K=(1 5)
    CASES_TAG=(mt_k1 mt_k5)
    IDX=${SLURM_ARRAY_TASK_ID}
    K="${CASES_K[$IDX]}"
    TAG="${CASES_TAG[$IDX]}"
    N="${N:-8}"
    STEPS="${STEPS:-200000}"
fi

: "${K:?K must be set}"
: "${TAG:?TAG must be set}"
: "${N:?N must be set (n_envs)}"
: "${STEPS:?STEPS must be set}"

echo "[start] $(date) K=${K} N=${N} TAG=${TAG} STEPS=${STEPS} host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python ppo_vec.py \
    --plan_k "${K}" \
    --n_envs "${N}" \
    --tag "${TAG}" \
    --total_timesteps "${STEPS}" \
    --device cuda

echo "[done]  $(date)"
