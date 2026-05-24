#!/bin/bash
#SBATCH --job-name=causal_rl
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Causal-curiosity RL: PPO + planning on live GPT-2 ablations.
# Magnolia (Ole Miss MCSR) — L40S nodes.
#
# Usage (single):
#   sbatch --export=ALL,K=1,STEPS=50000,TAG=k1 scripts/train.sbatch
#   sbatch --export=ALL,K=5,STEPS=50000,TAG=k5 scripts/train.sbatch
#
# Usage (array — both K's in parallel):
#   sbatch --array=0-1 scripts/train.sbatch

set -euo pipefail

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

PROJECT="${HOME}/causality-RL"
source "${PROJECT}/.venv/bin/activate"

cd "${PROJECT}/phase 5"
mkdir -p logs results

# Array mode: index 0 -> K=1, index 1 -> K=5
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    CASES_K=(1 5)
    CASES_TAG=(k1 k5)
    IDX=${SLURM_ARRAY_TASK_ID}
    K="${CASES_K[$IDX]}"
    TAG="${CASES_TAG[$IDX]}"
    STEPS="${STEPS:-50000}"
fi

: "${K:?K must be set (1 = vanilla PPO, 5 = best-of-5 planning)}"
: "${TAG:?TAG must be set (output file suffix)}"
: "${STEPS:?STEPS must be set (total PPO timesteps)}"

echo "[start] $(date) K=${K} TAG=${TAG} STEPS=${STEPS} host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python ppo_planning_real.py \
    --plan_k "${K}" \
    --tag "${TAG}" \
    --total_timesteps "${STEPS}" \
    --device cuda

echo "[done]  $(date)"
