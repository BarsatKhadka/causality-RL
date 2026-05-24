#!/bin/bash
#SBATCH --job-name=causal_rl_compare
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

# Generate the random baseline (needs GPU because the env loads GPT-2)
# and produce Phase 5 comparison plots. Run AFTER train.sbatch finishes.
#
# Usage:
#   sbatch --dependency=afterok:<train_job_id> scripts/compare.sbatch
# or simply, once K=1 and K=5 are both done:
#   sbatch scripts/compare.sbatch

set -euo pipefail

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

PROJECT="${HOME}/causality-RL"
source "${PROJECT}/.venv/bin/activate"

cd "${PROJECT}/phase 5"
mkdir -p logs

echo "[start] $(date) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python compare.py
echo "[done]  $(date)"
