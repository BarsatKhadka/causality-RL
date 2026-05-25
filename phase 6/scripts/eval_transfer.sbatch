#!/bin/bash
#SBATCH --job-name=causal_rl_transfer
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

# Held-out transfer eval: load a trained Phase 6 policy and test on docstring.
# Usage (after the train array job finishes):
#   sbatch --dependency=afterok:<train_job_id> scripts/eval_transfer.sbatch
# Or explicitly per tag:
#   sbatch --export=ALL,TAG=mt_k1 scripts/eval_transfer.sbatch
#   sbatch --export=ALL,TAG=mt_k5 scripts/eval_transfer.sbatch

set -euo pipefail

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

PROJECT="${HOME}/causality-RL"
source "${PROJECT}/.venv/bin/activate"

cd "${PROJECT}/phase 6"
mkdir -p logs results

# Default: eval both tags if TAG not set.
if [[ -z "${TAG:-}" ]]; then
    TAGS=(mt_k1 mt_k5)
else
    TAGS=("${TAG}")
fi

for T in "${TAGS[@]}"; do
    P="results/phase6_${T}_policy.pt"
    if [[ -f "$P" ]]; then
        echo "[start] $(date) transfer eval tag=${T} policy=${P}"
        python eval_transfer.py --policy_path "$P" --tag "$T" --device cuda
        echo "[done]  tag=${T}"
    else
        echo "[skip] no policy at $P"
    fi
done

echo "[end] $(date)"
