#!/bin/bash
#SBATCH --job-name=causal_rl_compare_heads
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

# Compares trained policy's picked heads against (a) oracle and
# (b) canonical literature heads, for each of induction / IOI / docstring.
#
# Usage:
#   sbatch scripts/compare_heads.sbatch
#   sbatch --export=ALL,TAG=mt_k1 scripts/compare_heads.sbatch

set -euo pipefail

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

PROJECT="${HOME}/causality-RL"
source "${PROJECT}/.venv/bin/activate"

cd "${PROJECT}/phase 6"
mkdir -p logs results

if [[ -z "${TAG:-}" ]]; then
    TAGS=(mt_k1 mt_k5)
else
    TAGS=("${TAG}")
fi

for T in "${TAGS[@]}"; do
    P="results/phase6_${T}_policy.pt"
    if [[ -f "$P" ]]; then
        echo "[start] $(date) compare_heads tag=${T}"
        python compare_heads.py --policy_path "$P" --tag "$T" --device cuda
        echo "[done]  tag=${T}"
    else
        echo "[skip] no policy at $P"
    fi
done

echo "[end] $(date)"
