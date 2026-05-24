# Phase 5 — fully live, varying-episode training

Builds on Phase 4 with two real changes:

1. **No precomputation, no persistent cache.** Every `env.step()` invokes
   GPT-2 with a real ablation hook. (A tiny in-memory cache exists *within*
   each episode so the planner doesn't re-evaluate the same candidate twice
   in the same step — that cache is cleared on `reset()`.)
2. **Per-episode randomness.** Each `reset(seed=...)` regenerates a fresh
   induction batch with that seed. Head scores drift episode-to-episode, so
   the agent cannot memorize "always pick head 22" — it must learn structural
   priors over the network.

Eval uses a held-out seed band (`seed_base = 10_000_000`) so there is zero
overlap with training seeds.

---

## Files

| File | What it does |
| --- | --- |
| `head_env_real.py` | Live env. Smoke-test it standalone: `python head_env_real.py` |
| `ppo_planning_real.py` | PPO with best-of-K planning. K=1 vanilla, K=5 model-based. |
| `compare.py` | Generates random baseline + makes both Phase 5 plots. |
| `requirements.txt` | Pip deps for HPC venv. |
| `scripts/train.sbatch` | SLURM job: one K config per submission, or array `0..1` for both. |
| `scripts/compare.sbatch` | SLURM job: random baseline + plots. Run after both trains. |

---

## HPC quickstart (Magnolia / L40S)

```bash
# One-time setup
git clone https://github.com/<you>/causality-RL.git ~/causality-RL
cd ~/causality-RL
module load python/2025.12-2 cuda12.8/toolkit/12.8.1
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r "phase 5/requirements.txt"

# Submit both K configs as a 2-job array
cd "phase 5"
mkdir -p logs results
sbatch --array=0-1 --export=ALL,STEPS=50000 scripts/train.sbatch

# After both finish, generate plots
sbatch --dependency=afterok:<train_job_id> scripts/compare.sbatch
```

---

## Speed expectations (rough)

| Phase 5 config | per-step (L40, batch=32) | 50k steps |
| --- | --- | --- |
| K=1 (vanilla PPO) | ~30-60 ms | ~30-50 min |
| K=5 (best-of-5 planning) | ~150-300 ms | ~2-4 h |

Total wall time for the array job ≤ 4 h on a single L40. The 6-hour SLURM
time-limit in `train.sbatch` is comfortable headroom.

---

## What success looks like

- **Random baseline** plateaus around 2.0-2.5 (depends on how scores fluctuate per episode).
- **K=1 PPO**: learning trend should rise from baseline level to a clear plateau above random.
- **K=5 planning**: should beat K=1 on the discovery curve (especially early steps),
  because the planner uses real GPT-2 to filter candidates regardless of how
  polished the policy is.
- Eval curves should **rise over training** — if they're flat, the policy isn't
  learning anything useful, which would itself be a meaningful Phase 5 finding.

---

## Output files (in `results/`)

- `real_k1_curves.npy` + `real_k1_summary.json` + `real_k1_policy.pt`
- `real_k5_curves.npy` + `real_k5_summary.json` + `real_k5_policy.pt`
- `real_random_curves.npy`
- `phase5_discovery_curves.png` — the headline plot
- `phase5_learning_trend.png` — eval-perf vs training-step
- `phase5_summary.json` — numbers
