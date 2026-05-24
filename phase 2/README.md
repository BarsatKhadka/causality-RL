# Phase 2 — Random search baseline

A dumb agent that picks heads uniformly at random. This is the floor your
Phase 3 RL agent has to beat.

## What it does
- Uses the Phase 1 reward function (cached as `phase 1/results/head_scores_logitdiff.npy`).
  Each "query" is now a constant-time lookup, so running 100 episodes is sub-second
  instead of 25 hours.
- Runs 100 random episodes of 50 steps each.
- For each episode, tracks the running-max score *and* how many steps it took to
  find the ground-truth top-1, top-3, top-5, top-10 heads.

## Why multiple K's
There's no canonical answer to "which heads ARE the induction circuit." We
report top-K for K ∈ {1, 3, 5, 10}:
- **top-1** — single most-important head. Cleanest headline metric.
- **top-3 / top-5** — does the agent find the dense core of the circuit?
- **top-10** — does the agent find the full circuit including weaker components?

## Run
```powershell
.\venv\Scripts\Activate.ps1
python ".\phase 2\evaluate.py"
```

## Outputs (in `phase 2/results/`)
- `random_curves.npy`              — [n_runs, n_steps] running-max curves
- `random_steps_to_topk.json`      — success rate + median steps per K
- `random_discovery_curve.png`     — mean curve with 25–75% band
