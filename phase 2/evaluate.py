"""Run random search many times and plot the average discovery curve.

Inputs: Phase 1's saved scores (`phase 1/results/head_scores_logitdiff.npy`).
Outputs: `phase 2/results/`
    - random_curves.npy            : [n_runs, n_steps] running max
    - random_steps_to_topk.json    : median/mean steps to find top-K, per K
    - random_discovery_curve.png   : the plot

Configurable: number of runs, episode length, which K's to track.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from random_agent import run_episode


PHASE1_SCORES = Path(__file__).parents[1] / "phase 1" / "results" / "head_scores_logitdiff.npy"
OUT_DIR = Path(__file__).parent / "results"

N_RUNS = 100
N_STEPS = 50          # per the plan: ~50 interventions per episode
KS = (1, 3, 5, 10)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    scores = np.load(PHASE1_SCORES).flatten()  # [144]
    n_heads = scores.shape[0]
    print(f"Loaded {n_heads} head scores. Top-5: {np.sort(scores)[-5:][::-1]}")

    # Print the ground-truth top-K so the user sees what we're chasing.
    ranking = np.argsort(-scores)
    print("\nGround-truth ranking (top 10):")
    for r, idx in enumerate(ranking[:10], 1):
        L, H = divmod(int(idx), 12)
        print(f"  {r:2d}. L{L}.H{H}  score={scores[idx]:+.4f}")

    curves = np.zeros((N_RUNS, N_STEPS), dtype=np.float32)
    steps_to: dict[int, list[int | None]] = {K: [] for K in KS}

    for seed in range(N_RUNS):
        rng = np.random.default_rng(seed)
        ep = run_episode(scores, n_steps=N_STEPS, rng=rng)
        curves[seed] = ep["running_max"]
        for K in KS:
            steps_to[K].append(ep[f"steps_to_top{K}"])

    np.save(OUT_DIR / "random_curves.npy", curves)

    # Summary stats. For "steps to find top-K", treat None (not found) as N_STEPS+1
    # for the mean — but also report the success rate honestly.
    summary = {"n_runs": N_RUNS, "n_steps": N_STEPS, "topK": {}}
    for K in KS:
        vals = steps_to[K]
        found = [v for v in vals if v is not None]
        success_rate = len(found) / N_RUNS
        median = float(np.median(found)) if found else None
        mean = float(np.mean(found)) if found else None
        summary["topK"][f"top{K}"] = {
            "success_rate": success_rate,
            "median_steps_if_found": median,
            "mean_steps_if_found": mean,
        }
    with open(OUT_DIR / "random_steps_to_topk.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nRandom search summary (100 runs, 50 steps each):")
    for K in KS:
        s = summary["topK"][f"top{K}"]
        med = s["median_steps_if_found"]
        sr = s["success_rate"]
        med_str = f"{med:.1f}" if med is not None else "n/a"
        print(f"  top-{K:<2}: found in {sr*100:5.1f}% of runs   median steps = {med_str}")

    try:
        import matplotlib.pyplot as plt
        mean_curve = curves.mean(axis=0)
        p25 = np.percentile(curves, 25, axis=0)
        p75 = np.percentile(curves, 75, axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(1, N_STEPS + 1)
        ax.plot(x, mean_curve, label="random search (mean)", color="C0", lw=2)
        ax.fill_between(x, p25, p75, color="C0", alpha=0.2, label="25–75% band")
        ax.axhline(scores.max(), ls="--", color="grey", label=f"true top-1 score = {scores.max():.2f}")
        ax.set_xlabel("interventions (steps)")
        ax.set_ylabel("best head-score found so far (logit-diff drop)")
        ax.set_title("Phase 2 baseline: random search discovery curve")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "random_discovery_curve.png", dpi=150)
        print(f"\nSaved {OUT_DIR / 'random_discovery_curve.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
