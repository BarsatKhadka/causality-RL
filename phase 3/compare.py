"""Phase 3 headline plot: PPO discovery curve vs. Phase 2 random baseline.

Reads:
    phase 2/results/random_curves.npy   [n_runs, 50]
    phase 3/results/ppo_curves.npy      [n_eval_episodes, 50]
    phase 1/results/head_scores_logitdiff.npy   (for the true-top-1 reference line)

Writes:
    phase 3/results/comparison_curve.png
    phase 3/results/comparison_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parents[1]
RANDOM_CURVES = ROOT / "phase 2" / "results" / "random_curves.npy"
PPO_CURVES = ROOT / "phase 3" / "results" / "ppo_curves.npy"
PHASE1_SCORES = ROOT / "phase 1" / "results" / "head_scores_logitdiff.npy"
OUT_DIR = ROOT / "phase 3" / "results"


def steps_to_threshold(curves: np.ndarray, threshold: float) -> np.ndarray:
    """For each row (episode), first step (1-indexed) where running_max >= threshold.

    Returns episode-length+1 if never reached (so the mean reflects misses).
    """
    n_runs, n_steps = curves.shape
    hits = curves >= threshold
    first_hit = np.where(
        hits.any(axis=1),
        hits.argmax(axis=1) + 1,
        n_steps + 1,
    )
    return first_hit


def main() -> None:
    scores = np.load(PHASE1_SCORES).flatten()
    true_top1 = float(scores.max())

    random_curves = np.load(RANDOM_CURVES)
    ppo_curves = np.load(PPO_CURVES)
    n_steps = random_curves.shape[1]
    x = np.arange(1, n_steps + 1)

    rand_mean = random_curves.mean(axis=0)
    rand_p25 = np.percentile(random_curves, 25, axis=0)
    rand_p75 = np.percentile(random_curves, 75, axis=0)

    ppo_mean = ppo_curves.mean(axis=0)
    ppo_p25 = np.percentile(ppo_curves, 25, axis=0)
    ppo_p75 = np.percentile(ppo_curves, 75, axis=0)

    # Steps-to-find-top-1 (allowing tiny numerical slack)
    eps = 1e-4
    rand_steps = steps_to_threshold(random_curves, true_top1 - eps)
    ppo_steps = steps_to_threshold(ppo_curves, true_top1 - eps)

    summary = {
        "true_top1_score": true_top1,
        "random": {
            "n_runs": int(random_curves.shape[0]),
            "top1_rate": float((rand_steps <= n_steps).mean()),
            "median_steps_to_top1": float(np.median(rand_steps[rand_steps <= n_steps]))
                if (rand_steps <= n_steps).any() else None,
            "final_running_max_mean": float(rand_mean[-1]),
        },
        "ppo": {
            "n_runs": int(ppo_curves.shape[0]),
            "top1_rate": float((ppo_steps <= n_steps).mean()),
            "median_steps_to_top1": float(np.median(ppo_steps[ppo_steps <= n_steps]))
                if (ppo_steps <= n_steps).any() else None,
            "final_running_max_mean": float(ppo_mean[-1]),
        },
    }
    with open(OUT_DIR / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"True top-1 score      : {true_top1:.4f}")
    print("\n             top-1 rate   median steps to top-1   final mean rmax")
    for name, s in [("random", summary["random"]), ("ppo", summary["ppo"])]:
        med = s["median_steps_to_top1"]
        med_str = f"{med:.1f}" if med is not None else "n/a"
        print(
            f"  {name:<6}    {s['top1_rate']*100:6.1f}%        {med_str:>5}              "
            f"{s['final_running_max_mean']:.3f}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, rand_mean, color="C0", lw=2, label=f"random (n={random_curves.shape[0]})")
    ax.fill_between(x, rand_p25, rand_p75, color="C0", alpha=0.2)
    ax.plot(x, ppo_mean, color="C3", lw=2, label=f"PPO (n={ppo_curves.shape[0]})")
    ax.fill_between(x, ppo_p25, ppo_p75, color="C3", alpha=0.2)
    ax.axhline(true_top1, ls="--", color="grey", label=f"true top-1 = {true_top1:.2f}")
    ax.set_xlabel("interventions (steps)")
    ax.set_ylabel("best head score found so far (logit-diff drop)")
    ax.set_title("Phase 3: PPO vs random — head-discovery curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_curve.png", dpi=150)
    print(f"\nSaved {OUT_DIR / 'comparison_curve.png'}")


if __name__ == "__main__":
    main()
