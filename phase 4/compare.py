"""Phase 4 plot: vanilla PPO (K=1) vs planning PPO (K=5) vs random baseline.

Reads:
    phase 2/results/random_curves.npy
    phase 4/results/ppo_planning_k1_curves.npy
    phase 4/results/ppo_planning_k5_curves.npy
    phase 4/results/ppo_planning_k1_summary.json
    phase 4/results/ppo_planning_k5_summary.json

Writes:
    phase 4/results/phase4_comparison.png
    phase 4/results/phase4_learning_trend.png
    phase 4/results/phase4_comparison_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parents[1]
RANDOM = ROOT / "phase 2" / "results" / "random_curves.npy"
OUT = Path(__file__).parent / "results"


def load_run(tag: str):
    curves = np.load(OUT / f"ppo_planning_{tag}_curves.npy")
    with open(OUT / f"ppo_planning_{tag}_summary.json") as f:
        summary = json.load(f)
    return curves, summary


def main() -> None:
    rand_curves = np.load(RANDOM)
    runs = {}
    for tag in ("k1", "k5"):
        path = OUT / f"ppo_planning_{tag}_curves.npy"
        if path.exists():
            runs[tag] = load_run(tag)
    if not runs:
        raise SystemExit("No PPO runs found yet. Run ppo_planning.py first.")

    n_steps = rand_curves.shape[1]
    x = np.arange(1, n_steps + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, rand_curves.mean(0), color="C0", lw=2,
            label=f"random (n={rand_curves.shape[0]})")
    ax.fill_between(x, np.percentile(rand_curves, 25, axis=0),
                    np.percentile(rand_curves, 75, axis=0), color="C0", alpha=0.15)

    colors = {"k1": "C1", "k5": "C3"}
    names = {"k1": "PPO  (K=1, no planning)", "k5": "PPO + planning (K=5)"}
    for tag, (curves, _) in runs.items():
        m = curves.mean(0)
        ax.plot(x, m, color=colors[tag], lw=2, label=f"{names[tag]} (n={curves.shape[0]})")
        ax.fill_between(x, np.percentile(curves, 25, axis=0),
                        np.percentile(curves, 75, axis=0), color=colors[tag], alpha=0.15)

    # True top-1 from whichever PPO run has the cache populated
    top1 = max((s["final_mean_running_max"] for _, s in runs.values()), default=None)
    if top1 is not None:
        ax.axhline(top1, ls="--", color="grey", label=f"true top-1 ≈ {top1:.2f}")

    ax.set_xlabel("interventions (steps)")
    ax.set_ylabel("best head score found so far (logit-diff drop)")
    ax.set_title("Phase 4: model-based planning on live GPT-2 ablations")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "phase4_comparison.png", dpi=150)

    # Learning-trend plot from eval_history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for tag, (_, summary) in runs.items():
        hist = summary["eval_history"]
        steps = [h["step"] for h in hist]
        top1 = [h["top1_rate"] * 100 for h in hist]
        rmax = [h["final_running_max_mean"] for h in hist]
        axes[0].plot(steps, top1, "-o", color=colors[tag], lw=2, label=names[tag])
        axes[1].plot(steps, rmax, "-o", color=colors[tag], lw=2, label=names[tag])
    axes[0].set_xlabel("training step"); axes[0].set_ylabel("eval top-1 rate (%)")
    axes[0].set_title("Learning trend — top-1 success"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("training step"); axes[1].set_ylabel("eval final running-max")
    axes[1].set_title("Learning trend — best score found"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "phase4_learning_trend.png", dpi=150)

    # Summary
    summary_out = {"random": {"final_rmax_mean": float(rand_curves.mean(0)[-1])}}
    for tag, (curves, s) in runs.items():
        summary_out[tag] = {
            "final_rmax_mean": float(curves.mean(0)[-1]),
            "final_top1_rate": s["final_top1_rate"],
            "final_median_steps_to_top1": s["final_median_steps_to_top1"],
            "cache_misses_total": s["cache_misses_total"],
        }
    with open(OUT / "phase4_comparison_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    print(json.dumps(summary_out, indent=2))
    print(f"\nSaved {OUT / 'phase4_comparison.png'}")
    print(f"Saved {OUT / 'phase4_learning_trend.png'}")


if __name__ == "__main__":
    main()
