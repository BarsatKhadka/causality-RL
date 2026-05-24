"""Phase 5 plots:
  - phase5_discovery_curves.png   : best-found-so-far vs step (random vs K=1 vs K=5)
  - phase5_learning_trend.png     : eval_mean_running_max vs training_step (K=1 vs K=5)
  - phase5_summary.json           : headline numbers

Random baseline is generated on-the-fly here using the held-out eval seeds,
so it shares the same per-episode randomness as PPO eval (apples-to-apples).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from head_env_real import RealHeadDiscoveryEnv  # noqa: E402


ROOT = Path(__file__).parent
OUT = ROOT / "results"
EVAL_SEED_BASE = 10_000_000
N_RANDOM_EPISODES = 30


def random_baseline_curves(n_episodes: int, seed_base: int) -> np.ndarray:
    env = RealHeadDiscoveryEnv(verbose=False)
    curves = np.zeros((n_episodes, env.max_steps), dtype=np.float32)
    rng = np.random.default_rng(0)
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_base + ep)
        running = -np.inf
        for t in range(env.max_steps):
            legal = np.flatnonzero(env.action_mask())
            a = int(rng.choice(legal))
            obs, r, term, trunc, info = env.step(a)
            running = max(running, info["running_max"])
            curves[ep, t] = running
            if term or trunc:
                break
    return curves


def main() -> None:
    OUT.mkdir(exist_ok=True)

    rand_path = OUT / "real_random_curves.npy"
    if rand_path.exists():
        rand_curves = np.load(rand_path)
        print(f"[compare] reusing cached random baseline: {rand_curves.shape}")
    else:
        print("[compare] generating random baseline (live GPT-2)...")
        rand_curves = random_baseline_curves(N_RANDOM_EPISODES, EVAL_SEED_BASE)
        np.save(rand_path, rand_curves)

    runs = {}
    for tag in ("k1", "k5"):
        cpath = OUT / f"real_{tag}_curves.npy"
        spath = OUT / f"real_{tag}_summary.json"
        if cpath.exists() and spath.exists():
            with open(spath) as f:
                summary = json.load(f)
            runs[tag] = (np.load(cpath), summary)

    if not runs:
        raise SystemExit("No PPO runs found. Run ppo_planning_real.py first.")

    n_steps = rand_curves.shape[1]
    x = np.arange(1, n_steps + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, rand_curves.mean(0), color="C0", lw=2, label=f"random (n={rand_curves.shape[0]})")
    ax.fill_between(x, np.percentile(rand_curves, 25, axis=0),
                    np.percentile(rand_curves, 75, axis=0), color="C0", alpha=0.15)

    colors = {"k1": "C1", "k5": "C3"}
    names = {"k1": "PPO (K=1, no planning)", "k5": "PPO + planning (K=5)"}
    for tag, (curves, _) in runs.items():
        m = curves.mean(0)
        ax.plot(x, m, color=colors[tag], lw=2, label=f"{names[tag]} (n={curves.shape[0]})")
        ax.fill_between(x, np.percentile(curves, 25, axis=0),
                        np.percentile(curves, 75, axis=0), color=colors[tag], alpha=0.15)
    ax.set_xlabel("interventions (steps)")
    ax.set_ylabel("best head score found so far (logit-diff drop)")
    ax.set_title("Phase 5: live GPT-2, varying induction batch each episode")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "phase5_discovery_curves.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, (_, summary) in runs.items():
        hist = summary["eval_history"]
        steps = [h["step"] for h in hist]
        rmax = [h["eval_mean_running_max"] for h in hist]
        ax.plot(steps, rmax, "-o", color=colors[tag], lw=2, label=names[tag])
    ax.axhline(rand_curves.mean(0)[-1], ls="--", color="C0",
               label=f"random baseline final = {rand_curves.mean(0)[-1]:.2f}")
    ax.set_xlabel("training step")
    ax.set_ylabel("eval mean running-max (held-out seeds)")
    ax.set_title("Learning trend — does the policy improve over training?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "phase5_learning_trend.png", dpi=150)

    summary_out = {
        "random": {
            "n_episodes": int(rand_curves.shape[0]),
            "mean_final_running_max": float(rand_curves.mean(0)[-1]),
        }
    }
    for tag, (curves, s) in runs.items():
        summary_out[tag] = {
            "n_episodes": int(curves.shape[0]),
            "mean_final_running_max": float(curves.mean(0)[-1]),
            "fwd_calls_total": s["fwd_calls_total"],
            "wall_time_sec": s["wall_time_sec"],
            "device": s["device"],
        }
    with open(OUT / "phase5_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    print(json.dumps(summary_out, indent=2))
    print(f"\nSaved {OUT/'phase5_discovery_curves.png'}")
    print(f"Saved {OUT/'phase5_learning_trend.png'}")


if __name__ == "__main__":
    main()
