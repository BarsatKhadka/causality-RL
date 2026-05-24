"""Policy autopsy: load a trained Phase 5 policy and answer the question
'which head is scoring the 3.51 we keep seeing?'

For each tag (k1, k5) and a handful of held-out eval seeds:
  - Roll out the policy.
  - Log every (step, action, reward) tuple.
  - Print the single head that achieved each episode's running_max.
  - Aggregate over episodes: which heads does the policy prefer?

Outputs a per-tag picks heatmap and a JSON dump of the trace.

Run on HPC:
    python policy_autopsy.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from head_env_real import RealHeadDiscoveryEnv
from ppo_planning_real import ActorCritic, plan_action


ROOT = Path(__file__).parent
OUT = ROOT / "results"
EVAL_SEED_BASE = 10_000_000
N_EPISODES = 10


def autopsy_one(tag: str, env: RealHeadDiscoveryEnv, device):
    weights_path = OUT / f"real_{tag}_policy.pt"
    if not weights_path.exists():
        print(f"[skip] {weights_path} not found")
        return None

    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)
    n_layers = env.n_layers
    n_heads = env.n_heads_per_layer

    agent = ActorCritic(obs_dim, n_actions).to(device)
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    agent.eval()

    K = 5 if tag == "k5" else 1
    episodes = []
    all_picks_layerhead = Counter()
    rmax_actions = []

    with torch.no_grad():
        for ep in range(N_EPISODES):
            obs, info = env.reset(seed=EVAL_SEED_BASE + ep)
            baseline = info["baseline"]
            running = -np.inf
            running_action = -1
            trace = []
            for t in range(env.max_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(env.action_mask(), dtype=torch.bool, device=device).unsqueeze(0)
                a, _ = plan_action(agent, obs_t, mask_t, env, K=K, deterministic=False)
                obs, r, term, trunc, info = env.step(a)
                L, H = divmod(a, n_heads)
                all_picks_layerhead[(L, H)] += 1
                trace.append({"step": t, "action": int(a), "layer": L, "head": H, "reward": float(r)})
                if r > running:
                    running = r
                    running_action = a
                if term or trunc:
                    break
            best_L, best_H = divmod(running_action, n_heads)
            episodes.append({
                "seed": EVAL_SEED_BASE + ep,
                "baseline": baseline,
                "running_max": float(running),
                "best_action": int(running_action),
                "best_layer_head": [best_L, best_H],
                "trace": trace,
            })
            rmax_actions.append(running_action)
            print(f"[{tag}] ep={ep:>2d}  seed={EVAL_SEED_BASE+ep}  baseline={baseline:.2f}  "
                  f"running_max={running:.3f}  achieved by L{best_L}.H{best_H}")

    pick_grid = np.zeros((n_layers, n_heads), dtype=np.int32)
    for (L, H), c in all_picks_layerhead.items():
        pick_grid[L, H] = c

    print(f"\n[{tag}] head most often achieving running_max:")
    rmax_counter = Counter(rmax_actions)
    for a, c in rmax_counter.most_common(5):
        L, H = divmod(a, n_heads)
        print(f"   L{L}.H{H}  -> top in {c}/{N_EPISODES} episodes")

    print(f"\n[{tag}] top-10 most-picked heads overall (across all episodes/all steps):")
    for (L, H), c in all_picks_layerhead.most_common(10):
        print(f"   L{L}.H{H}  picked {c} times")

    return {
        "tag": tag,
        "episodes": episodes,
        "pick_grid": pick_grid.tolist(),
        "top_rmax_actions": rmax_counter.most_common(10),
    }


def main():
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device = {device}")

    env = RealHeadDiscoveryEnv(verbose=True)

    results = {}
    for tag in ("k1", "k5"):
        r = autopsy_one(tag, env, device)
        if r is not None:
            results[tag] = r
        print("=" * 70)

    with open(OUT / "policy_autopsy.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[saved] {OUT/'policy_autopsy.json'}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
        for ax, (tag, r) in zip(axes[0], results.items()):
            grid = np.array(r["pick_grid"])
            im = ax.imshow(grid, aspect="auto", cmap="viridis")
            ax.set_xlabel("head")
            ax.set_ylabel("layer")
            ax.set_title(f"{tag}: head-pick frequency  (across {N_EPISODES} eval episodes)")
            plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(OUT / "policy_autopsy_heatmap.png", dpi=150)
        print(f"[saved] {OUT/'policy_autopsy_heatmap.png'}")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
