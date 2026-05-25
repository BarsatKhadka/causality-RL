"""Phase 6 — held-out transfer eval.

Loads a trained multi-task policy and tests it on the docstring task (never
seen during training). Compares against:
  - random baseline (same env, uniform action sampling)
  - oracle ceiling (per-episode best of all 144 heads)

For the trained policy we run three task-ID conditions:
  - unknown:        task_onehot = [0, 0]
  - pretend_induction: [1, 0]
  - pretend_ioi:       [0, 1]
The "unknown" condition is the strongest transfer claim: the policy must work
out it's a new task from the in-episode score feedback alone.

Usage:
    python eval_transfer.py --policy_path results/phase6_mt_k1_policy.pt --tag mt_k1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PHASE1 = Path(__file__).parents[1] / "phase 1"
sys.path.insert(0, str(PHASE1))
from induction_dataset import (  # noqa: E402
    control_mean_loss,
    load_natural_corpus,
    make_control_batch,
)
from ablation import (  # noqa: E402
    control_loss_with_head_ablated,
)

sys.path.insert(0, str(Path(__file__).parent))
from docstring_dataset import make_docstring_batch, docstring_logit_diff  # noqa: E402
from docstring_ablation import docstring_logit_diff_with_head_ablated  # noqa: E402
from ppo_vec import ActorCritic  # noqa: E402

from transformer_lens import HookedTransformer  # noqa: E402


SCORE_SCALE = 5.0
CONTROL_WEIGHT = 1.0
N_HEADS_PER_LAYER = 12
N_LAYERS = 12
N_ACTIONS = N_LAYERS * N_HEADS_PER_LAYER

EVAL_SEED_BASE = 10_000_000
N_TEST_SEQS = 32
SEQ_LEN = 60   # actual docstring prompt is longer but we don't need a fixed cap
MAX_STEPS = 50
N_EVAL_EPISODES = 30


# ---------------- standalone env-like state for docstring ----------------

class DocstringRunner:
    """A thin replacement for MultiTaskHeadDiscoveryEnv specialized to docstring.
    Same reward semantics (contrastive vs natural-text control)."""

    def __init__(self, model, device, corpus_tokens):
        self.model = model
        self.device = device
        self.corpus = corpus_tokens
        self.n_actions = N_ACTIONS
        self.max_steps = MAX_STEPS
        self._reset_episode_state()
        self._fwd_calls = 0

    def _reset_episode_state(self):
        self._tokens = None
        self._finals = None
        self._correct = None
        self._distractor = None
        self._baseline = 0.0
        self._control_tokens = None
        self._control_baseline_loss = 0.0
        self._cache: dict[int, float] = {}
        self._tried = np.zeros(self.n_actions, dtype=bool)
        self._tried_mask = np.zeros(self.n_actions, dtype=np.float32)
        self._score_vec = np.zeros(self.n_actions, dtype=np.float32)
        self._running_max = -np.inf
        self._step_count = 0

    def reset(self, seed: int):
        self._reset_episode_state()
        tokens, finals, c, d = make_docstring_batch(self.model, batch_size=N_TEST_SEQS, seed=seed)
        self._tokens = tokens.to(self.device)
        self._finals = finals.to(self.device)
        self._correct = c.to(self.device)
        self._distractor = d.to(self.device)
        self._baseline = docstring_logit_diff(
            self.model, self._tokens, self._finals, self._correct, self._distractor
        )
        self._control_tokens = make_control_batch(
            self.model, batch_size=N_TEST_SEQS, seq_len=SEQ_LEN,
            seed=seed + 13_103, corpus=self.corpus,
        ).to(self.device)
        self._control_baseline_loss = control_mean_loss(self.model, self._control_tokens)

    def query_score(self, action: int) -> float:
        if action in self._cache:
            return self._cache[action]
        layer, head = divmod(action, N_HEADS_PER_LAYER)
        ablated = docstring_logit_diff_with_head_ablated(
            self.model, self._tokens, self._finals, self._correct, self._distractor, layer, head
        )
        ablated_ctrl = control_loss_with_head_ablated(
            self.model, self._control_tokens, layer, head
        )
        task_damage = float(self._baseline - ablated)
        ctrl_damage = float(ablated_ctrl - self._control_baseline_loss)
        score = task_damage - CONTROL_WEIGHT * ctrl_damage
        self._cache[action] = score
        self._fwd_calls += 2
        return score

    def step(self, action: int):
        action = int(action)
        if self._tried[action]:
            reward = -1.0
        else:
            reward = self.query_score(action)
            self._tried_mask[action] = 1.0
            self._score_vec[action] = reward / SCORE_SCALE
            self._tried[action] = True
            if reward > self._running_max:
                self._running_max = reward
        self._step_count += 1
        return reward, self._step_count >= self.max_steps

    def action_mask(self):
        return ~self._tried


def build_obs(runner: DocstringRunner, task_onehot: np.ndarray) -> np.ndarray:
    return np.concatenate([task_onehot, runner._tried_mask, runner._score_vec]).astype(np.float32)


def run_policy(runner, agent, device, task_onehot, n_episodes):
    rmaxes = []
    pick_counter = Counter()
    rmax_action_counter = Counter()
    curves = np.zeros((n_episodes, MAX_STEPS), dtype=np.float32)
    agent.eval()
    with torch.no_grad():
        for ep in range(n_episodes):
            runner.reset(seed=EVAL_SEED_BASE + ep)
            best_action = -1
            for t in range(runner.max_steps):
                obs = build_obs(runner, task_onehot)
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(runner.action_mask(), dtype=torch.bool, device=device).unsqueeze(0)
                trunk = agent.trunk(obs_t)
                logits = agent.actor(trunk).masked_fill(~mask_t, -1e9)
                probs = torch.softmax(logits, dim=-1)
                a = int(torch.multinomial(probs[0], 1).item())
                prev_rmax = runner._running_max
                r, done = runner.step(a)
                pick_counter[a] += 1
                if runner._running_max > prev_rmax:
                    best_action = a
                curves[ep, t] = runner._running_max if np.isfinite(runner._running_max) else 0.0
                if done:
                    break
            rmaxes.append(runner._running_max)
            rmax_action_counter[best_action] += 1
    return {
        "mean_rmax": float(np.mean(rmaxes)),
        "per_episode_rmax": [float(x) for x in rmaxes],
        "rmax_actions": rmax_action_counter.most_common(10),
        "pick_top10": pick_counter.most_common(10),
        "discovery_curve_mean": curves.mean(axis=0).tolist(),
        "steps_to_within_90pct_of_final": _steps_to_threshold(curves, 0.9),
    }


def _steps_to_threshold(curves: np.ndarray, frac: float) -> float:
    """For each episode, how many picks until running_max reaches `frac` of its
    final value? Returns the mean across episodes."""
    out = []
    for ep in range(curves.shape[0]):
        final = curves[ep, -1]
        if final <= 0:
            continue
        thresh = frac * final
        hit = np.argmax(curves[ep] >= thresh) + 1
        out.append(int(hit))
    return float(np.mean(out)) if out else float("nan")


def run_random(runner, n_episodes):
    rng = np.random.default_rng(0)
    rmaxes = []
    curves = np.zeros((n_episodes, MAX_STEPS), dtype=np.float32)
    for ep in range(n_episodes):
        runner.reset(seed=EVAL_SEED_BASE + ep)
        for t in range(runner.max_steps):
            legal = np.flatnonzero(runner.action_mask())
            a = int(rng.choice(legal))
            _, done = runner.step(a)
            curves[ep, t] = runner._running_max if np.isfinite(runner._running_max) else 0.0
            if done:
                break
        rmaxes.append(runner._running_max)
    return {
        "mean_rmax": float(np.mean(rmaxes)),
        "discovery_curve_mean": curves.mean(axis=0).tolist(),
        "steps_to_within_90pct_of_final": _steps_to_threshold(curves, 0.9),
    }


def run_oracle(runner, n_episodes):
    """Ablate ALL 144 heads each episode. Reports the true per-episode best."""
    per_ep_best = []
    per_ep_best_action = []
    all_scores = np.zeros((n_episodes, runner.n_actions), dtype=np.float32)
    for ep in range(n_episodes):
        runner.reset(seed=EVAL_SEED_BASE + ep)
        scores = np.zeros(runner.n_actions, dtype=np.float32)
        for a in range(runner.n_actions):
            scores[a] = runner.query_score(a)
        all_scores[ep] = scores
        best = int(np.argmax(scores))
        per_ep_best.append(float(scores[best]))
        per_ep_best_action.append(best)
    return {
        "mean_best": float(np.mean(per_ep_best)),
        "per_episode_best": per_ep_best,
        "per_episode_best_action": per_ep_best_action,
        "all_scores": all_scores,
    }


def fmt_actions(counter):
    out = []
    for a, c in counter:
        L, H = divmod(int(a), N_HEADS_PER_LAYER)
        out.append(f"L{L}.H{H}:{c}")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True, type=str)
    ap.add_argument("--tag", default="mt_k1", type=str)
    ap.add_argument("--n_episodes", default=N_EVAL_EPISODES, type=int)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--out_dir", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", flush=True)

    print("[setup] loading GPT-2 ...", flush=True)
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    corpus = load_natural_corpus(model)
    runner = DocstringRunner(model, device, corpus)

    obs_dim = 2 + 2 * N_ACTIONS   # task_onehot + mask + scores  (n_tasks=2)
    agent = ActorCritic(obs_dim, N_ACTIONS).to(device)
    agent.load_state_dict(torch.load(args.policy_path, map_location=device))
    print(f"[setup] loaded policy from {args.policy_path}", flush=True)

    results = {"tag": args.tag, "n_episodes": args.n_episodes, "policy_path": args.policy_path}

    print("\n[1/5] random baseline on docstring ...", flush=True)
    t0 = time.time()
    results["random"] = run_random(runner, args.n_episodes)
    print(f"      mean_rmax={results['random']['mean_rmax']:.3f}  ({time.time()-t0:.0f}s)")

    for tag, oh in [("unknown", [0, 0]), ("pretend_induction", [1, 0]), ("pretend_ioi", [0, 1])]:
        print(f"\n[*] trained policy (task_onehot={oh}, '{tag}') ...", flush=True)
        t0 = time.time()
        r = run_policy(runner, agent, device, np.array(oh, dtype=np.float32), args.n_episodes)
        print(f"      mean_rmax={r['mean_rmax']:.3f}  ({time.time()-t0:.0f}s)")
        print(f"      top heads that achieved per-ep max: {fmt_actions(r['rmax_actions'])}")
        print(f"      most-picked overall:               {fmt_actions(r['pick_top10'])}")
        results[tag] = r

    print("\n[5/5] oracle (ablate all 144 heads per ep) ...", flush=True)
    t0 = time.time()
    oracle = run_oracle(runner, min(args.n_episodes, 15))
    print(f"      mean_best={oracle['mean_best']:.3f}  ({time.time()-t0:.0f}s)")
    top_actions = Counter(oracle["per_episode_best_action"]).most_common(10)
    print(f"      per-ep best heads: {fmt_actions(top_actions)}")
    results["oracle"] = {
        "n_episodes": int(oracle["all_scores"].shape[0]),
        "mean_best": oracle["mean_best"],
        "per_episode_best": oracle["per_episode_best"],
        "per_episode_best_action": oracle["per_episode_best_action"],
    }
    np.save(out_dir / f"transfer_{args.tag}_oracle_scores.npy", oracle["all_scores"])

    summary_path = out_dir / f"transfer_{args.tag}_summary.json"
    with open(summary_path, "w") as f:
        # numpy types -> python
        def _coerce(o):
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.integer): return int(o)
            return str(o)
        json.dump(results, f, indent=2, default=_coerce)
    print(f"\n[saved] {summary_path}")

    print("\n=== TRANSFER HEADLINE ({}) ===".format(args.tag))
    print(f"  random              : final={results['random']['mean_rmax']:.3f}  "
          f"steps_to_90pct={results['random']['steps_to_within_90pct_of_final']:.1f}")
    for tag in ("unknown", "pretend_induction", "pretend_ioi"):
        r = results[tag]
        print(f"  trained ({tag:<17}): final={r['mean_rmax']:.3f}  "
              f"steps_to_90pct={r['steps_to_within_90pct_of_final']:.1f}")
    print(f"  oracle ceiling      : {results['oracle']['mean_best']:.3f}")


if __name__ == "__main__":
    main()
