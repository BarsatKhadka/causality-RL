"""Compare the trained Phase 6 policy's picked heads against:
  (a) the per-task oracle (ablate all 144),
  (b) canonical heads from the interpretability literature.

For each of {induction, IOI, docstring}, runs:
  - Trained policy (with appropriate task priming) for N episodes
  - Oracle (full 144-head sweep) for fewer episodes (cheaper sanity)
  - Reports top-K picks, overlap with the canonical literature set,
    overlap between policy and oracle (the "did agent find what was findable").

Run:
    python compare_heads.py --policy_path results/phase6_mt_k1_policy.pt --tag mt_k1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PHASE1 = Path(__file__).parents[1] / "phase 1"
sys.path.insert(0, str(PHASE1))
from induction_dataset import (  # noqa: E402
    control_mean_loss,
    induction_logit_diff,
    load_natural_corpus,
    make_control_batch,
    make_distractor_tokens,
    make_induction_batch,
)
from ablation import (  # noqa: E402
    control_loss_with_head_ablated,
    logit_diff_with_head_ablated,
)

sys.path.insert(0, str(Path(__file__).parent))
from ioi_dataset import ioi_logit_diff, make_ioi_batch  # noqa: E402
from ioi_ablation import ioi_logit_diff_with_head_ablated  # noqa: E402
from docstring_dataset import docstring_logit_diff, make_docstring_batch  # noqa: E402
from docstring_ablation import docstring_logit_diff_with_head_ablated  # noqa: E402
from ppo_vec import ActorCritic  # noqa: E402

from transformer_lens import HookedTransformer  # noqa: E402


N_HEADS_PER_LAYER = 12
N_LAYERS = 12
N_ACTIONS = N_LAYERS * N_HEADS_PER_LAYER
SCORE_SCALE = 5.0
CONTROL_WEIGHT = 1.0
MAX_STEPS = 50
N_TEST_SEQS = 32
SEQ_LEN = 60
EVAL_SEED_BASE = 10_000_000

# ---------------- canonical heads from the literature ----------------

# ---- Verified canonical sets ----
#
# Induction heads in GPT-2 small: the most commonly cited set from Olsson 2022
# / Elhage 2021 / TransformerLens demos. Wang 2022 (Fig 7) confirms 5.5, 5.8,
# 5.9, 6.9 as the induction heads active in the IOI circuit.
CANONICAL_INDUCTION = {
    (5, 1), (5, 5), (6, 9), (7, 2), (7, 10),
}

# IOI heads in GPT-2 small (Wang et al. 2022, "Interpretability in the Wild",
# Figure 7 + Section 3). All heads grouped by sub-role.
IOI_NAME_MOVERS         = {(9, 6), (9, 9), (10, 0)}
IOI_NEGATIVE_NMS        = {(10, 7), (11, 10)}
IOI_S_INHIBITION        = {(7, 3), (7, 9), (8, 6), (8, 10)}
IOI_BACKUP_NMS          = {(10, 10), (10, 2), (10, 6), (11, 2), (11, 9), (9, 0), (9, 7), (10, 1)}
IOI_INDUCTION_IN_IOI    = {(5, 5), (5, 8), (5, 9), (6, 9)}
IOI_DUPLICATE_TOKEN     = {(0, 1), (0, 10), (3, 0)}
IOI_PREVIOUS_TOKEN      = {(2, 2), (4, 11)}

CANONICAL_IOI = (
    IOI_NAME_MOVERS | IOI_NEGATIVE_NMS | IOI_S_INHIBITION | IOI_BACKUP_NMS
    | IOI_INDUCTION_IN_IOI | IOI_DUPLICATE_TOKEN | IOI_PREVIOUS_TOKEN
)

# Docstring: Heimersheim & Janiak 2023 worked on a 4-layer attention-only toy
# model, NOT GPT-2 small. Their L0-L3 head indices do not correspond to GPT-2
# heads. We therefore have no published canonical set for the docstring task
# in GPT-2 small — we report only oracle-vs-policy agreement, not canonical
# overlap, for that task.
CANONICAL_DOCSTRING: set = set()

CANONICAL = {
    "induction": CANONICAL_INDUCTION,
    "ioi": CANONICAL_IOI,
    "docstring": CANONICAL_DOCSTRING,
}

# Sub-categories used to break down IOI overlap in the output.
IOI_SUBCATS = {
    "name_movers": IOI_NAME_MOVERS,
    "negative_NMs": IOI_NEGATIVE_NMS,
    "s_inhibition": IOI_S_INHIBITION,
    "backup_NMs": IOI_BACKUP_NMS,
    "induction_in_ioi": IOI_INDUCTION_IN_IOI,
    "duplicate_token": IOI_DUPLICATE_TOKEN,
    "previous_token": IOI_PREVIOUS_TOKEN,
}


# ---------------- generic runner for any task ----------------

class TaskRunner:
    """Replays the contrastive-reward env for an arbitrary task."""

    def __init__(self, model, device, corpus_tokens, task_name: str):
        self.model = model
        self.device = device
        self.corpus = corpus_tokens
        self.task = task_name
        self.n_actions = N_ACTIONS
        self.max_steps = MAX_STEPS
        self._fwd_calls = 0
        self._reset_state()

    def _reset_state(self):
        self._state: Dict = {}
        self._baseline = 0.0
        self._control_tokens = None
        self._control_baseline_loss = 0.0
        self._cache: Dict[int, float] = {}
        self._tried = np.zeros(self.n_actions, dtype=bool)
        self._tried_mask = np.zeros(self.n_actions, dtype=np.float32)
        self._score_vec = np.zeros(self.n_actions, dtype=np.float32)
        self._running_max = -np.inf
        self._step_count = 0

    def _make_task_batch(self, seed):
        if self.task == "induction":
            tokens, tp = make_induction_batch(self.model, batch_size=N_TEST_SEQS, seq_len=SEQ_LEN, seed=seed)
            d = make_distractor_tokens(self.model, tokens, seed=seed + 7_919)
            return {"tokens": tokens.to(self.device),
                    "target_positions": tp.to(self.device),
                    "distractors": d.to(self.device)}
        elif self.task == "ioi":
            tokens, finals, io, s = make_ioi_batch(self.model, batch_size=N_TEST_SEQS, seed=seed)
            return {"tokens": tokens.to(self.device),
                    "final_positions": finals.to(self.device),
                    "io_tokens": io.to(self.device),
                    "s_tokens": s.to(self.device)}
        elif self.task == "docstring":
            tokens, finals, c, dd = make_docstring_batch(self.model, batch_size=N_TEST_SEQS, seed=seed)
            return {"tokens": tokens.to(self.device),
                    "final_positions": finals.to(self.device),
                    "correct_tokens": c.to(self.device),
                    "distractor_tokens": dd.to(self.device)}
        else:
            raise ValueError(self.task)

    def _task_baseline(self):
        s = self._state
        if self.task == "induction":
            return induction_logit_diff(self.model, s["tokens"], s["target_positions"], s["distractors"])
        if self.task == "ioi":
            return ioi_logit_diff(self.model, s["tokens"], s["final_positions"], s["io_tokens"], s["s_tokens"])
        if self.task == "docstring":
            return docstring_logit_diff(self.model, s["tokens"], s["final_positions"],
                                        s["correct_tokens"], s["distractor_tokens"])

    def _task_ablated(self, layer, head):
        s = self._state
        if self.task == "induction":
            return logit_diff_with_head_ablated(
                self.model, s["tokens"], s["target_positions"], s["distractors"], layer, head)
        if self.task == "ioi":
            return ioi_logit_diff_with_head_ablated(
                self.model, s["tokens"], s["final_positions"], s["io_tokens"], s["s_tokens"], layer, head)
        if self.task == "docstring":
            return docstring_logit_diff_with_head_ablated(
                self.model, s["tokens"], s["final_positions"],
                s["correct_tokens"], s["distractor_tokens"], layer, head)

    def reset(self, seed):
        self._reset_state()
        self._state = self._make_task_batch(seed)
        self._baseline = self._task_baseline()
        self._control_tokens = make_control_batch(
            self.model, batch_size=N_TEST_SEQS, seq_len=SEQ_LEN,
            seed=seed + 13_103, corpus=self.corpus,
        ).to(self.device)
        self._control_baseline_loss = control_mean_loss(self.model, self._control_tokens)

    def query_score(self, action: int) -> float:
        if action in self._cache:
            return self._cache[action]
        layer, head = divmod(action, N_HEADS_PER_LAYER)
        ablated = self._task_ablated(layer, head)
        ablated_ctrl = control_loss_with_head_ablated(self.model, self._control_tokens, layer, head)
        score = float(self._baseline - ablated) - CONTROL_WEIGHT * float(ablated_ctrl - self._control_baseline_loss)
        self._cache[action] = score
        self._fwd_calls += 2
        return score

    def step(self, action: int):
        action = int(action)
        if self._tried[action]:
            r = -1.0
        else:
            r = self.query_score(action)
            self._tried_mask[action] = 1.0
            self._score_vec[action] = r / SCORE_SCALE
            self._tried[action] = True
            if r > self._running_max:
                self._running_max = r
        self._step_count += 1
        return r, self._step_count >= self.max_steps

    def action_mask(self):
        return ~self._tried


def build_obs(runner, task_onehot):
    return np.concatenate([task_onehot, runner._tried_mask, runner._score_vec]).astype(np.float32)


# ---------------- runners ----------------

def run_policy(runner, agent, device, task_onehot, n_episodes):
    pick_counter = Counter()
    rmax_action_counter = Counter()
    rmaxes = []
    agent.eval()
    with torch.no_grad():
        for ep in range(n_episodes):
            runner.reset(EVAL_SEED_BASE + ep)
            best_action = -1
            prev_rmax = -np.inf
            for t in range(runner.max_steps):
                obs = build_obs(runner, task_onehot)
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(runner.action_mask(), dtype=torch.bool, device=device).unsqueeze(0)
                trunk = agent.trunk(obs_t)
                logits = agent.actor(trunk).masked_fill(~mask_t, -1e9)
                probs = torch.softmax(logits, dim=-1)
                a = int(torch.multinomial(probs[0], 1).item())
                pick_counter[a] += 1
                prev = runner._running_max
                r, done = runner.step(a)
                if runner._running_max > prev:
                    best_action = a
                if done:
                    break
            rmaxes.append(runner._running_max)
            rmax_action_counter[best_action] += 1
    return {
        "mean_rmax": float(np.mean(rmaxes)),
        "top_picks": pick_counter.most_common(10),
        "top_rmax_actions": rmax_action_counter.most_common(10),
    }


def run_oracle(runner, n_episodes):
    scores_all = np.zeros((n_episodes, N_ACTIONS), dtype=np.float32)
    per_ep_best_action = []
    per_ep_best_score = []
    for ep in range(n_episodes):
        runner.reset(EVAL_SEED_BASE + ep)
        for a in range(N_ACTIONS):
            scores_all[ep, a] = runner.query_score(a)
        best = int(np.argmax(scores_all[ep]))
        per_ep_best_action.append(best)
        per_ep_best_score.append(float(scores_all[ep, best]))
    mean_per_head = scores_all.mean(axis=0)
    full_order = list(np.argsort(-mean_per_head))
    return {
        "mean_best": float(np.mean(per_ep_best_score)),
        "per_episode_best_action": per_ep_best_action,
        "top10_by_mean": full_order[:10],
        "full_order_by_mean": full_order,           # all 144, descending
        "mean_per_head": mean_per_head.tolist(),
    }


# ---------------- pretty printing ----------------

def fmt_action(a: int) -> str:
    L, H = divmod(int(a), N_HEADS_PER_LAYER)
    return f"L{L}.H{H}"


def fmt_picks(counter_or_list, k=10) -> str:
    items = counter_or_list if isinstance(counter_or_list, list) else counter_or_list
    parts = []
    for entry in items[:k]:
        if isinstance(entry, tuple):
            a, c = entry
            parts.append(f"{fmt_action(a)}({c})")
        else:
            parts.append(fmt_action(entry))
    return " ".join(parts)


def overlap(picked: List[int], canonical: set, k: int) -> Tuple[int, int]:
    top_k = [int(p) for p, _ in picked[:k]] if isinstance(picked[0], tuple) else [int(p) for p in picked[:k]]
    top_lh = {(divmod(p, N_HEADS_PER_LAYER)) for p in top_k}
    return len(top_lh & canonical), len(canonical)


def rank_of_canonicals(ordered_actions: List[int], canonical: set) -> Dict[str, int]:
    """For each canonical (L,H), find its rank in the ordered list.
    Rank starts at 1. Returns {LX.HY: rank}.  Heads not in the list get rank 145.
    `ordered_actions` may be a list of ints OR a list of (action, count) pairs."""
    if not canonical:
        return {}
    if ordered_actions and isinstance(ordered_actions[0], (list, tuple)):
        actions = [int(a) for a, _ in ordered_actions]
    else:
        actions = [int(a) for a in ordered_actions]
    rank_by_action = {a: i + 1 for i, a in enumerate(actions)}
    out = {}
    for L, H in canonical:
        a = L * N_HEADS_PER_LAYER + H
        out[f"L{L}.H{H}"] = rank_by_action.get(a, N_ACTIONS + 1)
    return out


def fmt_ranks(rank_dict: Dict[str, int]) -> str:
    items = sorted(rank_dict.items(), key=lambda kv: kv[1])
    return "  ".join(f"{name}=#{r}" for name, r in items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--tag", default="mt_k1")
    ap.add_argument("--n_policy_episodes", default=20, type=int)
    ap.add_argument("--n_oracle_episodes", default=10, type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")
    print("[setup] loading GPT-2 ...")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    corpus = load_natural_corpus(model)

    obs_dim = 2 + 2 * N_ACTIONS
    agent = ActorCritic(obs_dim, N_ACTIONS).to(device)
    agent.load_state_dict(torch.load(args.policy_path, map_location=device))
    print(f"[setup] loaded policy: {args.policy_path}\n")

    # Match the task priming to the task being tested.
    TASK_PRIMING = {
        "induction": np.array([1, 0], dtype=np.float32),
        "ioi":       np.array([0, 1], dtype=np.float32),
        "docstring": np.array([1, 0], dtype=np.float32),  # transferred via induction prior
    }

    out = {"tag": args.tag, "policy_path": args.policy_path}
    for task in ("induction", "ioi", "docstring"):
        print("=" * 78)
        print(f"TASK: {task}   priming={TASK_PRIMING[task].tolist()}")
        print("=" * 78)
        runner = TaskRunner(model, device, corpus, task_name=task)
        pol = run_policy(runner, agent, device, TASK_PRIMING[task], args.n_policy_episodes)
        ora = run_oracle(runner, args.n_oracle_episodes)

        # Canonical lookup
        canon = CANONICAL[task]
        canon_str = ", ".join(f"L{L}.H{H}" for L, H in sorted(canon))

        # Overlaps
        ov5_pol, n_canon = overlap(pol["top_picks"], canon, 5)
        ov10_pol, _      = overlap(pol["top_picks"], canon, 10)
        ov5_ora, _       = overlap(ora["top10_by_mean"], canon, 5)
        ov10_ora, _      = overlap(ora["top10_by_mean"], canon, 10)

        # Per-canonical-head ranks: where the policy/oracle place each canonical head
        policy_ranks = rank_of_canonicals(pol["top_picks"], canon)
        oracle_ranks = rank_of_canonicals(ora["full_order_by_mean"], canon)

        print(f"\n  Canonical heads ({task}): {canon_str}  [n={n_canon}]")
        print(f"\n  Policy top-10 most picked   : {fmt_picks(pol['top_picks'], 10)}")
        print(f"  Policy top heads at peak    : {fmt_picks(pol['top_rmax_actions'], 5)}")
        print(f"  Oracle top-10 by mean score : {fmt_picks(ora['top10_by_mean'], 10)}")
        print(f"\n  Policy mean rmax  : {pol['mean_rmax']:.3f}")
        print(f"  Oracle mean best  : {ora['mean_best']:.3f}")
        print(f"  Gap (policy-oracle): {pol['mean_rmax'] - ora['mean_best']:+.3f}")
        if n_canon:
            print(f"\n  Overlap with canonical literature (top-5 / top-10):")
            print(f"    policy : {ov5_pol}/{n_canon}  | {ov10_pol}/{n_canon}")
            print(f"    oracle : {ov5_ora}/{n_canon}  | {ov10_ora}/{n_canon}")
            print(f"\n  Canonical heads — ranking in agent's pick list (1 = most picked):")
            print(f"    {fmt_ranks(policy_ranks)}")
            print(f"  Canonical heads — ranking in oracle's mean-score order:")
            print(f"    {fmt_ranks(oracle_ranks)}")
        if task == "ioi":
            print(f"\n  IOI sub-category overlap (policy top-10 vs canon set):")
            for cat_name, cat_set in IOI_SUBCATS.items():
                ov, n = overlap(pol["top_picks"], cat_set, 10)
                print(f"    {cat_name:<20s}: {ov}/{n}")
        print()

        out[task] = {
            "policy_mean_rmax": pol["mean_rmax"],
            "oracle_mean_best": ora["mean_best"],
            "policy_top_picks": pol["top_picks"],
            "policy_top_rmax_actions": pol["top_rmax_actions"],
            "oracle_top10_by_mean": [int(a) for a in ora["top10_by_mean"]],
            "canonical_set": [list(t) for t in canon],
            "overlap_policy_top5": ov5_pol,
            "overlap_policy_top10": ov10_pol,
            "overlap_oracle_top5": ov5_ora,
            "overlap_oracle_top10": ov10_ora,
            "n_canonical": n_canon,
            "canonical_ranks_policy": policy_ranks,
            "canonical_ranks_oracle": oracle_ranks,
        }
        if task == "ioi":
            out[task]["ioi_subcat_overlap_policy"] = {
                cat: overlap(pol["top_picks"], cs, 10)[0] for cat, cs in IOI_SUBCATS.items()
            }

    summary_path = out_dir / f"compare_heads_{args.tag}.json"
    with open(summary_path, "w") as f:
        def _c(o):
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.integer): return int(o)
            return str(o)
        json.dump(out, f, indent=2, default=_c)
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
