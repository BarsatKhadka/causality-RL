"""Phase 6 — Multi-task environment for circuit-discovery RL.

Each episode samples one task from {induction, IOI}, regenerates a fresh batch
for that task, and computes contrastive reward = task_damage - control_damage
on a paired natural-text control batch.

Observation:
    [task_onehot (T), tried_mask (144), normalized_scores (144)]
    where T = number of training tasks (currently 2).

This lets the policy learn task-conditional behavior: "for induction, prioritize
heads around L5-L7; for IOI, also try L7-L10 (name movers / S-inhibition)."

The same env class will be used for held-out evaluation on a third task
(e.g. docstring), with the task-ID position untrained — that's the
generalization probe.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

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

from transformer_lens import HookedTransformer  # noqa: E402


REPICK_PENALTY = -1.0
SCORE_SCALE = 5.0
CONTROL_WEIGHT = 1.0


# A "task" packages: a batch generator, a clean-baseline scorer, and a
# head-ablated scorer. All return Python floats.
class Task:
    def __init__(self, name: str):
        self.name = name

    def resample(self, model, seed: int, device, n_seqs: int, seq_len: int) -> dict:
        """Return a dict of task-specific tensors (already on device)."""
        raise NotImplementedError

    def baseline(self, model, state: dict) -> float:
        raise NotImplementedError

    def ablated(self, model, state: dict, layer: int, head: int) -> float:
        raise NotImplementedError


class InductionTask(Task):
    def __init__(self):
        super().__init__("induction")

    def resample(self, model, seed, device, n_seqs, seq_len):
        tokens, tp = make_induction_batch(model, batch_size=n_seqs, seq_len=seq_len, seed=seed)
        distractors = make_distractor_tokens(model, tokens, seed=seed + 7_919)
        return {
            "tokens": tokens.to(device),
            "target_positions": tp.to(device),
            "distractors": distractors.to(device),
        }

    def baseline(self, model, state):
        return induction_logit_diff(
            model, state["tokens"], state["target_positions"], state["distractors"]
        )

    def ablated(self, model, state, layer, head):
        return logit_diff_with_head_ablated(
            model, state["tokens"], state["target_positions"],
            state["distractors"], layer, head,
        )


class IOITask(Task):
    def __init__(self):
        super().__init__("ioi")

    def resample(self, model, seed, device, n_seqs, seq_len):
        tokens, finals, io, s = make_ioi_batch(model, batch_size=n_seqs, seed=seed)
        return {
            "tokens": tokens.to(device),
            "final_positions": finals.to(device),
            "io_tokens": io.to(device),
            "s_tokens": s.to(device),
        }

    def baseline(self, model, state):
        return ioi_logit_diff(
            model, state["tokens"], state["final_positions"],
            state["io_tokens"], state["s_tokens"],
        )

    def ablated(self, model, state, layer, head):
        return ioi_logit_diff_with_head_ablated(
            model, state["tokens"], state["final_positions"],
            state["io_tokens"], state["s_tokens"], layer, head,
        )


class MultiTaskHeadDiscoveryEnv(gym.Env):
    """Per-episode random task. Reward = task_damage - control_damage.
    Observation = [task_onehot, tried_mask, normalized_scores]."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        tasks: Optional[list] = None,
        n_test_seqs: int = 32,
        seq_len: int = 60,
        max_steps: int = 50,
        device: Optional[str] = None,
        model_name: str = "gpt2",
        verbose: bool = False,
        model: Optional[HookedTransformer] = None,
        corpus_tokens: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        self.n_test_seqs = n_test_seqs
        self.seq_len = seq_len
        self.max_steps = max_steps

        if model is not None:
            self.model = model
        else:
            if self.verbose:
                print(f"[env] loading {model_name} on {self.device}...", flush=True)
            t0 = time.time()
            self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            if self.verbose:
                print(f"[env] model loaded in {time.time()-t0:.1f}s", flush=True)

        self.tasks = tasks if tasks is not None else [InductionTask(), IOITask()]
        self.n_tasks = len(self.tasks)
        self.task_index_by_name = {t.name: i for i, t in enumerate(self.tasks)}

        self.n_layers = int(self.model.cfg.n_layers)
        self.n_heads_per_layer = int(self.model.cfg.n_heads)
        self.n_actions = self.n_layers * self.n_heads_per_layer

        self.obs_dim = self.n_tasks + 2 * self.n_actions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        # Pre-load natural-text corpus for the control batch (or reuse).
        if corpus_tokens is not None:
            self._corpus_tokens = corpus_tokens
        else:
            t0 = time.time()
            self._corpus_tokens = load_natural_corpus(self.model)
            if self.verbose:
                print(f"[env] corpus tokens: {self._corpus_tokens.shape[0]} "
                      f"(loaded in {time.time()-t0:.1f}s)", flush=True)

        # Per-episode state
        self._task: Optional[Task] = None
        self._task_idx: int = -1
        self._task_state: Optional[dict] = None
        self._baseline: float = 0.0
        self._control_tokens: Optional[torch.Tensor] = None
        self._control_baseline_loss: float = 0.0
        self._episode_cache: Dict[int, float] = {}
        self._tried_mask: Optional[np.ndarray] = None
        self._score_vec: Optional[np.ndarray] = None
        self._tried: Optional[np.ndarray] = None
        self._step_count = 0
        self._running_max = -np.inf
        self._episode_idx = 0
        self._episode_seed: Optional[int] = None
        self._fwd_calls = 0

    # ---------------- per-episode setup ----------------

    def _resample(self, seed: int, task_index: int) -> None:
        self._task_idx = task_index
        self._task = self.tasks[task_index]
        self._task_state = self._task.resample(
            self.model, seed, self.device, self.n_test_seqs, self.seq_len
        )
        self._baseline = self._task.baseline(self.model, self._task_state)
        self._control_tokens = make_control_batch(
            self.model,
            batch_size=self.n_test_seqs,
            seq_len=self.seq_len,
            seed=seed + 13_103,
            corpus=self._corpus_tokens,
        ).to(self.device)
        self._control_baseline_loss = control_mean_loss(self.model, self._control_tokens)

    # ---------------- core ----------------

    def query_score(self, action: int) -> float:
        action = int(action)
        if action in self._episode_cache:
            return self._episode_cache[action]
        layer, head = divmod(action, self.n_heads_per_layer)
        ablated_task = self._task.ablated(self.model, self._task_state, layer, head)
        ablated_control_loss = control_loss_with_head_ablated(
            self.model, self._control_tokens, layer, head
        )
        task_damage = float(self._baseline - ablated_task)
        control_damage = float(ablated_control_loss - self._control_baseline_loss)
        score = task_damage - CONTROL_WEIGHT * control_damage
        self._episode_cache[action] = score
        self._fwd_calls += 2
        return score

    @property
    def fwd_calls(self) -> int:
        return self._fwd_calls

    @property
    def current_task_name(self) -> str:
        return self._task.name if self._task is not None else "<unset>"

    # ---------------- Gym API ----------------

    def _build_obs(self) -> np.ndarray:
        task_onehot = np.zeros(self.n_tasks, dtype=np.float32)
        if self._task_idx >= 0:
            task_onehot[self._task_idx] = 1.0
        return np.concatenate([task_onehot, self._tried_mask, self._score_vec]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        self._episode_seed = seed

        # Task selection: explicit override via options["task"] or "task_index",
        # else sampled uniformly per episode.
        if options is not None and "task_index" in options:
            task_index = int(options["task_index"])
        elif options is not None and "task" in options:
            task_index = self.task_index_by_name[options["task"]]
        else:
            # Use the env's RNG so it's deterministic with seed.
            task_index = int(self.np_random.integers(0, self.n_tasks))

        self._resample(seed, task_index)
        self._episode_cache.clear()
        self._tried_mask = np.zeros(self.n_actions, dtype=np.float32)
        self._score_vec = np.zeros(self.n_actions, dtype=np.float32)
        self._tried = np.zeros(self.n_actions, dtype=bool)
        self._step_count = 0
        self._running_max = -np.inf
        self._episode_idx += 1
        return self._build_obs(), self._info(repicked=False)

    def step(self, action: int):
        assert self._tried is not None, "call reset() first"
        action = int(action)
        repicked = bool(self._tried[action])
        if repicked:
            reward = REPICK_PENALTY
        else:
            reward = self.query_score(action)
            self._tried_mask[action] = 1.0
            self._score_vec[action] = reward / SCORE_SCALE
            self._tried[action] = True
            if reward > self._running_max:
                self._running_max = reward
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps
        return self._build_obs(), reward, terminated, truncated, self._info(repicked=repicked)

    def action_mask(self) -> np.ndarray:
        assert self._tried is not None
        return ~self._tried

    def _info(self, repicked: bool) -> dict:
        rmax = self._running_max if np.isfinite(self._running_max) else 0.0
        return {
            "running_max": float(rmax),
            "tried_count": int(self._tried.sum()) if self._tried is not None else 0,
            "repicked": repicked,
            "episode_seed": self._episode_seed,
            "baseline": float(self._baseline),
            "control_baseline": float(self._control_baseline_loss),
            "task": self.current_task_name,
            "task_index": self._task_idx,
        }


# ---------------- smoke test ----------------

if __name__ == "__main__":
    env = MultiTaskHeadDiscoveryEnv(verbose=True, n_test_seqs=8, seq_len=40)
    print(f"obs_dim={env.obs_dim}, n_tasks={env.n_tasks}, n_actions={env.n_actions}")
    t0 = time.time()
    for ep in range(3):
        obs, info = env.reset(seed=100 + ep)
        rng = np.random.default_rng(ep)
        for t in range(10):
            legal = np.flatnonzero(env.action_mask())
            a = int(rng.choice(legal))
            obs, r, term, trunc, info = env.step(a)
        print(f"[ep {ep}] task={info['task']:9s} baseline={info['baseline']:.2f} "
              f"control_baseline={info['control_baseline']:.2f} running_max={info['running_max']:.3f}")
    print(f"total wall = {time.time()-t0:.1f}s   fwd_calls={env.fwd_calls}")
