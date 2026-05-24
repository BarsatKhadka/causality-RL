"""Phase 5 env — fully live GPT-2 ablation, fresh induction batch each episode.

Key change from Phase 4: every reset() regenerates the induction sequences
using a fresh seed. This means the *scores fluctuate per episode* — the head
that's best for episode 17's sequences may not be best for episode 18's. The
agent cannot memorize "always pick head 22"; it must learn which *regions of
the network* tend to score high.

Within an episode the scores are deterministic (the test batch is fixed for
that episode), so we keep a small in-memory cache that gets cleared on reset.
This makes planning (K>1 candidates per step) efficient without leaking
information across episodes.

No persistent disk cache. No precomputation. Every score = real GPT-2 forward pass.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

PHASE1 = Path(__file__).parents[1] / "phase 1"
sys.path.insert(0, str(PHASE1))
from induction_dataset import (  # noqa: E402
    induction_logit_diff,
    make_distractor_tokens,
    make_induction_batch,
)
from ablation import logit_diff_with_head_ablated  # noqa: E402

from transformer_lens import HookedTransformer  # noqa: E402


REPICK_PENALTY = -1.0
# Score normalization: divide tried-head rewards by this so the obs channel
# lives in roughly [0, 1.5]. Picked from observed baselines (~9.5).
SCORE_SCALE = 10.0


class RealHeadDiscoveryEnv(gym.Env):
    """Fully live env. Per-episode induction batch + per-episode score cache."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_test_seqs: int = 32,
        seq_len: int = 60,
        max_steps: int = 50,
        device: Optional[str] = None,
        model_name: str = "gpt2",
        verbose: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        self.n_test_seqs = n_test_seqs
        self.seq_len = seq_len
        self.max_steps = max_steps

        if self.verbose:
            print(f"[env] loading {model_name} on {self.device}...", flush=True)
        t0 = time.time()
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if self.verbose:
            print(f"[env] model loaded in {time.time()-t0:.1f}s", flush=True)

        self.n_layers = int(self.model.cfg.n_layers)
        self.n_heads_per_layer = int(self.model.cfg.n_heads)
        self.n_actions = self.n_layers * self.n_heads_per_layer

        # 2-channel obs: [tried_mask (0/1), normalized_score (0 for untried, reward/SCORE_SCALE for tried)]
        self.obs_dim = 2 * self.n_actions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        # Set lazily on reset()
        self._tokens = None
        self._target_positions = None
        self._distractors = None
        self._baseline = 0.0
        self._episode_cache: dict[int, float] = {}
        self._tried_mask: Optional[np.ndarray] = None
        self._score_vec: Optional[np.ndarray] = None
        self._tried: Optional[np.ndarray] = None
        self._step_count = 0
        self._running_max = -np.inf
        self._episode_idx = 0
        self._episode_seed: Optional[int] = None
        self._fwd_calls = 0

    # ---------------- per-episode batch ----------------

    def _resample_batch(self, seed: int) -> None:
        """Generate a fresh induction batch using `seed`. Recomputes baseline."""
        tokens, tp = make_induction_batch(
            self.model, batch_size=self.n_test_seqs, seq_len=self.seq_len, seed=seed
        )
        self._tokens = tokens.to(self.device)
        self._target_positions = tp.to(self.device)
        # distractor seed offset so it doesn't collide with batch seed
        self._distractors = make_distractor_tokens(self.model, self._tokens, seed=seed + 7_919).to(self.device)
        self._baseline = induction_logit_diff(
            self.model, self._tokens, self._target_positions, self._distractors
        )

    # ---------------- core ----------------

    def query_score(self, action: int) -> float:
        """Real GPT-2 ablation score for the current episode's batch."""
        action = int(action)
        if action in self._episode_cache:
            return self._episode_cache[action]
        layer, head = divmod(action, self.n_heads_per_layer)
        ablated = logit_diff_with_head_ablated(
            self.model, self._tokens, self._target_positions, self._distractors, layer, head
        )
        score = float(self._baseline - ablated)
        self._episode_cache[action] = score
        self._fwd_calls += 1
        return score

    @property
    def fwd_calls(self) -> int:
        return self._fwd_calls

    # ---------------- Gym API ----------------

    def _build_obs(self) -> np.ndarray:
        return np.concatenate([self._tried_mask, self._score_vec]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Per-episode seed: prefer explicit, else derive from episode index + offset
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        self._episode_seed = seed
        self._resample_batch(seed)
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
        }


# ---------------- standalone smoke test ----------------

if __name__ == "__main__":
    env = RealHeadDiscoveryEnv(verbose=True)
    print(f"\n[smoke] n_actions={env.n_actions}, device={env.device}")
    t0 = time.time()
    for ep in range(2):
        obs, info = env.reset(seed=42 + ep)
        rng = np.random.default_rng(ep)
        rmax = -np.inf
        for t in range(env.max_steps):
            legal = np.flatnonzero(env.action_mask())
            a = int(rng.choice(legal))
            obs, r, term, trunc, info = env.step(a)
            rmax = max(rmax, info["running_max"])
            if term or trunc:
                break
        print(f"[smoke] ep={ep} seed={info['episode_seed']} baseline={info['baseline']:.3f} "
              f"running_max={rmax:.3f} fwd_calls_total={env.fwd_calls}")
    print(f"[smoke] total wall time = {time.time()-t0:.1f}s")
