"""Live GPT-2 ablation environment.

Each `step` actually runs a GPT-2 forward pass with the chosen attention head
zero-ablated on a small batch of induction sequences. Scores are deterministic
given the fixed test set, so we memoize on disk: a (layer, head) key seen
before is an O(1) lookup; an unseen key triggers a real forward pass.

This is the env Phase 4's model-based planner queries — including for the
candidate-simulation loop.
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

# Reuse the Phase 1 building blocks for batch construction + masked forward pass
PHASE1 = Path(__file__).parents[1] / "phase 1"
sys.path.insert(0, str(PHASE1))
from induction_dataset import (  # noqa: E402
    induction_logit_diff,
    make_distractor_tokens,
    make_induction_batch,
)
from ablation import logit_diff_with_head_ablated  # noqa: E402

from transformer_lens import HookedTransformer  # noqa: E402


SENTINEL = -1.0
REPICK_PENALTY = -1.0
DEFAULT_MAX_STEPS = 50
CACHE_PATH = Path(__file__).parent / "results" / "live_score_cache.npz"


class LiveHeadDiscoveryEnv(gym.Env):
    """Same interface as Phase 3 HeadDiscoveryEnv, but scores are computed live.

    Args:
        n_test_seqs: induction batch size per ablation. 32 is ~3x faster than
            Phase 1's 100 with very similar ranking.
        seq_len: length of induction sequences.
        max_steps: episode length.
        use_cache: if True, persist results to `CACHE_PATH` between processes.
            The cache is keyed by (layer, head) and is correct as long as
            n_test_seqs / seq_len / seeds don't change.
        device: "cpu" or "cuda".
        verbose: print every cache miss (so you can watch GPT-2 actually run).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_test_seqs: int = 32,
        seq_len: int = 60,
        max_steps: int = DEFAULT_MAX_STEPS,
        use_cache: bool = True,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose

        if self.verbose:
            print(f"[env] loading GPT-2 small on {self.device}...", flush=True)
        t0 = time.time()
        self.model = HookedTransformer.from_pretrained("gpt2", device=self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if self.verbose:
            print(f"[env] model loaded in {time.time()-t0:.1f}s", flush=True)

        self.n_layers = int(self.model.cfg.n_layers)
        self.n_heads_per_layer = int(self.model.cfg.n_heads)
        self.n_actions = self.n_layers * self.n_heads_per_layer  # 144
        self.max_steps = max_steps

        # Build fixed induction test set + distractors. Same seeds = same scores.
        self.tokens, self.target_positions = make_induction_batch(
            self.model, batch_size=n_test_seqs, seq_len=seq_len, seed=0
        )
        self.tokens = self.tokens.to(self.device)
        self.target_positions = self.target_positions.to(self.device)
        self.distractors = make_distractor_tokens(self.model, self.tokens, seed=1).to(self.device)

        # Compute baseline logit-diff once (every score is measured relative to this)
        self.baseline = induction_logit_diff(
            self.model, self.tokens, self.target_positions, self.distractors
        )
        if self.verbose:
            print(f"[env] baseline induction logit-diff = {self.baseline:.4f}", flush=True)

        # Score cache (loaded if exists, else fresh)
        self.use_cache = use_cache
        self._cache: dict[int, float] = {}
        if use_cache and CACHE_PATH.exists():
            data = np.load(CACHE_PATH)
            keys = data["keys"].tolist()
            vals = data["vals"].tolist()
            self._cache = dict(zip(keys, vals))
            if self.verbose:
                print(f"[env] loaded {len(self._cache)} cached scores from disk", flush=True)
        self._cache_misses = 0

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_actions,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        # Episode state
        self._obs: Optional[np.ndarray] = None
        self._tried: Optional[np.ndarray] = None
        self._step_count = 0
        self._running_max = -np.inf

    # ---------------- core scoring ----------------

    def query_score(self, action: int) -> float:
        """Return the logit-diff drop for ablating `action`. Cached after first call."""
        action = int(action)
        if action in self._cache:
            return self._cache[action]

        layer, head = divmod(action, self.n_heads_per_layer)
        t0 = time.time()
        ablated = logit_diff_with_head_ablated(
            self.model, self.tokens, self.target_positions, self.distractors, layer, head
        )
        score = float(self.baseline - ablated)
        self._cache[action] = score
        self._cache_misses += 1
        if self.verbose:
            print(
                f"[env] CACHE MISS L{layer}.H{head} -> score={score:+.4f} "
                f"({time.time()-t0:.2f}s, cache size={len(self._cache)}/{self.n_actions})",
                flush=True,
            )
        return score

    def save_cache(self) -> None:
        if not self.use_cache:
            return
        CACHE_PATH.parent.mkdir(exist_ok=True)
        keys = np.array(list(self._cache.keys()), dtype=np.int32)
        vals = np.array(list(self._cache.values()), dtype=np.float32)
        np.savez(CACHE_PATH, keys=keys, vals=vals)

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    # ---------------- Gym API ----------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._obs = np.full(self.n_actions, SENTINEL, dtype=np.float32)
        self._tried = np.zeros(self.n_actions, dtype=bool)
        self._step_count = 0
        self._running_max = -np.inf
        return self._obs.copy(), self._info(repicked=False)

    def step(self, action: int):
        assert self._obs is not None, "call reset() first"
        action = int(action)
        repicked = bool(self._tried[action])

        if repicked:
            reward = REPICK_PENALTY
        else:
            reward = self.query_score(action)
            self._obs[action] = reward
            self._tried[action] = True
            if reward > self._running_max:
                self._running_max = reward

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps
        return (
            self._obs.copy(),
            reward,
            terminated,
            truncated,
            self._info(repicked=repicked),
        )

    def action_mask(self) -> np.ndarray:
        assert self._tried is not None
        return ~self._tried

    def _info(self, repicked: bool) -> dict:
        rmax = self._running_max if np.isfinite(self._running_max) else 0.0
        return {
            "running_max": float(rmax),
            "tried_count": int(self._tried.sum()) if self._tried is not None else 0,
            "repicked": repicked,
        }


# ---------------- standalone warm-up: populate cache for all 144 heads ----------------

def warmup_cache() -> None:
    """Sweep every head once so subsequent training runs are fast.
    Optional convenience: PPO will trigger these on demand anyway."""
    env = LiveHeadDiscoveryEnv()
    t0 = time.time()
    for a in range(env.n_actions):
        env.query_score(a)
    env.save_cache()
    print(f"\n[warmup] all {env.n_actions} heads scored in {time.time()-t0:.1f}s, cache saved")


if __name__ == "__main__":
    warmup_cache()
