"""Gym environment: pick attention heads, get their causal scores as reward.

The environment wraps the cached Phase 1 reward function. No GPT-2 forward
passes happen here — every "query" is a numpy lookup. This makes training
fast: PPO updates are the only real cost.

Spaces
------
Observation : Box(144,)  — entry i = score of head i if tried, else SENTINEL (-1.0)
Action      : Discrete(144) — index of the head to ablate this step
Reward      : float — the head's logit-diff drop (from Phase 1). Picking a
                      previously-tried head returns a penalty (REPICK_PENALTY).
Done        : after MAX_STEPS picks
Info        : {"running_max": best score found so far,
               "tried_count": number of distinct heads tried,
               "repicked": True if the action was already tried}
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


SENTINEL = -1.0          # "not yet tried"
REPICK_PENALTY = -1.0    # discouragement for picking the same head twice
DEFAULT_MAX_STEPS = 50

PHASE1_SCORES = (
    Path(__file__).parents[1] / "phase 1" / "results" / "head_scores_logitdiff.npy"
)


class HeadDiscoveryEnv(gym.Env):
    """Discrete-action env where each action queries one attention head's score."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scores_path: str | Path = PHASE1_SCORES,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        super().__init__()
        self.scores = np.load(scores_path).flatten().astype(np.float32)
        self.n_heads = int(self.scores.shape[0])
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_heads,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_heads)

        self._obs: np.ndarray | None = None
        self._tried: np.ndarray | None = None        # bool mask
        self._step_count: int = 0
        self._running_max: float = 0.0

    # ------------------------------------------------------------------ Gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._obs = np.full(self.n_heads, SENTINEL, dtype=np.float32)
        self._tried = np.zeros(self.n_heads, dtype=bool)
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
            reward = float(self.scores[action])
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

    # --------------------------------------------------------------- Utilities

    def action_mask(self) -> np.ndarray:
        """Bool mask, True = legal action (head not yet tried).

        PPO doesn't use this directly without modification, but the wrapper
        in `ppo_heads.py` will pull it from `env.unwrapped.action_mask()`
        to zero out logits of already-tried heads.
        """
        assert self._tried is not None, "call reset() first"
        return ~self._tried

    def _info(self, repicked: bool) -> dict:
        running_max = self._running_max if np.isfinite(self._running_max) else 0.0
        return {
            "running_max": float(running_max),
            "tried_count": int(self._tried.sum()) if self._tried is not None else 0,
            "repicked": repicked,
        }


# --------------------------------------------------------------------- smoke test

def _smoke_test() -> None:
    env = HeadDiscoveryEnv()
    print(f"n_heads={env.n_heads}, max_steps={env.max_steps}")
    print(f"true top-1 score = {env.scores.max():.4f}  (head idx {env.scores.argmax()})")

    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=0)
    total_reward = 0.0
    for t in range(env.max_steps):
        legal = np.flatnonzero(env.action_mask())
        a = int(rng.choice(legal))
        obs, r, term, trunc, info = env.step(a)
        total_reward += r
        if term or trunc:
            break
    print(
        f"random rollout: steps={t+1}  total_reward={total_reward:.3f}  "
        f"running_max={info['running_max']:.3f}  tried={info['tried_count']}"
    )

    # Repick penalty check
    obs, _ = env.reset(seed=0)
    obs, r1, *_ = env.step(0)
    obs, r2, *_, info = env.step(0)
    print(f"first pick of head 0: reward={r1:.4f}   re-pick: reward={r2:.4f}  (penalty)")
    assert r2 == REPICK_PENALTY


if __name__ == "__main__":
    _smoke_test()
