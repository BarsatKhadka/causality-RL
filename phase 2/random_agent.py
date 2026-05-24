"""Phase 2 — Random search baseline.

The agent picks unseen heads uniformly at random and queries the Phase 1
reward function. We cache Phase 1's scores so each "query" is O(1) — no
need to re-run GPT-2. The science is the same; we're just not wasting GPUs.
"""

from __future__ import annotations

import numpy as np


def random_search_run(
    scores_flat: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One episode: pick `n_steps` unseen heads uniformly at random.

    Returns the running max-score curve, shape [n_steps].
    Entry i = best score among the first (i+1) picks.
    """
    n_heads = scores_flat.shape[0]
    order = rng.permutation(n_heads)[:n_steps]
    picked_scores = scores_flat[order]
    running_max = np.maximum.accumulate(picked_scores)
    return running_max


def steps_to_find_topk(
    picked_indices: np.ndarray,
    topk_set: set[int],
) -> int | None:
    """Return the first step (1-indexed) at which ALL of `topk_set` have been picked.

    None if not yet found within the episode.
    """
    remaining = set(topk_set)
    for i, idx in enumerate(picked_indices, start=1):
        remaining.discard(int(idx))
        if not remaining:
            return i
    return None


def run_episode(
    scores_flat: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
) -> dict:
    """One full episode. Returns the discovery curve plus steps-to-find for several K."""
    n_heads = scores_flat.shape[0]
    order = rng.permutation(n_heads)[:n_steps]
    picked_scores = scores_flat[order]
    running_max = np.maximum.accumulate(picked_scores)

    # Ground-truth top-K head indices (by Phase 1 score, descending)
    ranking = np.argsort(-scores_flat)
    out = {"running_max": running_max}
    for K in (1, 3, 5, 10):
        topk_set = set(int(i) for i in ranking[:K])
        out[f"steps_to_top{K}"] = steps_to_find_topk(order, topk_set)
    return out
