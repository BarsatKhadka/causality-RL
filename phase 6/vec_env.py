"""Minimal vectorized env: N MultiTaskHeadDiscoveryEnv instances sharing one
GPT-2 instance + one corpus tensor on the GPU.

Why not gymnasium SyncVectorEnv: each sub-env there would pickle-copy the
model. We want exactly one GPT-2 on GPU and many gym states pointing at it.

The wrapper exposes a small numpy-style API used by `ppo_vec.py`:
    reset(seeds=None)      -> obs [N, D]
    step(actions)          -> obs, rewards, terminated, truncated, infos
    action_masks()         -> bool array [N, n_actions]
    fwd_calls (property)   -> int total across all sub-envs
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from transformer_lens import HookedTransformer

from multitask_env import (
    InductionTask,
    IOITask,
    MultiTaskHeadDiscoveryEnv,
)

from induction_dataset import load_natural_corpus


class VecMultiTaskEnv:
    def __init__(
        self,
        n_envs: int,
        device: str = "cuda",
        model_name: str = "gpt2",
        n_test_seqs: int = 32,
        seq_len: int = 60,
        max_steps: int = 50,
        verbose: bool = True,
    ):
        self.n_envs = n_envs
        self.device = torch.device(device)

        if verbose:
            print(f"[vec] loading shared {model_name} on {self.device}...", flush=True)
        model = HookedTransformer.from_pretrained(model_name, device=self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        corpus = load_natural_corpus(model)
        if verbose:
            print(f"[vec] corpus tokens: {corpus.shape[0]}", flush=True)

        self.envs: List[MultiTaskHeadDiscoveryEnv] = []
        for i in range(n_envs):
            e = MultiTaskHeadDiscoveryEnv(
                tasks=[InductionTask(), IOITask()],
                n_test_seqs=n_test_seqs,
                seq_len=seq_len,
                max_steps=max_steps,
                device=str(self.device),
                model=model,
                corpus_tokens=corpus,
                verbose=False,
            )
            self.envs.append(e)
            # Seed each env's RNG distinctly so per-episode random task choice
            # is decorrelated across the vec stack.
            e.action_space.seed(1000 + i)

        e0 = self.envs[0]
        self.n_actions = e0.n_actions
        self.obs_dim = e0.obs_dim
        self.n_tasks = e0.n_tasks
        self.max_steps = e0.max_steps
        self.task_index_by_name = e0.task_index_by_name

    @property
    def fwd_calls(self) -> int:
        return sum(e.fwd_calls for e in self.envs)

    def reset(self, seeds: Optional[List[int]] = None, options=None):
        if seeds is None:
            seeds = [None] * self.n_envs
        obs_list, info_list = [], []
        for i, e in enumerate(self.envs):
            opt = options[i] if options is not None else None
            obs, info = e.reset(seed=seeds[i], options=opt)
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list, axis=0), info_list

    def reset_one(self, idx: int, seed: Optional[int] = None, options=None):
        obs, info = self.envs[idx].reset(seed=seed, options=options)
        return obs, info

    def step(self, actions):
        obs_list, rew, term, trunc, info_list = [], [], [], [], []
        for e, a in zip(self.envs, actions):
            obs, r, t1, t2, info = e.step(int(a))
            obs_list.append(obs)
            rew.append(r)
            term.append(t1)
            trunc.append(t2)
            info_list.append(info)
        return (
            np.stack(obs_list, axis=0),
            np.array(rew, dtype=np.float32),
            np.array(term, dtype=bool),
            np.array(trunc, dtype=bool),
            info_list,
        )

    def action_masks(self) -> np.ndarray:
        return np.stack([e.action_mask() for e in self.envs], axis=0)
