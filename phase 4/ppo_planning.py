"""PPO with model-based planning on the live GPT-2 env.

Each decision step:
  1. Policy outputs a distribution over the 144 legal heads.
  2. Sample K candidates from that distribution (without replacement).
  3. Use the env (= GPT-2) to actually score each candidate.
  4. Commit to the best-scoring candidate as the executed action.
  5. Update PPO with (state, executed_action, observed_reward).

K=1 reduces to vanilla PPO. K>1 is best-of-K model-based planning.

The bias note: best-of-K sampling differs from the policy's own sampling
distribution, so the PPO update is slightly biased. We accept that — empirically
showing that K>1 improves the discovery curve is the whole point.

Eval is dense (every 1000 steps) so the learning trend is visible.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from head_env_live import LiveHeadDiscoveryEnv


# --------------------------------------------------- args + network

@dataclass
class Args:
    seed: int = 1
    total_timesteps: int = 20_000      # smaller — env is real and slow on miss
    learning_rate: float = 2.5e-4
    num_steps: int = 50                # = episode length
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    anneal_lr: bool = True

    plan_k: int = 5                    # candidates considered per decision (1 = no planning)
    eval_every: int = 1_000
    eval_episodes: int = 10

    tag: str = "k5"                    # used in output filenames
    out_dir: Path = field(default_factory=lambda: Path(__file__).parent / "results")

    @property
    def batch_size(self) -> int:
        return self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(hidden, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def _masked_logits(self, obs, mask):
        return self.actor(self.trunk(obs)).masked_fill(~mask, -1e9)

    def value(self, obs):
        return self.critic(self.trunk(obs)).squeeze(-1)

    def distribution(self, obs, mask) -> Categorical:
        return Categorical(logits=self._masked_logits(obs, mask))

    def evaluate(self, obs, mask, action):
        masked = self._masked_logits(obs, mask)
        dist = Categorical(logits=masked)
        return dist.log_prob(action), dist.entropy(), self.value(obs)


# --------------------------------------------------- planner

def plan_action(
    agent: ActorCritic,
    obs_t: torch.Tensor,
    mask_t: torch.Tensor,
    env: LiveHeadDiscoveryEnv,
    K: int,
    deterministic: bool = False,
) -> tuple[int, float]:
    """Propose K candidates from the policy, score each via the env, pick best.

    Returns (executed_action, executed_score). Score lookups are cached so K>1
    is cheap after warm-up.
    """
    dist = agent.distribution(obs_t, mask_t)
    if K == 1:
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        a = int(action.item())
        # We don't pre-evaluate K=1; the env.step in the caller computes it.
        # But we DO query here so that planning and non-planning code paths are
        # symmetric, and the cache absorbs the cost.
        return a, env.query_score(a)

    # K candidates without replacement, sampled (or top-K logits if deterministic)
    if deterministic:
        cand = dist.probs[0].topk(min(K, int(mask_t.sum().item()))).indices.cpu().tolist()
    else:
        cand_set: list[int] = []
        probs = dist.probs[0].clone()
        for _ in range(min(K, int(mask_t.sum().item()))):
            a = int(torch.multinomial(probs, 1).item())
            cand_set.append(a)
            probs[a] = 0.0
            if probs.sum() == 0:
                break
        cand = cand_set

    scores = [env.query_score(a) for a in cand]
    best_idx = int(np.argmax(scores))
    return cand[best_idx], scores[best_idx]


# --------------------------------------------------- eval

def evaluate_policy(agent, env, n_episodes, device, K_plan):
    agent.eval()
    n_steps = env.max_steps
    curves = np.zeros((n_episodes, n_steps), dtype=np.float32)
    top1_idx = max(env._cache, key=env._cache.get) if env._cache else None
    top1_score = env._cache[top1_idx] if top1_idx is not None else None
    top1_hits = 0
    steps_to_top1 = []
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=10_000 + ep)
            running = -np.inf
            found_step = None
            for t in range(n_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(env.action_mask(), dtype=torch.bool, device=device).unsqueeze(0)
                a, _ = plan_action(agent, obs_t, mask_t, env, K=K_plan, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                if top1_idx is not None and a == top1_idx and found_step is None:
                    found_step = t + 1
                running = max(running, info["running_max"])
                curves[ep, t] = running
                if term or trunc:
                    break
            if found_step is not None:
                top1_hits += 1
                steps_to_top1.append(found_step)
    agent.train()
    return {
        "curves": curves,
        "mean_curve": curves.mean(axis=0),
        "top1_rate": top1_hits / n_episodes,
        "median_steps_to_top1": float(np.median(steps_to_top1)) if steps_to_top1 else None,
        "top1_idx": top1_idx,
        "top1_score": top1_score,
    }


# --------------------------------------------------- main

def main(args: Args | None = None) -> None:
    args = args or Args()
    args.out_dir.mkdir(exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    env = LiveHeadDiscoveryEnv()
    eval_env = env   # share the model and cache
    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)
    print(f"\n[train] env={obs_dim}d obs, {n_actions} actions, episode={env.max_steps}, K_plan={args.plan_k}")

    agent = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_buf = torch.zeros((args.num_steps, obs_dim), device=device)
    mask_buf = torch.zeros((args.num_steps, n_actions), dtype=torch.bool, device=device)
    act_buf = torch.zeros(args.num_steps, dtype=torch.long, device=device)
    logp_buf = torch.zeros(args.num_steps, device=device)
    rew_buf = torch.zeros(args.num_steps, device=device)
    done_buf = torch.zeros(args.num_steps, device=device)
    val_buf = torch.zeros(args.num_steps, device=device)

    obs, _ = env.reset(seed=args.seed)
    next_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    next_mask = torch.as_tensor(env.action_mask(), dtype=torch.bool, device=device)
    next_done = torch.zeros(1, device=device)

    n_updates = args.total_timesteps // args.num_steps
    eval_history = []
    global_step = 0
    start = time.time()
    next_eval = 0

    for update in range(1, n_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / n_updates
            for g in optimizer.param_groups:
                g["lr"] = frac * args.learning_rate

        ep_returns = []
        ep_running_max = []
        ep_return = 0.0
        for t in range(args.num_steps):
            global_step += 1
            obs_buf[t] = next_obs
            mask_buf[t] = next_mask
            done_buf[t] = next_done

            with torch.no_grad():
                # Planning: propose K, env scores each, pick best
                a, _ = plan_action(agent, next_obs.unsqueeze(0), next_mask.unsqueeze(0),
                                   env, K=args.plan_k, deterministic=False)
                # Re-evaluate to grab log_prob and value under current policy
                log_prob, _, value = agent.evaluate(
                    next_obs.unsqueeze(0),
                    next_mask.unsqueeze(0),
                    torch.tensor([a], device=device),
                )
            act_buf[t] = a
            logp_buf[t] = log_prob[0]
            val_buf[t] = value[0]

            obs_np, reward, term, trunc, info = env.step(a)
            rew_buf[t] = float(reward)
            ep_return += float(reward)
            done = term or trunc
            next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            next_mask = torch.as_tensor(env.action_mask(), dtype=torch.bool, device=device)
            next_done = torch.tensor([1.0 if done else 0.0], device=device)
            if done:
                ep_returns.append(ep_return)
                ep_running_max.append(info["running_max"])
                ep_return = 0.0
                obs_np, _ = env.reset()
                next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                next_mask = torch.as_tensor(env.action_mask(), dtype=torch.bool, device=device)

        # GAE
        with torch.no_grad():
            next_value = agent.value(next_obs.unsqueeze(0))[0]
            advantages = torch.zeros_like(rew_buf)
            last = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nt = 1.0 - next_done[0]; nv = next_value
                else:
                    nt = 1.0 - done_buf[t + 1]; nv = val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nv * nt - val_buf[t]
                last = delta + args.gamma * args.gae_lambda * nt * last
                advantages[t] = last
            returns = advantages + val_buf

        # PPO update
        b_inds = np.arange(args.batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for s in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[s:s + args.minibatch_size]
                new_logp, entropy, new_val = agent.evaluate(obs_buf[mb], mask_buf[mb], act_buf[mb])
                ratio = (new_logp - logp_buf[mb]).exp()
                adv = advantages[mb]
                if args.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg = torch.max(pg1, pg2).mean()
                vloss = 0.5 * ((new_val - returns[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = pg - args.ent_coef * ent + args.vf_coef * vloss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if global_step >= next_eval:
            ev = evaluate_policy(agent, eval_env, args.eval_episodes, device, K_plan=args.plan_k)
            mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            mean_rmax = float(np.mean(ep_running_max)) if ep_running_max else float("nan")
            eval_history.append(
                {
                    "step": global_step,
                    "top1_rate": ev["top1_rate"],
                    "median_steps_to_top1": ev["median_steps_to_top1"],
                    "final_running_max_mean": float(ev["mean_curve"][-1]),
                    "cache_misses": env.cache_misses,
                }
            )
            print(
                f"[train] step {global_step:5d}  train_ret={mean_ret:6.2f}  train_rmax={mean_rmax:5.2f}  "
                f"eval_top1={ev['top1_rate']*100:5.1f}%  eval_rmax={ev['mean_curve'][-1]:.3f}  "
                f"misses={env.cache_misses}/{env.n_actions}  "
                f"elapsed={time.time()-start:.0f}s",
                flush=True,
            )
            env.save_cache()
            next_eval += args.eval_every

    # Final eval
    final = evaluate_policy(agent, eval_env, n_episodes=30, device=device, K_plan=args.plan_k)
    np.save(args.out_dir / f"ppo_planning_{args.tag}_curves.npy", final["curves"])
    with open(args.out_dir / f"ppo_planning_{args.tag}_summary.json", "w") as f:
        json.dump(
            {
                "tag": args.tag,
                "plan_k": args.plan_k,
                "total_timesteps": args.total_timesteps,
                "final_top1_rate": final["top1_rate"],
                "final_median_steps_to_top1": final["median_steps_to_top1"],
                "final_mean_running_max": float(final["mean_curve"][-1]),
                "eval_history": eval_history,
                "cache_misses_total": env.cache_misses,
                "cache_size_final": len(env._cache),
            },
            f,
            indent=2,
        )
    env.save_cache()
    print(f"\n[done] tag={args.tag}  top1={final['top1_rate']*100:.1f}%  "
          f"median_steps_to_top1={final['median_steps_to_top1']}  "
          f"final_rmax={final['mean_curve'][-1]:.3f}  cache_misses={env.cache_misses}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
