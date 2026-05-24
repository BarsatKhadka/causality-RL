"""PPO adapted for HeadDiscoveryEnv.

Differences from `ppo_reference.py` (CleanRL's classic ppo.py):
  * single environment (no SyncVectorEnv) — env stepping is microseconds
  * action masking: illegal-action logits set to -1e9 before softmax
  * mask is stored in the rollout buffer so the update uses the same masked
    distribution that was sampled from
  * smaller MLP (144 -> 128 -> 128 -> {144 logits, 1 value})
  * num_steps = 50 (= episode length), so each rollout is exactly one episode
  * total_timesteps = 200k
  * CPU only, no W&B, no tensorboard — just print + numpy save
  * periodic deterministic eval, saving the discovery curve

Run:
    python "phase 3/ppo_heads.py"
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

from head_env import HeadDiscoveryEnv


# ---------------------------------------------------------------- Args / setup

@dataclass
class Args:
    seed: int = 1
    total_timesteps: int = 200_000
    learning_rate: float = 2.5e-4
    num_steps: int = 50              # = episode length
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

    eval_every: int = 10_000          # steps between eval rollouts
    eval_episodes: int = 20           # episodes per eval (deterministic)

    out_dir: Path = field(default_factory=lambda: Path(__file__).parent / "results")

    @property
    def batch_size(self) -> int:
        return self.num_steps           # single env

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


# ----------------------------------------------------------------- Network

def layer_init(layer, std: float = np.sqrt(2), bias_const: float = 0.0):
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

    def _masked_logits(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.actor(self.trunk(obs))
        return logits.masked_fill(~mask, -1e9)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.trunk(obs)).squeeze(-1)

    def act(self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False):
        """Returns (action, log_prob, entropy, value)."""
        masked = self._masked_logits(obs, mask)
        dist = Categorical(logits=masked)
        action = masked.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value(obs)
        return action, log_prob, entropy, value

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor, action: torch.Tensor):
        """Recompute log_prob, entropy, value for stored (obs, mask, action)."""
        masked = self._masked_logits(obs, mask)
        dist = Categorical(logits=masked)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value(obs)
        return log_prob, entropy, value


# ------------------------------------------------------------- Eval helper

def evaluate_policy(
    agent: ActorCritic, env: HeadDiscoveryEnv, n_episodes: int, device: torch.device
) -> dict:
    """Run deterministic rollouts; return discovery curves + summary stats."""
    agent.eval()
    n_steps = env.max_steps
    curves = np.zeros((n_episodes, n_steps), dtype=np.float32)
    top1_idx = int(env.scores.argmax())
    top1_hits = 0
    steps_to_top1: list[int] = []

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=10_000 + ep)
            running = -np.inf
            found_step: int | None = None
            for t in range(n_steps):
                mask = env.action_mask()
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                action, *_ = agent.act(obs_t, mask_t, deterministic=True)
                a = int(action.item())
                obs, r, term, trunc, info = env.step(a)
                if a == top1_idx and found_step is None:
                    found_step = t + 1
                running = max(running, info["running_max"])
                curves[ep, t] = running
                if term or trunc:
                    break
            if found_step is not None:
                top1_hits += 1
                steps_to_top1.append(found_step)

    agent.train()
    median_steps = float(np.median(steps_to_top1)) if steps_to_top1 else None
    return {
        "curves": curves,
        "mean_curve": curves.mean(axis=0),
        "top1_rate": top1_hits / n_episodes,
        "median_steps_to_top1": median_steps,
    }


# ----------------------------------------------------------------- Main

def main(args: Args | None = None) -> None:
    args = args or Args()
    args.out_dir.mkdir(exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    env = HeadDiscoveryEnv()
    eval_env = HeadDiscoveryEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)
    print(f"env: obs_dim={obs_dim}  n_actions={n_actions}  episode_len={env.max_steps}")
    print(f"true top-1 head idx = {env.scores.argmax()}  score = {env.scores.max():.4f}")

    agent = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"actor-critic params = {sum(p.numel() for p in agent.parameters())}")

    # Rollout storage (single env, num_steps per rollout)
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
    global_step = 0
    eval_history: list[dict] = []
    start = time.time()
    next_eval = 0

    for update in range(1, n_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / n_updates
            for g in optimizer.param_groups:
                g["lr"] = frac * args.learning_rate

        # ------ rollout ------
        ep_returns: list[float] = []
        ep_running_max: list[float] = []
        ep_return = 0.0
        for t in range(args.num_steps):
            global_step += 1
            obs_buf[t] = next_obs
            mask_buf[t] = next_mask
            done_buf[t] = next_done

            with torch.no_grad():
                action, log_prob, _, value = agent.act(
                    next_obs.unsqueeze(0), next_mask.unsqueeze(0)
                )
            act_buf[t] = action[0]
            logp_buf[t] = log_prob[0]
            val_buf[t] = value[0]

            obs_np, reward, term, trunc, info = env.step(int(action.item()))
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

        # ------ GAE ------
        with torch.no_grad():
            next_value = agent.value(next_obs.unsqueeze(0))[0]
            advantages = torch.zeros_like(rew_buf)
            last_gae = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_nonterminal = 1.0 - next_done[0]
                    next_v = next_value
                else:
                    next_nonterminal = 1.0 - done_buf[t + 1]
                    next_v = val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * next_v * next_nonterminal - val_buf[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + val_buf

        # ------ PPO update ------
        b_inds = np.arange(args.batch_size)
        clipfracs: list[float] = []
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start_i in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start_i : start_i + args.minibatch_size]
                mb_obs = obs_buf[mb]
                mb_mask = mask_buf[mb]
                mb_act = act_buf[mb]
                mb_logp_old = logp_buf[mb]
                mb_adv = advantages[mb]
                mb_ret = returns[mb]

                new_logp, entropy, new_val = agent.evaluate(mb_obs, mb_mask, mb_act)
                logratio = new_logp - mb_logp_old
                ratio = logratio.exp()

                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

        # ------ log ------
        mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
        mean_rmax = float(np.mean(ep_running_max)) if ep_running_max else float("nan")
        sps = int(global_step / (time.time() - start))

        if global_step >= next_eval:
            ev = evaluate_policy(agent, eval_env, args.eval_episodes, device)
            eval_history.append(
                {
                    "step": global_step,
                    "top1_rate": ev["top1_rate"],
                    "median_steps_to_top1": ev["median_steps_to_top1"],
                    "final_running_max_mean": float(ev["mean_curve"][-1]),
                }
            )
            print(
                f"step {global_step:6d}  train_ret={mean_ret:6.2f}  train_rmax={mean_rmax:5.2f}  "
                f"eval_top1={ev['top1_rate']*100:5.1f}%  eval_rmax={ev['mean_curve'][-1]:5.2f}  "
                f"sps={sps}"
            )
            next_eval += args.eval_every

    # ------ final eval + save ------
    final = evaluate_policy(agent, eval_env, n_episodes=100, device=device)
    np.save(args.out_dir / "ppo_curves.npy", final["curves"])
    with open(args.out_dir / "ppo_summary.json", "w") as f:
        json.dump(
            {
                "total_timesteps": args.total_timesteps,
                "final_top1_rate": final["top1_rate"],
                "final_median_steps_to_top1": final["median_steps_to_top1"],
                "final_mean_running_max": float(final["mean_curve"][-1]),
                "eval_history": eval_history,
            },
            f,
            indent=2,
        )
    print("\n=== FINAL (100 deterministic eval episodes) ===")
    print(f"top-1 success rate     : {final['top1_rate']*100:.1f}%   (random baseline = 38%)")
    print(f"median steps to top-1  : {final['median_steps_to_top1']}   (random baseline ~ 33)")
    print(f"final mean running-max : {final['mean_curve'][-1]:.3f}   (true top-1 = {env.scores.max():.3f})")
    print(f"\nsaved: {args.out_dir / 'ppo_curves.npy'}  +  ppo_summary.json")


if __name__ == "__main__":
    main()
