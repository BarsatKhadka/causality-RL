"""Phase 5: PPO with best-of-K planning on the FULLY LIVE env.

Each episode gets a fresh induction batch (new seed), so head scores drift
episode-to-episode. The agent must learn structural priors ("which regions
of the network tend to matter") rather than memorize a single ranking.

Eval uses a held-out band of seeds (>= 10_000_000) so no overlap with training.

Run:
    # Vanilla PPO (no planning)
    python ppo_planning_real.py --plan_k 1 --tag k1 --total_timesteps 50000

    # Model-based planning (best-of-5)
    python ppo_planning_real.py --plan_k 5 --tag k5 --total_timesteps 50000
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

from head_env_real import RealHeadDiscoveryEnv


# ---------------- args ----------------

@dataclass
class Args:
    seed: int = 1
    total_timesteps: int = 50_000
    learning_rate: float = 2.5e-4
    num_steps: int = 50                  # = episode length, so 1 update = 1 episode
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.05            # higher: avoid early collapse on stochastic env
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    anneal_lr: bool = True
    lr_min_frac: float = 0.2          # don't anneal LR all the way to 0

    plan_k: int = 5                       # 1 = vanilla PPO, >1 = best-of-K planning
    n_test_seqs: int = 32                 # induction batch size per ablation
    seq_len: int = 60

    eval_every: int = 2_000
    eval_episodes: int = 10
    eval_seed_base: int = 10_000_000      # held-out seed band

    tag: str = "k5"
    out_dir: Path = field(default_factory=lambda: Path(__file__).parent / "results")
    device: str = ""                       # "" -> auto (cuda if available)

    @property
    def batch_size(self) -> int:
        return self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


# ---------------- network ----------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
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


# ---------------- planner ----------------

def plan_action(agent, obs_t, mask_t, env, K, deterministic=False):
    dist = agent.distribution(obs_t, mask_t)
    n_legal = int(mask_t.sum().item())
    k_eff = min(K, n_legal)
    if k_eff == 0:
        a = int(torch.argmax(mask_t.float()).item())
        return a, env.query_score(a)
    if K == 1:
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        a = int(action.item())
        return a, env.query_score(a)
    if deterministic:
        cand = dist.probs[0].topk(k_eff).indices.cpu().tolist()
    else:
        probs = dist.probs[0].clone()
        cand = []
        for _ in range(k_eff):
            a = int(torch.multinomial(probs, 1).item())
            cand.append(a)
            probs[a] = 0.0
            if probs.sum() == 0:
                break
    scores = [env.query_score(a) for a in cand]
    best = int(np.argmax(scores))
    return cand[best], scores[best]


# ---------------- eval ----------------

def evaluate_policy(agent, env, n_episodes, device, K_plan, seed_base):
    agent.eval()
    n_steps = env.max_steps
    curves = np.zeros((n_episodes, n_steps), dtype=np.float32)
    baselines = []
    final_rmax = []
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed_base + ep)
            baselines.append(info["baseline"])
            running = -np.inf
            for t in range(n_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(env.action_mask(), dtype=torch.bool, device=device).unsqueeze(0)
                # Stochastic eval: with a near-deterministic argmax policy on this env,
                # K=1 collapses to a single fixed head-ordering across all eval episodes.
                # Sampling lets the policy's learned distribution interact with per-episode
                # score feedback in the obs.
                a, _ = plan_action(agent, obs_t, mask_t, env, K=K_plan, deterministic=False)
                obs, r, term, trunc, info = env.step(a)
                running = max(running, info["running_max"])
                curves[ep, t] = running
                if term or trunc:
                    break
            final_rmax.append(running)
    agent.train()
    return {
        "curves": curves,
        "mean_curve": curves.mean(axis=0),
        "mean_baseline": float(np.mean(baselines)),
        "mean_final_rmax": float(np.mean(final_rmax)),
    }


# ---------------- main ----------------

def main(args: Args | None = None) -> None:
    args = args or Args()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[setup] device = {device}", flush=True)

    env = RealHeadDiscoveryEnv(
        n_test_seqs=args.n_test_seqs, seq_len=args.seq_len, device=device_str, verbose=True
    )
    eval_env = env   # share model

    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)
    print(f"[setup] env: obs_dim={obs_dim}, n_actions={n_actions}, "
          f"episode_len={env.max_steps}, K_plan={args.plan_k}", flush=True)
    print(f"[setup] total_timesteps={args.total_timesteps}, eval_every={args.eval_every}", flush=True)

    agent = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"[setup] policy params = {sum(p.numel() for p in agent.parameters())}", flush=True)

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
    eval_history = []
    start = time.time()
    next_eval = 0

    for update in range(1, n_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / n_updates
            frac = max(frac, args.lr_min_frac)
            for g in optimizer.param_groups:
                g["lr"] = frac * args.learning_rate

        ep_returns, ep_rmax = [], []
        ep_return = 0.0
        for t in range(args.num_steps):
            global_step += 1
            obs_buf[t] = next_obs
            mask_buf[t] = next_mask
            done_buf[t] = next_done
            with torch.no_grad():
                a, _ = plan_action(agent, next_obs.unsqueeze(0), next_mask.unsqueeze(0),
                                   env, K=args.plan_k, deterministic=False)
                log_prob, _, value = agent.evaluate(
                    next_obs.unsqueeze(0), next_mask.unsqueeze(0),
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
                ep_rmax.append(info["running_max"])
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
            ev = evaluate_policy(agent, eval_env, args.eval_episodes, device,
                                 K_plan=args.plan_k, seed_base=args.eval_seed_base)
            mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            mean_rmax = float(np.mean(ep_rmax)) if ep_rmax else float("nan")
            eval_history.append(
                {
                    "step": global_step,
                    "train_mean_running_max": mean_rmax,
                    "eval_mean_running_max": ev["mean_final_rmax"],
                    "eval_mean_baseline": ev["mean_baseline"],
                    "fwd_calls": env.fwd_calls,
                    "elapsed_sec": time.time() - start,
                }
            )
            print(
                f"[train] step {global_step:6d}  train_rmax={mean_rmax:5.2f}  "
                f"eval_rmax={ev['mean_final_rmax']:5.2f}  "
                f"eval_base={ev['mean_baseline']:5.2f}  fwd={env.fwd_calls}  "
                f"elapsed={time.time()-start:.0f}s",
                flush=True,
            )
            next_eval += args.eval_every

    # Final eval (more episodes)
    final = evaluate_policy(agent, eval_env, n_episodes=50, device=device,
                            K_plan=args.plan_k, seed_base=args.eval_seed_base)
    np.save(args.out_dir / f"real_{args.tag}_curves.npy", final["curves"])
    with open(args.out_dir / f"real_{args.tag}_summary.json", "w") as f:
        json.dump(
            {
                "tag": args.tag,
                "plan_k": args.plan_k,
                "total_timesteps": args.total_timesteps,
                "n_test_seqs": args.n_test_seqs,
                "final_eval_mean_running_max": final["mean_final_rmax"],
                "final_eval_mean_baseline": final["mean_baseline"],
                "fwd_calls_total": env.fwd_calls,
                "wall_time_sec": time.time() - start,
                "eval_history": eval_history,
                "device": str(device),
            },
            f,
            indent=2,
        )
    # Save policy too — useful for downstream analysis / Phase 6
    torch.save(agent.state_dict(), args.out_dir / f"real_{args.tag}_policy.pt")
    print(f"\n[done] tag={args.tag}  final_eval_rmax={final['mean_final_rmax']:.3f}  "
          f"fwd_calls={env.fwd_calls}  wall={time.time()-start:.0f}s")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
