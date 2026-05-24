"""Phase 6 — Vectorized PPO on the multi-task head-discovery env.

N parallel envs share one GPT-2 on the GPU. Each PPO update sees
N * num_steps transitions (default 8 * 50 = 400) instead of 50.
This is the sample-efficiency fix for the Phase 5 collapse.

Run:
    python ppo_vec.py --n_envs 8 --total_timesteps 200000 --tag mt_k1
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

sys.path.insert(0, str(Path(__file__).parent))
from vec_env import VecMultiTaskEnv  # noqa: E402


# ---------------- args ----------------

@dataclass
class Args:
    seed: int = 1
    total_timesteps: int = 200_000
    n_envs: int = 8
    num_steps: int = 50                  # env episode length
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.1                # higher to fight collapse
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    anneal_lr: bool = True
    lr_min_frac: float = 0.2

    plan_k: int = 1                      # >1 = best-of-K planning per env
    n_test_seqs: int = 32
    seq_len: int = 60

    eval_every: int = 4_000              # transitions, not updates
    eval_episodes: int = 8               # per task
    eval_seed_base: int = 10_000_000

    tag: str = "mt"
    out_dir: Path = field(default_factory=lambda: Path(__file__).parent / "results")
    device: str = ""

    @property
    def batch_size(self) -> int:
        return self.n_envs * self.num_steps

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


# ---------------- planner (per-env best-of-K) ----------------

def plan_actions(agent, obs_t, mask_t, vec_env, K, deterministic=False):
    """Per-env best-of-K planning. obs_t, mask_t are [N, ...]. Returns numpy
    action array [N]."""
    dist = agent.distribution(obs_t, mask_t)
    N = obs_t.shape[0]
    actions = np.zeros(N, dtype=np.int64)
    for i in range(N):
        n_legal = int(mask_t[i].sum().item())
        k_eff = min(K, n_legal)
        if k_eff == 0:
            a = int(torch.argmax(mask_t[i].float()).item())
            actions[i] = a
            continue
        if K == 1:
            a_t = dist.probs[i].argmax() if deterministic else \
                  torch.distributions.Categorical(probs=dist.probs[i]).sample()
            actions[i] = int(a_t.item())
            continue
        if deterministic:
            cand = dist.probs[i].topk(k_eff).indices.cpu().tolist()
        else:
            probs = dist.probs[i].clone()
            cand = []
            for _ in range(k_eff):
                a = int(torch.multinomial(probs, 1).item())
                cand.append(a)
                probs[a] = 0.0
                if probs.sum() == 0:
                    break
        scores = [vec_env.envs[i].query_score(a) for a in cand]
        actions[i] = cand[int(np.argmax(scores))]
    return actions


# ---------------- eval ----------------

def evaluate_per_task(agent, vec_env, n_episodes_per_task, device, K_plan, seed_base):
    """Run held-out eval episodes for each known task; report mean running-max
    per task."""
    agent.eval()
    n_tasks = vec_env.n_tasks
    n_steps = vec_env.max_steps
    results = {}
    e0 = vec_env.envs[0]
    with torch.no_grad():
        for ti in range(n_tasks):
            task_name = next(name for name, idx in vec_env.task_index_by_name.items() if idx == ti)
            rmaxes = []
            for ep in range(n_episodes_per_task):
                obs, info = e0.reset(seed=seed_base + ti * 10_000 + ep,
                                     options={"task_index": ti})
                running = -np.inf
                for t in range(n_steps):
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    mask_t = torch.as_tensor(e0.action_mask(), dtype=torch.bool, device=device).unsqueeze(0)
                    a = plan_actions(agent, obs_t, mask_t,
                                     _SingleWrap(e0), K=K_plan, deterministic=False)[0]
                    obs, r, term, trunc, info = e0.step(int(a))
                    running = max(running, info["running_max"])
                    if term or trunc:
                        break
                rmaxes.append(running)
            results[task_name] = float(np.mean(rmaxes))
    agent.train()
    return results


class _SingleWrap:
    """Tiny adapter: makes a single env look like vec_env for plan_actions."""
    def __init__(self, env):
        self.envs = [env]


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

    vec = VecMultiTaskEnv(
        n_envs=args.n_envs,
        device=device_str,
        n_test_seqs=args.n_test_seqs,
        seq_len=args.seq_len,
    )
    obs_dim = vec.obs_dim
    n_actions = vec.n_actions
    print(f"[setup] N_envs={args.n_envs}  obs_dim={obs_dim}  n_actions={n_actions}  "
          f"episode_len={vec.max_steps}  K_plan={args.plan_k}", flush=True)
    print(f"[setup] batch_size={args.batch_size}  minibatch={args.minibatch_size}  "
          f"total_timesteps={args.total_timesteps}", flush=True)

    agent = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"[setup] policy params = {sum(p.numel() for p in agent.parameters())}", flush=True)

    obs_buf = torch.zeros((args.num_steps, args.n_envs, obs_dim), device=device)
    mask_buf = torch.zeros((args.num_steps, args.n_envs, n_actions), dtype=torch.bool, device=device)
    act_buf = torch.zeros((args.num_steps, args.n_envs), dtype=torch.long, device=device)
    logp_buf = torch.zeros((args.num_steps, args.n_envs), device=device)
    rew_buf = torch.zeros((args.num_steps, args.n_envs), device=device)
    done_buf = torch.zeros((args.num_steps, args.n_envs), device=device)
    val_buf = torch.zeros((args.num_steps, args.n_envs), device=device)

    obs_np, _ = vec.reset(seeds=[args.seed + 100 + i for i in range(args.n_envs)])
    next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    next_mask = torch.as_tensor(vec.action_masks(), dtype=torch.bool, device=device)
    next_done = torch.zeros(args.n_envs, device=device)

    n_updates = args.total_timesteps // (args.n_envs * args.num_steps)
    global_step = 0
    eval_history = []
    next_eval = 0
    start = time.time()

    for update in range(1, n_updates + 1):
        if args.anneal_lr:
            frac = max(1.0 - (update - 1) / n_updates, args.lr_min_frac)
            for g in optimizer.param_groups:
                g["lr"] = frac * args.learning_rate

        ep_rmax = []
        for t in range(args.num_steps):
            global_step += args.n_envs
            obs_buf[t] = next_obs
            mask_buf[t] = next_mask
            done_buf[t] = next_done

            with torch.no_grad():
                actions_np = plan_actions(agent, next_obs, next_mask, vec,
                                          K=args.plan_k, deterministic=False)
                act_t = torch.as_tensor(actions_np, dtype=torch.long, device=device)
                logp, _, value = agent.evaluate(next_obs, next_mask, act_t)
            act_buf[t] = act_t
            logp_buf[t] = logp
            val_buf[t] = value

            obs_np, rew, term, trunc, infos = vec.step(actions_np)
            rew_buf[t] = torch.as_tensor(rew, dtype=torch.float32, device=device)

            done = term | trunc
            # Auto-reset finished envs (per-env)
            for i, d in enumerate(done):
                if d:
                    ep_rmax.append(infos[i]["running_max"])
                    obs_i, _ = vec.reset_one(i)
                    obs_np[i] = obs_i

            next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            next_mask = torch.as_tensor(vec.action_masks(), dtype=torch.bool, device=device)
            next_done = torch.as_tensor(done.astype(np.float32), device=device)

        # GAE
        with torch.no_grad():
            next_value = agent.value(next_obs)
            advantages = torch.zeros_like(rew_buf)
            last = torch.zeros(args.n_envs, device=device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nt = 1.0 - next_done
                    nv = next_value
                else:
                    nt = 1.0 - done_buf[t + 1]
                    nv = val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nv * nt - val_buf[t]
                last = delta + args.gamma * args.gae_lambda * nt * last
                advantages[t] = last
            returns = advantages + val_buf

        # Flatten for PPO update
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_mask = mask_buf.reshape(-1, n_actions)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)

        b_inds = np.arange(args.batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for s in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[s:s + args.minibatch_size]
                new_logp, entropy, new_val = agent.evaluate(b_obs[mb], b_mask[mb], b_act[mb])
                ratio = (new_logp - b_logp[mb]).exp()
                adv = b_adv[mb]
                if args.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg = torch.max(pg1, pg2).mean()
                vloss = 0.5 * ((new_val - b_ret[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = pg - args.ent_coef * ent + args.vf_coef * vloss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if global_step >= next_eval:
            ev = evaluate_per_task(agent, vec, args.eval_episodes, device,
                                    K_plan=args.plan_k, seed_base=args.eval_seed_base)
            train_rmax = float(np.mean(ep_rmax)) if ep_rmax else float("nan")
            rec = {
                "step": global_step,
                "train_rmax": train_rmax,
                "eval": ev,
                "fwd_calls": vec.fwd_calls,
                "elapsed_sec": time.time() - start,
            }
            eval_history.append(rec)
            ev_str = "  ".join(f"{k}={v:5.2f}" for k, v in ev.items())
            print(f"[train] step {global_step:7d}  train_rmax={train_rmax:5.2f}  "
                  f"{ev_str}  fwd={vec.fwd_calls}  elapsed={time.time()-start:.0f}s",
                  flush=True)
            next_eval += args.eval_every

    # Final per-task eval (more episodes)
    final = evaluate_per_task(agent, vec, n_episodes_per_task=30, device=device,
                               K_plan=args.plan_k, seed_base=args.eval_seed_base)
    summary = {
        "tag": args.tag,
        "n_envs": args.n_envs,
        "plan_k": args.plan_k,
        "total_timesteps": args.total_timesteps,
        "final_eval_per_task": final,
        "fwd_calls_total": vec.fwd_calls,
        "wall_time_sec": time.time() - start,
        "eval_history": eval_history,
        "device": str(device),
    }
    with open(args.out_dir / f"phase6_{args.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(agent.state_dict(), args.out_dir / f"phase6_{args.tag}_policy.pt")
    print(f"\n[done] tag={args.tag}  per-task final eval={final}  "
          f"fwd_calls={vec.fwd_calls}  wall={time.time()-start:.0f}s")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
