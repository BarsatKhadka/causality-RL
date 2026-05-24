# Phase 3 — PPO agent for head discovery

Build an RL agent that learns *which heads to investigate next*, beating the
Phase 2 random baseline.

## Layout
- `ppo_reference.py` — untouched copy of CleanRL's classic `ppo.py` (discrete-action,
  CartPole-style). Reference only — do not edit. The `cleanrl/` submodule is the
  upstream source.
- *(coming)* `head_env.py` — Gym-compatible environment wrapping the Phase 1
  reward function:
    - **State**: 144-dim vector. Each entry = score-if-tried (logit-diff drop) or
      `-1` if untried.
    - **Action**: Discrete(144) — pick a head to ablate next.
    - **Reward**: the head's causal score from Phase 1 (cached, O(1) lookup).
    - **Episode end**: after N=50 steps, or when the same head is picked twice
      (with a penalty).
- *(coming)* `ppo_heads.py` — adapted PPO. Changes from `ppo_reference.py`:
    - Single env, not vectorized (each step is cheap)
    - Action masking so the policy doesn't repick a tried head
    - Smaller MLP (input is just the 144-d state vector)
    - Eval callback that records the same discovery-curve metric as Phase 2
- *(coming)* `compare.py` — overlays the PPO discovery curve on the Phase 2
  random curve. The headline plot for the paper.

## Why an environment at all
The Phase 1 reward function maps `head -> score`. To turn it into RL, we need
state (what the agent has seen so far) and dynamics (state changes after each
pick). The environment is just bookkeeping around the cached scores — no GPT-2
forward passes happen inside it. That means training PPO is fast: the slow part
is just the policy updates.

## Done-when
PPO's discovery curve sits clearly above the random baseline on the comparison
plot. Specifically: top-1 success rate >> 38%, median steps-to-top-1 << 33.
