# Preprint Plan — Causal Curiosity RL for Circuit Discovery

## Core experiment: train on two tasks, test on a third

**Train:** induction + IOI, alternating episodes.
**Test (held-out):** docstring circuit (Heimersheim & Janiak 2023) *or* greater-than
circuit (Hanna et al. 2023). Both have published ground-truth heads. Use whichever
has the cleaner reference dataset.

## Why multi-task training (the key insight)

Single-task RL lets the agent memorize the answer for that task. Two simultaneous
tasks force the agent to find what is *common*: which layers/positions tend to
matter across tasks. Induction's important heads sit in layers 5–7. IOI's important
heads sit in layers 7–10. They overlap, but not perfectly. The agent has to learn
the *general prior over GPT-2's geometry* rather than one task's answer.

This isn't a hack — it's the correct training setup for the question we're actually
asking ("can RL learn a transferable causal-discovery strategy?").

## Architecture change

Add a task ID to the observation:

```
obs = [task_onehot (2), tried_mask (144), normalized_scores (144)]  =  290-dim
```

The policy can now condition on which task this episode is, so it learns
task-conditional strategies that share underlying structure.

## Training setup

- Alternate episodes: induction → IOI → induction → IOI → ...
- Each episode: fresh random seed → fresh batch → fresh scores. No persistent cache.
  Every step is a real GPT-2 ablation forward pass.
- **8 parallel envs per PPO update** (400 transitions per update instead of 50).
  This is the sample-efficiency fix; multi-task without this still won't learn.
- ent_coef = 0.1 to start, annealed.
- 200k total steps. L40S GPU, overnight.

## Result table

| Method | Induction Recall@20 | IOI Recall@20 | Docstring Recall@20 |
|---|---|---|---|
| Random | ~14% | ~14% | ~14% |
| K=1 PPO (ours) | ? | ? | ? |
| K=5 planning (ours) | ? | ? | ? |

`Recall@K` = fraction of ground-truth heads recovered in the first K picks.

Induction & IOI columns demonstrate the agent learned the training tasks.
Docstring column demonstrates **transfer to a held-out circuit it never saw**.

## Definitions / ground truth

- Induction heads: Olsson et al. 2022 (L5.H1, L5.H5, L6.H9, L7.H2 in GPT-2 small).
- IOI heads: Wang et al. 2022 (name movers, S-inhibition heads, etc.).
- Docstring: Heimersheim & Janiak 2023.
- Greater-than: Hanna et al. 2023.

## Headline claim (target)

> A single PPO agent trained on causal interventions across two known circuits
> learns a generalizable head-discovery policy that transfers to a held-out
> circuit, recovering ground-truth heads with higher recall than random search
> using a fraction of the forward passes.

---

## What needs to be built (engineering)

1. **Task registry** with two trainers (induction, IOI) + per-task batch generators
   and metrics. Keep the interface identical so adding the test task is trivial.
2. **Task ID in obs.** One-hot, prepended.
3. **Vectorized env.** 8 `RealHeadDiscoveryEnv` instances sharing a single GPT-2
   on GPU. Batched ablation when possible.
4. **Recall@K eval.** Replace running-max metric with set-overlap against
   published ground-truth head lists.
5. **Random + (ideally) one attribution baseline** (e.g. activation-patching score
   ranking) for comparison.

## Estimated cost
- Code: 1.5 days.
- Training: one overnight L40S run (200k steps × 8 envs).
- Eval + plots: a few hours.

---

## Honest preprint risk register

- **Vec-env PPO might still plateau.** If it does, the multi-task framing alone
  won't save the result. Fallback: longer training, higher entropy, or fall back
  to behavioral cloning from oracle-discovered heads (still a paper, different
  claim).
- **Recall@20 might just track "the agent picks mid-layer heads."** Need to compare
  against a baseline that also picks mid-layer heads (e.g. uniform over layers
  3–9) to show the agent is doing more than learning "skip layer 0 and 11."
- **Docstring transfer might fail.** That would itself be publishable: "this method
  recovers training-task circuits but doesn't transfer," and is honest. Don't
  paper over a negative result.
- **Comparison to existing methods.** Path patching / EAP / ACDC already do
  circuit discovery. We need at least one head-to-head: "RL agent finds the same
  heads with K× fewer forward passes" or similar. Without this the contribution
  is unclear.
