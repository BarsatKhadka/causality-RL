"""Step 1.3 — Zero-ablation of a single attention head via TransformerLens hooks."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


def make_head_ablation_hook(head_index: int):
    """Zero out the output of `head_index` at hook_z.

    hook_z has shape [batch, seq, n_heads, d_head].
    """
    def hook_fn(z: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        z[:, :, head_index, :] = 0.0
        return z
    return hook_fn


def loss_with_head_ablated(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_positions: torch.Tensor,
    layer: int,
    head: int,
) -> float:
    hook_name = f"blocks.{layer}.attn.hook_z"
    hook_fn = make_head_ablation_hook(head)
    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens,
            return_type="logits",
            fwd_hooks=[(hook_name, hook_fn)],
        )
    batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
    pred_logits = logits[batch_idx, target_positions, :]
    targets = tokens[:, 2]
    loss = torch.nn.functional.cross_entropy(pred_logits, targets, reduction="mean")
    return loss.item()


def logit_diff_with_head_ablated(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_positions: torch.Tensor,
    distractor_tokens: torch.Tensor,
    layer: int,
    head: int,
) -> float:
    """Logit-diff (correct - distractor) at the induction position with one head zeroed."""
    hook_name = f"blocks.{layer}.attn.hook_z"
    hook_fn = make_head_ablation_hook(head)
    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens,
            return_type="logits",
            fwd_hooks=[(hook_name, hook_fn)],
        )
    batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
    pred_logits = logits[batch_idx, target_positions, :]
    correct = tokens[:, 2]
    diff = pred_logits[batch_idx, correct] - pred_logits[batch_idx, distractor_tokens]
    return diff.mean().item()


def control_loss_with_head_ablated(
    model: HookedTransformer,
    control_tokens: torch.Tensor,
    layer: int,
    head: int,
) -> float:
    """Mean next-token CE on the control batch with one head zeroed.

    Used contrastively with induction logit-diff to isolate induction-specific
    heads from heads that just affect general processing.
    """
    hook_name = f"blocks.{layer}.attn.hook_z"
    hook_fn = make_head_ablation_hook(head)
    with torch.no_grad():
        logits = model.run_with_hooks(
            control_tokens,
            return_type="logits",
            fwd_hooks=[(hook_name, hook_fn)],
        )
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = control_tokens[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
        reduction="mean",
    )
    return loss.item()
