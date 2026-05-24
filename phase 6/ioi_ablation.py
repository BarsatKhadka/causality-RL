"""Zero-ablation of one attention head, scored on the IOI batch."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

# Reuse the head-ablation hook from phase 1.
PHASE1 = Path(__file__).parents[1] / "phase 1"
sys.path.insert(0, str(PHASE1))
from ablation import make_head_ablation_hook  # noqa: E402


def ioi_logit_diff_with_head_ablated(
    model: HookedTransformer,
    tokens: torch.Tensor,
    final_positions: torch.Tensor,
    io_tokens: torch.Tensor,
    s_tokens: torch.Tensor,
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
    pred = logits[batch_idx, final_positions, :]
    diff = pred[batch_idx, io_tokens] - pred[batch_idx, s_tokens]
    return diff.mean().item()
