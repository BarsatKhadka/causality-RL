"""Step 1.4 — Score every attention head in GPT-2 small using logit-diff.

Metric: drop in induction logit-diff when a head is zero-ablated.
    score(L,H) = baseline_logit_diff - ablated_logit_diff(L,H)
Higher score = head contributes more to the induction behavior.

Known induction heads in GPT-2 small: 5.5 and 6.9 — should rank near the top.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from ablation import logit_diff_with_head_ablated
from induction_dataset import (
    induction_logit_diff,
    make_distractor_tokens,
    make_induction_batch,
)


def score_all_heads(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_positions: torch.Tensor,
    distractors: torch.Tensor,
) -> tuple[np.ndarray, float]:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    baseline = induction_logit_diff(model, tokens, target_positions, distractors)
    print(f"Baseline induction logit-diff: {baseline:.4f}")

    scores = np.zeros((n_layers, n_heads), dtype=np.float32)
    for layer in tqdm(range(n_layers), desc="layers"):
        for head in range(n_heads):
            ablated = logit_diff_with_head_ablated(
                model, tokens, target_positions, distractors, layer, head
            )
            scores[layer, head] = baseline - ablated
    return scores, baseline


def main() -> None:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GPT-2 small on {device}...")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()

    tokens, target_positions = make_induction_batch(
        model, batch_size=100, seq_len=60, seed=0
    )
    tokens = tokens.to(device)
    target_positions = target_positions.to(device)
    distractors = make_distractor_tokens(model, tokens, seed=1).to(device)

    scores, baseline = score_all_heads(model, tokens, target_positions, distractors)

    np.save(out_dir / "head_scores_logitdiff.npy", scores)
    with open(out_dir / "summary_logitdiff.json", "w") as f:
        json.dump(
            {
                "metric": "logit_diff_drop",
                "baseline_logit_diff": float(baseline),
                "n_layers": int(model.cfg.n_layers),
                "n_heads": int(model.cfg.n_heads),
            },
            f,
            indent=2,
        )

    flat = scores.flatten()
    top_idx = np.argsort(-flat)[:10]
    print("\nTop 10 heads by logit-diff drop on induction:")
    for rank, idx in enumerate(top_idx, 1):
        layer = idx // scores.shape[1]
        head = idx % scores.shape[1]
        marker = "  <-- known" if (layer, head) in [(5, 5), (6, 9)] else ""
        print(f"  {rank:2d}. L{layer}.H{head}  +{flat[idx]:.4f}{marker}")

    # Rank of known heads
    order = np.argsort(-flat)
    rank_map = {int(idx): r for r, idx in enumerate(order, 1)}
    print("\nRank of known induction heads:")
    for (l, h) in [(5, 5), (6, 9)]:
        idx = l * scores.shape[1] + h
        print(f"  L{l}.H{h}: rank {rank_map[idx]}  (score +{flat[idx]:.4f})")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(scores, aspect="auto", cmap="viridis")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title("Induction logit-diff drop when head zero-ablated")
        plt.colorbar(im, ax=ax)
        for (l, h) in [(5, 5), (6, 9)]:
            ax.add_patch(
                plt.Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False, edgecolor="red", lw=2)
            )
        fig.tight_layout()
        fig.savefig(out_dir / "head_scores_logitdiff_heatmap.png", dpi=150)
        print(f"\nSaved heatmap to {out_dir / 'head_scores_logitdiff_heatmap.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
