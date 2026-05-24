"""Step 1.2 — Build induction test sequences.

An induction sequence has the structure:
    [BOS] [random prefix] [A] [B] [random fillers...] [A] -> model should predict [B]

We measure baseline induction performance as the loss at the position predicting
the second [B] (i.e., the logit position right after the second [A]).
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer


def make_induction_batch(
    model: HookedTransformer,
    batch_size: int = 100,
    seq_len: int = 60,
    seed: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of induction sequences.

    Layout per sequence (length = seq_len + 3, including BOS):
        BOS, r1, r2, ..., r_{seq_len-3}, A, B, f1, f2, ..., A
    The induction target is B, predicted at the final A position.

    Returns:
        tokens: LongTensor [batch, seq_len_total]
        target_positions: LongTensor [batch] — position of the final A (whose
            *next-token* prediction should be B).
    """
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
    else:
        g = None

    vocab_size = model.cfg.d_vocab
    # avoid special tokens at the very bottom of the vocab (BOS etc.)
    low, high = 10, vocab_size

    # Two random tokens A, B per sequence
    A = torch.randint(low, high, (batch_size,), generator=g)
    B = torch.randint(low, high, (batch_size,), generator=g)

    # Filler region length: place A,B early, then fillers, then final A
    filler_len = seq_len - 3  # we'll use: A, B, fillers..., A_repeat
    fillers = torch.randint(low, high, (batch_size, filler_len), generator=g)

    bos = torch.full((batch_size, 1), model.tokenizer.bos_token_id or 50256, dtype=torch.long)
    tokens = torch.cat(
        [
            bos,                       # 0: BOS
            A.unsqueeze(1),            # 1: A
            B.unsqueeze(1),            # 2: B
            fillers,                   # 3 .. 3+filler_len-1
            A.unsqueeze(1),            # final: A (repeat)
        ],
        dim=1,
    )

    # Final A position (where the next-token prediction should be B)
    target_positions = torch.full((batch_size,), tokens.shape[1] - 1, dtype=torch.long)
    return tokens, target_positions


def induction_loss(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_positions: torch.Tensor,
) -> float:
    """Average cross-entropy loss for predicting B at the final-A position."""
    with torch.no_grad():
        logits = model(tokens, return_type="logits")  # [B, T, V]
    batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
    pred_logits = logits[batch_idx, target_positions, :]  # [B, V]
    targets = tokens[:, 2]
    loss = torch.nn.functional.cross_entropy(pred_logits, targets, reduction="mean")
    return loss.item()


def induction_logit_diff(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_positions: torch.Tensor,
    distractor_tokens: torch.Tensor,
) -> float:
    """Mean logit-diff: logit[B] - logit[distractor] at the final-A position.

    Higher = stronger induction. This metric is less sensitive to overall
    distribution shifts than CE loss, so it isolates the induction effect
    from generic damage to early-layer representations.
    """
    with torch.no_grad():
        logits = model(tokens, return_type="logits")
    batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
    pred_logits = logits[batch_idx, target_positions, :]
    correct = tokens[:, 2]
    diff = pred_logits[batch_idx, correct] - pred_logits[batch_idx, distractor_tokens]
    return diff.mean().item()


def make_distractor_tokens(
    model: HookedTransformer,
    tokens: torch.Tensor,
    seed: int = 1,
) -> torch.Tensor:
    """Per-sequence distractor token != B and != A, for logit-diff baseline."""
    g = torch.Generator().manual_seed(seed)
    batch = tokens.shape[0]
    vocab_size = model.cfg.d_vocab
    distractors = torch.randint(10, vocab_size, (batch,), generator=g)
    # Resample any collisions with A or B
    A = tokens[:, 1].cpu()
    B = tokens[:, 2].cpu()
    for i in range(batch):
        while distractors[i].item() == A[i].item() or distractors[i].item() == B[i].item():
            distractors[i] = torch.randint(10, vocab_size, (1,), generator=g).item()
    return distractors.to(tokens.device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GPT-2 small on {device}...")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()

    tokens, target_positions = make_induction_batch(model, batch_size=100, seq_len=60, seed=0)
    tokens = tokens.to(device)
    target_positions = target_positions.to(device)

    baseline = induction_loss(model, tokens, target_positions)
    print(f"Baseline induction CE loss: {baseline:.4f}")

    distractors = make_distractor_tokens(model, tokens, seed=1).to(device)
    diff = induction_logit_diff(model, tokens, target_positions, distractors)
    print(f"Baseline induction logit-diff (correct - distractor): {diff:.4f}")
