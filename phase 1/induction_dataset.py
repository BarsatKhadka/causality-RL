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


_NATURAL_CORPUS_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def load_natural_corpus(
    model: HookedTransformer,
    n_tokens: int = 200_000,
    cache_key: str = "wikitext-2",
) -> torch.Tensor:
    """Tokenize a slice of natural English text into one long token tensor.

    Cached per-process. The control batch sampler picks random sub-spans from
    this tensor, so ablations are scored against text where previous-token
    heads, attention patterns, and general syntax all matter — unlike random
    tokens, where they don't.
    """
    key = (cache_key, n_tokens)
    if key in _NATURAL_CORPUS_CACHE:
        return _NATURAL_CORPUS_CACHE[key]
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(t for t in ds["text"] if t.strip())[: n_tokens * 8]
    except Exception:
        # Fallback: a hardcoded English paragraph repeated. Crude but real text.
        # Used if the HPC node has no internet for datasets / no datasets pkg.
        text = (
            "The agent learns by interacting with the environment. Each action "
            "produces feedback used to update the policy. Over many episodes, "
            "the agent gradually improves. Reinforcement learning is the study "
            "of how agents ought to take actions to maximize cumulative reward. "
            "Language models predict the next token given the preceding context. "
        ) * 4000
    toks = model.to_tokens(text, prepend_bos=False)[0][:n_tokens]
    _NATURAL_CORPUS_CACHE[key] = toks
    return toks


def make_control_batch(
    model: HookedTransformer,
    batch_size: int = 32,
    seq_len: int = 60,
    seed: int = 0,
    corpus: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample a batch of natural-text token spans for the contrastive control.

    If `corpus` is None, lazily loads wikitext via `load_natural_corpus`.
    Picks `batch_size` random starting positions, takes `seq_len-1` tokens from
    each, and prepends BOS. Used as a stronger contrastive baseline than
    random tokens, because previous-token heads and general syntax matter here.
    """
    if corpus is None:
        corpus = load_natural_corpus(model)
    g = torch.Generator().manual_seed(seed)
    max_start = corpus.shape[0] - (seq_len - 1) - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=g)
    bos_id = model.tokenizer.bos_token_id or 50256
    out = torch.empty((batch_size, seq_len), dtype=torch.long)
    out[:, 0] = bos_id
    for i, s in enumerate(starts.tolist()):
        out[i, 1:] = corpus[s : s + seq_len - 1]
    return out


def control_mean_loss(
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> float:
    """Mean next-token cross-entropy across non-BOS positions on the control batch.

    A clean GPT-2 small gets ~6-9 on random tokens (high because tokens are
    unpredictable random ints). Ablating a generally-important head pushes this
    up substantially; ablating an induction-specific head barely moves it.
    """
    with torch.no_grad():
        logits = model(tokens, return_type="logits")  # [B, T, V]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = tokens[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
        reduction="mean",
    )
    return loss.item()


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
