"""Docstring task (Heimersheim & Janiak 2023) — held-out test for Phase 6.

A function signature lists N parameter names. The docstring lists `:param`
entries for the first N-1 of them. The model must predict the Nth parameter
name at the position right after the final `:param `.

Example prompt (with newline separators):
    def fn(self, alpha, beta, gamma, delta, epsilon):
        \"\"\"
        :param alpha: example value
        :param beta: example value
        :param gamma: example value
        :param delta: example value
        :param  <- predict ' epsilon' here

Correct answer = the name of the unmentioned final parameter.
Distractor   = the name of an earlier mentioned parameter (model should
                NOT repeat one it's already covered).

We expose `make_docstring_batch` returning (tokens, final_positions,
correct_tokens, distractor_tokens), exactly the same signature shape as IOI,
so the rest of the pipeline plugs in unchanged.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
from transformer_lens import HookedTransformer


# Pool of identifier-like words that tokenize cleanly. Each must be a SINGLE
# leading-space GPT-2 token (so logit-diff is at the immediate next position).
PARAM_POOL = [
    "files", "data", "size", "shape", "state", "option", "value", "name",
    "items", "config", "path", "mode", "flag", "key", "token", "user",
    "color", "type", "index", "label", "limit", "scale", "rate", "weight",
    "depth", "count", "offset", "result", "target", "source",
]

DESCRIPTIONS = [
    "example value", "the input value", "an integer flag", "a list of items",
    "configuration entry", "the requested item", "string identifier",
    "metadata field", "an optional argument", "the source object",
]


def _build_one(rng: random.Random, n_params: int = 5) -> Tuple[str, str, str]:
    """Return (prompt_text, correct_param, distractor_param)."""
    params = rng.sample(PARAM_POOL, n_params)
    *given, missing = params  # all but last appear in docstring
    distractor = rng.choice(given)
    lines = ["def fn(self, " + ", ".join(params) + "):",
             '    """']
    for p in given:
        lines.append(f"    :param {p}: {rng.choice(DESCRIPTIONS)}")
    lines.append("    :param")
    text = "\n".join(lines)
    return text, missing, distractor


def make_docstring_batch(
    model: HookedTransformer,
    batch_size: int = 32,
    seed: int = 0,
    n_params: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id or 50256

    prompts: List[str] = []
    corrects: List[str] = []
    distractors: List[str] = []
    for _ in range(batch_size):
        text, c, d = _build_one(rng, n_params=n_params)
        prompts.append(text)
        corrects.append(c)
        distractors.append(d)

    seqs = []
    finals = []
    for p in prompts:
        ids = model.to_tokens(p, prepend_bos=True)[0]
        seqs.append(ids)
        finals.append(ids.shape[0] - 1)

    T = max(s.shape[0] for s in seqs)
    tokens = torch.full((batch_size, T), pad_id, dtype=torch.long)
    final_positions = torch.zeros(batch_size, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        tokens[i, :L] = s
        final_positions[i] = L - 1

    correct_tokens = torch.tensor(
        [model.tokenizer.encode(" " + n, add_special_tokens=False)[0] for n in corrects],
        dtype=torch.long,
    )
    distractor_tokens = torch.tensor(
        [model.tokenizer.encode(" " + n, add_special_tokens=False)[0] for n in distractors],
        dtype=torch.long,
    )
    return tokens, final_positions, correct_tokens, distractor_tokens


def docstring_logit_diff(
    model: HookedTransformer,
    tokens: torch.Tensor,
    final_positions: torch.Tensor,
    correct_tokens: torch.Tensor,
    distractor_tokens: torch.Tensor,
) -> float:
    with torch.no_grad():
        logits = model(tokens, return_type="logits")
    batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
    pred = logits[batch_idx, final_positions, :]
    diff = pred[batch_idx, correct_tokens] - pred[batch_idx, distractor_tokens]
    return diff.mean().item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GPT-2 small on {device}...")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    tokens, finals, c, d = make_docstring_batch(model, batch_size=32, seed=0)
    tokens = tokens.to(device); finals = finals.to(device)
    c = c.to(device); d = d.to(device)
    diff = docstring_logit_diff(model, tokens, finals, c, d)
    print(f"Baseline docstring logit-diff: {diff:.4f}  (expect ~1-3 on GPT-2 small)")
