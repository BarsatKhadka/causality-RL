"""IOI (Indirect Object Identification) task — Wang et al. 2022.

Sentences of the form:
    "When John and Mary went to the store, John gave a drink to ___"
The model should predict Mary (the indirect object), not John (the subject).

We construct sentences with two distinct names. The correct answer is the
indirect object (IO); the distractor is the subject (S). Logit-diff metric:
    diff = logit[IO] - logit[S]   at the final position
A clean GPT-2 small gets a positive diff (it prefers Mary > John). Ablating
a head that breaks the IOI circuit reduces this diff.

Templates and names follow the conventions of Wang et al. (ABBA / BABA orders).
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
from transformer_lens import HookedTransformer


NAMES = [
    "John", "Mary", "Tom", "James", "Sarah", "Michael", "Robert", "Jennifer",
    "David", "Linda", "William", "Patricia", "Richard", "Susan", "Joseph",
    "Karen", "Charles", "Nancy", "Daniel", "Lisa", "Anna", "Mark", "Paul",
    "Emily", "Andrew", "George",
]

OBJECTS = [
    "ring", "kiss", "bone", "basketball", "computer", "necklace", "drink",
    "snack", "book", "letter", "ball", "guitar", "pen", "shirt", "watch",
]

PLACES = [
    "store", "garden", "restaurant", "school", "park", "hospital", "library",
    "office", "station", "market", "cafe", "gym",
]

# ABBA: "When A and B went to X, A gave Y to" -> predicts B
# BABA: "When B and A went to X, A gave Y to" -> predicts B
TEMPLATES = [
    "When {A} and {B} went to the {place}, {S} gave the {obj} to",
    "When {A} and {B} were at the {place}, {S} gave the {obj} to",
    "After {A} and {B} arrived at the {place}, {S} handed the {obj} to",
    "While {A} and {B} were talking, {S} passed the {obj} to",
]


def _build_one(rng: random.Random) -> Tuple[str, str, str]:
    """Return (prompt_text, IO_name, S_name).

    Per Wang et al., we balance ABBA/BABA orderings: half the time the subject
    is the second name introduced, half the time it's the first. The IO is
    always the *other* name (the one that's not the subject).
    """
    a, b = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECTS)
    place = rng.choice(PLACES)
    template = rng.choice(TEMPLATES)
    # Pick which of the two introduced names is the subject (the one giving).
    subject_first = rng.random() < 0.5
    if subject_first:
        S, IO = a, b
    else:
        S, IO = b, a
    text = template.format(A=a, B=b, S=S, place=place, obj=obj)
    return text, IO, S


def make_ioi_batch(
    model: HookedTransformer,
    batch_size: int = 32,
    seed: int = 0,
    max_len: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a batch of IOI prompts, right-padded to the same length.

    Returns
    -------
    tokens : LongTensor [batch, T]
    final_positions : LongTensor [batch] — index of the *last real token* per
        prompt. The next-token prediction at this position is what we score.
    io_tokens : LongTensor [batch] — token id of the IO name (with leading space).
    s_tokens : LongTensor [batch] — token id of the subject name (distractor).
    """
    rng = random.Random(seed)
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    bos_id = model.tokenizer.bos_token_id or pad_id

    prompts: List[str] = []
    ios: List[str] = []
    subjs: List[str] = []
    for _ in range(batch_size):
        text, io, s = _build_one(rng)
        prompts.append(text)
        ios.append(io)
        subjs.append(s)

    # Tokenize each prompt separately so we can locate the true last position.
    seqs = []
    finals = []
    for p in prompts:
        ids = model.to_tokens(p, prepend_bos=True)[0]
        seqs.append(ids)
        finals.append(ids.shape[0] - 1)

    T = max(s.shape[0] for s in seqs)
    if max_len is not None:
        T = min(T, max_len)
    tokens = torch.full((batch_size, T), pad_id, dtype=torch.long)
    final_positions = torch.zeros(batch_size, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = min(s.shape[0], T)
        tokens[i, :L] = s[:L]
        final_positions[i] = L - 1

    # Tokenize names WITH leading space (matches how they'd appear after "to ").
    io_tokens = torch.tensor(
        [model.tokenizer.encode(" " + n, add_special_tokens=False)[0] for n in ios],
        dtype=torch.long,
    )
    s_tokens = torch.tensor(
        [model.tokenizer.encode(" " + n, add_special_tokens=False)[0] for n in subjs],
        dtype=torch.long,
    )
    return tokens, final_positions, io_tokens, s_tokens


def ioi_logit_diff(
    model: HookedTransformer,
    tokens: torch.Tensor,
    final_positions: torch.Tensor,
    io_tokens: torch.Tensor,
    s_tokens: torch.Tensor,
) -> float:
    """Mean (logit[IO] - logit[S]) at the final-real-token position.

    Positive on a clean model (it prefers the indirect object over the subject).
    """
    with torch.no_grad():
        logits = model(tokens, return_type="logits")  # [B, T, V]
    batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
    pred = logits[batch_idx, final_positions, :]
    diff = pred[batch_idx, io_tokens] - pred[batch_idx, s_tokens]
    return diff.mean().item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GPT-2 small on {device}...")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    tokens, finals, io, s = make_ioi_batch(model, batch_size=64, seed=0)
    tokens = tokens.to(device); finals = finals.to(device)
    io = io.to(device); s = s.to(device)
    diff = ioi_logit_diff(model, tokens, finals, io, s)
    print(f"Baseline IOI logit-diff (IO - S): {diff:.4f}  (expect ~3-4 on GPT-2 small)")
