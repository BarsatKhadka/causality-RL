# Phase 1 — Causal scoring infrastructure

Goal: given any attention head in GPT-2 small, measure how causally important
it is for the induction behaviour. This score is the reward function for the
later RL agent.

## Files
- `induction_dataset.py` — Step 1.2: build induction test sequences and measure baseline loss.
- `ablation.py` — Step 1.3: zero-ablation hook for a single attention head.
- `score_all_heads.py` — Step 1.4: loop over all 144 heads, score, save, plot.
- `requirements.txt` — torch, transformer_lens, numpy, matplotlib, tqdm.

## Setup
From the project root (with the existing `venv` activated):

```powershell
.\venv\Scripts\Activate.ps1
pip install -r ".\phase 1\requirements.txt"
```

## Run
```powershell
python ".\phase 1\score_all_heads.py"
```

Outputs to `phase 1/results/`:
- `head_scores.npy` — [12, 12] array of Δloss when each head is ablated
- `summary.json` — baseline loss + config
- `head_scores_heatmap.png` — visual; known induction heads 5.5 and 6.9 boxed in red

## Done-when (from plan.md)
The heatmap should show a few heads with much higher scores than the rest, and
heads **5.5** and **6.9** should rank near the top.
