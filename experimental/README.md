# Experimental: Recurrent-Depth Decoder for Neural TSP

This directory contains **experimental code** that extends the main paper's
architecture with a Recurrent-Depth Transformer (RDT) decoder.

## Key Properties

- **Self-contained.** Only imports from the main codebase, never modifies it.
- **Isolated artifacts.** All checkpoints go to `experimental/checkpoints/`,
  all results go to `experimental/results/`. Main paper's `checkpoints/` and
  `results/` directories are never touched.
- **Removable.** `rm -rf experimental/` reverts everything with zero side effects.
- **Future work.** Not part of the main paper's claims.

## What's Here

| File | Purpose |
|------|---------|
| `model_rdt.py` | `RecurrentGlimpseDecoder` with depth-wise LoRA and optional KL halting |
| `train_rdt.py` | Training script; supports warm-start from main paper checkpoints |
| `evaluate_rdt.py` | Full evaluation across 4 strategies (greedy, POMO-20, POMO+2opt, best-of-128) |
| `run_all_rdt.sh` | Orchestrated pipeline across all 4 metrics and T ∈ {2, 4} |
| `RESEARCH_PLAN.md` | Hypothesis, math, experimental protocol, continuation prompt |
| `README.md` | This file |

## Experimental Protocol (10 models, mirrors main paper)

To maintain parity with the main paper's 10-configuration rigor, the RDT
experiment is structured as three phases answering distinct questions:

### Phase 1: Euclidean Depth Sweep (5 models)
Parallels the main paper's 5 Euclidean variants. Answers:
*"How does solution quality scale with thinking depth T?"*

| # | Config | Warm-start source |
|---|--------|------------------|
| 1 | Euclidean T=1 (control) | `glimpse_rollout_euclidean_best.pt` |
| 2 | Euclidean T=2 | `glimpse_rollout_euclidean_best.pt` |
| 3 | Euclidean T=4 | `glimpse_rollout_euclidean_best.pt` |
| 4 | Euclidean T=8 (deep probe) | `glimpse_rollout_euclidean_best.pt` |
| 5 | Euclidean T=4 + entropy | `glimpse_rollout_euclidean_ent_best.pt` |

### Phase 2: Cross-Metric (2 models)
Parallels the main paper's Manhattan + Weighted Euclidean configs. Answers:
*"Does RDT help on metrics where attention struggles (Manhattan) or where axis
asymmetry matters (Weighted Euclidean)?"*

| # | Config | Warm-start source |
|---|--------|------------------|
| 6 | Manhattan T=4 | `glimpse_rollout_manhattan_best.pt` |
| 7 | Weighted Euclidean T=4 | `glimpse_rollout_weighted_euclidean_best.pt` |

### Phase 3: Mahalanobis 3-Way Ablation (3 models)
Parallels the main paper's original/whitened/whitened+spatial ablation. Answers:
*"Does RDT compound with whitening? With spatial encoding? Or is it orthogonal?"*

| # | Config | Warm-start source |
|---|--------|------------------|
| 8 | Mahalanobis original (no whitening) T=4 | `glimpse_rollout_mahalanobis_best.pt` |
| 9 | Mahalanobis whitened T=4 | `glimpse_rollout_mahalanobis_whiten_best.pt` |
| 10 | Mahalanobis whitened + spatial T=4 | `glimpse_rollout_mahalanobis_whiten_pomo_spatial_best.pt` |

All 10 models warm-start from main paper checkpoints and train for only 15
epochs each, because the encoder's representations are already learned — only
the LoRA adapters need to learn the refinement pattern. Total compute: ~2
hours vs. the main paper's ~90 minutes (comparable).

## Quick Start

**Full pipeline (recommended):**
```bash
chmod +x experimental/run_all_rdt.sh
./experimental/run_all_rdt.sh
```

**Individual commands:**
```bash
# Train one model
python experimental/train_rdt.py --metric mahalanobis \
    --thinking-steps 4 --epochs 15 \
    --init-from checkpoints/glimpse_rollout_mahalanobis_whiten_best.pt

# Evaluate all T values for one metric
python experimental/evaluate_rdt.py --metric mahalanobis --thinking-steps 2 4

# Quick evaluation (skip slow strategies)
python experimental/evaluate_rdt.py --metric mahalanobis \
    --thinking-steps 2 4 --skip-sampling --skip-2opt
```

## Output Files

All RDT artifacts live in `experimental/`:

```
experimental/
├── checkpoints/
│   ├── rdt_T2_r8_euclidean_best.pt
│   ├── rdt_T4_r8_euclidean_best.pt
│   ├── rdt_T2_r8_mahalanobis_best.pt
│   └── ... (8 models total)
└── results/
    ├── rdt_T*_*_history.json       # training curves
    ├── rdt_<metric>_full.json       # full evaluation results
    └── rdt_<metric>_eval.log        # evaluation logs
```

## What Gets Compared

The evaluation script **automatically reads** `results/evaluation_full.json`
from the main paper and prints a side-by-side comparison:

```
COMPARISON: RDT vs. Main Paper (glimpse decoder, mahalanobis)
─────────────────────────────────────────────────────
Strategy         Main Paper       Best RDT        Δ
─────────────────────────────────────────────────────
Greedy             15.4368        15.XXXX    +X.XXXX
POMO-20            15.0927        15.XXXX    +X.XXXX
POMO+2opt          15.0613        15.XXXX    +X.XXXX
```

Success criterion for the primary hypothesis: RDT POMO+2opt < 15.061
(i.e., beats the main paper's best Mahalanobis result of 0.31% gap).

## Parameter Overhead

For $d = 128$, LoRA rank $r = 8$, $T = 4$:
- Base model: 708,608 params
- RDT model: 716,800 params (**+1.2%**)

Only LoRA adapters are added. Base MHA weights are **shared across thinking
iterations**, so the model size is essentially unchanged.

## Rolling Back

```bash
rm -rf experimental/
```

Main paper codebase and artifacts are completely unaffected.

## Full Research Plan

See `RESEARCH_PLAN.md` for:
- Hypothesis and why this is novel
- Complete mathematical formulation (LaTeX)
- Phased experimental protocol
- Risk analysis and mitigation
- Continuation prompt for future sessions
- Citations to add if results warrant a paper section
