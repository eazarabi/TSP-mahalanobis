═══════════════════════════════════════════════════════════════════════════════
  Reinforcement Learning with Transformers for the Traveling Salesman Problem
  COSC6354 — Spring 2026
═══════════════════════════════════════════════════════════════════════════════

REQUIREMENTS
────────────
- Python 3.12+
- PyTorch (with CUDA recommended)
- scipy, numpy, matplotlib
- ortools (Google OR-Tools)

Install:
  pip install torch numpy scipy matplotlib ortools

QUICK START
───────────
To run the full pipeline (train all models + evaluate + generate plots):

  chmod +x run_all.sh
  ./run_all.sh

This takes ~4-5 hours (mostly OR-Tools evaluation).
Requires a GPU with >= 8GB VRAM for training.

PROJECT STRUCTURE
─────────────────
  config.py       - All hyperparameters and experiment settings
  data.py         - TSP instance generation, tour length calculation, whitening
  model.py        - Transformer encoder/decoder architecture
  train.py        - REINFORCE training with rollout/critic baseline
  evaluate.py     - Full evaluation pipeline with all decoding strategies
  baselines.py    - Classical solvers (NN, 2-opt, OR-Tools)
  run_all.sh      - Master script to run everything

  checkpoints/    - Saved model weights (.pt files)
  data/           - Cached TSP datasets
  results/        - Output: JSON results, PNG plots, training logs

TRAINING INDIVIDUAL MODELS
──────────────────────────
# Euclidean (standard TSP)
python train.py --metric euclidean --glimpse --baseline rollout --epochs 100 --no-pomo --no-spatial --no-whiten

# Mahalanobis (original)
python train.py --metric mahalanobis --glimpse --baseline rollout --epochs 100 --no-pomo --no-spatial --no-whiten

# Mahalanobis (improved with whitening + spatial encoding)
python train.py --metric mahalanobis --glimpse --baseline rollout --epochs 100 --no-pomo --no-spatial

# Manhattan
python train.py --metric manhattan --glimpse --baseline rollout --epochs 100 --no-pomo --no-spatial --no-whiten

# Weighted Euclidean
python train.py --metric weighted_euclidean --glimpse --baseline rollout --epochs 100 --no-pomo --no-spatial --no-whiten

EVALUATION
──────────
After training all models:
  python evaluate.py

Outputs:
  results/evaluation_full.json         - All numerical results
  results/tours_*.png                  - Tour visualizations (3 instances per metric)
  results/attention_*.png              - Decoder attention heatmaps
  results/sample_efficiency_*.png      - Training curves (val length vs epoch)
  results/length_distribution_*.png    - Tour length histograms


METHODS
───────
1. Whitening Transform: Σ^(-1/2) preprocessing converts Mahalanobis to Euclidean
2. POMO Multi-Start: Try all N starting nodes, keep best tour (inference-time)
3. 8x Augmentation in Whitened Space: Dihedral group symmetries (novel for Mahalanobis)
4. 2-opt Post-Processing: Refine RL solutions with local search
5. Spatial Encoding Bias: Distance-indexed learnable attention bias

DISTANCE METRICS
────────────────
  euclidean         - Standard L2 norm
  manhattan         - L1 / grid distance
  mahalanobis       - Correlation-weighted (Gaussian data)
  weighted_euclidean - Axis-asymmetric (directional costs)
