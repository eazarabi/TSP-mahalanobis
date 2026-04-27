═══════════════════════════════════════════════════════════════════════════════
  Reinforcement Learning with Transformers for the Traveling Salesman Problem
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

TRAINING INDIVIDUAL MODELS
──────────────────────────
# Euclidean (standard TSP)
python train.py --metric euclidean --glimpse --baseline rollout --epochs 5--no-pomo --no-spatial --no-whiten

# Mahalanobis (original)
python train.py --metric mahalanobis --glimpse --baseline rollout --epochs 100 --no-pomo --no-spatial --no-whiten

# Mahalanobis 
python train.py --metric mahalanobis --glimpse --baseline rollout --epochs 5 --no-pomo --no-spatial

# Manhattan
python train.py --metric manhattan --glimpse --baseline rollout --epochs 100 --no-pomo --no-spatial --no-whiten

# Weighted Euclidean
python train.py --metric weighted_euclidean --glimpse --baseline rollout --epochs 100 --no-pomo --no-spatial --no-whiten

DISTANCE METRICS
────────────────
  euclidean         - Standard L2 norm
  manhattan         - L1 / grid distance
  mahalanobis       - Correlation-weighted (Gaussian data)
  weighted_euclidean - Axis-asymmetric (directional costs)
