"""
Config for TSP Transformer Reinforcement Learning Project

Hyperparameters and other settings are defined here.
"""

import torch

class Config:
    # Problem definition
    num_cities = 20 # N >= 20 per spec
    coord_dim = 2 # 2D Euclidean (xi all elements of R^2)

    # Dataset
    train_size = 50000
    val_size = 1000
    test_size = 1000
    val_seed = 4321
    test_seed = 1234

    # Model Architecture
    embed_dim = 128
    num_heads = 8
    num_encoder_layers = 3
    ff_dim = 512
    tanh_clipping = 10.0
    dropout = 0.0

    # Training
    batch_size = 512
    num_epochs = 100
    lr = 1e-4
    max_grad_norm = 1.0
    
    baseline_type = "rollout"
    rollout_update_threshold = 0.05 

    critic_lr = 1e-4
    critic_embed_dim = 128
    critic_num_layers = 3
    
    # ─── Entropy Regularization ───
    # Set > 0 to enable entropy bonus during training: loss -= β * H(π)
    # Higher β → more exploration. 0.01–0.05 is typical range.
    entropy_coef = 0.00   # 0 = disabled (existing behavior unchanged)
    
    # ─── Beam Search ───
    beam_width = 10

    # Evaluation
    eval_batch_size = 256

    # Paths
    data_dir = "data"
    checkpoint_dir = "checkpoints"
    results_dir = "results"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ═══════════════════════════════════════════════════════════════
    # Distance Metric Experiment Settings
    # ═══════════════════════════════════════════════════════════════
    # Options: "euclidean", "manhattan", "mahalanobis", "weighted_euclidean"
    distance_metric = "euclidean"
    
    # Data distribution:
    #   "uniform"  → U[0,1]² (use with Euclidean, Manhattan, Weighted Euclidean)
    #   "gaussian" → multivariate normal (use with Mahalanobis)
    data_distribution = "uniform"
    
    # ─── Mahalanobis covariance matrix (Σ) ───
    # Positive-definite; eigenvalues [0.008, 0.102] for elongated clusters
    mahalanobis_sigma = [[0.08, 0.04],
                         [0.04, 0.03]]

    # ─── Whitening Transform (for improved Mahalanobis) ───
    # When True, pre-transforms Gaussian coords by Σ^(-1/2) before encoder.
    # In whitened space, Mahalanobis distance = Euclidean distance.
    # This lets the model use its Euclidean-distance intuition directly.
    use_whitening = True

    # ─── POMO-style multi-start decoding ───
    # Number of starting nodes for multi-start sampling during training.
    # At test time, all N cities are used as start nodes.
    pomo_start_nodes = 20  # = num_cities for full POMO

    # ─── Spatial Encoding Bias ───
    # When True, adds a learnable distance-indexed bias to attention logits.
    # Inspired by Zhao & Wong (PLOS One, 2025).
    use_spatial_encoding = True
    num_distance_bins = 32  # discretize distances into this many bins

    # ─── Weighted Euclidean weights (w_x, w_y) ───
    # Models directional asymmetry (e.g., drone against wind, sloped terrain)
    # d(a,b) = √(w_x·dx² + w_y·dy²)
    # (4, 1) means horizontal moves cost 4× vertical moves
    euclidean_weights = [4.0, 1.0]