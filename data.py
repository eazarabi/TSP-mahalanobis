"""
Random TSP instance generation (uniform or Gaussian)
Tour length L(pi) calculation under 4 metrics:
  - Euclidean: standard straight-line distance
  - Manhattan: L1/grid distance
  - Mahalanobis: correlation-weighted distance (for Gaussian data)
  - Weighted Euclidean: axis-asymmetric Euclidean distance

Includes whitening transform: pre-multiplies Gaussian coords by Σ^(-1/2)
so that Mahalanobis distance in the original space = Euclidean in whitened space.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader


# ════════════════════════════════════════════════════════════════════
# WHITENING UTILITIES
# ════════════════════════════════════════════════════════════════════

def compute_whitening_matrix(sigma):
    """
    Compute Σ^(-1/2) via eigendecomposition.
    After whitening: d_Mah(a, b) = d_Euc(W·a, W·b)

    Args:
        sigma: (2, 2) covariance matrix as list or tensor
    Returns:
        W: (2, 2) whitening matrix Σ^(-1/2)
        W_inv: (2, 2) inverse whitening matrix Σ^(1/2) (for de-whitening)
    """
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, dtype=torch.float32)
    # Eigendecomposition: Σ = V Λ Vᵀ
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
    # Σ^(-1/2) = V Λ^(-1/2) Vᵀ
    W = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.T
    # Σ^(1/2) = V Λ^(1/2) Vᵀ  (for inverse transform)
    W_inv = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
    return W, W_inv


def whiten_coords(coords, W):
    """
    Apply whitening transform to coordinates.
    coords: (batch, n, 2) or (n, 2)
    W: (2, 2) whitening matrix
    Returns whitened coords in same shape.
    """
    return coords @ W.T


def augment_8fold_whitened(coords):
    """
    Apply 8-fold dihedral augmentation in whitened space.
    In whitened space, Euclidean distances are preserved under rotations/reflections,
    which correspond to Mahalanobis-preserving transforms in the original space.

    coords: (batch, n, 2) whitened coordinates
    Returns: (batch*8, n, 2)
    """
    # Center around mean (whitened coords aren't necessarily in [0,1])
    center = coords.mean(dim=1, keepdim=True)  # (batch, 1, 2)
    centered = coords - center

    x, y = centered[..., 0:1], centered[..., 1:2]

    augs = [
        centered,                                    # identity
        torch.cat([-x, y], dim=-1),                  # reflect x
        torch.cat([x, -y], dim=-1),                  # reflect y
        torch.cat([-x, -y], dim=-1),                 # 180° rotation
        torch.cat([y, x], dim=-1),                   # swap axes (diagonal reflection)
        torch.cat([-y, x], dim=-1),                  # 90° rotation
        torch.cat([y, -x], dim=-1),                  # 270° rotation
        torch.cat([-y, -x], dim=-1),                 # anti-diagonal reflection
    ]

    # Re-center each augmentation to original center
    return torch.cat([a + center for a in augs], dim=0)


class TSPDataset(Dataset):
    def __init__(self, coords):
        self.data = coords

    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ════════════════════════════════════════════════════════════════════
# INSTANCE GENERATION
# ════════════════════════════════════════════════════════════════════

def generate_instances(num_instances, num_cities, seed=None,
                       distribution="uniform", sigma=None):
    """
    Generate TSP instances.
    
    Args:
        num_instances: number of TSP problems
        num_cities:    cities per problem
        seed:          for reproducibility
        distribution:  "uniform" → U[0,1]²  (for Euclidean/Manhattan/Weighted)
                       "gaussian" → N(mu, Sigma) (for Mahalanobis)
        sigma:         2x2 covariance matrix (required if distribution="gaussian")
    """
    if distribution == "uniform":
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            return torch.rand(num_instances, num_cities, 2, generator=generator)
        return torch.rand(num_instances, num_cities, 2)
    
    elif distribution == "gaussian":
        assert sigma is not None, "Must provide sigma for Gaussian distribution"
        sigma_t = torch.tensor(sigma, dtype=torch.float32)
        mean = torch.tensor([0.5, 0.5])
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=sigma_t)
        
        if seed is not None:
            torch.manual_seed(seed)
        samples = dist.sample((num_instances, num_cities))
        return samples
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ════════════════════════════════════════════════════════════════════
# TOUR LENGTH — four metrics
# ════════════════════════════════════════════════════════════════════

def tour_length(coords, tours, metric="euclidean", sigma_inv=None, weights=None):
    """
    Compute total tour length for a batch of tours under any of 4 metrics.
    L(π) = Σ d(x_{π_t}, x_{π_{t+1}}) + d(x_{π_n}, x_{π_1})
    
    Args:
        coords:    (batch, n, 2) city coordinates
        tours:     (batch, n) permutation of city indices
        metric:    "euclidean", "manhattan", "mahalanobis", or "weighted_euclidean"
        sigma_inv: (2, 2) inverse covariance (for mahalanobis)
        weights:   (2,) per-axis weights (for weighted_euclidean)
    
    Returns:
        (batch,) tensor of tour lengths
    """
    idx = tours.unsqueeze(-1).expand_as(coords)
    ordered = coords.gather(1, idx)
    rolled = torch.roll(ordered, shifts=-1, dims=1)
    diffs = ordered - rolled                             # (batch, n, 2)
    
    if metric == "euclidean":
        dists = diffs.norm(p=2, dim=-1)
    
    elif metric == "manhattan":
        dists = diffs.abs().sum(dim=-1)
    
    elif metric == "mahalanobis":
        assert sigma_inv is not None, "Mahalanobis requires sigma_inv"
        if not isinstance(sigma_inv, torch.Tensor):
            sigma_inv = torch.tensor(sigma_inv, dtype=coords.dtype,
                                      device=coords.device)
        transformed = torch.einsum("bnd,de->bne", diffs, sigma_inv)
        quadratic = (diffs * transformed).sum(dim=-1)
        dists = torch.sqrt(quadratic.clamp(min=1e-12))
    
    elif metric == "weighted_euclidean":
        assert weights is not None, "Weighted Euclidean requires weights"
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=coords.dtype,
                                    device=coords.device)
        # d = √(w_x·dx² + w_y·dy²)
        weighted_sq = weights * (diffs ** 2)             # (batch, n, 2)
        dists = torch.sqrt(weighted_sq.sum(dim=-1).clamp(min=1e-12))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return dists.sum(dim=-1)


# ════════════════════════════════════════════════════════════════════
# SAVE / LOAD (unchanged)
# ════════════════════════════════════════════════════════════════════

def save_dataset(coords, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(coords, path)


def load_dataset(path):
    return torch.load(path, weights_only=True)


def prepare_datasets(config):
    """Prepare val/test datasets based on config.data_distribution."""
    dist = config.data_distribution
    sigma = config.mahalanobis_sigma if dist == "gaussian" else None
    
    suffix = f"_{dist}" if dist != "uniform" else ""
    val_path = os.path.join(
        config.data_dir,
        f"tsp{config.num_cities}_val_seed{config.val_seed}{suffix}.pt"
    )
    test_path = os.path.join(
        config.data_dir,
        f"tsp{config.num_cities}_test_seed{config.test_seed}{suffix}.pt"
    )
    
    if os.path.exists(val_path):
        val_coords = load_dataset(val_path)
        print(f"Loaded validation set from {val_path}")
    else:
        val_coords = generate_instances(
            config.val_size, config.num_cities,
            seed=config.val_seed, distribution=dist, sigma=sigma
        )
        save_dataset(val_coords, val_path)
        print(f"Generated and saved validation set to {val_path}")
    
    if os.path.exists(test_path):
        test_coords = load_dataset(test_path)
        print(f"Loaded test set from {test_path}")
    else:
        test_coords = generate_instances(
            config.test_size, config.num_cities,
            seed=config.test_seed, distribution=dist, sigma=sigma
        )
        save_dataset(test_coords, test_path)
        print(f"Generated and saved test set to {test_path}")
    
    return val_coords, test_coords


def get_train_loader(config):
    dist = config.data_distribution
    sigma = config.mahalanobis_sigma if dist == "gaussian" else None
    train_coords = generate_instances(
        config.train_size, config.num_cities,
        distribution=dist, sigma=sigma
    )
    dataset = TSPDataset(train_coords)
    return DataLoader(dataset, batch_size=config.batch_size,
                       shuffle=True, num_workers=0)


def get_eval_loader(coords, config):
    dataset = TSPDataset(coords)
    return DataLoader(dataset, batch_size=config.eval_batch_size,
                       shuffle=False, num_workers=0)