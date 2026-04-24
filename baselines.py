"""
Classical TSP baselines (project spec §2.5):
  - Nearest Neighbor (NN): greedy heuristic
  - 2-opt Local Search: improvement from an initial tour
  - OR-Tools: near-optimal solver for Optimality Gap

Supports 4 distance metrics:
  - "euclidean", "manhattan", "mahalanobis", "weighted_euclidean"

OR-Tools uses a precomputed distance matrix, allowing arbitrary metrics.
"""
import time
import numpy as np
import torch

from data import tour_length


# ════════════════════════════════════════════════════════════════════
# DISTANCE HELPERS
# ════════════════════════════════════════════════════════════════════

def _pairwise_distance(a, b, metric="euclidean", sigma_inv=None, weights=None):
    """Distance between two 2D numpy points."""
    diff = a - b
    if metric == "euclidean":
        return float(np.linalg.norm(diff))
    elif metric == "manhattan":
        return float(np.abs(diff).sum())
    elif metric == "mahalanobis":
        assert sigma_inv is not None
        return float(np.sqrt(max(diff @ sigma_inv @ diff, 0.0)))
    elif metric == "weighted_euclidean":
        assert weights is not None
        return float(np.sqrt(max((weights * diff ** 2).sum(), 0.0)))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _distance_from_point(coords_np, point_idx, metric="euclidean",
                          sigma_inv=None, weights=None):
    """Distance from coords_np[point_idx] to all others. Returns (n,)."""
    diffs = coords_np - coords_np[point_idx]
    if metric == "euclidean":
        return np.linalg.norm(diffs, axis=1)
    elif metric == "manhattan":
        return np.abs(diffs).sum(axis=1)
    elif metric == "mahalanobis":
        assert sigma_inv is not None
        quad = np.einsum("nd,de,ne->n", diffs, sigma_inv, diffs)
        return np.sqrt(np.maximum(quad, 0.0))
    elif metric == "weighted_euclidean":
        assert weights is not None
        weighted_sq = weights * (diffs ** 2)   # (n, 2)
        return np.sqrt(np.maximum(weighted_sq.sum(axis=1), 0.0))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _distance_matrix(coords_np, metric="euclidean", sigma_inv=None, weights=None):
    """Full n×n distance matrix. Used by OR-Tools."""
    n = coords_np.shape[0]
    diffs = coords_np[:, None, :] - coords_np[None, :, :]   # (n, n, 2)
    
    if metric == "euclidean":
        return np.linalg.norm(diffs, axis=-1)
    elif metric == "manhattan":
        return np.abs(diffs).sum(axis=-1)
    elif metric == "mahalanobis":
        assert sigma_inv is not None
        quad = np.einsum("ijd,de,ije->ij", diffs, sigma_inv, diffs)
        return np.sqrt(np.maximum(quad, 0.0))
    elif metric == "weighted_euclidean":
        assert weights is not None
        weighted_sq = weights * (diffs ** 2)    # (n, n, 2)
        return np.sqrt(np.maximum(weighted_sq.sum(axis=-1), 0.0))
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ════════════════════════════════════════════════════════════════════
# NEAREST NEIGHBOR
# ════════════════════════════════════════════════════════════════════

def nearest_neighbor_tour(coords_np, start=0, metric="euclidean",
                           sigma_inv=None, weights=None):
    n = coords_np.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True

    for _ in range(n - 1):
        current = tour[-1]
        dists = _distance_from_point(coords_np, current, metric,
                                      sigma_inv, weights)
        dists[visited] = np.inf
        next_city = int(np.argmin(dists))
        tour.append(next_city)
        visited[next_city] = True

    return tour


def nearest_neighbor_batch(coords, metric="euclidean", sigma_inv=None, weights=None):
    coords_np = coords.cpu().numpy()
    tours = [
        nearest_neighbor_tour(coords_np[i], metric=metric,
                               sigma_inv=sigma_inv, weights=weights)
        for i in range(coords.size(0))
    ]
    tours = torch.tensor(tours, dtype=torch.long)
    sigma_inv_t = torch.tensor(sigma_inv, dtype=torch.float32) if sigma_inv is not None else None
    weights_t = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
    lengths = tour_length(coords.cpu(), tours, metric=metric,
                           sigma_inv=sigma_inv_t, weights=weights_t)
    return tours, lengths


# ════════════════════════════════════════════════════════════════════
# 2-OPT
# ════════════════════════════════════════════════════════════════════

def two_opt_improve(coords_np, tour, metric="euclidean", sigma_inv=None,
                     weights=None, max_passes=1000):
    tour = list(tour)
    n = len(tour)
    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a = coords_np[tour[i - 1]]
                b = coords_np[tour[i]]
                c = coords_np[tour[j]]
                d = coords_np[tour[(j + 1) % n]]
                
                old_cost = (_pairwise_distance(a, b, metric, sigma_inv, weights)
                            + _pairwise_distance(c, d, metric, sigma_inv, weights))
                new_cost = (_pairwise_distance(a, c, metric, sigma_inv, weights)
                            + _pairwise_distance(b, d, metric, sigma_inv, weights))
                
                if new_cost < old_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True

    return tour


def two_opt_batch(coords, initial_tours=None, metric="euclidean",
                   sigma_inv=None, weights=None):
    coords_np = coords.cpu().numpy()
    batch_size = coords.size(0)
    
    if initial_tours is None:
        initial = [
            nearest_neighbor_tour(coords_np[i], metric=metric,
                                   sigma_inv=sigma_inv, weights=weights)
            for i in range(batch_size)
        ]
    else:
        initial = initial_tours.cpu().tolist()
    
    improved = [
        two_opt_improve(coords_np[i], initial[i], metric=metric,
                         sigma_inv=sigma_inv, weights=weights)
        for i in range(batch_size)
    ]
    tours = torch.tensor(improved, dtype=torch.long)
    sigma_inv_t = torch.tensor(sigma_inv, dtype=torch.float32) if sigma_inv is not None else None
    weights_t = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
    lengths = tour_length(coords.cpu(), tours, metric=metric,
                           sigma_inv=sigma_inv_t, weights=weights_t)
    return tours, lengths


# ════════════════════════════════════════════════════════════════════
# OR-TOOLS — supports all 4 metrics via precomputed distance matrix
# ════════════════════════════════════════════════════════════════════

def _tour_length_np(coords_np, tour, metric="euclidean", sigma_inv=None, weights=None):
    n = len(tour)
    total = 0.0
    for i in range(n):
        a = coords_np[tour[i]]
        b = coords_np[tour[(i + 1) % n]]
        total += _pairwise_distance(a, b, metric, sigma_inv, weights)
    return total


def solve_ortools(coords_np, time_limit_seconds=5, metric="euclidean",
                   sigma_inv=None, weights=None):
    """Solve one TSP with OR-Tools under any of 4 metrics."""
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    
    n = coords_np.shape[0]
    
    # Pre-compute full distance matrix
    dist_matrix = _distance_matrix(coords_np, metric=metric,
                                    sigma_inv=sigma_inv, weights=weights)
    scale = 100_000
    int_matrix = (dist_matrix * scale).astype(np.int64)
    
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(int_matrix[from_node][to_node])
    
    transit_cb_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)
    
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = time_limit_seconds
    
    solution = routing.SolveWithParameters(search_params)
    
    if solution is None:
        fallback = nearest_neighbor_tour(coords_np, metric=metric,
                                          sigma_inv=sigma_inv, weights=weights)
        return fallback, _tour_length_np(coords_np, fallback, metric,
                                          sigma_inv, weights)
    
    tour = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        tour.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    
    length = _tour_length_np(coords_np, tour, metric, sigma_inv, weights)
    return tour, length


def ortools_batch(coords, time_limit_seconds=5, metric="euclidean",
                   sigma_inv=None, weights=None, verbose=True):
    coords_np = coords.cpu().numpy()
    batch_size = coords.size(0)
    tours = []
    lengths = []
    
    for i in range(batch_size):
        if verbose and i % 100 == 0 and i > 0:
            print(f"  OR-Tools ({metric}): {i}/{batch_size} instances solved")
        tour, length = solve_ortools(coords_np[i], time_limit_seconds,
                                      metric=metric, sigma_inv=sigma_inv,
                                      weights=weights)
        tours.append(tour)
        lengths.append(length)
    
    return tours, torch.tensor(lengths, dtype=torch.float32)


# ════════════════════════════════════════════════════════════════════
# Sanity test
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data import generate_instances
    
    print("=" * 60)
    print("TESTING ALL 4 METRICS WITH ALL 3 SOLVERS")
    print("=" * 60)
    
    torch.manual_seed(42)
    coords = torch.rand(10, 20, 2)
    
    # Euclidean
    print("\n── EUCLIDEAN (uniform data) ──")
    t0 = time.time(); _, nn_e = nearest_neighbor_batch(coords, metric="euclidean")
    print(f"NN:       Mean={nn_e.mean():.4f}, Time={time.time()-t0:.2f}s")
    t0 = time.time(); _, to_e = two_opt_batch(coords, metric="euclidean")
    print(f"2-opt:    Mean={to_e.mean():.4f}, Time={time.time()-t0:.2f}s")
    t0 = time.time(); _, or_e = ortools_batch(coords, time_limit_seconds=2, metric="euclidean", verbose=False)
    print(f"OR-Tools: Mean={or_e.mean():.4f}, Time={time.time()-t0:.2f}s")
    
    # Manhattan
    print("\n── MANHATTAN (uniform data) ──")
    t0 = time.time(); _, nn_m = nearest_neighbor_batch(coords, metric="manhattan")
    print(f"NN:       Mean={nn_m.mean():.4f}, Time={time.time()-t0:.2f}s")
    t0 = time.time(); _, to_m = two_opt_batch(coords, metric="manhattan")
    print(f"2-opt:    Mean={to_m.mean():.4f}, Time={time.time()-t0:.2f}s")
    t0 = time.time(); _, or_m = ortools_batch(coords, time_limit_seconds=2, metric="manhattan", verbose=False)
    print(f"OR-Tools: Mean={or_m.mean():.4f}, Time={time.time()-t0:.2f}s")
    
    # Mahalanobis (with Gaussian data)
    print("\n── MAHALANOBIS (correlated Gaussian data) ──")
    sigma = [[0.08, 0.04], [0.04, 0.03]]
    sigma_inv = np.linalg.inv(np.array(sigma))
    gauss_coords = generate_instances(10, 20, seed=42,
                                       distribution="gaussian", sigma=sigma)
    t0 = time.time(); _, nn_mh = nearest_neighbor_batch(gauss_coords, metric="mahalanobis", sigma_inv=sigma_inv)
    print(f"NN:       Mean={nn_mh.mean():.4f}, Time={time.time()-t0:.2f}s")
    t0 = time.time(); _, to_mh = two_opt_batch(gauss_coords, metric="mahalanobis", sigma_inv=sigma_inv)
    print(f"2-opt:    Mean={to_mh.mean():.4f}, Time={time.time()-t0:.2f}s")
    t0 = time.time(); _, or_mh = ortools_batch(gauss_coords, time_limit_seconds=2, metric="mahalanobis", sigma_inv=sigma_inv, verbose=False)
    print(f"OR-Tools: Mean={or_mh.mean():.4f}, Time={time.time()-t0:.2f}s")
    
    # Weighted Euclidean (4x horizontal penalty)
    print("\n── WEIGHTED EUCLIDEAN (uniform data, weights [4, 1]) ──")
    weights = np.array([4.0, 1.0])
    t0 = time.time(); _, nn_we = nearest_neighbor_batch(coords, metric="weighted_euclidean", weights=weights)
    print(f"NN:       Mean={nn_we.mean():.4f}, Time={time.time()-t0:.2f}s")
    t0 = time.time(); _, to_we = two_opt_batch(coords, metric="weighted_euclidean", weights=weights)
    print(f"2-opt:    Mean={to_we.mean():.4f}, Time={time.time()-t0:.2f}s")
    t0 = time.time(); _, or_we = ortools_batch(coords, time_limit_seconds=2, metric="weighted_euclidean", weights=weights, verbose=False)
    print(f"OR-Tools: Mean={or_we.mean():.4f}, Time={time.time()-t0:.2f}s")
    
    print("\n" + "=" * 60)
    print("ALL 4 METRICS + ALL 3 SOLVERS WORKING")
    print("=" * 60)