"""
Final evaluation pipeline.

Decoding strategies:
  - Greedy (1 tour per instance)
  - Best-of-128 sampling
  - Beam search (widths 10, 25)
  - 8x instance augmentation (for Euclidean, Weighted Euclidean, and Mahalanobis via whitening)
  - POMO multi-start (N=20 start nodes, greedy from each)
  - RL + 2-opt post-processing

Cross-metric comparison:
  Euclidean / Manhattan / Mahalanobis / Weighted Euclidean

Additional analyses:
  - Attention/probability heatmap
  - Tour visualizations per method
  - Length distribution histograms
  - Sample-efficiency learning curves

Run ONCE after all training complete:
    python evaluate.py
"""
import os
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from data import (prepare_datasets, get_eval_loader, tour_length, generate_instances,
                  compute_whitening_matrix, whiten_coords, augment_8fold_whitened)
from model import TSPTransformer
from baselines import (
    nearest_neighbor_batch,
    two_opt_batch,
    ortools_batch,
    _pairwise_distance,
)


# ════════════════════════════════════════════════════════════════════
# DECODING STRATEGIES
# ════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path, config, use_glimpse):
    model = TSPTransformer(config, use_glimpse=use_glimpse).to(config.device)
    ckpt = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def evaluate_greedy(model, test_loader, device, metric, sigma_inv, weights,
                    W_whiten=None):
    """Greedy decoding — one tour per instance."""
    model.eval()
    all_tours, all_lengths = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            enc_input = batch
            if W_whiten is not None:
                enc_input = whiten_coords(batch, W_whiten.to(device))
            tours, _ = model(enc_input, decode_type="greedy")
            lengths = tour_length(batch, tours, metric=metric,
                                   sigma_inv=sigma_inv, weights=weights)
            all_tours.append(tours.cpu())
            all_lengths.append(lengths.cpu())
    return torch.cat(all_tours), torch.cat(all_lengths), time.time() - t0


def evaluate_sampling(model, test_loader, device, metric, sigma_inv, weights,
                      W_whiten=None, N=128):
    """Best-of-N sampling — sample N tours per instance, keep shortest."""
    model.eval()
    all_best_tours, all_best_lengths = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            bs = batch.size(0)
            enc_input = batch
            if W_whiten is not None:
                enc_input = whiten_coords(batch, W_whiten.to(device))
            best_lengths = torch.full((bs,), float('inf'), device=device)
            best_tours = torch.zeros((bs, batch.size(1)), dtype=torch.long, device=device)
            for _ in range(N):
                tours, _ = model(enc_input, decode_type="sample")
                lengths = tour_length(batch, tours, metric=metric,
                                       sigma_inv=sigma_inv, weights=weights)
                improved = lengths < best_lengths
                best_lengths = torch.where(improved, lengths, best_lengths)
                best_tours[improved] = tours[improved]
            all_best_tours.append(best_tours.cpu())
            all_best_lengths.append(best_lengths.cpu())
    return torch.cat(all_best_tours), torch.cat(all_best_lengths), time.time() - t0


def evaluate_beam_search(model, test_loader, device, metric, sigma_inv, weights,
                         W_whiten=None, beam_width=10):
    """Beam search decoding."""
    model.eval()
    all_tours, all_lengths = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            enc_input = batch
            if W_whiten is not None:
                enc_input = whiten_coords(batch, W_whiten.to(device))
            tours, _ = model(enc_input, decode_type="beam_search", beam_width=beam_width)
            lengths = tour_length(batch, tours, metric=metric,
                                   sigma_inv=sigma_inv, weights=weights)
            all_tours.append(tours.cpu())
            all_lengths.append(lengths.cpu())
    return torch.cat(all_tours), torch.cat(all_lengths), time.time() - t0


def augment_instance(coords):
    """Apply 8 augmentations (4 rotations x 2 reflections). For Euclidean metrics."""
    centered = coords - 0.5
    augs = [
        centered,
        torch.stack([-centered[..., 1], centered[..., 0]], dim=-1),
        -centered,
        torch.stack([centered[..., 1], -centered[..., 0]], dim=-1),
    ]
    flipped = centered.clone()
    flipped[..., 0] = -flipped[..., 0]
    augs.extend([
        flipped,
        torch.stack([-flipped[..., 1], flipped[..., 0]], dim=-1),
        -flipped,
        torch.stack([flipped[..., 1], -flipped[..., 0]], dim=-1),
    ])
    return torch.cat([a + 0.5 for a in augs], dim=0)


def evaluate_augmented(model, test_coords, config, metric, sigma_inv, weights,
                       W_whiten=None):
    """
    8x augmentation: best of 8 transformed versions per instance.
    For Euclidean/Weighted Euclidean: augment in original space.
    For Mahalanobis: augment in WHITENED space (novel technique).
    """
    model.eval()
    device = config.device
    n_inst = test_coords.size(0)
    bs_eval = config.eval_batch_size
    all_best = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_inst, bs_eval):
            batch = test_coords[i:i + bs_eval].to(device)
            bs = batch.size(0)

            if metric == "mahalanobis" and W_whiten is not None:
                # Augment in whitened space (preserves Mahalanobis distances)
                whitened = whiten_coords(batch, W_whiten.to(device))
                aug_batch = augment_8fold_whitened(whitened)  # (bs*8, n, 2)
                aug_tours, _ = model(aug_batch, decode_type="greedy")
                # Compute lengths using original coords expanded
                batch_exp = batch.repeat(8, 1, 1)
                aug_lengths = tour_length(batch_exp, aug_tours, metric=metric,
                                           sigma_inv=sigma_inv, weights=weights)
            else:
                # Standard augmentation (Euclidean/Weighted Euclidean)
                aug_batch = augment_instance(batch)  # (bs*8, n, 2)
                enc_input = aug_batch
                if W_whiten is not None:
                    enc_input = whiten_coords(aug_batch, W_whiten.to(device))
                aug_tours, _ = model(enc_input, decode_type="greedy")
                aug_lengths = tour_length(aug_batch, aug_tours, metric=metric,
                                           sigma_inv=sigma_inv, weights=weights)

            aug_lengths = aug_lengths.view(8, bs)
            best = aug_lengths.min(dim=0).values
            all_best.append(best.cpu())
    return torch.cat(all_best), time.time() - t0


def evaluate_pomo(model, test_loader, device, metric, sigma_inv, weights,
                  W_whiten=None, num_starts=20):
    """POMO: greedy decode from N start nodes, keep best per instance."""
    model.eval()
    all_best_lengths = []
    all_best_tours = []
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            bs, n, _ = batch.shape
            enc_input = batch
            if W_whiten is not None:
                enc_input = whiten_coords(batch, W_whiten.to(device))

            best_lengths = torch.full((bs,), float('inf'), device=device)
            best_tours = torch.zeros(bs, n, dtype=torch.long, device=device)

            for start in range(min(num_starts, n)):
                start_nodes = torch.full((bs,), start, dtype=torch.long, device=device)
                tours, _ = model(enc_input, decode_type="greedy", start_node=start_nodes)
                lengths = tour_length(batch, tours, metric=metric,
                                       sigma_inv=sigma_inv, weights=weights)
                improved = lengths < best_lengths
                best_lengths = torch.where(improved, lengths, best_lengths)
                best_tours[improved] = tours[improved]

            all_best_tours.append(best_tours.cpu())
            all_best_lengths.append(best_lengths.cpu())
    return torch.cat(all_best_tours), torch.cat(all_best_lengths), time.time() - t0


def evaluate_rl_2opt(tours_tensor, test_coords, metric, sigma_inv_np, weights_np):
    """Apply 2-opt refinement to existing RL tours."""
    coords_np = test_coords.numpy()
    t0 = time.time()
    refined_lengths = []

    for i in range(len(coords_np)):
        cities = coords_np[i]
        tour = tours_tensor[i].numpy().tolist()
        n = len(tour)
        improved = True
        it = 0
        while improved and it < 500:
            improved = False
            it += 1
            for a in range(1, n - 1):
                for b in range(a + 1, n):
                    old_d = (_pairwise_distance(cities[tour[a-1]], cities[tour[a]],
                                               metric, sigma_inv_np, weights_np) +
                             _pairwise_distance(cities[tour[b]], cities[tour[(b+1) % n]],
                                               metric, sigma_inv_np, weights_np))
                    new_d = (_pairwise_distance(cities[tour[a-1]], cities[tour[b]],
                                               metric, sigma_inv_np, weights_np) +
                             _pairwise_distance(cities[tour[a]], cities[tour[(b+1) % n]],
                                               metric, sigma_inv_np, weights_np))
                    if new_d < old_d - 1e-10:
                        tour[a:b+1] = tour[a:b+1][::-1]
                        improved = True
        total = sum(_pairwise_distance(cities[tour[j]], cities[tour[(j+1) % n]],
                                       metric, sigma_inv_np, weights_np)
                    for j in range(n))
        refined_lengths.append(total)

    return torch.tensor(refined_lengths), time.time() - t0


# ════════════════════════════════════════════════════════════════════
# ATTENTION VISUALIZATION
# ════════════════════════════════════════════════════════════════════

def visualize_attention(model, instance_coords, config, metric, sigma_inv, weights,
                        output_path, W_whiten=None):
    """Visualize decoder attention distribution per step for one instance."""
    model.eval()
    device = config.device
    coords = instance_coords.unsqueeze(0).to(device)
    n = coords.size(1)

    enc_input = coords
    if W_whiten is not None:
        enc_input = whiten_coords(coords, W_whiten.to(device))

    with torch.no_grad():
        out = model(enc_input, decode_type="greedy", return_probs=True)
        if len(out) == 3:
            tours, _, probs = out
        else:
            tours, _, probs = out[0], out[1], out[-1]

    tours_np = tours[0].cpu().numpy()
    probs_np = probs[0].cpu().numpy()
    coords_np = coords[0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im = axes[0].imshow(probs_np, cmap="viridis", aspect="auto")
    axes[0].set_xlabel("City Index")
    axes[0].set_ylabel("Decoding Step")
    axes[0].set_title("Decoder Attention Distribution per Step")
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n - 1))
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    for step in range(n - 1):
        chosen = tours_np[step + 1]
        axes[0].plot(chosen, step, "r*", markersize=12, markeredgecolor="white")

    cmap = plt.cm.plasma
    for i in range(n):
        a, b = tours_np[i], tours_np[(i + 1) % n]
        axes[1].plot([coords_np[a, 0], coords_np[b, 0]],
                     [coords_np[a, 1], coords_np[b, 1]],
                     color=cmap(i / n), linewidth=2, zorder=1)
    axes[1].scatter(coords_np[:, 0], coords_np[:, 1], c="darkred", s=60, zorder=2)
    axes[1].scatter([coords_np[tours_np[0], 0]], [coords_np[tours_np[0], 1]],
                    c="gold", s=180, marker="*", edgecolors="black", zorder=3)
    for idx, (x, y) in enumerate(coords_np):
        axes[1].annotate(str(idx), (x, y), fontsize=8,
                         xytext=(4, 4), textcoords='offset points')
    axes[1].set_title("Tour (color = visit order)")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved attention viz: {output_path}")


# ════════════════════════════════════════════════════════════════════
# PLOTTING UTILS
# ════════════════════════════════════════════════════════════════════

def plot_sample_efficiency(histories, output_path, title):
    plt.figure(figsize=(10, 6))
    for run_name, history in histories.items():
        plt.plot(history["epoch"], history["val_length"],
                 label=run_name.replace("_", " "), linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Tour Length")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_tour(ax, coords_np, tour, title=""):
    n = len(tour)
    x = [coords_np[tour[i]][0] for i in range(n)] + [coords_np[tour[0]][0]]
    y = [coords_np[tour[i]][1] for i in range(n)] + [coords_np[tour[0]][1]]
    ax.plot(x, y, "-", color="steelblue", linewidth=1.5, zorder=1)
    ax.scatter(coords_np[:, 0], coords_np[:, 1], c="darkred", s=40, zorder=2)
    ax.scatter([coords_np[tour[0]][0]], [coords_np[tour[0]][1]],
               c="gold", s=120, marker="*", edgecolors="black", zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def plot_tour_comparisons(test_coords, tour_dict, output_path, num_instances=3):
    coords_np = test_coords.cpu().numpy()
    methods = list(tour_dict.keys())
    fig, axes = plt.subplots(num_instances, len(methods),
                              figsize=(3 * len(methods), 3 * num_instances))
    if num_instances == 1:
        axes = axes[np.newaxis, :]
    for row in range(num_instances):
        ci = coords_np[row]
        for col, method in enumerate(methods):
            tours, lengths = tour_dict[method]
            tour = tours[row] if isinstance(tours, list) else tours[row].cpu().tolist()
            length = lengths[row].item()
            plot_tour(axes[row, col], ci, tour, title=f"{method}\nL={length:.3f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_length_distribution(results_dict, output_path, title):
    plt.figure(figsize=(10, 6))
    for method_name, lengths in results_dict.items():
        plt.hist(lengths.numpy(), bins=40, alpha=0.5, label=method_name,
                 edgecolor="black", linewidth=0.3)
    plt.xlabel("Tour Length")
    plt.ylabel("Number of Test Instances")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


# ════════════════════════════════════════════════════════════════════
# PER-METRIC EVALUATION
# ════════════════════════════════════════════════════════════════════

def evaluate_metric(metric, model_configs, base_config):
    """Run all decoding strategies for all models on this distance metric."""
    print()
    print("=" * 78)
    print(f"  METRIC: {metric}")
    print("=" * 78)

    # Configure
    config = Config()
    for attr in dir(base_config):
        if not attr.startswith('_') and not callable(getattr(base_config, attr)):
            setattr(config, attr, getattr(base_config, attr))
    config.distance_metric = metric
    config.data_distribution = "gaussian" if metric == "mahalanobis" else "uniform"

    # Metric-specific constants
    sigma_inv, weights = None, None
    sigma_inv_np, weights_np = None, None
    W_whiten = None

    if metric == "mahalanobis":
        sigma_t = torch.tensor(config.mahalanobis_sigma, dtype=torch.float32)
        sigma_inv = torch.inverse(sigma_t).to(config.device)
        sigma_inv_np = torch.inverse(sigma_t).numpy()
        if config.use_whitening:
            W_whiten, _ = compute_whitening_matrix(config.mahalanobis_sigma)
    if metric == "weighted_euclidean":
        weights = torch.tensor(config.euclidean_weights, dtype=torch.float32).to(config.device)
        weights_np = np.array(config.euclidean_weights)

    # Test data
    _, test_coords = prepare_datasets(config)
    test_loader = get_eval_loader(test_coords, config)
    print(f"Test set: {test_coords.size(0)} instances, N={test_coords.size(1)}")

    device = config.device
    results = {}
    tours_dict = {}

    # ── RL models ──
    for run_name, use_glimpse in model_configs:
        ckpt_path = os.path.join(config.checkpoint_dir, f"{run_name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  SKIP: {run_name} (no checkpoint)")
            continue

        print(f"\nEvaluating {run_name}...")

        # Determine if this model uses spatial encoding (from name)
        model_config = Config()
        model_config.use_spatial_encoding = "spatial" in run_name
        model_config.distance_metric = metric
        model = load_model(ckpt_path, model_config, use_glimpse)

        # Determine if whitening is used for this model
        model_whiten = W_whiten if "whiten" in run_name else None

        # Greedy
        g_tours, g_lengths, g_time = evaluate_greedy(
            model, test_loader, device, metric, sigma_inv, weights, model_whiten
        )
        key = f"RL {run_name} (greedy)"
        results[key] = {"lengths": g_lengths, "time": g_time}
        tours_dict[key] = (g_tours, g_lengths)
        print(f"  Greedy:        Mean L = {g_lengths.mean():.4f}, Time = {g_time:.2f}s")

        # Premium decoding for flagship models
        is_flagship = use_glimpse and "rollout" in run_name
        if is_flagship:
            # Best-of-128 sampling
            s_tours, s_lengths, s_time = evaluate_sampling(
                model, test_loader, device, metric, sigma_inv, weights, model_whiten, N=128
            )
            key = f"RL {run_name} (best-of-128)"
            results[key] = {"lengths": s_lengths, "time": s_time}
            tours_dict[key] = (s_tours, s_lengths)
            print(f"  Best-of-128:   Mean L = {s_lengths.mean():.4f}, Time = {s_time:.2f}s")

            # Beam search width 10
            b10_tours, b10_lengths, b10_time = evaluate_beam_search(
                model, test_loader, device, metric, sigma_inv, weights, model_whiten, 10
            )
            key = f"RL {run_name} (beam-10)"
            results[key] = {"lengths": b10_lengths, "time": b10_time}
            tours_dict[key] = (b10_tours, b10_lengths)
            print(f"  Beam (k=10):   Mean L = {b10_lengths.mean():.4f}, Time = {b10_time:.2f}s")

            # Beam search width 25
            b25_tours, b25_lengths, b25_time = evaluate_beam_search(
                model, test_loader, device, metric, sigma_inv, weights, model_whiten, 25
            )
            key = f"RL {run_name} (beam-25)"
            results[key] = {"lengths": b25_lengths, "time": b25_time}
            tours_dict[key] = (b25_tours, b25_lengths)
            print(f"  Beam (k=25):   Mean L = {b25_lengths.mean():.4f}, Time = {b25_time:.2f}s")

            # 8x augmentation — now works for Mahalanobis too (via whitening)
            if metric in ["euclidean", "weighted_euclidean"] or (metric == "mahalanobis" and model_whiten is not None):
                a_lengths, a_time = evaluate_augmented(
                    model, test_coords, config, metric, sigma_inv, weights, model_whiten
                )
                key = f"RL {run_name} (8x aug)"
                results[key] = {"lengths": a_lengths, "time": a_time}
                print(f"  8x augmentation: Mean L = {a_lengths.mean():.4f}, Time = {a_time:.2f}s")

            # POMO multi-start (all metrics)
            p_tours, p_lengths, p_time = evaluate_pomo(
                model, test_loader, device, metric, sigma_inv, weights,
                model_whiten, num_starts=config.num_cities
            )
            key = f"RL {run_name} (POMO-{config.num_cities})"
            results[key] = {"lengths": p_lengths, "time": p_time}
            tours_dict[key] = (p_tours, p_lengths)
            print(f"  POMO-{config.num_cities}:      Mean L = {p_lengths.mean():.4f}, Time = {p_time:.2f}s")

            # RL+2opt (on POMO tours for best starting point)
            r2_lengths, r2_time = evaluate_rl_2opt(
                p_tours, test_coords, metric, sigma_inv_np, weights_np
            )
            key = f"RL {run_name} (POMO+2opt)"
            results[key] = {"lengths": r2_lengths, "time": p_time + r2_time}
            print(f"  POMO+2opt:     Mean L = {r2_lengths.mean():.4f}, Time = {p_time + r2_time:.2f}s")

            # Attention visualization
            viz_path = os.path.join(config.results_dir, f"attention_{run_name}.png")
            visualize_attention(model, test_coords[0], config, metric,
                                sigma_inv, weights, viz_path, model_whiten)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Classical baselines ──
    print(f"\nClassical baselines for {metric}...")

    print("  Nearest Neighbor...")
    t0 = time.time()
    nn_tours, nn_lengths = nearest_neighbor_batch(
        test_coords, metric=metric, sigma_inv=sigma_inv_np, weights=weights_np
    )
    nn_time = time.time() - t0
    results["NN"] = {"lengths": nn_lengths, "time": nn_time}
    tours_dict["NN"] = (nn_tours, nn_lengths)
    print(f"    Mean L = {nn_lengths.mean():.4f}, Time = {nn_time:.2f}s")

    print("  2-opt...")
    t0 = time.time()
    to_tours, to_lengths = two_opt_batch(
        test_coords, metric=metric, sigma_inv=sigma_inv_np, weights=weights_np
    )
    to_time = time.time() - t0
    results["2-opt"] = {"lengths": to_lengths, "time": to_time}
    tours_dict["2-opt"] = (to_tours, to_lengths)
    print(f"    Mean L = {to_lengths.mean():.4f}, Time = {to_time:.2f}s")

    print("  OR-Tools...")
    t0 = time.time()
    or_tours, or_lengths = ortools_batch(
        test_coords, time_limit_seconds=2, metric=metric,
        sigma_inv=sigma_inv_np, weights=weights_np, verbose=True
    )
    or_time = time.time() - t0
    results["OR-Tools"] = {"lengths": or_lengths, "time": or_time}
    tours_dict["OR-Tools"] = (or_tours, or_lengths)
    print(f"    Mean L = {or_lengths.mean():.4f}, Time = {or_time:.2f}s")

    # ── Build summary ──
    optimal = or_lengths
    nn_mean = nn_lengths.mean().item()
    opt_mean = optimal.mean().item()
    denom = nn_mean - opt_mean

    summary = {}
    for method, data in results.items():
        L = data["lengths"]
        mean_L = L.mean().item()
        gap = ((L - optimal) / optimal * 100).mean().item()
        ratio = (L / optimal).mean().item()
        impr = ((nn_mean - mean_L) / denom * 100) if denom > 0 else 0.0
        summary[method] = {
            "mean_length": mean_L,
            "gap_percent": gap,
            "approx_ratio": ratio,
            "improvement_over_nn_percent": impr,
            "time_sec": data["time"],
        }

    return {
        "metric": metric,
        "results": results,
        "tours": tours_dict,
        "summary": summary,
        "test_coords": test_coords,
    }


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    config = Config()
    print("=" * 78)
    print("FINAL EVALUATION PIPELINE")
    print(f"Device: {config.device}")
    print("=" * 78)

    # Models per metric
    euclidean_models = [
        ("simple_rollout_euclidean", False),
        ("simple_critic_euclidean", False),
        ("glimpse_rollout_euclidean", True),
        ("glimpse_critic_euclidean", True),
        ("glimpse_rollout_euclidean_ent", True),
    ]
    # Fallback names (for old runs without _euclidean suffix)
    fallback_names = {
        "simple_rollout_euclidean": "simple_rollout",
        "simple_critic_euclidean": "simple_critic",
        "glimpse_critic_euclidean": "glimpse_critic",
    }
    resolved = []
    for name, glimpse in euclidean_models:
        ckpt = os.path.join(config.checkpoint_dir, f"{name}_best.pt")
        if not os.path.exists(ckpt):
            alt = fallback_names.get(name, name)
            alt_ckpt = os.path.join(config.checkpoint_dir, f"{alt}_best.pt")
            if os.path.exists(alt_ckpt):
                resolved.append((alt, glimpse))
                continue
        resolved.append((name, glimpse))
    euclidean_models = resolved

    manhattan_models = [("glimpse_rollout_manhattan", True)]

    # Mahalanobis: both original and improved (whitening) models
    mahalanobis_models = [
        ("glimpse_rollout_mahalanobis", True),          # original
        ("glimpse_rollout_mahalanobis_whiten", True),   # whitening only
    ]
    # Add POMO+spatial if available
    pomo_ckpt = os.path.join(config.checkpoint_dir,
                             "glimpse_rollout_mahalanobis_whiten_pomo_spatial_best.pt")
    if os.path.exists(pomo_ckpt):
        mahalanobis_models.append(("glimpse_rollout_mahalanobis_whiten_pomo_spatial", True))

    weighted_models = [("glimpse_rollout_weighted_euclidean", True)]

    all_results = {}
    all_results["euclidean"] = evaluate_metric("euclidean", euclidean_models, config)
    all_results["manhattan"] = evaluate_metric("manhattan", manhattan_models, config)
    all_results["mahalanobis"] = evaluate_metric("mahalanobis", mahalanobis_models, config)
    all_results["weighted_euclidean"] = evaluate_metric("weighted_euclidean", weighted_models, config)

    # ── Print final tables ──
    print()
    print("#" * 90)
    print("# FINAL RESULTS (TSP-20, 1000 test instances per metric)")
    print("#" * 90)

    for metric, md in all_results.items():
        print(f"\n=== {metric.upper()} ===")
        print(f"{'Method':<55} {'Mean L':>10} {'Gap %':>8} {'Ratio':>7} {'ImpNN%':>8} {'Time':>10}")
        print("-" * 102)
        baseline_order = ["NN", "2-opt", "OR-Tools"]
        all_methods = list(md["summary"].keys())
        rl = [m for m in all_methods if m not in baseline_order]
        ordered = [m for m in baseline_order if m in all_methods] + rl
        for method in ordered:
            s = md["summary"][method]
            print(f"{method:<55} {s['mean_length']:>10.4f} "
                  f"{s['gap_percent']:>7.2f}% {s['approx_ratio']:>7.4f} "
                  f"{s['improvement_over_nn_percent']:>7.1f}% "
                  f"{s['time_sec']:>9.1f}s")

    # ── Save raw results ──
    os.makedirs(config.results_dir, exist_ok=True)
    serializable = {m: {"summary": d["summary"]} for m, d in all_results.items()}
    results_path = os.path.join(config.results_dir, "evaluation_full.json")
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved full results: {results_path}")

    # ── Plots ──
    print("\nGenerating plots...")

    # Load all training histories
    all_histories = {}
    history_files = {
        "Simple+Rollout (Euc)": "simple_rollout_euclidean_history.json",
        "Simple+Critic (Euc)": "simple_critic_euclidean_history.json",
        "Glimpse+Rollout (Euc)": "glimpse_rollout_euclidean_history.json",
        "Glimpse+Critic (Euc)": "glimpse_critic_euclidean_history.json",
        "Glimpse+Rollout+Ent (Euc)": "glimpse_rollout_euclidean_ent_history.json",
        "Glimpse+Rollout (Manhattan)": "glimpse_rollout_manhattan_history.json",
        "Glimpse+Rollout (Mahalanobis)": "glimpse_rollout_mahalanobis_history.json",
        "Glimpse+Rollout (Mahal+Whiten)": "glimpse_rollout_mahalanobis_whiten_history.json",
        "Glimpse+Rollout (Weighted Euc)": "glimpse_rollout_weighted_euclidean_history.json",
    }
    fallback_history = {
        "simple_rollout_euclidean_history.json": "simple_rollout_history.json",
        "simple_critic_euclidean_history.json": "simple_critic_history.json",
        "glimpse_critic_euclidean_history.json": "glimpse_critic_history.json",
    }
    for label, fname in history_files.items():
        path = os.path.join(config.results_dir, fname)
        if not os.path.exists(path):
            alt = fallback_history.get(fname)
            if alt:
                path = os.path.join(config.results_dir, alt)
        if os.path.exists(path):
            with open(path) as f:
                all_histories[label] = json.load(f)

    # Architecture ablation (Euclidean only)
    arch_keys = ["Simple+Rollout (Euc)", "Simple+Critic (Euc)",
                 "Glimpse+Rollout (Euc)", "Glimpse+Critic (Euc)"]
    arch_hist = {k: all_histories[k] for k in arch_keys if k in all_histories}
    if arch_hist:
        plot_sample_efficiency(
            arch_hist,
            os.path.join(config.results_dir, "sample_efficiency_architecture.png"),
            "Sample Efficiency: Architecture Ablation (Euclidean TSP-20)"
        )

    # Entropy comparison
    ent_keys = ["Glimpse+Rollout (Euc)", "Glimpse+Rollout+Ent (Euc)"]
    ent_hist = {k: all_histories[k] for k in ent_keys if k in all_histories}
    if len(ent_hist) == 2:
        plot_sample_efficiency(
            ent_hist,
            os.path.join(config.results_dir, "sample_efficiency_entropy.png"),
            "Sample Efficiency: Entropy Regularization Comparison"
        )

    # Cross-metric comparison (including improved Mahalanobis)
    metric_keys = ["Glimpse+Rollout (Euc)", "Glimpse+Rollout (Manhattan)",
                   "Glimpse+Rollout (Mahalanobis)", "Glimpse+Rollout (Mahal+Whiten)",
                   "Glimpse+Rollout (Weighted Euc)"]
    metric_hist = {k: all_histories[k] for k in metric_keys if k in all_histories}
    if len(metric_hist) >= 2:
        plot_sample_efficiency(
            metric_hist,
            os.path.join(config.results_dir, "sample_efficiency_metrics.png"),
            "Sample Efficiency Across Distance Metrics"
        )

    # Tour comparison plots per metric
    for metric, md in all_results.items():
        td = md["tours"]
        keys = list(td.keys())
        plot_keys = [k for k in keys if k in ["NN", "2-opt", "OR-Tools"]]
        plot_keys += [k for k in keys if "greedy" in k][:2]
        plot_keys += [k for k in keys if "POMO" in k and "2opt" not in k][:1]
        subset = {k: td[k] for k in plot_keys if k in td}
        if subset:
            plot_tour_comparisons(
                md["test_coords"], subset,
                os.path.join(config.results_dir, f"tours_{metric}.png"),
                num_instances=3,
            )

    # Length distribution (Euclidean)
    if "euclidean" in all_results:
        dd = {m: d["lengths"] for m, d in all_results["euclidean"]["results"].items()
              if "beam" not in m}
        plot_length_distribution(
            dd,
            os.path.join(config.results_dir, "length_distribution_euclidean.png"),
            "Tour Length Distribution Across Methods (Euclidean TSP-20)"
        )

    print()
    print("=" * 78)
    print("EVALUATION COMPLETE")
    print(f"Results: {results_path}")
    print(f"Plots:   {config.results_dir}/*.png")
    print("=" * 78)


if __name__ == "__main__":
    main()
