"""
Improved REINFORCE training for Mahalanobis TSP with:
  1. Whitening Transform — feed Σ^(-1/2)·coords to encoder
  2. POMO Multi-Start — N start nodes with shared baseline
  3. Spatial Encoding Bias — distance-indexed attention bias
  4. 8x Augmentation in whitened space (at evaluation time)
  5. 2-opt post-processing (at evaluation time)

Key insight: Mahalanobis distance = Euclidean distance in whitened space.
By whitening, the model's learned Euclidean intuition transfers directly.

References:
  - Kwon et al. "POMO" (NeurIPS 2020) — multi-start + shared baseline
  - Zhao & Wong (PLOS One, 2025) — spatial encoding bias
  - Oursland "Neural Networks Learn Distance Metrics" (2025) — whitening proof
"""

import os
import copy
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ttest_rel

from config import Config
from data import (get_train_loader, get_eval_loader, prepare_datasets,
                  tour_length, compute_whitening_matrix, whiten_coords)
from model import TSPTransformer, CriticNetwork


def validate(model, val_loader, device, metric="euclidean", sigma_inv=None,
             weights=None, W_whiten=None):
    """Validate with optional whitening."""
    model.eval()
    all_lengths = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            # Whiten for encoder if enabled
            encoder_input = batch
            if W_whiten is not None:
                encoder_input = whiten_coords(batch, W_whiten.to(device))
            tours, _ = model(encoder_input, decode_type="greedy")
            # Compute tour length in ORIGINAL space with Mahalanobis metric
            lengths = tour_length(batch, tours, metric=metric,
                                   sigma_inv=sigma_inv, weights=weights)
            all_lengths.append(lengths.cpu())
    all_lengths = torch.cat(all_lengths)
    return all_lengths.mean().item(), all_lengths


def validate_pomo(model, val_loader, device, metric, sigma_inv, weights,
                  W_whiten=None, num_starts=None):
    """POMO validation: try all N start nodes, keep best per instance."""
    model.eval()
    all_lengths = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            bs, n, _ = batch.shape
            if num_starts is None:
                num_starts = n

            encoder_input = batch
            if W_whiten is not None:
                encoder_input = whiten_coords(batch, W_whiten.to(device))

            best_lengths = torch.full((bs,), float('inf'), device=device)

            for start in range(num_starts):
                start_nodes = torch.full((bs,), start, dtype=torch.long, device=device)
                tours, _ = model(encoder_input, decode_type="greedy",
                                 start_node=start_nodes)
                lengths = tour_length(batch, tours, metric=metric,
                                       sigma_inv=sigma_inv, weights=weights)
                best_lengths = torch.minimum(best_lengths, lengths)

            all_lengths.append(best_lengths.cpu())
    all_lengths = torch.cat(all_lengths)
    return all_lengths.mean().item(), all_lengths


def save_checkpoint(model, optimizer, epoch, val_length, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_length": val_length,
    }, path)


def should_update_baseline(current_lengths, baseline_lengths, alpha=0.05):
    current_mean = current_lengths.mean().item()
    baseline_mean = baseline_lengths.mean().item()
    if current_mean >= baseline_mean:
        return False
    t_stat, p_value = ttest_rel(current_lengths.numpy(), baseline_lengths.numpy())
    one_sided_p = p_value / 2 if t_stat < 0 else 1.0
    return one_sided_p < alpha


def train_improved(config, use_glimpse=True, baseline_type="rollout",
                   run_name=None, use_pomo=True):
    """
    Improved training loop with whitening + POMO multi-start + spatial encoding.
    """
    if run_name is None:
        decoder_name = "glimpse" if use_glimpse else "simple"
        metric_tag = config.distance_metric
        pomo_tag = "_pomo" if use_pomo else ""
        spatial_tag = "_spatial" if config.use_spatial_encoding else ""
        whiten_tag = "_whiten" if config.use_whitening and metric_tag == "mahalanobis" else ""
        run_name = f"{decoder_name}_{baseline_type}_{metric_tag}{whiten_tag}{pomo_tag}{spatial_tag}"

    device = config.device
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    # ── Distance metric setup ──
    metric = config.distance_metric
    sigma_inv = None
    weights = None
    W_whiten = None

    if metric == "mahalanobis":
        sigma_t = torch.tensor(config.mahalanobis_sigma, dtype=torch.float32)
        sigma_inv = torch.inverse(sigma_t).to(device)
        if config.use_whitening:
            W_whiten, _ = compute_whitening_matrix(config.mahalanobis_sigma)
            print(f"Whitening matrix Σ^(-1/2):\n{W_whiten}")
    elif metric == "weighted_euclidean":
        weights = torch.tensor(config.euclidean_weights, dtype=torch.float32).to(device)

    # ── Data ──
    val_coords, _ = prepare_datasets(config)
    val_loader = get_eval_loader(val_coords, config)

    # ── Model (with spatial encoding) ──
    model = TSPTransformer(config, use_glimpse=use_glimpse).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # ── Baseline setup ──
    if baseline_type == "rollout":
        baseline_model = copy.deepcopy(model).to(device)
        baseline_model.eval()
        critic = None
        critic_optimizer = None
    elif baseline_type == "critic":
        critic = CriticNetwork(config).to(device)
        critic_optimizer = optim.Adam(critic.parameters(), lr=config.critic_lr)
        baseline_model = None
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")

    # POMO settings — reduce batch size to fit in GPU memory
    num_starts = config.pomo_start_nodes if use_pomo else 1
    if use_pomo and num_starts > 1:
        # Effective batch = batch_size * num_starts_per_chunk
        # For 8GB GPU: batch=512, process 4 starts at a time
        config.batch_size = min(config.batch_size, 256)

    history = {
        "epoch": [], "train_loss": [], "train_length": [],
        "val_length": [], "baseline_updates": [], "epoch_time": [],
    }
    best_val_length = float("inf")

    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 70)
    print(f"RUN (IMPROVED): {run_name}")
    print(f"  Decoder: {'glimpse' if use_glimpse else 'simple'}")
    print(f"  Baseline: {baseline_type}")
    print(f"  Distance: {metric}")
    print(f"  Whitening: {W_whiten is not None}")
    print(f"  POMO starts: {num_starts}")
    print(f"  Spatial encoding: {config.use_spatial_encoding}")
    print(f"  Data: {config.data_distribution}")
    print(f"  Device: {device}")
    print(f"  Model params: {total_params:,}")
    print(f"  Epochs: {config.num_epochs}")
    print("=" * 70)

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        model.train()
        train_loader = get_train_loader(config)

        epoch_loss = 0.0
        epoch_length = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            bs, n, _ = batch.shape

            # Whiten coordinates for encoder
            encoder_input = batch
            if W_whiten is not None:
                encoder_input = whiten_coords(batch, W_whiten.to(device))

            if use_pomo and num_starts > 1:
                # POMO: sample from multiple start nodes
                # Memory-efficient: process starts one at a time for safety
                chunk_size = 1  # one start per forward pass to avoid OOM
                all_lengths = []
                all_log_probs = []

                for s_start in range(0, num_starts, chunk_size):
                    s_end = min(s_start + chunk_size, num_starts)
                    s_count = s_end - s_start

                    encoder_exp = encoder_input.repeat(s_count, 1, 1)
                    batch_exp = batch.repeat(s_count, 1, 1)
                    start_nodes = torch.arange(s_start, s_end, device=device).repeat_interleave(bs)

                    tours_chunk, lp_chunk = model(encoder_exp, decode_type="sample",
                                                  start_node=start_nodes)
                    lengths_chunk = tour_length(batch_exp, tours_chunk, metric=metric,
                                               sigma_inv=sigma_inv, weights=weights)
                    all_lengths.append(lengths_chunk.view(s_count, bs))
                    all_log_probs.append(lp_chunk.view(s_count, bs))

                # Reshape: (num_starts, bs)
                lengths_reshaped = torch.cat(all_lengths, dim=0)
                log_probs_reshaped = torch.cat(all_log_probs, dim=0)

                # POMO shared baseline: mean length across all starts for same instance
                baseline_val = lengths_reshaped.mean(dim=0, keepdim=True).expand_as(lengths_reshaped)

                # REINFORCE with shared baseline
                advantage = (lengths_reshaped - baseline_val).detach()
                reinforce_loss = (advantage * log_probs_reshaped).mean()

                # Track average length (best across starts per instance)
                best_lengths = lengths_reshaped.min(dim=0).values
                batch_mean_length = best_lengths.mean().item()

            else:
                # Standard single-start training
                if config.entropy_coef > 0:
                    tours, log_probs_single, entropy = model(
                        encoder_input, decode_type="sample", return_entropy=True
                    )
                else:
                    tours, log_probs_single = model(encoder_input, decode_type="sample")
                    entropy = None

                lengths = tour_length(batch, tours, metric=metric,
                                       sigma_inv=sigma_inv, weights=weights)

                if baseline_type == "rollout":
                    with torch.no_grad():
                        bl_input = encoder_input
                        bl_tours, _ = baseline_model(bl_input, decode_type="greedy")
                        baseline_val = tour_length(batch, bl_tours, metric=metric,
                                                    sigma_inv=sigma_inv, weights=weights)
                else:
                    # Critic takes original coords (or whitened — be consistent)
                    baseline_val = critic(encoder_input)

                advantage = (lengths - baseline_val).detach()
                reinforce_loss = (advantage * log_probs_single).mean()

                if config.entropy_coef > 0 and entropy is not None:
                    reinforce_loss = reinforce_loss - config.entropy_coef * entropy.mean()

                batch_mean_length = lengths.mean().item()

            optimizer.zero_grad()
            reinforce_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            if baseline_type == "critic" and not (use_pomo and num_starts > 1):
                critic_pred = critic(encoder_input)
                critic_loss = ((critic_pred - lengths.detach()) ** 2).mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
                critic_optimizer.step()

            epoch_loss += reinforce_loss.item()
            epoch_length += batch_mean_length
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_train_length = epoch_length / num_batches

        # Validation (standard greedy from node 0)
        val_length, val_lengths_tensor = validate(
            model, val_loader, device, metric, sigma_inv, weights, W_whiten
        )

        baseline_updated = False
        if baseline_type == "rollout" and not (use_pomo and num_starts > 1):
            _, baseline_val_tensor = validate(
                baseline_model, val_loader, device, metric, sigma_inv, weights, W_whiten
            )
            if should_update_baseline(val_lengths_tensor, baseline_val_tensor,
                                       alpha=config.rollout_update_threshold):
                baseline_model = copy.deepcopy(model).to(device)
                baseline_model.eval()
                baseline_updated = True

        if val_length < best_val_length:
            best_val_length = val_length
            best_path = os.path.join(config.checkpoint_dir, f"{run_name}_best.pt")
            save_checkpoint(model, optimizer, epoch, val_length, best_path)

        epoch_time = time.time() - epoch_start
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["train_length"].append(avg_train_length)
        history["val_length"].append(val_length)
        history["baseline_updates"].append(baseline_updated)
        history["epoch_time"].append(epoch_time)

        marker = ""
        if val_length == best_val_length:
            marker = " (best)"
        if baseline_updated:
            marker += " [baseline updated]"

        print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
              f"train_L={avg_train_length:.4f} | "
              f"val_L={val_length:.4f} | "
              f"loss={avg_loss:.4f} | "
              f"time={epoch_time:.1f}s{marker}")

    final_path = os.path.join(config.checkpoint_dir, f"{run_name}_final.pt")
    save_checkpoint(model, optimizer, config.num_epochs, val_length, final_path)

    history_path = os.path.join(config.results_dir, f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 70)
    print(f"Training complete. Best val length: {best_val_length:.4f}")
    print(f"Model saved to: {final_path}")
    print(f"History saved to: {history_path}")
    print("=" * 70)

    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--glimpse", action="store_true", default=False,
                        help="Use glimpse decoder (default: simple decoder)")
    parser.add_argument("--baseline", type=str, default="rollout",
                        choices=["rollout", "critic"])
    parser.add_argument("--metric", type=str, default="euclidean",
                        choices=["euclidean", "manhattan", "mahalanobis", "weighted_euclidean"])
    parser.add_argument("--distribution", type=str, default=None,
                        choices=["uniform", "gaussian"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--no-whiten", action="store_true",
                        help="Disable whitening transform")
    parser.add_argument("--no-pomo", action="store_true",
                        help="Disable POMO multi-start")
    parser.add_argument("--no-spatial", action="store_true",
                        help="Disable spatial encoding bias")
    parser.add_argument("--pomo-starts", type=int, default=None,
                        help="Number of POMO start nodes (default: num_cities)")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.train_size:
        cfg.train_size = args.train_size
    if args.metric:
        cfg.distance_metric = args.metric
    if args.distribution:
        cfg.data_distribution = args.distribution
    if args.no_whiten:
        cfg.use_whitening = False
    if args.no_spatial:
        cfg.use_spatial_encoding = False
    if args.pomo_starts:
        cfg.pomo_start_nodes = args.pomo_starts

    # Auto-pair Mahalanobis with Gaussian data
    if cfg.distance_metric == "mahalanobis" and args.distribution is None:
        cfg.data_distribution = "gaussian"
        print("Note: auto-enabled gaussian data for mahalanobis metric")

    train_improved(cfg, use_glimpse=args.glimpse, baseline_type=args.baseline,
                   use_pomo=not args.no_pomo)
