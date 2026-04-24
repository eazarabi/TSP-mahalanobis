"""
Experimental: Training script for Recurrent-Depth TSP models.

Reuses the main training loop (train.train_improved) but swaps the decoder
for RecurrentGlimpseDecoder. Designed for minimal invasion of the main codebase.

Usage:
    python experimental/train_rdt.py --metric euclidean --thinking-steps 4 --epochs 50
    python experimental/train_rdt.py --metric mahalanobis --thinking-steps 4 --epochs 50
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ttest_rel

from config import Config
from data import get_train_loader, get_eval_loader, prepare_datasets, tour_length, compute_whitening_matrix, whiten_coords
from model import TSPTransformer
from experimental.model_rdt import RecurrentGlimpseDecoder


def build_rdt_model(config, num_thinking_steps=4, lora_rank=8, use_halting=False):
    """Construct a TSPTransformer then replace its decoder with the RDT variant."""
    model = TSPTransformer(config, use_glimpse=True)
    model.decoder = RecurrentGlimpseDecoder(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        tanh_clipping=config.tanh_clipping,
        num_thinking_steps=num_thinking_steps,
        lora_rank=lora_rank,
        use_halting=use_halting,
    )
    return model


def validate(model, val_loader, device, metric, sigma_inv, weights, W_whiten):
    model.eval()
    all_lengths = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            enc_input = batch if W_whiten is None else whiten_coords(batch, W_whiten.to(device))
            tours, _ = model(enc_input, decode_type="greedy")
            lengths = tour_length(batch, tours, metric=metric, sigma_inv=sigma_inv, weights=weights)
            all_lengths.append(lengths.cpu())
    all_lengths = torch.cat(all_lengths)
    return all_lengths.mean().item(), all_lengths


def should_update_baseline(cur, base, alpha=0.05):
    if cur.mean().item() >= base.mean().item():
        return False
    t_stat, p = ttest_rel(cur.numpy(), base.numpy())
    one_sided = p / 2 if t_stat < 0 else 1.0
    return one_sided < alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="euclidean",
                        choices=["euclidean", "manhattan", "mahalanobis", "weighted_euclidean"])
    parser.add_argument("--thinking-steps", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--use-halting", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--init-from", type=str, default=None,
                        help="Optional path to pre-trained checkpoint to warm-start from")
    parser.add_argument("--entropy-coef", type=float, default=0.0,
                        help="Entropy regularization coefficient (0 = disabled)")
    parser.add_argument("--no-whitening", action="store_true",
                        help="Disable whitening for Mahalanobis (use raw coords)")
    parser.add_argument("--no-spatial", action="store_true",
                        help="Disable spatial encoding bias in encoder")
    parser.add_argument("--run-name-suffix", type=str, default="",
                        help="Optional suffix to distinguish runs (e.g., '_ent', '_nospatial')")
    args = parser.parse_args()

    cfg = Config()
    cfg.num_epochs = args.epochs
    cfg.train_size = args.train_size
    cfg.distance_metric = args.metric
    cfg.entropy_coef = args.entropy_coef
    if args.no_spatial:
        cfg.use_spatial_encoding = False
    if args.no_whitening:
        cfg.use_whitening = False
    if args.metric == "mahalanobis":
        cfg.data_distribution = "gaussian"

    # Redirect all RDT artifacts to experimental/ subdirectories
    exp_root = os.path.dirname(os.path.abspath(__file__))
    cfg.checkpoint_dir = os.path.join(exp_root, "checkpoints")
    cfg.results_dir = os.path.join(exp_root, "results")

    device = cfg.device
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Metric-specific setup
    sigma_inv, weights, W_whiten = None, None, None
    if args.metric == "mahalanobis":
        sigma_t = torch.tensor(cfg.mahalanobis_sigma, dtype=torch.float32)
        sigma_inv = torch.inverse(sigma_t).to(device)
        if cfg.use_whitening:
            W_whiten, _ = compute_whitening_matrix(cfg.mahalanobis_sigma)
    elif args.metric == "weighted_euclidean":
        weights = torch.tensor(cfg.euclidean_weights, dtype=torch.float32).to(device)

    # Data
    val_coords, _ = prepare_datasets(cfg)
    val_loader = get_eval_loader(val_coords, cfg)

    # Build model
    model = build_rdt_model(cfg, args.thinking_steps, args.lora_rank, args.use_halting).to(device)

    # Optionally warm-start from a pre-trained (non-RDT) checkpoint
    if args.init_from and os.path.exists(args.init_from):
        print(f"Warm-starting shared weights from {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        state = ckpt["model_state_dict"]
        # Load only matching keys (encoder + non-LoRA decoder params)
        own_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k].copy_(v)
                loaded += 1
        print(f"Loaded {loaded} matching params")

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    baseline_model = copy.deepcopy(model).to(device)
    baseline_model.eval()

    run_name = f"rdt_T{args.thinking_steps}_r{args.lora_rank}_{args.metric}"
    if args.no_whitening and args.metric == "mahalanobis":
        run_name += "_nowhiten"
    if args.no_spatial:
        run_name += "_nospatial"
    if args.entropy_coef > 0:
        run_name += "_ent"
    if args.use_halting:
        run_name += "_halt"
    if args.run_name_suffix:
        run_name += args.run_name_suffix

    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 70)
    print(f"RDT TRAINING: {run_name}")
    print(f"  Thinking steps: {args.thinking_steps} | LoRA rank: {args.lora_rank}")
    print(f"  Metric: {args.metric} | Device: {device}")
    print(f"  Model params: {total_params:,}")
    print("=" * 70)

    history = {"epoch": [], "train_loss": [], "train_length": [], "val_length": []}
    best_val = float("inf")

    for epoch in range(cfg.num_epochs):
        t0 = time.time()
        model.train()
        train_loader = get_train_loader(cfg)
        ep_loss, ep_len, nb = 0.0, 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            enc_input = batch if W_whiten is None else whiten_coords(batch, W_whiten.to(device))

            if args.entropy_coef > 0:
                tours, log_probs, entropy = model(enc_input, decode_type="sample", return_entropy=True)
            else:
                tours, log_probs = model(enc_input, decode_type="sample")
                entropy = None
            lengths = tour_length(batch, tours, metric=args.metric, sigma_inv=sigma_inv, weights=weights)

            with torch.no_grad():
                bl_tours, _ = baseline_model(enc_input, decode_type="greedy")
                baseline_len = tour_length(batch, bl_tours, metric=args.metric, sigma_inv=sigma_inv, weights=weights)

            advantage = (lengths - baseline_len).detach()
            loss = (advantage * log_probs).mean()
            if entropy is not None:
                loss = loss - args.entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            ep_loss += loss.item(); ep_len += lengths.mean().item(); nb += 1

        val_len, val_tensor = validate(model, val_loader, device, args.metric, sigma_inv, weights, W_whiten)

        _, bl_tensor = validate(baseline_model, val_loader, device, args.metric, sigma_inv, weights, W_whiten)
        if should_update_baseline(val_tensor, bl_tensor):
            baseline_model = copy.deepcopy(model).to(device)
            baseline_model.eval()
            marker = " [baseline updated]"
        else:
            marker = ""

        if val_len < best_val:
            best_val = val_len
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_length": val_len,
                "args": vars(args),
            }, os.path.join(cfg.checkpoint_dir, f"{run_name}_best.pt"))
            marker += " (best)"

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(ep_loss / nb)
        history["train_length"].append(ep_len / nb)
        history["val_length"].append(val_len)

        print(f"Epoch {epoch+1:3d}/{cfg.num_epochs} | train_L={ep_len/nb:.4f} | "
              f"val_L={val_len:.4f} | loss={ep_loss/nb:.4f} | time={time.time()-t0:.1f}s{marker}")

    with open(os.path.join(cfg.results_dir, f"{run_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 70)
    print(f"Training complete. Best val: {best_val:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()