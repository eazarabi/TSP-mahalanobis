"""
Unified trainer for all three RDT variants: decoder-only, encoder-only, combined.
Saves artifacts to the appropriate subdirectory under experimental/.

Usage:
    python experimental/train_rdt_any.py --variant encoder --metric euclidean --T 4 --epochs 15
    python experimental/train_rdt_any.py --variant combined --metric mahalanobis --T-enc 4 --T-dec 4 --epochs 15
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, copy, json, time
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ttest_rel

from config import Config
from data import get_train_loader, get_eval_loader, prepare_datasets, tour_length, compute_whitening_matrix, whiten_coords
from experimental.model_rdt import RecurrentGlimpseDecoder
from experimental.encoder_rdt.model import build_encoder_rdt_model
from experimental.combined_rdt.model import build_combined_rdt_model
from experimental.train_rdt import build_rdt_model as build_decoder_rdt_model


def _setup_dirs(variant):
    exp_root = os.path.dirname(os.path.abspath(__file__))
    if variant == "decoder":
        ckpt = os.path.join(exp_root, "checkpoints")
        res = os.path.join(exp_root, "results")
    else:
        ckpt = os.path.join(exp_root, f"{variant}_rdt", "checkpoints")
        res = os.path.join(exp_root, f"{variant}_rdt", "results")
    os.makedirs(ckpt, exist_ok=True); os.makedirs(res, exist_ok=True)
    return ckpt, res


def _build(variant, cfg, T_enc, T_dec, lora_rank):
    if variant == "decoder":
        return build_decoder_rdt_model(cfg, num_thinking_steps=T_dec, lora_rank=lora_rank)
    if variant == "encoder":
        return build_encoder_rdt_model(cfg, num_encoder_loops=T_enc, lora_rank=lora_rank)
    if variant == "combined":
        return build_combined_rdt_model(cfg, num_encoder_loops=T_enc,
                                        num_decoder_loops=T_dec, lora_rank=lora_rank)
    raise ValueError(f"Unknown variant: {variant}")


def _validate(model, loader, device, metric, sigma_inv, weights, W):
    model.eval()
    all_l = []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            enc = b if W is None else whiten_coords(b, W.to(device))
            tours, _ = model(enc, decode_type="greedy")
            l = tour_length(b, tours, metric=metric, sigma_inv=sigma_inv, weights=weights)
            all_l.append(l.cpu())
    all_l = torch.cat(all_l)
    return all_l.mean().item(), all_l


def _should_update(cur, base, alpha=0.05):
    if cur.mean().item() >= base.mean().item(): return False
    t_stat, p = ttest_rel(cur.numpy(), base.numpy())
    return (p / 2 if t_stat < 0 else 1.0) < alpha


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--variant", type=str, required=True, choices=["decoder", "encoder", "combined"])
    P.add_argument("--metric", type=str, default="euclidean",
                   choices=["euclidean", "manhattan", "mahalanobis", "weighted_euclidean"])
    P.add_argument("--T", type=int, default=4, help="Thinking steps (applies to decoder-only and encoder-only)")
    P.add_argument("--T-enc", type=int, default=None, help="Encoder loops (combined mode)")
    P.add_argument("--T-dec", type=int, default=None, help="Decoder thinking steps (combined mode)")
    P.add_argument("--lora-rank", type=int, default=8)
    P.add_argument("--epochs", type=int, default=15)
    P.add_argument("--train-size", type=int, default=50000)
    P.add_argument("--init-from", type=str, default=None)
    P.add_argument("--entropy-coef", type=float, default=0.0)
    P.add_argument("--no-whitening", action="store_true")
    P.add_argument("--no-spatial", action="store_true")
    args = P.parse_args()

    # Resolve T_enc / T_dec
    if args.variant == "decoder":
        T_enc, T_dec = 1, args.T
    elif args.variant == "encoder":
        T_enc, T_dec = args.T, 1
    else:  # combined
        T_enc = args.T_enc if args.T_enc is not None else args.T
        T_dec = args.T_dec if args.T_dec is not None else args.T

    cfg = Config()
    cfg.num_epochs = args.epochs
    cfg.train_size = args.train_size
    cfg.distance_metric = args.metric
    cfg.entropy_coef = args.entropy_coef
    if args.no_spatial: cfg.use_spatial_encoding = False
    if args.no_whitening: cfg.use_whitening = False
    if args.metric == "mahalanobis": cfg.data_distribution = "gaussian"

    ckpt_dir, res_dir = _setup_dirs(args.variant)
    cfg.checkpoint_dir = ckpt_dir
    cfg.results_dir = res_dir

    device = cfg.device
    sigma_inv, weights, W = None, None, None
    if args.metric == "mahalanobis":
        sigma_t = torch.tensor(cfg.mahalanobis_sigma, dtype=torch.float32)
        sigma_inv = torch.inverse(sigma_t).to(device)
        if cfg.use_whitening:
            W, _ = compute_whitening_matrix(cfg.mahalanobis_sigma)
    elif args.metric == "weighted_euclidean":
        weights = torch.tensor(cfg.euclidean_weights, dtype=torch.float32).to(device)

    val_coords, _ = prepare_datasets(cfg)
    val_loader = get_eval_loader(val_coords, cfg)

    model = _build(args.variant, cfg, T_enc, T_dec, args.lora_rank).to(device)

    if args.init_from and os.path.exists(args.init_from):
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        state = ckpt["model_state_dict"]
        own = model.state_dict()
        loaded = 0
        for k, v in state.items():
            # Encoder-only variants wrap the encoder, so source keys need prefix adjustment
            target_keys = [k, f"encoder.base.{k[len('encoder.'):]}" if k.startswith("encoder.") else k]
            for tk in target_keys:
                if tk in own and own[tk].shape == v.shape:
                    own[tk].copy_(v); loaded += 1; break
        print(f"Warm-start: loaded {loaded} matching params")

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    baseline_model = copy.deepcopy(model).to(device); baseline_model.eval()

    # Run name
    suffix_parts = [f"{args.variant}"]
    if args.variant == "combined":
        suffix_parts.append(f"Te{T_enc}_Td{T_dec}")
    else:
        suffix_parts.append(f"T{args.T}")
    suffix_parts.append(f"r{args.lora_rank}")
    suffix_parts.append(args.metric)
    if args.no_whitening and args.metric == "mahalanobis": suffix_parts.append("nowhiten")
    if args.no_spatial: suffix_parts.append("nospatial")
    if args.entropy_coef > 0: suffix_parts.append("ent")
    run_name = "rdt_" + "_".join(suffix_parts)

    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 70)
    print(f"RDT {args.variant.upper()}: {run_name}")
    print(f"  T_enc={T_enc}  T_dec={T_dec}  lora_rank={args.lora_rank}")
    print(f"  Metric: {args.metric}   Params: {n_params:,}")
    print(f"  Output: {ckpt_dir}")
    print("=" * 70)

    history = {"epoch": [], "train_loss": [], "train_length": [], "val_length": []}
    best_val = float("inf")

    for epoch in range(cfg.num_epochs):
        t0 = time.time()
        model.train()
        loader = get_train_loader(cfg)
        ep_loss, ep_len, nb = 0.0, 0.0, 0

        for batch in loader:
            batch = batch.to(device)
            enc = batch if W is None else whiten_coords(batch, W.to(device))

            if args.entropy_coef > 0:
                tours, lp, entropy = model(enc, decode_type="sample", return_entropy=True)
            else:
                tours, lp = model(enc, decode_type="sample")
                entropy = None
            lengths = tour_length(batch, tours, metric=args.metric, sigma_inv=sigma_inv, weights=weights)

            with torch.no_grad():
                bl_tours, _ = baseline_model(enc, decode_type="greedy")
                bl_len = tour_length(batch, bl_tours, metric=args.metric, sigma_inv=sigma_inv, weights=weights)

            adv = (lengths - bl_len).detach()
            loss = (adv * lp).mean()
            if entropy is not None:
                loss = loss - args.entropy_coef * entropy.mean()

            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            ep_loss += loss.item(); ep_len += lengths.mean().item(); nb += 1

        val_len, val_t = _validate(model, val_loader, device, args.metric, sigma_inv, weights, W)
        _, bl_t = _validate(baseline_model, val_loader, device, args.metric, sigma_inv, weights, W)
        if _should_update(val_t, bl_t):
            baseline_model = copy.deepcopy(model).to(device); baseline_model.eval()
            marker = " [bl]"
        else:
            marker = ""

        if val_len < best_val:
            best_val = val_len
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_length": val_len, "args": vars(args)},
                       os.path.join(ckpt_dir, f"{run_name}_best.pt"))
            marker += " (best)"

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(ep_loss / nb)
        history["train_length"].append(ep_len / nb)
        history["val_length"].append(val_len)

        print(f"Epoch {epoch+1:3d}/{cfg.num_epochs} | train_L={ep_len/nb:.4f} | "
              f"val_L={val_len:.4f} | time={time.time()-t0:.1f}s{marker}")

    with open(os.path.join(res_dir, f"{run_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val: {best_val:.4f}")


if __name__ == "__main__":
    main()
