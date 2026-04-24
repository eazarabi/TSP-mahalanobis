"""
Experimental: Full evaluation of RDT decoder against main paper results.

Mirrors the decoding strategies used in the main evaluate.py:
  - Greedy
  - Best-of-128 sampling
  - POMO-20 multi-start
  - POMO-20 + 2-opt refinement
  - 8x dihedral augmentation (for rotation-invariant metrics)

Compares each RDT variant (T=1, 2, 4, ...) against the main paper's glimpse
decoder result for the same metric, reading the pre-computed baselines from
results/evaluation_full.json in the project root.

Saves all output to experimental/results/ to keep main paper artifacts clean.

Usage:
    python experimental/evaluate_rdt.py --metric euclidean --thinking-steps 1 2 4
    python experimental/evaluate_rdt.py --metric mahalanobis --thinking-steps 1 2 4
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import torch

from config import Config
from data import prepare_datasets, get_eval_loader, tour_length, compute_whitening_matrix, whiten_coords
from baselines import _pairwise_distance
from experimental.train_rdt import build_rdt_model


# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_CHECKPOINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
EXP_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MAIN_RESULTS_JSON = os.path.join(PROJECT_ROOT, "results", "evaluation_full.json")


def _load_main_baseline(metric: str):
    """Read the main paper's results for comparison."""
    if not os.path.exists(MAIN_RESULTS_JSON):
        return None
    with open(MAIN_RESULTS_JSON) as f:
        data = json.load(f)
    if metric not in data:
        return None
    summary = data[metric]["summary"]
    or_tools_mean = summary["OR-Tools"]["mean_length"]
    return {"or_tools": or_tools_mean, "summary": summary}


def evaluate_greedy(model, test_coords, cfg, metric, sigma_inv, weights, W_whiten):
    model.eval()
    loader = get_eval_loader(test_coords, cfg)
    device = cfg.device
    all_tours, all_lengths = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            enc = batch if W_whiten is None else whiten_coords(batch, W_whiten.to(device))
            tours, _ = model(enc, decode_type="greedy")
            lengths = tour_length(batch, tours, metric=metric, sigma_inv=sigma_inv, weights=weights)
            all_tours.append(tours.cpu()); all_lengths.append(lengths.cpu())
    return torch.cat(all_tours), torch.cat(all_lengths), time.time() - t0


def evaluate_sampling(model, test_coords, cfg, metric, sigma_inv, weights, W_whiten, N=128):
    model.eval()
    loader = get_eval_loader(test_coords, cfg)
    device = cfg.device
    all_lengths = []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            bs = batch.size(0)
            enc = batch if W_whiten is None else whiten_coords(batch, W_whiten.to(device))
            best = torch.full((bs,), float("inf"), device=device)
            for _ in range(N):
                tours, _ = model(enc, decode_type="sample")
                l = tour_length(batch, tours, metric=metric, sigma_inv=sigma_inv, weights=weights)
                best = torch.minimum(best, l)
            all_lengths.append(best.cpu())
    return torch.cat(all_lengths), time.time() - t0


def evaluate_pomo(model, test_coords, cfg, metric, sigma_inv, weights, W_whiten):
    model.eval()
    loader = get_eval_loader(test_coords, cfg)
    device = cfg.device
    all_tours, all_lengths = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            bs, n = batch.size(0), batch.size(1)
            enc = batch if W_whiten is None else whiten_coords(batch, W_whiten.to(device))
            best_l = torch.full((bs,), float("inf"), device=device)
            best_t = torch.zeros(bs, n, dtype=torch.long, device=device)
            for s in range(n):
                sn = torch.full((bs,), s, dtype=torch.long, device=device)
                tours, _ = model(enc, decode_type="greedy", start_node=sn)
                l = tour_length(batch, tours, metric=metric, sigma_inv=sigma_inv, weights=weights)
                mask = l < best_l
                best_l = torch.where(mask, l, best_l)
                best_t[mask] = tours[mask]
            all_tours.append(best_t.cpu()); all_lengths.append(best_l.cpu())
    return torch.cat(all_tours), torch.cat(all_lengths), time.time() - t0


def apply_2opt(tours_tensor, test_coords, metric, sigma_inv_np, weights_np):
    """2-opt refinement of given tours."""
    coords_np = test_coords.numpy()
    refined_lengths = []
    t0 = time.time()
    for i in range(len(coords_np)):
        cities = coords_np[i]
        tour = tours_tensor[i].numpy().tolist()
        n = len(tour)
        improved = True
        it = 0
        while improved and it < 500:
            improved = False; it += 1
            for a in range(1, n - 1):
                for b in range(a + 1, n):
                    old = (_pairwise_distance(cities[tour[a-1]], cities[tour[a]], metric, sigma_inv_np, weights_np) +
                           _pairwise_distance(cities[tour[b]], cities[tour[(b+1)%n]], metric, sigma_inv_np, weights_np))
                    new = (_pairwise_distance(cities[tour[a-1]], cities[tour[b]], metric, sigma_inv_np, weights_np) +
                           _pairwise_distance(cities[tour[a]], cities[tour[(b+1)%n]], metric, sigma_inv_np, weights_np))
                    if new < old - 1e-10:
                        tour[a:b+1] = tour[a:b+1][::-1]
                        improved = True
        total = sum(_pairwise_distance(cities[tour[j]], cities[tour[(j+1)%n]], metric, sigma_inv_np, weights_np) for j in range(n))
        refined_lengths.append(total)
    return torch.tensor(refined_lengths), time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="euclidean",
                        choices=["euclidean", "manhattan", "mahalanobis", "weighted_euclidean"])
    parser.add_argument("--thinking-steps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--skip-sampling", action="store_true", help="Skip best-of-128 (slow)")
    parser.add_argument("--skip-2opt", action="store_true", help="Skip 2-opt refinement (slow)")
    args = parser.parse_args()

    cfg = Config()
    cfg.distance_metric = args.metric
    if args.metric == "mahalanobis":
        cfg.data_distribution = "gaussian"

    device = cfg.device
    sigma_inv = sigma_inv_np = weights = weights_np = W_whiten = None
    if args.metric == "mahalanobis":
        sigma_t = torch.tensor(cfg.mahalanobis_sigma, dtype=torch.float32)
        sigma_inv = torch.inverse(sigma_t).to(device)
        sigma_inv_np = torch.inverse(sigma_t).numpy()
        W_whiten, _ = compute_whitening_matrix(cfg.mahalanobis_sigma)
    elif args.metric == "weighted_euclidean":
        import numpy as np
        weights = torch.tensor(cfg.euclidean_weights, dtype=torch.float32).to(device)
        weights_np = np.array(cfg.euclidean_weights)

    _, test_coords = prepare_datasets(cfg)

    # Pull main paper baseline for this metric
    baseline = _load_main_baseline(args.metric)
    or_ref = baseline["or_tools"] if baseline else None
    if or_ref is None:
        print(f"WARNING: no main baseline found in {MAIN_RESULTS_JSON}; gaps not computed")

    os.makedirs(EXP_RESULTS, exist_ok=True)

    header = f"{'T':>3} {'Greedy':>9} {'Gap%':>6} {'POMO-20':>9} {'Gap%':>6} {'POMO+2opt':>11} {'Gap%':>6}"
    if not args.skip_sampling:
        header += f" {'Best-128':>10} {'Gap%':>6}"
    header += f" {'Time':>7}"
    print(f"\n{'='*90}")
    print(f"RDT EVALUATION: {args.metric.upper()} TSP-20")
    if or_ref:
        print(f"OR-Tools reference: {or_ref:.4f}")
    print(f"{'='*90}")
    print(header)
    print("-" * 90)

    all_results = {}
    for T in args.thinking_steps:
        ckpt_path = os.path.join(EXP_CHECKPOINTS, f"rdt_T{T}_r8_{args.metric}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"T={T}: SKIP (no checkpoint at {ckpt_path})")
            continue

        model = build_rdt_model(cfg, num_thinking_steps=T).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        t_start = time.time()
        _, g_len, _ = evaluate_greedy(model, test_coords, cfg, args.metric, sigma_inv, weights, W_whiten)
        p_tours, p_len, _ = evaluate_pomo(model, test_coords, cfg, args.metric, sigma_inv, weights, W_whiten)

        result = {
            "greedy_mean": g_len.mean().item(),
            "pomo_mean": p_len.mean().item(),
        }

        if not args.skip_2opt:
            r_len, _ = apply_2opt(p_tours, test_coords, args.metric, sigma_inv_np, weights_np)
            result["pomo_2opt_mean"] = r_len.mean().item()
        else:
            result["pomo_2opt_mean"] = None

        if not args.skip_sampling:
            s_len, _ = evaluate_sampling(model, test_coords, cfg, args.metric, sigma_inv, weights, W_whiten, N=128)
            result["bestof128_mean"] = s_len.mean().item()
        else:
            result["bestof128_mean"] = None

        result["total_time"] = time.time() - t_start

        def gap(mean):
            return (mean - or_ref) / or_ref * 100 if or_ref and mean is not None else None

        row = f"{T:>3} {result['greedy_mean']:>9.4f} {gap(result['greedy_mean']):>5.2f}% "
        row += f"{result['pomo_mean']:>9.4f} {gap(result['pomo_mean']):>5.2f}% "
        if result['pomo_2opt_mean'] is not None:
            row += f"{result['pomo_2opt_mean']:>11.4f} {gap(result['pomo_2opt_mean']):>5.2f}% "
        else:
            row += f"{'—':>11} {'—':>6} "
        if result['bestof128_mean'] is not None:
            row += f"{result['bestof128_mean']:>10.4f} {gap(result['bestof128_mean']):>5.2f}% "
        row += f"{result['total_time']:>6.1f}s"
        print(row)

        all_results[T] = result

    # Save full results
    out_path = os.path.join(EXP_RESULTS, f"rdt_{args.metric}_full.json")
    with open(out_path, "w") as f:
        json.dump({"or_tools_ref": or_ref, "results": all_results, "metric": args.metric}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Side-by-side comparison with main paper
    if baseline:
        print(f"\n{'='*70}")
        print(f"COMPARISON: RDT vs. Main Paper (glimpse decoder, {args.metric})")
        print(f"{'='*70}")
        main_summary = baseline["summary"]
        # Find the main paper's best greedy / POMO / POMO+2opt for this metric
        def find_main(substr, default=None):
            for k, v in main_summary.items():
                if substr in k.lower():
                    return v["mean_length"], v["gap_percent"]
            return default

        main_greedy = find_main("(greedy)")
        main_pomo = find_main("(pomo-20)")
        main_pomo_2opt = find_main("(pomo+2opt)")
        print(f"{'Strategy':<15} {'Main Paper':>14} {'Best RDT':>14} {'Δ':>8}")
        print("-" * 55)
        if main_greedy and all_results:
            best_rdt_greedy = min((r["greedy_mean"] for r in all_results.values()), default=None)
            if best_rdt_greedy:
                delta = best_rdt_greedy - main_greedy[0]
                print(f"{'Greedy':<15} {main_greedy[0]:>14.4f} {best_rdt_greedy:>14.4f} {delta:>+8.4f}")
        if main_pomo and all_results:
            best_rdt_pomo = min((r["pomo_mean"] for r in all_results.values()), default=None)
            if best_rdt_pomo:
                delta = best_rdt_pomo - main_pomo[0]
                print(f"{'POMO-20':<15} {main_pomo[0]:>14.4f} {best_rdt_pomo:>14.4f} {delta:>+8.4f}")
        if main_pomo_2opt and all_results and not args.skip_2opt:
            best_rdt_2opt = min((r["pomo_2opt_mean"] for r in all_results.values() if r["pomo_2opt_mean"]), default=None)
            if best_rdt_2opt:
                delta = best_rdt_2opt - main_pomo_2opt[0]
                print(f"{'POMO+2opt':<15} {main_pomo_2opt[0]:>14.4f} {best_rdt_2opt:>14.4f} {delta:>+8.4f}")


if __name__ == "__main__":
    main()
