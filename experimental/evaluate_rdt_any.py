"""
Unified evaluator for all three RDT variants. Reads checkpoints from the
appropriate subdirectory and compares against main paper results.

Usage:
    python experimental/evaluate_rdt_any.py --variant encoder --metric euclidean --T 1 2 4 8
    python experimental/evaluate_rdt_any.py --variant combined --metric mahalanobis --T-enc 4 --T-dec 4
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, time
import torch
import numpy as np

from config import Config
from data import prepare_datasets, get_eval_loader, tour_length, compute_whitening_matrix, whiten_coords
from baselines import _pairwise_distance
from experimental.train_rdt_any import _build, _setup_dirs


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_JSON = os.path.join(PROJECT_ROOT, "results", "evaluation_full.json")


def _main_ref(metric):
    if not os.path.exists(MAIN_JSON): return None
    with open(MAIN_JSON) as f: data = json.load(f)
    return data.get(metric, {}).get("summary", {}).get("OR-Tools", {}).get("mean_length")


def _greedy(model, coords, cfg, m, si, w, W):
    model.eval(); loader = get_eval_loader(coords, cfg); dev = cfg.device
    out = []
    with torch.no_grad():
        for b in loader:
            b = b.to(dev); enc = b if W is None else whiten_coords(b, W.to(dev))
            t, _ = model(enc, decode_type="greedy")
            out.append(tour_length(b, t, metric=m, sigma_inv=si, weights=w).cpu())
    return torch.cat(out)


def _pomo(model, coords, cfg, m, si, w, W):
    model.eval(); loader = get_eval_loader(coords, cfg); dev = cfg.device
    tours, lengths = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(dev); bs, n = b.size(0), b.size(1)
            enc = b if W is None else whiten_coords(b, W.to(dev))
            best_l = torch.full((bs,), float("inf"), device=dev)
            best_t = torch.zeros(bs, n, dtype=torch.long, device=dev)
            for s in range(n):
                sn = torch.full((bs,), s, dtype=torch.long, device=dev)
                t, _ = model(enc, decode_type="greedy", start_node=sn)
                l = tour_length(b, t, metric=m, sigma_inv=si, weights=w)
                mask = l < best_l
                best_l = torch.where(mask, l, best_l); best_t[mask] = t[mask]
            tours.append(best_t.cpu()); lengths.append(best_l.cpu())
    return torch.cat(tours), torch.cat(lengths)


def _two_opt(tours_t, coords, metric, si_np, w_np):
    coords_np = coords.numpy(); out = []
    for i in range(len(coords_np)):
        cities = coords_np[i]; tour = tours_t[i].numpy().tolist(); n = len(tour)
        improved, it = True, 0
        while improved and it < 500:
            improved, it = False, it + 1
            for a in range(1, n - 1):
                for b in range(a + 1, n):
                    old = (_pairwise_distance(cities[tour[a-1]], cities[tour[a]], metric, si_np, w_np) +
                           _pairwise_distance(cities[tour[b]], cities[tour[(b+1)%n]], metric, si_np, w_np))
                    new = (_pairwise_distance(cities[tour[a-1]], cities[tour[b]], metric, si_np, w_np) +
                           _pairwise_distance(cities[tour[a]], cities[tour[(b+1)%n]], metric, si_np, w_np))
                    if new < old - 1e-10:
                        tour[a:b+1] = tour[a:b+1][::-1]; improved = True
        out.append(sum(_pairwise_distance(cities[tour[j]], cities[tour[(j+1)%n]], metric, si_np, w_np) for j in range(n)))
    return torch.tensor(out)


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--variant", required=True, choices=["decoder", "encoder", "combined"])
    P.add_argument("--metric", default="euclidean",
                   choices=["euclidean", "manhattan", "mahalanobis", "weighted_euclidean"])
    P.add_argument("--T", type=int, nargs="+", default=[4])
    P.add_argument("--T-enc", type=int, nargs="+", default=None)
    P.add_argument("--T-dec", type=int, nargs="+", default=None)
    P.add_argument("--no-whitening", action="store_true")
    P.add_argument("--no-spatial", action="store_true")
    args = P.parse_args()

    cfg = Config()
    cfg.distance_metric = args.metric
    if args.no_spatial: cfg.use_spatial_encoding = False
    if args.no_whitening: cfg.use_whitening = False
    if args.metric == "mahalanobis": cfg.data_distribution = "gaussian"

    dev = cfg.device
    si = si_np = w = w_np = W = None
    if args.metric == "mahalanobis":
        st = torch.tensor(cfg.mahalanobis_sigma, dtype=torch.float32)
        si = torch.inverse(st).to(dev); si_np = torch.inverse(st).numpy()
        if cfg.use_whitening: W, _ = compute_whitening_matrix(cfg.mahalanobis_sigma)
    elif args.metric == "weighted_euclidean":
        w = torch.tensor(cfg.euclidean_weights, dtype=torch.float32).to(dev)
        w_np = np.array(cfg.euclidean_weights)

    _, test_coords = prepare_datasets(cfg)
    ckpt_dir, res_dir = _setup_dirs(args.variant)
    or_ref = _main_ref(args.metric)

    # Build T combinations
    if args.variant == "combined":
        T_enc_list = args.T_enc or args.T
        T_dec_list = args.T_dec or args.T
        configs = [(te, td) for te in T_enc_list for td in T_dec_list]
    else:
        configs = [(t, 1) if args.variant == "encoder" else (1, t) for t in args.T]

    print(f"\n{'='*90}")
    print(f"RDT {args.variant.upper()} EVALUATION: {args.metric.upper()} TSP-20")
    if or_ref: print(f"OR-Tools reference: {or_ref:.4f}")
    print(f"{'='*90}")
    print(f"{'T_enc':>5} {'T_dec':>5} {'Greedy':>9} {'Gap%':>6} {'POMO-20':>9} {'Gap%':>6} {'POMO+2opt':>11} {'Gap%':>6} {'Time':>7}")
    print("-" * 90)

    all_results = {}
    for T_enc, T_dec in configs:
        if args.variant == "decoder":
            name = f"rdt_decoder_T{T_dec}_r8_{args.metric}"
        elif args.variant == "encoder":
            name = f"rdt_encoder_T{T_enc}_r8_{args.metric}"
        else:
            name = f"rdt_combined_Te{T_enc}_Td{T_dec}_r8_{args.metric}"
        if args.no_whitening and args.metric == "mahalanobis": name += "_nowhiten"
        if args.no_spatial: name += "_nospatial"

        ckpt_path = os.path.join(ckpt_dir, f"{name}_best.pt")
        if not os.path.exists(ckpt_path):
            # Fallback: try old decoder-only naming
            if args.variant == "decoder":
                alt = os.path.join(ckpt_dir, f"rdt_T{T_dec}_r8_{args.metric}_best.pt")
                if os.path.exists(alt): ckpt_path = alt
            if not os.path.exists(ckpt_path):
                print(f"T_enc={T_enc} T_dec={T_dec}: SKIP ({ckpt_path} not found)")
                continue

        model = _build(args.variant, cfg, T_enc, T_dec, 8).to(dev)
        ck = torch.load(ckpt_path, map_location=dev, weights_only=True)
        model.load_state_dict(ck["model_state_dict"]); model.eval()

        t_start = time.time()
        g_len = _greedy(model, test_coords, cfg, args.metric, si, w, W)
        p_tours, p_len = _pomo(model, test_coords, cfg, args.metric, si, w, W)
        r_len = _two_opt(p_tours, test_coords, args.metric, si_np, w_np)
        t_total = time.time() - t_start

        def gap(mean): return (mean - or_ref) / or_ref * 100 if or_ref else 0.0

        gm, pm, rm = g_len.mean().item(), p_len.mean().item(), r_len.mean().item()
        print(f"{T_enc:>5} {T_dec:>5} {gm:>9.4f} {gap(gm):>5.2f}% {pm:>9.4f} {gap(pm):>5.2f}% {rm:>11.4f} {gap(rm):>5.2f}% {t_total:>6.1f}s")
        all_results[f"Te{T_enc}_Td{T_dec}"] = {
            "greedy_mean": gm, "pomo_mean": pm, "pomo_2opt_mean": rm,
            "greedy_gap": gap(gm), "pomo_gap": gap(pm), "pomo_2opt_gap": gap(rm),
            "time": t_total,
        }

    out_path = os.path.join(res_dir, f"{args.variant}_{args.metric}_full.json")
    with open(out_path, "w") as f:
        json.dump({"or_tools_ref": or_ref, "variant": args.variant, "metric": args.metric,
                   "results": all_results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
