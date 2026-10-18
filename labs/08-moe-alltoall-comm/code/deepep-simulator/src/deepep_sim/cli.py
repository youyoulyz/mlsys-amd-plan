"""Command line entry points.

    python -m deepep_sim.cli plan   --config configs/default.yaml
    python -m deepep_sim.cli sweep  --config configs/default.yaml
    torchrun --nproc_per_node=2 -m deepep_sim.cli bench --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

from .config import SimConfig, load_config
from .cost_model import CostModel, LinkParams
from .routing import imbalance_factor, make_routing_logits, route_tokens


def _load(args) -> SimConfig:
    cfg = load_config(args.config) if args.config else SimConfig()
    if args.tag:
        cfg.tag = args.tag
    return cfg


def cmd_plan(args) -> int:
    cfg = _load(args)
    link = LinkParams(
        bandwidth_gbps=args.bandwidth,
        latency_us=args.latency,
    )
    model = CostModel(cfg, link)
    pred = model.predict()
    print(json.dumps(pred, indent=2, ensure_ascii=False))
    return 0


def cmd_sweep(args) -> int:
    """Grid over tokens/hidden/topk/distribution, print predicted cost."""
    base = _load(args)
    rows = []
    for tokens in args.tokens:
        for hidden in args.hidden:
            for topk in args.topk:
                for dist in args.distributions:
                    cfg = copy.deepcopy(base)
                    cfg.num_tokens = tokens
                    cfg.hidden = hidden
                    cfg.topk = topk
                    cfg.distribution = dist
                    logits = make_routing_logits(cfg, cfg.num_tokens, device="cpu", rank=0)
                    _, idx = route_tokens(cfg, logits)
                    imb = imbalance_factor(idx, cfg)
                    pred = CostModel(cfg).predict()
                    rows.append(
                        {
                            "num_tokens": tokens,
                            "hidden": hidden,
                            "topk": topk,
                            "distribution": dist,
                            "imbalance_factor": round(imb, 3),
                            "model_dispatch_us": round(pred["dispatch_us"], 2),
                            "model_combine_us": round(pred["combine_us"], 2),
                            "dispatch_bytes_per_rank": pred["dispatch_bytes_per_rank"][0],
                        }
                    )
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    if args.output:
        import csv
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[sweep] wrote {args.output}", file=sys.stderr)
    return 0


def cmd_bench(args) -> int:
    from .bench import run_rank
    cfg = _load(args)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    run_rank(cfg, local_rank, results_dir=args.results_dir)
    return 0


def cmd_selftest(args) -> int:
    """CPU-only correctness check of the algorithmic core."""
    import torch
    from .reference import roundtrip_identity

    cfg = SimConfig(num_tokens=512, hidden=128, num_experts=16, topk=4, dtype="fp32")
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(cfg.num_tokens, cfg.hidden, generator=gen)
    logits = make_routing_logits(cfg, cfg.num_tokens, generator=gen, device="cpu")
    w, idx = route_tokens(cfg, logits)
    out = roundtrip_identity(cfg, x, idx, w, check=True)
    print("[selftest] identity round-trip OK; max|out-x| =", float((out - x).abs().max()))
    print("[selftest] imbalance_factor =", round(imbalance_factor(idx, cfg), 3))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deepep_sim")
    p.add_argument("command", choices=["plan", "sweep", "bench", "selftest"])
    p.add_argument("--config", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--results-dir", default="results")
    # plan
    p.add_argument("--bandwidth", type=float, default=100.0, help="GB/s per rank")
    p.add_argument("--latency", type=float, default=5.0, help="us per collective")
    # sweep
    p.add_argument("--tokens", type=int, nargs="+", default=[2048, 4096, 8192])
    p.add_argument("--hidden", type=int, nargs="+", default=[2048, 4096, 7168])
    p.add_argument("--topk", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--distributions", nargs="+", default=["uniform", "hotspot", "rank_skew"])
    p.add_argument("--output", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "plan": cmd_plan,
        "sweep": cmd_sweep,
        "bench": cmd_bench,
        "selftest": cmd_selftest,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
