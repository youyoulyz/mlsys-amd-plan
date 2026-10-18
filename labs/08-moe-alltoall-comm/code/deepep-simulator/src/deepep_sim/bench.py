"""Timing harness for the distributed dispatch/combine path.

Runs inside a ``torchrun`` job; each process owns one rank. Times dispatch and
combine separately with CUDA events, and records the observed imbalance.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .config import SimConfig, load_config
from .cost_model import CostModel
from .dispatch_combine import SimEP
from .gemm import grouped_gemm_ref
from .routing import imbalance_factor, make_routing_logits, route_tokens


def _time_fn(fn, warmup: int, iters: int, use_cuda: bool = False) -> tuple[float, float, float]:
    """Return (avg, min, max) seconds using CUDA events (or wall time on CPU)."""
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        secs = [m / 1e3 for m in ms]
    else:
        import time
        secs = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            secs.append(time.perf_counter() - t0)
    import numpy as np
    arr = np.array(secs)
    return float(arr.mean()), float(arr.min()), float(arr.max())


def run_rank(cfg: SimConfig, local_rank: int, results_dir: str = "results") -> dict:
    """Execute the benchmark on one rank. Call under torchrun."""
    dist.init_process_group(backend=cfg.backend)
    rank = dist.get_rank()
    group = dist.new_group(list(range(cfg.world_size)))
    if cfg.device == "cuda":
        torch.cuda.set_device(local_rank)
    device = cfg.device

    torch.manual_seed(cfg.seed + rank)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + rank)

    tokens_per_rank = cfg.tokens_per_rank()
    t_local = tokens_per_rank[rank]

    logits = make_routing_logits(cfg, t_local, generator=gen, device=device, rank=rank)
    if device == "cuda":
        logits = logits.cuda()
    topk_w, topk_idx = route_tokens(cfg, logits)
    x = torch.randn((t_local, cfg.hidden), generator=gen, dtype=torch.float32).to(
        getattr(torch, cfg.torch_dtype)
    )
    if device == "cuda":
        x = x.cuda()
        topk_w = topk_w.cuda()
        topk_idx = topk_idx.cuda()

    expert_w = torch.randn(
        (cfg.num_local_experts, cfg.hidden, cfg.hidden_out),
        generator=gen,
        dtype=torch.float32,
    ).to(x.dtype)
    if device == "cuda":
        expert_w = expert_w.cuda()

    ep = SimEP(cfg, rank, group, device=device)

    def do_dispatch():
        return ep.dispatch(x, topk_idx, topk_w)

    def do_combine():
        recv_x, handle = ep.dispatch(x, topk_idx, topk_w)
        expert_out = grouped_gemm_ref(recv_x, handle.counts, handle.offsets, expert_w)
        return ep.combine(expert_out, handle)

    use_cuda = cfg.device == "cuda" and torch.cuda.is_available()
    d_avg, d_min, d_max = _time_fn(do_dispatch, cfg.warmup, cfg.iters, use_cuda)
    c_avg, c_min, c_max = _time_fn(do_combine, cfg.warmup, cfg.iters, use_cuda)

    model = CostModel(cfg)
    pred = model.predict(tokens_per_rank)
    imb = imbalance_factor(topk_idx, cfg)

    row = {
        "tag": cfg.tag,
        "world_size": cfg.world_size,
        "num_tokens": cfg.num_tokens,
        "hidden": cfg.hidden,
        "num_experts": cfg.num_experts,
        "topk": cfg.topk,
        "dtype": cfg.dtype,
        "distribution": cfg.distribution,
        "dispatch_avg_us": round(d_avg * 1e6, 2),
        "dispatch_min_us": round(d_min * 1e6, 2),
        "dispatch_max_us": round(d_max * 1e6, 2),
        "combine_avg_us": round(c_avg * 1e6, 2),
        "combine_min_us": round(c_min * 1e6, 2),
        "combine_max_us": round(c_max * 1e6, 2),
        "imbalance_factor": round(imb, 3),
        "dispatch_bytes_per_rank": pred["dispatch_bytes_per_rank"][rank],
        "model_dispatch_us": round(pred["dispatch_us_per_rank"][rank], 2),
        "model_combine_us": round(pred["combine_us_per_rank"][rank], 2),
        "rank": rank,
    }
    dist.barrier(group=group)
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"{cfg.tag}.csv")
        write_header = not os.path.exists(out_path)
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"[bench] wrote {out_path}")
    dist.destroy_process_group()
    return row
