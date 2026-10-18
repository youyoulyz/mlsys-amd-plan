"""Distributed correctness check for SimEP: identity experts must round-trip.

Run under torchrun (defaults to gloo/CPU so it works without free GPUs):

    PYTHONPATH=src python -m torch.distributed.run --nproc_per_node=2 \
        --standalone scripts/check_distributed.py

On GPUs pass --backend nccl --device cuda --nproc_per_node=2.
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from deepep_sim.config import SimConfig
from deepep_sim.dispatch_combine import SimEP
from deepep_sim.gemm import grouped_gemm_ref
from deepep_sim.routing import make_routing_logits, route_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gloo")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-tokens", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--num-experts", type=int, default=16)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--distribution", default="hotspot")
    ap.add_argument("--rank-skew", type=float, default=0.25)
    args = ap.parse_args()

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    group = dist.new_group(list(range(dist.get_world_size())))
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", rank))
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)

    cfg = SimConfig(
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        num_experts=args.num_experts,
        topk=args.topk,
        dtype=args.dtype,
        world_size=dist.get_world_size(),
        distribution=args.distribution,
        rank_skew=args.rank_skew,
        backend=args.backend,
        device=args.device,
    )

    torch.manual_seed(cfg.seed + rank)
    gen = torch.Generator().manual_seed(cfg.seed + rank)
    t_local = cfg.tokens_per_rank()[rank]
    device = args.device

    logits = make_routing_logits(cfg, t_local, generator=gen, device="cpu", rank=rank)
    if device == "cuda":
        logits = logits.cuda()
    w, idx = route_tokens(cfg, logits)
    x = torch.randn(t_local, cfg.hidden, generator=gen, dtype=torch.float32)
    x = x.to(getattr(torch, cfg.torch_dtype))
    if device == "cuda":
        x, w, idx = x.cuda(), w.cuda(), idx.cuda()

    # Identity experts: W_e = I, so expert compute is a no-op.
    eye = torch.eye(cfg.hidden, dtype=x.dtype)
    if device == "cuda":
        eye = eye.cuda()
    expert_w = eye.unsqueeze(0).repeat(cfg.num_local_experts, 1, 1)  # [E_local, H, H]

    ep = SimEP(cfg, rank, group, device=device)
    recv_x, handle = ep.dispatch(x, idx, w)
    expert_out = grouped_gemm_ref(recv_x, handle.counts, handle.offsets, expert_w)
    out = ep.combine(expert_out, handle)

    err = float((out - x).abs().max()) if out.numel() else 0.0
    err_t = torch.tensor([err], device=device)
    dist.all_reduce(err_t, op=dist.ReduceOp.MAX, group=group)
    if rank == 0:
        print(f"[check] world={dist.get_world_size()} dist={cfg.distribution} "
              f"rank_skew={cfg.rank_skew} max|combine(dispatch(x))-x| = {float(err_t):.3e}")
        assert float(err_t) < 1e-4, "round-trip is not identity"
        print("[check] OK")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
