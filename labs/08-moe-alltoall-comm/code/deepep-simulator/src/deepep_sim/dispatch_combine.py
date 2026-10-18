"""End-to-end MoE EP dispatch / combine over a real process group.

Semantics mirror DeepEP:

dispatch
    1. top-k route local tokens         -> expert ids + weights
    2. send each (token, k) pair to the rank that owns its expert
    3. group received rows by local expert (permutation for grouped GEMM)

combine
    1. take per-expert outputs
    2. send each row back to the rank that produced the token
    3. weight and scatter-add into the original token rows

Communication uses ``all_to_all_single`` (see ``comm.py``). Every routed pair
carries a 4-int64 tag ``(src_rank, token_row, k, expert_id)`` so combine can be
inverted exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from .comm import all_to_all_variable, exchange_counts, order_by_destination
from .config import SimConfig
from .permutation import group_by_expert
from .routing import expert_to_rank


@dataclass
class DispatchHandle:
    """Everything combine needs to invert a dispatch."""

    topk_weights: torch.Tensor   # [T, K] local
    recv_meta: torch.Tensor      # [M, 4] grouped order
    perm: torch.Tensor           # [M] grouped -> recv order
    counts: torch.Tensor         # [E_local]
    offsets: torch.Tensor        # [E_local]
    num_local_tokens: int
    send_counts: list
    recv_counts: list


class SimEP:
    """Minimal expert-parallel dispatch/combine engine."""

    def __init__(self, cfg: SimConfig, rank: int, group, device: Optional[str] = None):
        self.cfg = cfg
        self.rank = rank
        self.world = cfg.world_size
        self.group = group
        self.device = device or cfg.device
        self.dtype = getattr(torch, cfg.torch_dtype)

    # -- public API ----------------------------------------------------------
    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, DispatchHandle]:
        """Route ``x`` and return ``(recv_x_grouped, handle)``."""
        t, k = topk_idx.shape
        dest = expert_to_rank(topk_idx, self.cfg)                       # [T, K]
        pairs = t * k
        rows = torch.arange(pairs, device=x.device) // k               # token row
        ks = torch.arange(pairs, device=x.device) % k                  # top-k slot
        dest_flat = dest.reshape(-1)

        order, send_counts = order_by_destination(dest_flat, self.world)
        row_sorted = rows[order]
        k_sorted = ks[order]
        expert_sorted = topk_idx.reshape(-1)[order]

        send_x = x.index_select(0, row_sorted)
        send_meta = torch.stack(
            [
                torch.full_like(row_sorted, self.rank),
                row_sorted,
                k_sorted,
                expert_sorted,
            ],
            dim=1,
        )

        recv_counts = exchange_counts(send_counts, self.group, self.device)
        recv_counts_list = [int(c) for c in recv_counts.tolist()]
        send_counts_list = [int(c) for c in send_counts.tolist()]

        recv_x = all_to_all_variable(send_x, send_counts_list, recv_counts_list, self.group)
        recv_meta = all_to_all_variable(send_meta, send_counts_list, recv_counts_list, self.group)

        # Received rows all target this rank, so group by *local* expert index.
        lo = self.rank * self.cfg.num_local_experts
        local_expert = recv_meta[:, 3] - lo
        perm, counts, offsets = group_by_expert(local_expert, self.cfg.num_local_experts)
        recv_x_grouped = recv_x.index_select(0, perm)
        recv_meta_grouped = recv_meta.index_select(0, perm)

        handle = DispatchHandle(
            topk_weights=topk_weights,
            recv_meta=recv_meta_grouped,
            perm=perm,
            counts=counts,
            offsets=offsets,
            num_local_tokens=t,
            send_counts=send_counts_list,
            recv_counts=recv_counts_list,
        )
        return recv_x_grouped, handle

    def combine(self, expert_out_grouped: torch.Tensor, handle: DispatchHandle) -> torch.Tensor:
        """Invert a dispatch and weight-sum expert outputs back to tokens."""
        # Undo the expert permutation so meta and values share the recv order.
        inv = torch.empty_like(handle.perm)
        inv[handle.perm] = torch.arange(handle.perm.numel(), device=handle.perm.device)
        out_recv = expert_out_grouped.index_select(0, inv)
        meta_recv = handle.recv_meta.index_select(0, inv)

        src_rank = meta_recv[:, 0]
        order, send_counts = order_by_destination(src_rank, self.world)
        send_counts_list = [int(c) for c in send_counts.tolist()]

        send_out = out_recv.index_select(0, order)
        send_meta = meta_recv.index_select(0, order)

        recv_counts = exchange_counts(send_counts, self.group, self.device)
        recv_counts_list = [int(c) for c in recv_counts.tolist()]

        patch = all_to_all_variable(send_out, send_counts_list, recv_counts_list, self.group)
        patch_meta = all_to_all_variable(send_meta, send_counts_list, recv_counts_list, self.group)

        row = patch_meta[:, 1].to(torch.long)
        slot = patch_meta[:, 2].to(torch.long)
        weight = handle.topk_weights[row, slot].to(patch.dtype)

        out = torch.zeros(
            (handle.num_local_tokens, patch.shape[1]),
            dtype=patch.dtype,
            device=patch.device,
        )
        out.index_add_(0, row, patch * weight.unsqueeze(1))
        return out

    # -- convenience ---------------------------------------------------------
    def local_expert_ids(self) -> list[int]:
        lo = self.rank * self.cfg.num_local_experts
        return list(range(lo, lo + self.cfg.num_local_experts))

    def barrier(self) -> None:
        dist.barrier(group=self.group)
