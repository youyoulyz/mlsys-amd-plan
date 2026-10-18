"""Collective helpers for variable-size token exchange.

Everything here is built on ``torch.distributed.all_to_all_single`` so it works
with NCCL on GPUs and gloo on CPU. DeepEP uses hand-written P2P/NVLink kernels;
using the generic collective is intentional — it makes the *cost model* and the
*algorithm* the things under study, not the kernel.
"""

from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist


def exchange_counts(
    send_counts: torch.Tensor,
    group,
    device: str,
) -> torch.Tensor:
    """All-to-all a per-destination count vector.

    ``send_counts[d]`` is how many rows this rank sends to peer ``d``. The
    returned ``recv_counts[d]`` is how many rows this rank receives from ``d``.
    """
    world = send_counts.numel()
    out = torch.empty(world, dtype=torch.long, device=device)
    dist.all_to_all_single(
        out,
        send_counts.to(torch.long),
        output_split_sizes=[1] * world,
        input_split_sizes=[1] * world,
        group=group,
    )
    return out


def all_to_all_variable(
    send: torch.Tensor,
    send_counts: List[int],
    recv_counts: List[int],
    group,
) -> torch.Tensor:
    """Variable-size all-to-all of row-blocks.

    ``send`` must already be ordered so that all rows destined for rank 0 come
    first, then rank 1, ... ``send_counts[d]`` gives the row count per peer.
    """
    total_recv = int(sum(recv_counts))
    out = torch.empty((total_recv, *send.shape[1:]), dtype=send.dtype, device=send.device)
    dist.all_to_all_single(
        out,
        send.contiguous(),
        output_split_sizes=[int(c) for c in recv_counts],
        input_split_sizes=[int(c) for c in send_counts],
        group=group,
    )
    return out


def order_by_destination(dest: torch.Tensor, world_size: int):
    """Stable-sort rows by destination rank.

    Returns ``(order, counts)`` where ``order`` is the permutation that groups
    rows by destination and ``counts[d]`` is the number of rows for rank ``d``.
    """
    order = torch.argsort(dest, stable=True)
    counts = torch.bincount(dest, minlength=world_size)
    return order, counts
