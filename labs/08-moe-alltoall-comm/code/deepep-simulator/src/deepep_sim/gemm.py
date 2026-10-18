"""Grouped expert GEMM, kept simple and explicit.

The real cost of the expert compute is a grouped GEMM. Here we implement a
straightforward per-expert loop (correctness reference) and, optionally, a
batched matmul that mirrors the grouped layout. On V100 the grouped GEMM is the
*compute* half of an overlap study; we do not try to make it fast.
"""

from __future__ import annotations

from typing import Optional

import torch


def grouped_gemm_ref(
    x: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    expert_weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-expert loop GEMM.

    Arguments:
        x: [M, K] rows already grouped by expert.
        counts: [E_local] rows per local expert.
        offsets: [E_local] exclusive prefix sum of ``counts``.
        expert_weight: [E_local, K, N].
        bias: optional [E_local, N].

    Returns:
        [M, N] with the same grouping as ``x``.
    """
    out = torch.empty((x.shape[0], expert_weight.shape[-1]), dtype=x.dtype, device=x.device)
    for e in range(counts.numel()):
        n = int(counts[e])
        if n == 0:
            continue
        s = int(offsets[e])
        acc = x[s:s + n] @ expert_weight[e]
        if bias is not None:
            acc = acc + bias[e]
        out[s:s + n] = acc
    return out


def grouped_gemm_bmm(
    x: torch.Tensor,
    counts: torch.Tensor,
    expert_weight: torch.Tensor,
) -> torch.Tensor:
    """Batched-matmul grouping (pads to the max expert count).

    Faster on GPU than a Python loop but wastes work on ragged experts; useful
    to contrast "grouped GEMM" implementations in an overlap study.
    """
    e = counts.numel()
    n = int(counts.max()) if e > 0 else 0
    k = x.shape[1]
    hidden_out = expert_weight.shape[-1]
    padded = x.new_zeros((e, n, k))
    mask = x.new_zeros((e, n), dtype=torch.bool)
    pos = 0
    for i in range(e):
        c = int(counts[i])
        if c:
            padded[i, :c] = x[pos:pos + c]
            mask[i, :c] = True
        pos += c
    out = torch.bmm(padded, expert_weight)
    return out[mask]
