"""Token permutation helpers (the "sort by expert" step of MoE).

DeepEP fuses permutation into its kernels; here we keep it explicit so its cost
can be measured separately from communication and from the expert GEMM.
"""

from __future__ import annotations

import torch


def group_by_expert(indices: torch.Tensor, num_experts: int):
    """Stable-sort rows so all rows of the same expert are contiguous.

    Arguments:
        indices: [N] expert id per row.
        num_experts: total experts.

    Returns:
        (perm, counts, offsets) where ``perm`` maps sorted position -> original
        row, ``counts`` is per-expert row count, ``offsets`` is the exclusive
        prefix sum of ``counts``.
    """
    flat = indices.reshape(-1).to(torch.long)
    counts = torch.bincount(flat, minlength=num_experts)
    offsets = torch.cumsum(counts, dim=0) - counts
    perm = torch.argsort(flat, stable=True)
    return perm, counts, offsets


def apply_perm(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Gather rows: ``out[i] = x[perm[i]]``."""
    return x.index_select(0, perm)


def invert_perm(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv


def unpermute(y: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`apply_perm`: scatter sorted rows back to original order."""
    return y.index_select(0, invert_perm(perm))


def scatter_back(
    values: torch.Tensor,
    row_index: torch.Tensor,
    weights: torch.Tensor,
    out_rows: int,
) -> torch.Tensor:
    """Weighted scatter-add for combine.

    ``values`` [N, H] are expert outputs; ``row_index`` [N] are the destination
    token rows; ``weights`` [N] are the routing weights. Rows that receive
    multiple contributions are summed.
    """
    out = torch.zeros((out_rows, values.shape[1]), dtype=values.dtype, device=values.device)
    out.index_add_(0, row_index.to(torch.long), values * weights.unsqueeze(1))
    return out
