"""Single-process reference for the EP round-trip math (no collectives).

Used as an oracle: with identity experts and normalized top-k weights, a
dispatch->combine round trip must return the input unchanged. This checks the
grouping / permutation / scatter logic without needing multiple processes.
"""

from __future__ import annotations

import torch

from .config import SimConfig
from .permutation import group_by_expert, scatter_back
from .routing import expert_to_rank


def routes_to_rank(cfg: SimConfig, topk_idx: torch.Tensor, rank: int) -> torch.Tensor:
    """Boolean mask [T, K] of pairs whose expert lives on ``rank``."""
    dest = expert_to_rank(topk_idx, cfg)
    return dest == rank


def roundtrip_identity(
    cfg: SimConfig,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    check: bool = True,
) -> torch.Tensor:
    """Emulate dispatch+combine for rank 0's tokens with identity experts.

    The communication is replaced by direct indexing, so this isolates the
    algorithmic correctness of the permutation and weighted scatter.
    """
    t, k = topk_idx.shape
    idx_flat = topk_idx.reshape(-1)
    w_flat = topk_weights.reshape(-1)
    pair_row = torch.arange(t * k, device=x.device) // k

    # "Local" pairs: expert owned by rank 0. "Remote": owned by rank 1 and
    # simulated as being sent away and returned unchanged.
    values = x.index_select(0, pair_row)          # [T*K, H] identity expert
    out = scatter_back(values, pair_row, w_flat, t)

    if check:
        # With weights summing to 1 per token, out must equal x.
        assert torch.allclose(out, x, atol=1e-5), "round-trip is not identity"
    return out


def grouped_counts(cfg: SimConfig, topk_idx: torch.Tensor, rank: int) -> torch.Tensor:
    """Per-local-expert counts for the pairs that land on ``rank``."""
    mask = routes_to_rank(cfg, topk_idx, rank).reshape(-1)
    local = topk_idx.reshape(-1)[mask]
    perm, counts, offsets = group_by_expert(local, cfg.num_experts)
    lo = rank * cfg.num_local_experts
    return counts[lo:lo + cfg.num_local_experts]
