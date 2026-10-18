"""Top-k routing and load-distribution generation.

The point of these helpers is to control the *load imbalance* that drives the
interesting EP behaviour: a hot expert makes one rank wait for another.
"""

from __future__ import annotations

from typing import Optional

import torch

from .config import SimConfig


def make_routing_logits(
    cfg: SimConfig,
    num_tokens: int,
    generator: Optional[torch.Generator] = None,
    device: str = "cpu",
    rank: int = 0,
) -> torch.Tensor:
    """Create raw gate logits [num_tokens, num_experts].

    * ``uniform``  : i.i.d. standard normal -> balanced-ish routing.
    * ``hotspot``  : a fraction of tokens get a large bias toward a few hot
      experts (global hot experts, shared across ranks).
    * ``rank_skew``: the local hotspot experts are shifted per rank, so a
      different expert is hot on each rank (worst case for all-to-all).
    """
    logits = torch.randn(
        (num_tokens, cfg.num_experts), generator=generator, device=device
    )

    if cfg.distribution == "uniform":
        return logits

    n_hot = max(1, cfg.num_experts // 16)
    if cfg.distribution == "hotspot":
        hot = torch.arange(n_hot, device=device)
    elif cfg.distribution == "rank_skew":
        # Rotate the hot set per rank so remote traffic is maximized.
        offset = (rank * n_hot) % cfg.num_experts
        hot = (torch.arange(n_hot, device=device) + offset) % cfg.num_experts
    else:
        raise ValueError(f"unknown distribution {cfg.distribution!r}")

    mask = torch.rand(num_tokens, generator=generator, device=device) < cfg.hotspot_frac
    rows = mask.nonzero(as_tuple=True)[0]
    if rows.numel() > 0:
        # Bias the selected (token, hot-expert) positions, leaving others normal.
        logits[rows.unsqueeze(1), hot.unsqueeze(0).expand(rows.numel(), -1)] += cfg.hotspot_gain
    return logits


def route_tokens(
    cfg: SimConfig,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k route ``logits`` and return ``(weights, indices)``.

    ``weights`` is [T, topk], normalized to sum to 1 per token (as in most
    MoE gates). ``indices`` are expert ids [T, topk].
    """
    weights, indices = torch.topk(logits, cfg.topk, dim=-1, largest=True, sorted=False)
    weights = torch.softmax(weights, dim=-1)
    return weights, indices


def expert_to_rank(expert_ids: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    """Map expert ids to the EP rank that owns them (contiguous split)."""
    return expert_ids * cfg.world_size // cfg.num_experts


def count_tokens_per_expert(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Histogram of routed (token, k) pairs per expert."""
    return torch.bincount(indices.reshape(-1), minlength=num_experts)


def imbalance_factor(indices: torch.Tensor, cfg: SimConfig) -> float:
    """max_rank_load / mean_rank_load for the *routed* load.

    A value of 1.0 means perfectly balanced; large values mean the slowest rank
    stalls the whole all-to-all.
    """
    per_expert = count_tokens_per_expert(indices, cfg.num_experts).float()
    per_rank = per_expert.reshape(cfg.world_size, cfg.num_local_experts).sum(dim=1)
    mean = per_rank.mean().clamp_min(1e-9)
    return float(per_rank.max() / mean)
