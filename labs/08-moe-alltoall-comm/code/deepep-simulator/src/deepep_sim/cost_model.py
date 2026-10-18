"""Analytical dispatch/combine cost model.

Predicts per-rank transfer time from shape + link parameters, and exposes the
*imbalance* term: the slowest rank bounds the collective.

    time ≈ latency + bytes / bandwidth

This is the tool you validate against the simulator on V100 and later re-fit on
8-card Hopper/Blackwell. It deliberately separates fixed latency from
bandwidth so both regimes are visible.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import SimConfig


@dataclass
class LinkParams:
    """Link parameters. Defaults are *placeholder* V100-class numbers."""

    bandwidth_gbps: float = 100.0     # per-rank unidirectional, GB/s
    latency_us: float = 5.0           # fixed per-collective latency
    meta_bytes_per_pair: int = 32     # 4 * int64 tag per routed pair

    @property
    def bandwidth_bytes_per_us(self) -> float:
        return self.bandwidth_gbps * 1e9 / 1e6


class CostModel:
    def __init__(self, cfg: SimConfig, link: LinkParams | None = None):
        self.cfg = cfg
        self.link = link or LinkParams()

    def _rank_bytes(self, tokens_per_rank: list[int]) -> tuple[list[int], list[int]]:
        h = self.cfg.hidden
        dp = self.cfg.dtype_bytes
        k = self.cfg.topk
        meta = self.link.meta_bytes_per_pair
        # Dispatch: send topk*T rows, receive on average the same volume.
        dispatch = [t * k * (h * dp + meta) for t in tokens_per_rank]
        # Combine sends the expert outputs back (hidden_out).
        ho = self.cfg.hidden_out or h
        combine = [t * k * (ho * dp + meta) for t in tokens_per_rank]
        return dispatch, combine

    def predict(self, tokens_per_rank: list[int] | None = None) -> dict:
        if tokens_per_rank is None:
            tokens_per_rank = self.cfg.tokens_per_rank()
        dispatch, combine = self._rank_bytes(tokens_per_rank)
        bw = self.link.bandwidth_bytes_per_us
        lat = self.link.latency_us

        disp_us = [lat + b / bw for b in dispatch]
        comb_us = [lat + b / bw for b in combine]
        return {
            "tokens_per_rank": tokens_per_rank,
            "dispatch_bytes_per_rank": dispatch,
            "combine_bytes_per_rank": combine,
            "dispatch_us_per_rank": disp_us,
            "combine_us_per_rank": comb_us,
            "dispatch_us": max(disp_us),
            "combine_us": max(comb_us),
            "imbalance_factor": max(dispatch) / (sum(dispatch) / len(dispatch)),
        }

    def predict_balanced(self) -> dict:
        t = self.cfg.num_tokens // self.cfg.world_size
        return self.predict([t] * self.cfg.world_size)
