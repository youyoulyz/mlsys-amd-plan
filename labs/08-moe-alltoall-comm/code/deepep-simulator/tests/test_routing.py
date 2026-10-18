import torch

from deepep_sim.config import SimConfig
from deepep_sim.routing import (
    count_tokens_per_expert,
    expert_to_rank,
    imbalance_factor,
    make_routing_logits,
    route_tokens,
)


def _cfg(**kw):
    base = dict(num_tokens=1024, hidden=64, num_experts=16, topk=4, dtype="fp32")
    base.update(kw)
    return SimConfig(**base)


def test_weights_sum_to_one():
    cfg = _cfg()
    gen = torch.Generator().manual_seed(0)
    logits = make_routing_logits(cfg, cfg.num_tokens, generator=gen, device="cpu")
    w, idx = route_tokens(cfg, logits)
    assert w.shape == (cfg.num_tokens, cfg.topk)
    assert idx.shape == (cfg.num_tokens, cfg.topk)
    assert torch.allclose(w.sum(dim=-1), torch.ones(cfg.num_tokens), atol=1e-5)


def test_expert_to_rank_contiguous():
    cfg = _cfg(num_experts=16, world_size=2)
    experts = torch.arange(16)
    ranks = expert_to_rank(experts, cfg)
    assert ranks[:8].eq(0).all() and ranks[8:].eq(1).all()


def test_hotspot_is_more_imbalanced_than_uniform():
    uniform = _cfg(distribution="uniform", hotspot_frac=0.0)
    hot = _cfg(distribution="hotspot", hotspot_frac=0.6, hotspot_gain=8.0)
    gen = torch.Generator().manual_seed(1)
    _, idx_u = route_tokens(uniform, make_routing_logits(uniform, uniform.num_tokens, generator=gen))
    gen = torch.Generator().manual_seed(1)
    _, idx_h = route_tokens(hot, make_routing_logits(hot, hot.num_tokens, generator=gen))
    assert imbalance_factor(idx_h, hot) > imbalance_factor(idx_u, uniform)


def test_counts_sum_matches_pairs():
    cfg = _cfg()
    gen = torch.Generator().manual_seed(2)
    logits = make_routing_logits(cfg, cfg.num_tokens, generator=gen)
    _, idx = route_tokens(cfg, logits)
    counts = count_tokens_per_expert(idx, cfg.num_experts)
    assert int(counts.sum()) == cfg.num_tokens * cfg.topk
