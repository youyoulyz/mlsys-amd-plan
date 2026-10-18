import torch

from deepep_sim.config import SimConfig
from deepep_sim.reference import grouped_counts, roundtrip_identity
from deepep_sim.routing import expert_to_rank, make_routing_logits, route_tokens


def test_identity_roundtrip_recovers_input():
    for distribution in ["uniform", "hotspot", "rank_skew"]:
        cfg = SimConfig(
            num_tokens=2000,
            hidden=256,
            num_experts=32,
            topk=6,
            dtype="fp32",
            world_size=2,
            distribution=distribution,
        )
        gen = torch.Generator().manual_seed(7)
        x = torch.randn(cfg.num_tokens, cfg.hidden, generator=gen)
        logits = make_routing_logits(cfg, cfg.num_tokens, generator=gen, device="cpu")
        w, idx = route_tokens(cfg, logits)
        out = roundtrip_identity(cfg, x, idx, w, check=True)
        assert torch.allclose(out, x, atol=1e-5), distribution


def test_grouped_counts_are_per_local_expert():
    cfg = SimConfig(
        num_tokens=1024, hidden=64, num_experts=16, topk=4, dtype="fp32", world_size=2
    )
    gen = torch.Generator().manual_seed(3)
    logits = make_routing_logits(cfg, cfg.num_tokens, generator=gen)
    _, idx = route_tokens(cfg, logits)
    counts = grouped_counts(cfg, idx, rank=0)
    assert counts.numel() == cfg.num_local_experts
    # Only the pairs whose expert lives on rank 0 are counted here.
    expected = int((expert_to_rank(idx, cfg) == 0).sum())
    assert int(counts.sum()) == expected
