from deepep_sim.config import SimConfig
from deepep_sim.cost_model import CostModel, LinkParams


def test_more_tokens_costs_more():
    small = CostModel(SimConfig(num_tokens=1024, hidden=4096, num_experts=16, topk=4))
    big = CostModel(SimConfig(num_tokens=8192, hidden=4096, num_experts=16, topk=4))
    assert big.predict()["dispatch_us"] > small.predict()["dispatch_us"]


def test_fp8_is_cheaper_than_bf16():
    bf16 = CostModel(SimConfig(num_tokens=4096, hidden=4096, num_experts=16, topk=4, dtype="bf16"))
    fp8 = CostModel(SimConfig(num_tokens=4096, hidden=4096, num_experts=16, topk=4, dtype="fp8"))
    assert fp8.predict()["dispatch_us"] < bf16.predict()["dispatch_us"]


def test_rank_skew_raises_imbalance():
    cfg = SimConfig(
        num_tokens=4096, hidden=4096, num_experts=16, topk=4,
        world_size=2, rank_skew=0.5,
    )
    pred = CostModel(cfg).predict()
    assert pred["imbalance_factor"] > 1.0


def test_bandwidth_scales_time_inversely():
    slow = CostModel(
        SimConfig(num_tokens=4096, hidden=4096, num_experts=16, topk=4),
        LinkParams(bandwidth_gbps=50.0, latency_us=0.0),
    )
    fast = CostModel(
        SimConfig(num_tokens=4096, hidden=4096, num_experts=16, topk=4),
        LinkParams(bandwidth_gbps=200.0, latency_us=0.0),
    )
    assert slow.predict()["dispatch_us"] > fast.predict()["dispatch_us"]
