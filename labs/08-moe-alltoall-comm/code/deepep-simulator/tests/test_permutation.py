import torch

from deepep_sim.permutation import (
    apply_perm,
    group_by_expert,
    invert_perm,
    scatter_back,
    unpermute,
)


def test_group_by_expert_is_stable_and_complete():
    idx = torch.tensor([2, 0, 2, 1, 0, 2])
    perm, counts, offsets = group_by_expert(idx, num_experts=3)
    assert counts.tolist() == [2, 1, 3]
    assert offsets.tolist() == [0, 2, 3]
    grouped = idx[perm]
    assert torch.equal(grouped, torch.tensor([0, 0, 1, 2, 2, 2]))
    # stable: within expert 2 the original order (rows 0,2,5) is preserved
    assert perm[grouped == 2].tolist() == [0, 2, 5]


def test_apply_unpermute_roundtrip():
    idx = torch.tensor([3, 0, 1, 3, 2])
    x = torch.arange(5 * 4).reshape(5, 4).float()
    perm, _, _ = group_by_expert(idx, num_experts=4)
    assert torch.equal(unpermute(apply_perm(x, perm), perm), x)
    assert torch.equal(invert_perm(invert_perm(perm)), perm)


def test_scatter_back_weighted_sum():
    values = torch.tensor([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
    rows = torch.tensor([0, 0, 1])
    weights = torch.tensor([0.25, 0.75, 1.0])
    out = scatter_back(values, rows, weights, out_rows=2)
    assert torch.allclose(out[0], torch.tensor([1.75, 1.75]))
    assert torch.allclose(out[1], torch.tensor([4.0, 4.0]))
