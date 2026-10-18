from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def apply_policy_kernel(
    ids_ptr,
    weights_ptr,
    drop_ptr,
    slot_ptr,
    num_rows,
    threshold,
    low_floor,
    high_floor,
    LOW_MODE: tl.constexpr,
    MID_MODE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TOPK)
    mask = offs < TOPK
    ids = tl.load(ids_ptr + row * TOPK + offs, mask=mask, other=0).to(tl.int32)
    weights = tl.load(weights_ptr + row * TOPK + offs, mask=mask, other=0.0).to(tl.float32)
    slots = tl.load(slot_ptr + ids, mask=mask, other=-1).to(tl.int32)
    resident = mask & (slots >= 0)
    resident_count = tl.sum(resident.to(tl.int32), axis=0)
    resident_mass = tl.sum(tl.where(resident, weights, 0.0), axis=0)
    any_resident = resident_count > 0
    mass_gate = resident_mass >= threshold

    if LOW_MODE == 1:
        low_drop = weights < low_floor
    elif LOW_MODE == 2:
        low_drop = (weights < low_floor) & mass_gate
    else:
        low_drop = tl.zeros((BLOCK_TOPK,), dtype=tl.int1)

    if MID_MODE == 1:
        mid_drop = (weights <= high_floor) & mass_gate
    else:
        mid_drop = tl.zeros((BLOCK_TOPK,), dtype=tl.int1)

    drop = mask & (~resident) & any_resident & (low_drop | mid_drop)
    kept = mask & (~drop)
    kept_sum = tl.sum(tl.where(kept, weights, 0.0), axis=0)
    first_idx = tl.min(tl.where(resident, offs, TOPK), axis=0)
    anchor = tl.load(ids_ptr + row * TOPK + first_idx, mask=first_idx < TOPK, other=0)
    new_weights = tl.where(drop, 0.0, weights) / tl.maximum(kept_sum, 1e-12)
    new_ids = tl.where(drop, anchor, ids)
    tl.store(weights_ptr + row * TOPK + offs, new_weights, mask=mask)
    tl.store(drop_ptr + row * TOPK + offs, drop.to(tl.uint8), mask=mask)
    tl.store(ids_ptr + row * TOPK + offs, new_ids.to(tl.int32), mask=mask)
