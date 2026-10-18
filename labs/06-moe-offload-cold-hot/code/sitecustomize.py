import atexit
import json
import os
import threading
import time
from pathlib import Path

PROBE = os.environ.get("FREETOKEN_MOE_PROBE") == "1"
POLICY = os.environ.get("FREETOKEN_MOE_POLICY", "").strip().lower()

if PROBE or POLICY:
    import torch
    import triton

    TOPK = int(os.environ.get("FREETOKEN_PROBE_TOPK", "8"))
    _state = {"decode_step": -1, "chunk": -1}
    _lock = threading.Lock()
    _log_file = None
    _log_path = None


def _csv_ints(xs):
    return ",".join(str(int(x)) for x in xs)


def _csv_floats(xs):
    return ",".join(f"{float(x):.6f}" for x in xs)


def _resident_hex(mask):
    bits = mask.detach().to(torch.int64).cpu().tolist()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:064x}"


def _route_summary(ids, pre_mask, num_experts):
    ids = ids.detach().reshape(-1).long()
    counts = torch.bincount(ids.clamp_min(0), minlength=num_experts)[:num_experts]
    active = counts > 0
    resident_active = active & pre_mask
    missing_active = active & ~pre_mask
    return (
        _csv_ints(counts.cpu().tolist()),
        int(active.sum().item()),
        int(resident_active.sum().item()),
        int(missing_active.sum().item()),
    )


def _install_probe():
    global _log_file, _log_path

    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    if getattr(OffloadMoELayer, "_freetoken_weight_probe_installed", False):
        return
    setattr(OffloadMoELayer, "_freetoken_weight_probe_installed", True)

    probe_dir = Path(os.environ["FREETOKEN_PROBE_DIR"])
    probe_dir.mkdir(parents=True, exist_ok=True)
    _log_path = probe_dir / f"trace-{os.getpid()}.tsv"
    _log_file = open(_log_path, "a", buffering=1)
    if _log_path.stat().st_size == 0:
        _log_file.write(
            "event\tstep\tlayer\tbatch\traw\tpre_resident\tpost_slot\tfetched\tcpu_routed"
            "\tmissing\tresident_count\tresident_hex\troute_counts\tactive_unique"
            "\tresident_unique\tmissing_unique\tdt_ms\traw_weights\tpre_resident_weights"
            "\tpolicy_dropped\tresident_weight_mass\n"
        )

    orig_decode = OffloadMoELayer._decode_routed
    orig_prefill = OffloadMoELayer._prefill_routed
    orig_wait = OffloadMoeCache.wait_prefill_layer

    def emit(row):
        row = list(row)
        row.extend([""] * (21 - len(row)))
        with _lock:
            _log_file.write("\t".join(str(x) for x in row) + "\n")

    def probe_decode(self, hidden_states, topk_weights, topk_ids):
        cache = self.offload_cache
        if cache is None:
            return orig_decode(self, hidden_states, topk_weights, topk_ids)

        raw = topk_ids.detach().clone()
        raw_cpu = raw.cpu()
        weights = topk_weights.detach().reshape(raw.shape)
        pre_slots = cache.slot_for_id[self.layer_id].index_select(0, raw.reshape(-1))
        pre_resident = (pre_slots >= 0).reshape(raw.shape)
        pre_mask = cache.slot_for_id[self.layer_id] >= 0
        summary = _route_summary(raw, pre_mask, cache.num_experts)
        raw_weights = weights.cpu()
        pre_weights = (weights * pre_resident.to(weights.dtype)).cpu()
        resident_mass = pre_weights.sum(dim=-1)
        resident_mask = _resident_hex(pre_mask)
        resident_count = int(pre_mask.sum().item())

        if self.layer_id == 0:
            _state["decode_step"] += 1
        t0 = time.perf_counter()
        out = orig_decode(self, hidden_states, topk_weights, topk_ids)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        post = topk_ids.detach()
        post_cpu = post.cpu()
        drop_mask = getattr(self, "_freetoken_last_drop", None)
        changed = drop_mask.to(torch.bool) if drop_mask is not None else torch.zeros_like(pre_resident)
        self._freetoken_last_drop = None
        dropped = int(changed.sum().item())
        fetched = int((~pre_resident & (post >= 0) & ~changed).sum().item())
        cpu_routed = int((~pre_resident & (post < 0) & ~changed).sum().item())
        missing = int((~pre_resident).sum().item())
        emit([
            "D", _state["decode_step"], self.layer_id, int(raw.numel()) // TOPK,
            _csv_ints(raw_cpu.reshape(-1).tolist()), _csv_ints(pre_resident.cpu().reshape(-1).tolist()),
            _csv_ints(post_cpu.reshape(-1).tolist()), fetched, cpu_routed, missing,
            resident_count, resident_mask, *summary, f"{dt_ms:.3f}",
            _csv_floats(raw_weights.reshape(-1).tolist()),
            _csv_floats(pre_weights.reshape(-1).tolist()), dropped,
            _csv_floats(resident_mass.reshape(-1).tolist()),
        ])
        return out

    def probe_prefill(self, hidden_states, topk_weights, topk_ids):
        cache = self.offload_cache
        if cache is None:
            return orig_prefill(self, hidden_states, topk_weights, topk_ids)
        if self.layer_id == 0:
            _state["chunk"] += 1
        raw = topk_ids.detach()
        weights = topk_weights.detach().reshape(raw.shape)
        pre_slots = cache.slot_for_id[self.layer_id].index_select(0, raw.reshape(-1))
        pre_resident = (pre_slots >= 0).reshape(raw.shape)
        pre_mask = cache.slot_for_id[self.layer_id] >= 0
        summary = _route_summary(raw, pre_mask, cache.num_experts)
        before_hex = _resident_hex(pre_mask)
        before_count = int(pre_mask.sum().item())
        t0 = time.perf_counter()
        out = orig_prefill(self, hidden_states, topk_weights, topk_ids)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        after = cache.slot_for_id[self.layer_id] >= 0
        after_count = int(after.sum().item())
        emit([
            "P", _state["chunk"], self.layer_id, int(hidden_states.shape[0]),
            "", "", "", "", "", "", before_count, before_hex, *summary, f"{dt_ms:.3f}",
            "", "", "", "",
        ])
        emit([
            "P_after", _state["chunk"], self.layer_id, int(hidden_states.shape[0]),
            "", "", "", "", "", "", after_count, _resident_hex(after), "", "", "", "",
            f"{dt_ms:.3f}",
        ])
        return out

    def probe_wait(self, layer_id):
        t0 = time.perf_counter()
        result = orig_wait(self, layer_id)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        emit(["W", _state["chunk"], layer_id, "", "", "", "", "", "", "", "", "", "",
              "", "", "", f"{dt_ms:.3f}"])
        return result

    OffloadMoELayer._decode_routed = probe_decode
    OffloadMoELayer._prefill_routed = probe_prefill
    OffloadMoeCache.wait_prefill_layer = probe_wait


def _install_policy():
    from freetoken.layers.moe import OffloadMoELayer
    from policy_kernel import apply_policy_kernel

    if getattr(OffloadMoELayer, "_freetoken_policy_installed", False):
        return
    setattr(OffloadMoELayer, "_freetoken_policy_installed", True)

    orig_decode = OffloadMoELayer._decode_routed
    floor = float(os.environ.get("FREETOKEN_MOE_POLICY_FLOOR", "0.05"))
    high = float(os.environ.get("FREETOKEN_MOE_POLICY_HIGH", "0.30"))
    shallow = int(os.environ.get("FREETOKEN_MOE_POLICY_SHALLOW", "5"))
    deep = {
        int(x) for x in os.environ.get("FREETOKEN_MOE_POLICY_DEEP", "20,38").split(",")
        if x.strip()
    }

    def modes(layer_id):
        if POLICY in {"noop", "kernel_noop"}:
            return 0, 0, 0.85, floor, high
        if POLICY == "top85":
            return 0, 1, 0.85, floor, 1.0
        if POLICY == "top90":
            return 0, 1, 0.90, floor, 1.0
        if POLICY == "floor":
            return 1, 0, 0.85, floor, high
        if POLICY in {"top85_floor", "combined"}:
            return 1, 1, 0.85, floor, high
        if POLICY == "layer":
            if layer_id < shallow:
                return 2, 0, 0.85, floor, high
            if layer_id in deep:
                return 0, 1, 0.85, floor, high
            return 1, 1, 0.90, floor, high
        raise ValueError(f"unknown FREETOKEN_MOE_POLICY={POLICY!r}")

    def policy_decode(self, hidden_states, topk_weights, topk_ids):
        cache = self.offload_cache
        if cache is None:
            return orig_decode(self, hidden_states, topk_weights, topk_ids)

        low_mode, mid_mode, threshold, low_floor, high_floor = modes(self.layer_id)
        ids = topk_ids.contiguous()
        weights = topk_weights.contiguous()
        rows, topk = ids.shape
        drop = torch.empty_like(ids, dtype=torch.uint8)
        apply_policy_kernel[(rows,)](
            ids,
            weights,
            drop,
            cache.slot_for_id[self.layer_id],
            rows,
            threshold,
            low_floor,
            high_floor,
            LOW_MODE=low_mode,
            MID_MODE=mid_mode,
            TOPK=topk,
            BLOCK_TOPK=triton.next_power_of_2(topk),
        )
        self._freetoken_last_drop = drop
        self._freetoken_last_ids = ids
        self._freetoken_last_weights = weights
        return orig_decode(self, hidden_states, weights, ids)

    OffloadMoELayer._decode_routed = policy_decode


if PROBE or POLICY:
    if PROBE:
        _install_probe()
    if POLICY:
        _install_policy()
        if PROBE:
            _install_probe()

    def _dump_probe(_signum=None, _frame=None):
        if _log_file is not None:
            _log_file.flush()

    atexit.register(_dump_probe)
