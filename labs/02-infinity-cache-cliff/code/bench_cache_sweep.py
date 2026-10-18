#!/usr/bin/env python3
"""Working-set size sweep to expose the Infinity Cache (GL2C/L2) bandwidth knee.

For each working-set size we run four streaming primitives and record the
*effective* logical bandwidth.  Operand size starts well below the 96 MiB
Infinity Cache and grows past it.  A sharp knee between the IF-resident and
HBM-bound regimes is the signature we are looking for.

Each timed sample batches several inner iterations so that kernel-launch
overhead does not dominate the small, cache-resident cases.
"""
import json
import os
import platform
import statistics
import sys

import torch


def med_stats(xs):
    xs = sorted(xs)
    return {
        "median": statistics.median(xs),
        "mean": statistics.fmean(xs),
        "min": min(xs),
        "max": max(xs),
        "p10": xs[int(0.10 * (len(xs) - 1))],
        "p90": xs[int(0.90 * (len(xs) - 1))],
        "samples": xs,
    }


def bench_repeat(fn, logical_bytes, inner, warmup_samples=3, samples=25):
    for _ in range(warmup_samples * inner):
        fn()
    torch.cuda.synchronize()

    per_iter = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        per_iter.append(start.elapsed_time(end) / 1000.0 / inner)

    gbps = [logical_bytes / t / 1e9 for t in per_iter]
    return {
        "seconds_per_iter": med_stats(per_iter),
        "effective_gbps": med_stats(gbps),
        "inner_iters": inner,
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device is not available")
    torch.cuda.init()
    torch.cuda.synchronize()

    sizes_mib = [
        8, 16, 32, 48, 64, 80, 96, 112, 128,
        160, 192, 256, 384, 512, 768, 1024,
    ]
    samples = int(os.environ.get("SAMPLES", "25"))
    target_sample_mib = float(os.environ.get("TARGET_SAMPLE_MIB", "1024"))

    out = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch.version.hip": getattr(torch.version, "hip", None),
            "device_name": torch.cuda.get_device_name(0),
            "multi_processor_count": torch.cuda.get_device_properties(0).multi_processor_count,
        },
        "method": {
            "timer": "torch.cuda.Event",
            "samples": samples,
            "target_sample_mib": target_sample_mib,
            "logical_bytes": {
                "read_sum": "N bytes read",
                "write_zero": "N bytes written",
                "copy": "N bytes read + N bytes written",
                "triad_add": "2N bytes read + N bytes written",
            },
            "note": "inner iters choose so each timed sample moves >= target_sample_mib logical bytes",
        },
        "results": [],
    }

    for size_mib in sizes_mib:
        n = size_mib * 1024 * 1024 // 4
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        z = torch.empty_like(x)
        z.copy_(x)
        torch.cuda.synchronize()

        target_bytes = target_sample_mib * 1024 * 1024
        cases = [
            ("read_sum", lambda: torch.sum(x), n * 4),
            ("write_zero", lambda: y.zero_(), n * 4),
            ("copy", lambda: z.copy_(x), 2 * n * 4),
            ("triad_add", lambda: y.add_(x, alpha=1.0), 3 * n * 4),
        ]
        for name, fn, logical_bytes in cases:
            inner = max(1, min(512, int(round(target_bytes / logical_bytes))))
            result = bench_repeat(fn, logical_bytes, inner, samples=samples)
            result.update(
                {
                    "size_mib": size_mib,
                    "operation": name,
                    "logical_bytes_per_iter": logical_bytes,
                }
            )
            out["results"].append(result)
            print(
                f"{size_mib:5d} MiB {name:11s} "
                f"{result['effective_gbps']['median']:9.1f} GB/s  "
                f"inner={inner}",
                flush=True,
            )

        del x, y, z
        torch.cuda.empty_cache()

    out_path = os.environ.get("OUT")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
