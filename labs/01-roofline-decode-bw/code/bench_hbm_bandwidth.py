#!/usr/bin/env python3
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


def bench(fn, logical_bytes, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)

    gbps = [logical_bytes / t / 1e9 for t in times]
    return {"seconds": med_stats(times), "effective_gbps": med_stats(gbps)}


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device is not available")
    torch.cuda.init()
    torch.cuda.synchronize()

    sizes_mib = [256, 512, 1024]
    reps = int(os.environ.get("REPS", "30"))
    out = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch.version.hip": getattr(torch.version, "hip", None),
            "torch.version.cuda": getattr(torch.version, "cuda", None),
            "device_name": torch.cuda.get_device_name(0),
            "device_properties": str(torch.cuda.get_device_properties(0)),
            "multi_processor_count": torch.cuda.get_device_properties(0).multi_processor_count,
        },
        "method": {
            "timer": "torch.cuda.Event",
            "warmup": 5,
            "repetitions": reps,
            "logical_bytes": {
                "read_sum": "N bytes read",
                "write_zero": "N bytes written",
                "copy": "N bytes read + N bytes written",
                "triad_add": "2N bytes read + N bytes written",
            },
        },
        "results": [],
    }

    for size_mib in sizes_mib:
        n = size_mib * 1024 * 1024 // 4
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        z = torch.empty_like(x)

        # Form one persistent operation graph / memory mapping before timing.
        z.copy_(x)
        torch.cuda.synchronize()

        cases = [
            ("read_sum", lambda: torch.sum(x), n * 4),
            ("write_zero", lambda: y.zero_(), n * 4),
            ("copy", lambda: z.copy_(x), 2 * n * 4),
            ("triad_add", lambda: y.add_(x, alpha=1.0), 3 * n * 4),
        ]
        for name, fn, logical_bytes in cases:
            result = bench(fn, logical_bytes, warmup=5, reps=reps)
            result.update(
                {
                    "size_mib": size_mib,
                    "operation": name,
                    "logical_bytes": logical_bytes,
                }
            )
            out["results"].append(result)
            print(json.dumps(result), flush=True)

        del x, y, z
        torch.cuda.empty_cache()

    out_path = os.environ.get("OUT")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
