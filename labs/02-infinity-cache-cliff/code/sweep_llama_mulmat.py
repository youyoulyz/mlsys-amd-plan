#!/usr/bin/env python3
"""Sweep MUL_MAT *weight size* with the real llama.cpp ROCm kernels.

Decode-style GEMV: m (output rows) is fixed so kernel parallelism is constant,
k (reduction dim) is scaled so the weight footprint |W| = m*k*bpe walks across
the 96 MiB Infinity Cache boundary.  The same weight buffer is replayed across
runs, so whether |W| stays resident is purely a function of its size.

Runs against a locally patched test-backend-ops that adds the sweep shapes
(see scripts/test-backend-ops.cpp.orig for the upstream file).
"""
import json
import os
import platform
import re
import subprocess
import sys

BIN = os.environ.get(
    "LLAMA_TEST", "/opt/llamacpp/llama.cpp/build-gfx1100-rocm10/bin/test-backend-ops")
M = int(os.environ.get("M", "4096"))
BACKEND = os.environ.get("BACKEND", "ROCm0")

BPE = {  # bytes per element for type_a (the weight)
    "f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 1.0625, "q4_K": 0.5625,
}
SIZES_MIB = [int(x) for x in os.environ.get(
    "SIZES", "8,16,32,48,64,80,96,112,128,160,192,256,384,512").split(",")]
TYPES = os.environ.get("TYPES", "f16,q8_0,q4_K,f32").split(",")
NTOKS = [int(x) for x in os.environ.get("NTOKS", "1,16").split(",")]

LINE = re.compile(
    r"MUL_MAT\(type_a=(?P<ta>[^,)]+),type_b=[^,]+"
    r"[,)][^)]*?m=(?P<m>\d+),n=(?P<n>\d+),k=(?P<k>\d+)[^)]*\):\s*"
    r"(?P<runs>\d+) runs\s+-\s+(?P<us>[\d.]+) us/run")


def k_for(size_mib, bpe):
    k = int(round(size_mib * 1024 * 1024 / (M * bpe) / 256.0)) * 256
    return max(256, k)


def run_case(type_a, k, n):
    param = f"type_a={type_a},type_b=f32,m={M},n={n},k={k}"
    p = subprocess.run([BIN, "perf", "-b", BACKEND, "-o", "MUL_MAT", "-p", param],
                       capture_output=True, text=True, timeout=1800)
    txt = (p.stdout or "") + (p.stderr or "")
    for line in txt.splitlines():
        mo = LINE.search(line)
        if mo:
            return {"ok": True, "runs": int(mo["runs"]), "us_per_run": float(mo["us"]),
                    "us_raw": mo["us"]}
    return {"ok": False, "raw": txt[-400:]}


def main():
    out = {
        "environment": {"platform": platform.platform(),
                        "binary": BIN, "m": M, "backend": BACKEND},
        "method": {
            "note": "weight buffer replayed across runs -> steady-state streaming; "
                    "|W| = m*k*bpe determines IF residency",
            "bytes_per_element": BPE,
        },
        "results": [],
    }
    for type_a in TYPES:
        bpe = BPE[type_a]
        for size_mib in SIZES_MIB:
            k = k_for(size_mib, bpe)
            wb = M * k * bpe
            for n in NTOKS:
                r = run_case(type_a, k, n)
                rec = {"type_a": type_a, "target_size_mib": size_mib, "m": M, "k": k,
                       "n": n, "weight_bytes": wb, "weight_mib": wb / 1024 / 1024}
                if r["ok"]:
                    rec["us_per_run"] = r["us_per_run"]
                    rec["weight_gbps"] = wb / (r["us_per_run"] * 1e-6) / 1e9
                    rec["tflops"] = 2 * M * n * k / (r["us_per_run"] * 1e-6) / 1e12
                    print(f"{type_a:5s} n={n:2d} {rec['weight_mib']:7.1f} MiB "
                          f"k={k:7d} {r['us_per_run']:9.2f} us "
                          f"{rec['weight_gbps']:8.1f} GB/s(wt) {rec['tflops']:6.2f} TF",
                          flush=True)
                else:
                    rec["error"] = r["raw"]
                    print(f"{type_a:5s} n={n:2d} {rec['weight_mib']:7.1f} MiB FAILED",
                          flush=True)
                out["results"].append(rec)
    out_path = os.environ.get("OUT")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
