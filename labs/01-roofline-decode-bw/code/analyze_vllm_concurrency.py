#!/usr/bin/env python3
"""Analyze the vLLM static-batch concurrency sweep for Qwen2.5-7B.

Same methodology as the 4B / 0.5B experiments, but with a larger model so the
model-level weight-read arithmetic intensity is no longer dominated by tiny
fixed overheads.

Metric (identical definition to the 4B / 0.5B runs):

    total_output_tok_s         = N * output_len / avg_latency
    per_request_tok_s           = output_len / avg_latency
    weight_only_FLOP_per_Byte   = N * TOKEN_MATMUL_FLOPS / per_token_weight_bytes
    weight_only_Byte_per_FLOP   = 1 / weight_only_FLOP_per_Byte
    effective_weight_read_gbps  = per_token_weight_bytes * per_request_tok_s / 1e9
    attained_logical_tflops     = TOKEN_MATMUL_FLOPS * total_output_tok_s / 1e12
    throughput_scaling_vs_b1    = total_output_tok_s(N) / total_output_tok_s(1)
    batch_efficiency_vs_b1      = throughput_scaling_vs_b1 / N

This is a *logical weight-read* ratio (one weight read serves the whole batch in
a decode step); it is NOT a DRAM counter measurement.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "vllm_concurrency"
OUT = RES
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- Qwen2.5-7B architecture ----
H = 3584
INTERMEDIATE = 18944
LAYERS = 28
VOCAB = 152064
Q_HEADS = 28
KV_HEADS = 4
HEAD_DIM = 128
# Qwen2.5-7B Q/O width is 28*128=3584 (== H); KV width 4*128=512.
# tie_word_embeddings=false -> separate lm_head.weight table.
LOGICAL_MATMUL_PARAMS = VOCAB * H + LAYERS * (
    2 * H * Q_HEADS * HEAD_DIM + 2 * H * KV_HEADS * HEAD_DIM + 3 * H * INTERMEDIATE
)
TOKEN_MATMUL_FLOPS = 2 * LOGICAL_MATMUL_PARAMS

HBM_THEORETICAL_GBPS = 960.0
PEAKS = {
    "FP16": {"ops_per_s": 123e12, "ridge": 123e12 / 960e9},
    "Q8_0": {"ops_per_s": 123e12, "ridge": 123e12 / 960e9},
    "Q4_K": {"ops_per_s": 246e12, "ridge": 246e12 / 960e9},
    "FP8": {"ops_per_s": None, "ridge": None},
}
RIDGE = {
    "base_fp16": PEAKS["FP16"]["ridge"],
    "gptq_int4": PEAKS["Q4_K"]["ridge"],
    "awq_int4": PEAKS["Q4_K"]["ridge"],
    "gptq_int8": PEAKS["Q8_0"]["ridge"],
    "fp8_dynamic": None,
}
PEAK_KEY = {"base_fp16": "FP16", "gptq_int4": "Q4_K", "awq_int4": "Q4_K",
            "gptq_int8": "Q8_0", "fp8_dynamic": "FP8"}

INPUT_LEN = 512
OUTPUT_LEN = 128
BATCHES = [1, 2, 4, 8, 16]
CASE_ORDER = ["base_fp16", "gptq_int4", "awq_int4", "gptq_int8", "fp8_dynamic"]

TYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I64": 8, "I8": 1,
              "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1}

MODEL_DIR = ROOT / "models/hf"
MODELS = {
    "base_fp16": MODEL_DIR / "Qwen2.5-7B-Instruct",
    "awq_int4": MODEL_DIR / "Qwen2.5-7B-Instruct-AWQ",
    "gptq_int4": MODEL_DIR / "Qwen2.5-7B-Instruct-GPTQ-Int4",
    "gptq_int8": MODEL_DIR / "Qwen2.5-7B-Instruct-GPTQ-Int8",
    "fp8_dynamic": MODEL_DIR / "Qwen2.5-7B-Instruct-FP8-dynamic",
}
META = {
    "base_fp16": {"precision": "FP16", "dtype": "float16", "method": "Qwen2.5-7B BF16 checkpoint cast to FP16"},
    "awq_int4": {"precision": "AWQ-INT4", "dtype": "float16", "method": "AWQ W4A16"},
    "gptq_int4": {"precision": "GPTQ-INT4", "dtype": "float16", "method": "GPTQ W4A16"},
    "gptq_int8": {"precision": "GPTQ-INT8", "dtype": "float16", "method": "GPTQ Int8 weight-only"},
    "fp8_dynamic": {"precision": "FP8", "dtype": "mixed", "method": "dynamic FP8 (W8A8); unsupported on gfx1100"},
}


def _numel(shape):
    n = 1
    for x in shape:
        n *= int(x)
    return n


def safetensors_stats(folder: Path) -> dict:
    total = 0
    type_bytes = {}
    type_counts = {}
    names = {}
    for f in sorted(folder.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for k in h.keys():
                sl = h.get_slice(k)
                typ = sl.get_dtype()
                shape = tuple(int(x) for x in sl.get_shape())
                nb = _numel(shape) * TYPE_BYTES[typ]
                total += nb
                type_bytes[typ] = type_bytes.get(typ, 0) + nb
                type_counts[typ] = type_counts.get(typ, 0) + 1
                names[k] = nb
    emb = names.get("model.embed_tokens.weight", 0)
    head = "lm_head.weight" if "lm_head.weight" in names else "model.embed_tokens.weight"
    # Unified rule (identical to the corrected 0.5B accounting):
    #   per_token = transformer bytes + the output-projection table read once.
    # The input embedding lookup only touches a single row and is negligible.
    # When the output projection reuses the tied embedding table, the read is
    # already inside `total`; when a distinct lm_head.weight exists we add it.
    if head == "model.embed_tokens.weight":
        # Tied checkpoint: output projection reuses the in-`total` embedding table.
        out_table = emb
        per_token = total
    else:
        # Untied: `total` already holds the distinct lm_head table; subtract only
        # the input-embedding table (a single-row gather at decode).
        out_table = names[head]
        per_token = total - emb
    return {"total": total, "type_bytes": type_bytes, "type_counts": type_counts,
            "embedding_bytes": emb, "head": head, "out_table": out_table,
            "per_token": per_token}


def per_token_weight_bytes() -> dict:
    out = {}
    for key, folder in MODELS.items():
        if not folder.exists():
            continue
        out[key] = safetensors_stats(folder)["per_token"]
    return out


def load_rows(weights: dict) -> list[dict]:
    rows = []
    for case in CASE_ORDER:
        if case not in weights:
            continue
        meta = META[case]
        wbytes = weights[case]
        for batch in BATCHES:
            tag = f"{case}_b{batch}"
            jpath = RES / f"{tag}.json"
            rcpath = RES / f"{tag}.rc"
            row = {"case": case, "precision": meta["precision"], "runtime_dtype": meta["dtype"],
                   "method": meta["method"], "batch_size": batch, "input_len": INPUT_LEN,
                   "output_len": OUTPUT_LEN, "per_token_weight_mib": wbytes / 1024**2,
                   "theoretical_ridge_ops_per_byte": RIDGE.get(case)}
            if jpath.exists():
                data = json.loads(jpath.read_text())
                lat = float(data["avg_latency"])
                lats = data.get("latencies", [])
                total_tok_s = batch * OUTPUT_LEN / lat
                per_req = OUTPUT_LEN / lat
                ai = batch * TOKEN_MATMUL_FLOPS / wbytes
                pk = PEAKS[PEAK_KEY[case]]["ops_per_s"]
                row.update({
                    "avg_latency_s": lat,
                    "p50_latency_s": float(data["percentiles"]["50"]),
                    "min_latency_s": min(lats) if lats else None,
                    "max_latency_s": max(lats) if lats else None,
                    "total_output_tok_s": total_tok_s,
                    "per_request_output_tok_s": per_req,
                    "weight_only_ai_ops_per_byte_batch_scaled": ai,
                    "weight_only_bytes_per_flop": 1.0 / ai,
                    "effective_weight_read_gbps_equiv": wbytes * per_req / 1e9,
                    "attained_logical_tflops_output_equiv": TOKEN_MATMUL_FLOPS * total_tok_s / 1e12,
                    "bandwidth_roofline_gflops_at_960": (min(pk, ai * HBM_THEORETICAL_GBPS * 1e9) / 1e9) if pk else None,
                    "status": "ok",
                })
            else:
                rc = rcpath.read_text().strip() if rcpath.exists() else "missing"
                row.update({k: None for k in (
                    "avg_latency_s", "p50_latency_s", "min_latency_s", "max_latency_s",
                    "total_output_tok_s", "per_request_output_tok_s",
                    "weight_only_ai_ops_per_byte_batch_scaled", "weight_only_bytes_per_flop",
                    "effective_weight_read_gbps_equiv", "attained_logical_tflops_output_equiv",
                    "bandwidth_roofline_gflops_at_960")})
                row["status"] = f"failed_rc={rc}"
            rows.append(row)
    return rows


def add_scaling(rows):
    base = {r["case"]: r["total_output_tok_s"] for r in rows
            if r["status"] == "ok" and r["batch_size"] == 1}
    for r in rows:
        if r["status"] != "ok" or not base.get(r["case"]):
            r["throughput_scaling_vs_b1"] = None
            r["batch_efficiency_vs_b1"] = None
            continue
        sc = r["total_output_tok_s"] / base[r["case"]]
        r["throughput_scaling_vs_b1"] = sc
        r["batch_efficiency_vs_b1"] = sc / r["batch_size"]


def make_plots(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    cases = [c for c in CASE_ORDER if any(r["case"] == c and r["status"] == "ok" for r in rows)]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for c in cases:
        pts = sorted([r for r in ok if r["case"] == c], key=lambda r: r["batch_size"])
        ax.plot([r["batch_size"] for r in pts],
                [r["weight_only_ai_ops_per_byte_batch_scaled"] for r in pts],
                marker="o", label=META[c]["precision"])
    for key, col in (("FP16", "tab:blue"), ("Q8_0", "tab:green"), ("Q4_K", "tab:red")):
        ax.axhline(PEAKS[key]["ridge"], ls="--", lw=1.0, color=col, alpha=0.6,
                   label=f"{key} ridge = {PEAKS[key]['ridge']:.1f} OP/B")
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCHES); ax.set_xticklabels([str(b) for b in BATCHES])
    ax.set_xlabel("static batch size N (concurrency)")
    ax.set_ylabel("logical weight-only FLOP/Byte (batch scaled)")
    ax.set_title("Qwen2.5-7B weight-only arithmetic intensity vs concurrency (RX 7900 XTX)")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "concurrency_ai_vs_batch.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for c in cases:
        pts = sorted([r for r in ok if r["case"] == c], key=lambda r: r["batch_size"])
        ax.plot([r["batch_size"] for r in pts], [r["total_output_tok_s"] for r in pts],
                marker="o", label=META[c]["precision"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCHES); ax.set_xticklabels([str(b) for b in BATCHES])
    ax.set_xlabel("static batch size N (concurrency)")
    ax.set_ylabel("total output tok/s")
    ax.set_title("Qwen2.5-7B aggregate decode throughput vs concurrency")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "concurrency_throughput_vs_batch.png", dpi=170); plt.close(fig)

    # comparison vs the previous Qwen3-4B run
    small_csv = ROOT.parent / "arith_intensity_4b_7900xtx_20260923/results/vllm_concurrency/vllm_concurrency.csv"
    if small_csv.exists():
        small = list(csv.DictReader(open(small_csv)))
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        pan = {"base_fp16": "FP16", "gptq_int4": "GPTQ-INT4", "awq_int4": "AWQ-INT4", "gptq_int8": "GPTQ-INT8"}
        for c, lab in pan.items():
            big = sorted([r for r in ok if r["case"] == c], key=lambda r: r["batch_size"])
            if big:
                ax.plot([r["batch_size"] for r in big],
                        [r["weight_only_ai_ops_per_byte_batch_scaled"] for r in big],
                        marker="o", ls="-", label=f"Qwen2.5-7B {lab}")
            sm = sorted([r for r in small if r["case"] == c and r["status"] == "ok"],
                        key=lambda r: int(r["batch_size"]))
            if sm:
                ax.plot([int(r["batch_size"]) for r in sm],
                        [float(r["weight_only_ai_ops_per_byte_batch_scaled"]) for r in sm],
                        marker="s", ls="--", alpha=0.5, label=f"Qwen3-4B {lab}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCHES); ax.set_xticklabels([str(b) for b in BATCHES])
        ax.set_xlabel("static batch size N (concurrency)")
        ax.set_ylabel("logical weight-only FLOP/Byte (batch scaled)")
        ax.set_title("Weight-only arithmetic intensity: Qwen2.5-7B vs Qwen3-4B")
        ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(FIG / "concurrency_ai_7b_vs_4b.png", dpi=170); plt.close(fig)

    # theoretical DRAM roofline ceiling vs measured (effective weight-read bandwidth)
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    labels = [META[c]["precision"] for c in cases]
    x = np.arange(len(cases)); wdt = 0.36
    # directly comparable unit: effective weight-read GB/s vs the 960 GB/s roof
    meas_gbps = []
    meas_batch = []
    for c in cases:
        pts = sorted([r for r in ok if r["case"] == c], key=lambda r: r["batch_size"])
        best = max(pts, key=lambda r: r["effective_weight_read_gbps_equiv"])
        meas_gbps.append(best["effective_weight_read_gbps_equiv"])
        meas_batch.append(best["batch_size"])
    ax.bar(x - wdt / 2, [HBM_THEORETICAL_GBPS] * len(cases), wdt,
           label="theoretical HBM roof @960 GB/s", color="tab:gray", alpha=0.5)
    ax.bar(x + wdt / 2, meas_gbps, wdt,
           label="measured peak effective weight-read", color="tab:blue")
    for xi, v, nb in zip(x + wdt / 2, meas_gbps, meas_batch):
        ax.text(xi, v + 12, f"{v:.0f} @N={nb}\n({100*v/HBM_THEORETICAL_GBPS:.0f}%)",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("effective weight-read bandwidth (GB/s)")
    ax.set_title("Theoretical HBM roof vs measured effective weight-read bandwidth")
    ax.grid(True, axis="y", alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "concurrency_theory_vs_measured.png", dpi=170); plt.close(fig)


def main():
    weights = per_token_weight_bytes()
    rows = load_rows(weights)
    add_scaling(rows)

    fields = ["case", "precision", "runtime_dtype", "method", "batch_size", "input_len",
              "output_len", "per_token_weight_mib", "avg_latency_s", "p50_latency_s",
              "min_latency_s", "max_latency_s", "total_output_tok_s", "per_request_output_tok_s",
              "weight_only_ai_ops_per_byte_batch_scaled", "weight_only_bytes_per_flop",
              "effective_weight_read_gbps_equiv", "attained_logical_tflops_output_equiv",
              "bandwidth_roofline_gflops_at_960", "theoretical_ridge_ops_per_byte",
              "throughput_scaling_vs_b1", "batch_efficiency_vs_b1", "status"]
    with (OUT / "vllm_concurrency.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    summary = {
        "environment": {"gpu": "AMD Radeon RX 7900 XTX", "arch": "gfx1100",
                        "hbm_theoretical_gbps": HBM_THEORETICAL_GBPS,
                        "input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
                        "num_iters": 5, "num_iters_warmup": 2, "concurrency": BATCHES,
                        "definition": "static batch via vllm bench latency --batch-size N; "
                                      "weight-only logical arithmetic intensity"},
        "logical_model": {"hidden_size": H, "intermediate_size": INTERMEDIATE, "layers": LAYERS,
                          "vocab_size": VOCAB, "q_heads": Q_HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
                          "logical_matmul_params": LOGICAL_MATMUL_PARAMS,
                          "token_matmul_flops": TOKEN_MATMUL_FLOPS},
        "ridges": {k: PEAKS[k]["ridge"] for k in ("FP16", "Q8_0", "Q4_K")},
        "per_token_weight_bytes": weights,
        "rows": rows,
    }
    (OUT / "vllm_concurrency_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    make_plots(rows)

    print("wrote", OUT)
    print(f"LOGICAL_MATMUL_PARAMS={LOGICAL_MATMUL_PARAMS}  TOKEN_MATMUL_FLOPS={TOKEN_MATMUL_FLOPS}")
    print(f"{'case':<12}{'N':>4}{'lat_s':>9}{'tot_tok/s':>11}{'AI op/B':>10}{'B/FLOP':>10}{'scale':>7}{'eff':>7}  status")
    for r in sorted(rows, key=lambda r: (CASE_ORDER.index(r["case"]), r["batch_size"])):
        if r["status"] == "ok":
            print(f"{r['case']:<12}{r['batch_size']:>4}{r['avg_latency_s']:>9.4f}"
                  f"{r['total_output_tok_s']:>11.2f}{r['weight_only_ai_ops_per_byte_batch_scaled']:>10.3f}"
                  f"{r['weight_only_bytes_per_flop']:>10.4f}"
                  f"{(r['throughput_scaling_vs_b1'] or float('nan')):>7.3f}"
                  f"{(r['batch_efficiency_vs_b1'] or float('nan')):>7.3f}  ok")
        else:
            print(f"{r['case']:<12}{r['batch_size']:>4}{'':>9}{'':>11}{'':>10}{'':>10}{'':>7}{'':>7}  {r['status']}")


if __name__ == "__main__":
    main()
