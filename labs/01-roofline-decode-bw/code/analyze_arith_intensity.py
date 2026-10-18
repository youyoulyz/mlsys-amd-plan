#!/usr/bin/env python3
"""Analyze measured arithmetic intensity / roofline data for RX 7900 XTX.

The script deliberately separates:
  1. hardware theoretical ridge points;
  2. llama.cpp operator-level measurements;
  3. model-level end-to-end throughput converted to logical arithmetic intensity.

It does not claim that algorithmic FLOPs / stored bytes equals bytes read from HBM.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from gguf import GGUFReader
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "common"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# AMD's published RX 7900 XTX product-page values. BF16 is not separately listed
# and is treated as FP16-equivalent (the usual RDNA3 matrix-rate convention).
PEAKS = {
    "FP32": {"ops_per_s": 61.4e12, "ridge": 61.4e12 / 960e9, "basis": "FP32 vector"},
    "FP16": {"ops_per_s": 123e12, "ridge": 123e12 / 960e9, "basis": "FP16 matrix"},
    "BF16": {"ops_per_s": 123e12, "ridge": 123e12 / 960e9, "basis": "BF16 assumed FP16-equivalent; not separately listed"},
    "Q8_0": {"ops_per_s": 123e12, "ridge": 123e12 / 960e9, "basis": "official INT8 matrix rate"},
    "Q4_K": {"ops_per_s": 246e12, "ridge": 246e12 / 960e9, "basis": "official INT4 matrix rate"},
    "FP8": {"ops_per_s": None, "ridge": None, "basis": "gfx1100/RDNA3 has no native FP8 WMMA; no supported peak used"},
}
HBM_THEORETICAL_GBPS = 960.0

GGUF_PATHS = {
    "FP32": ROOT / "models/gguf/qwen2.5-7b-instruct-f32.gguf",
    "FP16": ROOT / "models/gguf/qwen2.5-7b-instruct-fp16.gguf",
    "BF16": ROOT / "models/gguf/qwen2.5-7b-instruct-bf16.gguf",
    "Q8_0": ROOT / "models/gguf/qwen2.5-7b-instruct-q8_0.gguf",
    "Q4_K_M": ROOT / "models/gguf/qwen2.5-7b-instruct-q4_k_m.gguf",
}

VLLM_MODEL_PATHS = {
    "base_bf16": ROOT / "models/hf/Qwen2.5-7B-Instruct",
    "base_fp16": ROOT / "models/hf/Qwen2.5-7B-Instruct",
    "base_fp32": ROOT / "models/hf/Qwen2.5-7B-Instruct",
    "awq_int4": ROOT / "models/hf/Qwen2.5-7B-Instruct-AWQ",
    "gptq_int4": ROOT / "models/hf/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "gptq_int8": ROOT / "models/hf/Qwen2.5-7B-Instruct-GPTQ-Int8",
    "fp8_dynamic": ROOT / "models/hf/Qwen2.5-7B-Instruct-FP8-dynamic",
}

VLLM_META = {
    "base_bf16": {"precision": "BF16", "dtype": "bfloat16", "method": "Qwen2.5-7B native BF16 checkpoint"},
    "base_fp16": {"precision": "FP16", "dtype": "float16", "method": "Qwen2.5-7B BF16 checkpoint cast to FP16"},
    "base_fp32": {"precision": "FP32", "dtype": "float32", "method": "Qwen2.5-7B BF16 checkpoint upcast to FP32"},
    "awq_int4": {"precision": "AWQ-INT4", "dtype": "float16", "method": "AWQ W4A16, Triton path"},
    "gptq_int4": {"precision": "GPTQ-INT4", "dtype": "float16", "method": "GPTQ W4A16, RDNA3W4A16LinearKernel"},
    "gptq_int8": {"precision": "GPTQ-INT8", "dtype": "float16", "method": "GPTQ Int8, ConchLinearKernel"},
    "fp8_dynamic": {"precision": "FP8", "dtype": "mixed", "method": "dynamic FP8 compressed-tensors; unsupported on gfx1100"},
}

# Qwen2.5-7B computation shape. Decode matmul FLOPs follow
# 2 * (output-projection parameters + per-layer matrix parameters).
H = 3584
INTERMEDIATE = 18944
LAYERS = 28
VOCAB = 152064
Q_HEADS = 28
KV_HEADS = 4
HEAD_DIM = 128
# Qwen2.5-7B Q/O width is 28*128=3584 (== hidden_size); KV width is 4*128=512.
# tie_word_embeddings=false, so lm_head.weight is a distinct 152064x3584 table.
LOGICAL_MATMUL_PARAMS = VOCAB * H + LAYERS * (
    2 * H * Q_HEADS * HEAD_DIM + 2 * H * KV_HEADS * HEAD_DIM + 3 * H * INTERMEDIATE
)
TOKEN_MATMUL_FLOPS = 2 * LOGICAL_MATMUL_PARAMS

TYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I64": 8, "I8": 1, "U8": 1, "F8_E4M3": 1, "BOOL": 1}


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tensor_numel(shape: Iterable[int]) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n


def gguf_stats(path: Path) -> dict[str, Any]:
    reader = GGUFReader(path)
    type_counts: Counter[str] = Counter()
    type_bytes: defaultdict[str, int] = defaultdict(int)
    total_data = 0
    category_bytes: defaultdict[str, int] = defaultdict(int)
    category_counts: Counter[str] = Counter()
    param_total = 0
    matmul_params = 0
    named: dict[str, dict[str, Any]] = {}
    for t in reader.tensors:
        typ = t.tensor_type.name
        nbytes = int(t.n_bytes)
        nparam = tensor_numel(t.shape)
        total_data += nbytes
        param_total += nparam
        type_counts[typ] += 1
        type_bytes[typ] += nbytes
        named[t.name] = {"type": typ, "shape": [int(x) for x in t.shape], "bytes": nbytes, "params": nparam}
        if t.name.startswith("blk."):
            cat = "transformer_layers"
        elif t.name == "token_embd.weight":
            cat = "token_embedding_table"
        elif t.name == "output.weight":
            cat = "output_head"
        else:
            cat = "other"
        category_bytes[cat] += nbytes
        category_counts[cat] += 1
        if t.name == "output.weight" or any(x in t.name for x in (
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
            "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        )):
            matmul_params += nparam
    per_token = total_data - category_bytes["token_embedding_table"]
    if "output.weight" not in named:
        per_token += category_bytes["token_embedding_table"]
    return {
        "file_bytes": path.stat().st_size,
        "tensor_data_bytes": total_data,
        "parameter_slots_in_file": param_total,
        "matmul_logical_params": matmul_params,
        "per_decode_token_weight_bytes": per_token,
        "transformer_layer_bytes": category_bytes["transformer_layers"],
        "token_embedding_bytes": category_bytes["token_embedding_table"],
        "output_head_bytes": category_bytes["output_head"],
        "type_counts": dict(type_counts),
        "type_bytes": dict(type_bytes),
        "bits_per_logical_parameter": 8.0 * per_token / LOGICAL_MATMUL_PARAMS,
        "tensors": named,
    }


def safetensors_stats(folder: Path) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    type_bytes: defaultdict[str, int] = defaultdict(int)
    total = 0
    names: dict[str, dict[str, Any]] = {}
    for f in sorted(folder.glob("*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as h:
            for name in h.keys():
                sl = h.get_slice(name)
                typ = sl.get_dtype()
                shape = tuple(int(x) for x in sl.get_shape())
                nbytes = tensor_numel(shape) * TYPE_BYTES[typ]
                total += nbytes
                type_counts[typ] += 1
                type_bytes[typ] += nbytes
                names[name] = {"type": typ, "shape": shape, "bytes": nbytes}
    emb_name = "model.embed_tokens.weight"
    head_name = "lm_head.weight" if "lm_head.weight" in names else emb_name
    emb_bytes = names.get(emb_name, {}).get("bytes", 0)
    # Unified weight-accounting rule (identical to analyze_vllm_concurrency.py):
    #   per_token = transformer bytes + output-projection table read once.
    # Tied checkpoints -> output projection reuses the embedding table already
    # inside `total`; untied (fp8) -> add the distinct lm_head table.
    head_bytes = names.get(head_name, {}).get("bytes", 0)
    if head_name == emb_name:
        # Tied checkpoint: the output projection reuses the embedding table that
        # is already counted inside `total`, so a full table read happens once.
        out_table = emb_bytes
        per_token = total
    else:
        # Untied checkpoint: `total` already contains the distinct lm_head table.
        # Decode reads transformer core + output projection once; the input
        # embedding is a single-row gather and is subtracted here.
        out_table = head_bytes
        per_token = total - emb_bytes
    return {
        "file_bytes": sum(f.stat().st_size for f in sorted(folder.glob("*.safetensors"))),
        "tensor_data_bytes": total,
        "type_counts": dict(type_counts),
        "type_bytes": dict(type_bytes),
        "embedding_bytes": emb_bytes,
        "output_head_bytes": out_table,
        "output_head_name": head_name,
        "per_decode_token_weight_bytes": per_token,
        "tensors": names,
    }


def parse_operator_log(path: Path, source: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    pattern = re.compile(
        r"MUL_MAT\(type_a=(?P<typ>\w+),type_b=\w+,m=(?P<m>\d+),n=(?P<n>\d+),k=(?P<k>\d+),[^\n]*\):"
        r"[^\n]*(?:\n\s+)?(?P<runs>\d+) runs\s+-\s+(?P<us>[0-9.]+) us/run\s+-\s+"
        r"(?P<flop>[0-9.]+) (?P<flop_unit>[MG])FLOP/run\s+-\s+\s*(?P<perf>[0-9.]+) (?P<unit>[GT])FLOPS"
    )
    rows: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        g = match.groupdict()
        m, n, k = int(g["m"]), int(g["n"]), int(g["k"])
        typ = g["typ"]
        if typ == "q8_0":
            block, block_bytes = 32, 34
        elif typ == "q4_K":
            block, block_bytes = 256, 144
        else:
            block, block_bytes = 1, TYPE_BYTES.get(typ.upper(), 4)
        weight_bytes = (m * k // block) * block_bytes
        flops = 2 * m * n * k
        ai = flops / weight_bytes
        tflops = float(g["perf"]) if g["unit"] == "T" else float(g["perf"]) / 1000.0
        peaks_key = {"f32": "FP32", "f16": "FP16", "bf16": "BF16", "q8_0": "Q8_0", "q4_K": "Q4_K"}[typ]
        peak = PEAKS[peaks_key]["ops_per_s"]
        rows.append({
            "source": source,
            "type": typ,
            "m": m,
            "n": n,
            "k": k,
            "us_per_run": float(g["us"]),
            "logical_mflop_per_run": float(g["flop"]) * (1000.0 if g["flop_unit"] == "G" else 1.0),
            "logical_gflops": tflops * 1000.0,
            "weight_bytes": weight_bytes,
            "weight_mib": weight_bytes / 1024**2,
            "workload_ai_ops_per_byte_weight_only": ai,
            "measured_effective_weight_gbps": weight_bytes / (float(g["us"]) * 1e-6) / 1e9,
            "reference_peak_ops_per_s": peak,
            "attained_percent_of_reference_peak": 100.0 * tflops * 1e12 / peak if peak else None,
            "bandwidth_roofline_ops_per_s_at_960GBps": min(peak, ai * HBM_THEORETICAL_GBPS * 1e9) if peak else None,
        })
    return rows

def median(values: list[float]) -> float:
    return float(statistics.median(values))


def llama_model_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gguf = {key: gguf_stats(path) for key, path in GGUF_PATHS.items()}
    rows: list[dict[str, Any]] = []
    for key, path in GGUF_PATHS.items():
        result_stem = {"FP32": "f32", "FP16": "fp16", "BF16": "bf16", "Q8_0": "q8_0", "Q4_K_M": "q4_k_m"}[key]
        st = gguf[key]
        weight_bytes = st["per_decode_token_weight_bytes"]
        # Storage-derived quantities are always available, even when the model
        # cannot be loaded. Throughput fields stay None for such cases so that no
        # measurement is fabricated.
        row: dict[str, Any] = {
            "case": key,
            "file_mib": st["file_bytes"] / 1024**2,
            "per_token_weight_mib": weight_bytes / 1024**2,
            "bits_per_logical_parameter": st["bits_per_logical_parameter"],
            "token_matmul_gflops": TOKEN_MATMUL_FLOPS / 1e9,
            "weight_only_ai_ops_per_byte": TOKEN_MATMUL_FLOPS / weight_bytes,
            "prompt_tok_s_avg_all5": None,
            "prompt_tok_s_median": None,
            "decode_tok_s_avg_all5": None,
            "decode_tok_s_median": None,
            "decode_tok_s_stddev": None,
            "effective_weight_read_gbps_from_median": None,
            "attained_logical_tflops_from_median": None,
            "status": "ok",
        }
        result_json = ROOT / "results/llamacpp" / f"{result_stem}.json"
        try:
            raw = result_json.read_text().strip()
            data = json.loads(raw)
            prompt = next(x for x in data if x["n_prompt"] == 512 and x["n_gen"] == 0)
            decode = next(x for x in data if x["n_prompt"] == 0 and x["n_gen"] == 128)
        except (OSError, ValueError, StopIteration, TypeError):
            # e.g. FP32 GGUF is 30.47 GB > 24 GiB VRAM, so llama-bench never
            # wrote a usable JSON file. Keep the storage statistics only.
            row["status"] = "failed_over_vram_no_measurement"
            rows.append(row)
            continue
        prompt_med = median(prompt["samples_ts"])
        decode_med = median(decode["samples_ts"])
        row.update({
            "prompt_tok_s_avg_all5": prompt["avg_ts"],
            "prompt_tok_s_median": prompt_med,
            "decode_tok_s_avg_all5": decode["avg_ts"],
            "decode_tok_s_median": decode_med,
            "decode_tok_s_stddev": decode["stddev_ts"],
            "effective_weight_read_gbps_from_median": weight_bytes * decode_med / 1e9,
            "attained_logical_tflops_from_median": TOKEN_MATMUL_FLOPS * decode_med / 1e12,
        })
        rows.append(row)
    metadata = {
        "model_file_bytes": {k: v["file_bytes"] for k, v in gguf.items()},
        "per_token_weight_bytes": {k: v["per_decode_token_weight_bytes"] for k, v in gguf.items()},
        "gguf_type_counts": {k: v["type_counts"] for k, v in gguf.items()},
        "gguf_type_bytes": {k: v["type_bytes"] for k, v in gguf.items()},
    }
    return rows, metadata


def vllm_storage_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for key, path in VLLM_MODEL_PATHS.items():
        if not path.exists():
            continue
        st = safetensors_stats(path)
        meta = VLLM_META[key]
        # Qwen2.5-7B is untied (separate lm_head.weight); safetensors_stats
        # already applies the unified rule (transformer + output head once).
        per_token = st["per_decode_token_weight_bytes"]
        if key == "base_fp32":
            per_token = per_token * 2
        rows.append({
            "case": key,
            "precision": meta["precision"],
            "runtime_dtype": meta["dtype"],
            "method": meta["method"],
            "checkpoint_data_mib": st["tensor_data_bytes"] / 1024**2,
            "per_token_weight_mib": per_token / 1024**2,
            "bits_per_logical_parameter": 8.0 * per_token / LOGICAL_MATMUL_PARAMS,
            "type_bytes": st["type_bytes"],
            "output_head_name": st["output_head_name"],
        })
        details[key] = {
            "path": str(path),
            "tensor_data_bytes": st["tensor_data_bytes"],
            "per_token_weight_bytes": per_token,
            "type_counts": st["type_counts"],
            "type_bytes": st["type_bytes"],
        }
    return rows, details


def vllm_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    storage, details = vllm_storage_rows()
    by_key = {x["case"]: x for x in storage}
    rows: list[dict[str, Any]] = []
    for batch, result_dir in ((1, ROOT / "results/vllm_batch1"), (8, ROOT / "results/vllm")):
        for path in sorted(result_dir.glob("*.json")):
            key = path.stem
            data = json.loads(path.read_text())
            meta = VLLM_META[key]
            st = by_key[key]
            weight_bytes = st["per_token_weight_mib"] * 1024**2
            latency = float(data["avg_latency"])
            p50 = float(data["percentiles"]["50"])
            total_tok_s = batch * 128.0 / latency
            per_request_tok_s = 128.0 / latency
            rows.append({
                "case": key,
                "batch_size": batch,
                "precision": meta["precision"],
                "runtime_dtype": meta["dtype"],
                "method": meta["method"],
                "input_len": 512,
                "output_len": 128,
                "avg_latency_s": latency,
                "p50_latency_s": p50,
                "total_output_tok_s": total_tok_s,
                "per_request_output_tok_s": per_request_tok_s,
                "per_token_weight_mib": weight_bytes / 1024**2,
                "weight_only_ai_ops_per_byte_batch_scaled": batch * TOKEN_MATMUL_FLOPS / weight_bytes,
                "effective_weight_read_gbps_equiv": weight_bytes * per_request_tok_s / 1e9,
                "attained_logical_tflops_output_equiv": TOKEN_MATMUL_FLOPS * total_tok_s / 1e12,
                "status": "ok",
            })
    for batch, result_dir in ((1, ROOT / "results/vllm_batch1"), (8, ROOT / "results/vllm")):
        for path in sorted(result_dir.glob("*.rc")):
            key = path.stem
            if (result_dir / f"{key}.json").exists():
                continue
            meta = VLLM_META.get(key, {"precision": key, "dtype": "unknown", "method": ""})
            st = by_key.get(key)
            weight_bytes = st["per_token_weight_mib"] * 1024**2 if st else None
            rows.append({
                "case": key,
                "batch_size": batch,
                "precision": meta["precision"],
                "runtime_dtype": meta["dtype"],
                "method": meta["method"],
                "input_len": 512,
                "output_len": 128,
                "avg_latency_s": None,
                "p50_latency_s": None,
                "total_output_tok_s": None,
                "per_request_output_tok_s": None,
                "per_token_weight_mib": (weight_bytes / 1024**2) if weight_bytes else None,
                "weight_only_ai_ops_per_byte_batch_scaled": (batch * TOKEN_MATMUL_FLOPS / weight_bytes) if weight_bytes else None,
                "effective_weight_read_gbps_equiv": None,
                "attained_logical_tflops_output_equiv": None,
                "status": f"failed_rc={path.read_text().strip()}",
            })
    return rows, [{"case": k, **v} for k, v in details.items()]


def hbm_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads((ROOT / "results/common/hbm_bandwidth.json").read_text())
    rows: list[dict[str, Any]] = []
    for r in data["results"]:
        rows.append({
            "size_mib": r["size_mib"],
            "operation": r["operation"],
            "logical_bytes": r["logical_bytes"],
            "median_gbps": r["effective_gbps"]["median"],
            "mean_gbps": r["effective_gbps"]["mean"],
            "p10_gbps": r["effective_gbps"]["p10"],
            "p90_gbps": r["effective_gbps"]["p90"],
        })
    return rows, data


def make_plots(operator_rows: list[dict[str, Any]], llama_rows: list[dict[str, Any]], vu_rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    styles = {"f32": ("tab:blue", "F32"), "f16": ("tab:orange", "F16"), "bf16": ("tab:green", "BF16"), "q8_0": ("tab:red", "Q8_0"), "q4_K": ("tab:purple", "Q4_K")}
    for ax, n, title in ((axes[0], 1, "n=1: weight-bandwidth dominated"), (axes[1], 512, "n=512: compute oriented")):
        rows = [r for r in operator_rows if r["n"] == n]
        for r in rows:
            c, label = styles[r["type"]]
            ax.scatter(r["workload_ai_ops_per_byte_weight_only"], r["logical_gflops"] / 1000, color=c, s=48, label=label, zorder=4)
            ax.annotate(label, (r["workload_ai_ops_per_byte_weight_only"], r["logical_gflops"] / 1000), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("logical arithmetic intensity (ops / weight byte)")
        ax.set_ylabel("measured logical TFLOPS")
        ax.set_title(title)
    x = np.logspace(-0.5, 0.7, 100)
    axes[0].plot(x, np.minimum(960 * x, 3000) / 1000, "k--", linewidth=1, label="960 GB/s roof")
    peak_lines = []
    for peak, label in ((61.4, "FP32 peak"), (123, "FP16/BF16/INT8 peak"), (246, "INT4 peak")):
        peak_lines.append(axes[1].axhline(peak, linestyle="--", linewidth=1, alpha=0.55, label=label))
        axes[1].text(210, peak * 1.03, label, fontsize=8)
    axes[0].set_xlim(0.3, 5.0)
    axes[0].set_ylim(0.2, 3.5)
    axes[1].set_xlim(30, 2500)
    axes[1].set_ylim(10, 300)
    axes[0].legend(loc="lower right", fontsize=8)
    from matplotlib.lines import Line2D
    type_handles = [Line2D([0], [0], marker="o", linestyle="", color=c, label=label) for c, label in styles.values()]
    axes[1].legend(
        handles=type_handles + peak_lines,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        fontsize=8,
    )
    fig.savefig(FIG / "operator_roofline.png", dpi=170)
    plt.close(fig)

    # Model-level algorithmic AI vs reference ridge points. Ridge values are
    # precision-path reference points, not measured HBM bytes.
    labels: list[str] = []
    ais: list[float] = []
    ridges: list[float] = []
    colors: list[str] = []
    for r in llama_rows:
        labels.append("llama " + r["case"])
        ais.append(r["weight_only_ai_ops_per_byte"])
        ridge = {"FP32": 64.0, "FP16": 128.0, "BF16": 128.0, "Q8_0": 128.0, "Q4_K_M": 256.0}[r["case"]]
        ridges.append(ridge)
        colors.append("tab:blue")
    for r in vu_rows:
        if r["status"] != "ok":
            continue
        labels.append(f"vLLM b{r['batch_size']} " + r["case"])
        ais.append(r["weight_only_ai_ops_per_byte_batch_scaled"])
        p = r["precision"]
        if p == "FP32":
            ridge = 64.0
        elif p in {"FP16", "BF16", "GPTQ-INT8"}:
            ridge = 128.0
        elif p in {"AWQ-INT4", "GPTQ-INT4"}:
            ridge = 256.0
        else:
            ridge = math.nan
        ridges.append(ridge)
        colors.append("tab:red" if r["batch_size"] == 1 else "tab:green")
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 5.4))
    ax.bar(x, ais, color=colors, alpha=0.75)
    ridge_artist = ax.scatter(x, ridges, marker="x", s=55, color="black", label="theoretical ridge (native-path reference)")
    ax.set_yscale("log")
    ax.set_ylabel("ops / weight byte (log scale)")
    ax.set_title("Model-level weight-only arithmetic intensity vs theoretical roofline ridge")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.25)
    from matplotlib.patches import Patch
    handles = [
        Patch(color="tab:blue", alpha=0.75, label="llama.cpp model end-to-end"),
        Patch(color="tab:red", alpha=0.75, label="vLLM batch=1"),
        Patch(color="tab:green", alpha=0.75, label="vLLM batch=8"),
        ridge_artist,
    ]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "model_ai_vs_ridge.png", dpi=170)
    plt.close(fig)


def main() -> None:
    hbm, hbm_raw = hbm_rows()
    operator = parse_operator_log(ROOT / "logs/llamacpp/test-backend-ops-n1-console.txt", "llama.cpp n=1") + \
        parse_operator_log(ROOT / "logs/llamacpp/test-backend-ops-n512-console.txt", "llama.cpp n=512")
    llama_rows, llama_meta = llama_model_rows()
    vstorage, vdetails = vllm_storage_rows()
    vrows, vdetails2 = vllm_rows()
    make_plots(operator, llama_rows, vrows)

    theoretical_rows = []
    for label, p in PEAKS.items():
        theoretical_rows.append({
            "precision": label,
            "peak_ops_per_s": p["ops_per_s"],
            "peak_tflops_or_tops": p["ops_per_s"] / 1e12 if p["ops_per_s"] else None,
            "theoretical_hbm_gbps": HBM_THEORETICAL_GBPS,
            "ridge_point_ops_per_byte": p["ridge"],
            "basis": p["basis"],
        })

    csv_write(OUT / "theoretical_roofline.csv", theoretical_rows)
    csv_write(OUT / "hbm_bandwidth.csv", hbm)
    csv_write(OUT / "operator_perf.csv", operator)
    csv_write(OUT / "llamacpp_model.csv", llama_rows)
    csv_write(OUT / "vllm_model_storage.csv", vstorage)
    csv_write(OUT / "vllm_latency.csv", vrows)
    with (OUT / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "environment": {
                "gpu": "AMD Radeon RX 7900 XTX",
                "arch": "gfx1100",
                "rocm": "ROCm 10",
                "hbm_theoretical_gbps": HBM_THEORETICAL_GBPS,
                "llama_cpp": "/opt/llamacpp/llama.cpp/build-gfx1100-rocm10 commit 1bc7a5af0 build 10973",
                "vllm": "0.27.1.dev5+gf46a9dfe2.d20260827; torch 2.12.0+rocm10.0.0; HIP 7.15.26333",
            },
            "logical_model": {
                "hidden_size": H,
                "intermediate_size": INTERMEDIATE,
                "layers": LAYERS,
                "vocab_size": VOCAB,
                "q_heads": Q_HEADS,
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "logical_matmul_params": LOGICAL_MATMUL_PARAMS,
                "token_matmul_flops": TOKEN_MATMUL_FLOPS,
            },
            "peaks": PEAKS,
            "gguf_storage": llama_meta,
            "vllm_storage": vdetails2,
            "hbm_raw_environment": hbm_raw["environment"],
        }, f, indent=2, ensure_ascii=False)

    print("wrote", OUT)
    print("llama rows:")
    for r in llama_rows:
        if r["status"] == "ok":
            print(r["case"], "AI", f"{r['weight_only_ai_ops_per_byte']:.3f}", "decode tok/s median", f"{r['decode_tok_s_median']:.2f}")
        else:
            print(r["case"], "AI", f"{r['weight_only_ai_ops_per_byte']:.3f}", r["status"])
    print("vllm rows:")
    for r in vrows:
        if r["status"] == "ok":
            print(r["case"], "batch", r["batch_size"], "AI", f"{r['weight_only_ai_ops_per_byte_batch_scaled']:.3f}", "latency", f"{r['avg_latency_s']:.3f}")
        else:
            print(r["case"], r["status"])


if __name__ == "__main__":
    main()
