#!/usr/bin/env python3
"""Cross-check the published Qwen2.5-7B reports against their CSV sources.

Expected values are read from the analysis CSV/JSON files at runtime.  The
report tables are parsed from Markdown, so this checker catches accidental
copy/paste drift rather than maintaining a second hard-coded result table.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
COMMON = RES / "common"
CONC = RES / "vllm_concurrency"
REPORT = ROOT / "REPORT.md"
CONC_REPORT = ROOT / "VLLM_CONCURRENCY_REPORT.md"

# The archived copies live next to the other reports in the repository.
ARCHIVE_CONC_REPORT = (
    ROOT.parents[1]
    / "results/reports/VLLM_CONCURRENCY_QWEN2.5_7B_7900XTX_20260924.md"
)
ARCHIVE_MAIN_REPORT = (
    ROOT.parents[1]
    / "results/reports/ARITH_INTENSITY_QWEN2.5_7B_7900XTX_20260924.md"
)

# Historical comparison sources.  They are read-only inputs to the 7B report.
SMALL_CONC = (
    ROOT.parent / "arith_intensity_7900xtx_20260923"
    / "results/vllm_concurrency/vllm_concurrency.csv"
)
MID_CONC = (
    ROOT.parent / "arith_intensity_4b_7900xtx_20260923"
    / "results/vllm_concurrency/vllm_concurrency.csv"
)
SMALL_SUMMARY = (
    ROOT.parent / "arith_intensity_7900xtx_20260923"
    / "results/vllm_concurrency/vllm_concurrency_summary.json"
)
MID_SUMMARY = (
    ROOT.parent / "arith_intensity_4b_7900xtx_20260923"
    / "results/vllm_concurrency/vllm_concurrency_summary.json"
)

failures: list[str] = []
n_checks = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global n_checks
    n_checks += 1
    if condition:
        print(f"PASS  {name}")
    else:
        failures.append(name)
        print(f"FAIL  {name}  {detail}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean(s: str) -> str:
    return s.strip().strip("`").replace("**", "").strip()


def number(s: str) -> float | None:
    if s is None:
        return None
    t = clean(s).replace(",", "")
    if t in {"", "-", "—", "失败", "不适用", "N/A", "n/a"}:
        return None
    # Do not interpret prose such as "失败（OOM）" as a measurement.
    if "失败" in t:
        return None
    m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", t)
    return float(m.group(0)) if m else None


def numbers(s: str) -> list[float]:
    return [
        float(x)
        for x in re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", clean(s).replace(",", "")
        )
    ]


def close(got: float | None, want: float, tol: float) -> bool:
    return got is not None and abs(got - want) <= tol


def check_value(name: str, got: float | None, want: float | None, tol: float) -> None:
    if want is None:
        check(name, got is None, f"got={got!r} want=None tol={tol!r}")
    else:
        check(
            name,
            close(got, want, tol),
            f"got={got!r} want={want!r} tol={tol!r}",
        )


def markdown_tables(text: str):
    lines = text.splitlines()
    i = 0
    while i + 1 < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        header = [clean(x) for x in lines[i].strip().strip("|").split("|")]
        sep = [clean(x) for x in lines[i + 1].strip().strip("|").split("|")]
        is_sep = len(sep) == len(header) and all(
            re.fullmatch(r":?-{3,}:?", x.replace(" ", "")) for x in sep
        )
        if not is_sep:
            i += 1
            continue
        rows: list[list[str]] = []
        j = i + 2
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            rows.append([clean(x) for x in lines[j].strip().strip("|").split("|")])
            j += 1
        yield header, rows
        i = j


def table_after(text: str, marker: str, required: list[str]):
    if marker not in text:
        raise KeyError(f"marker not found: {marker}")
    segment = text.split(marker, 1)[1]
    for header, rows in markdown_tables(segment):
        if all(any(req in c for c in header) for req in required):
            return header, rows
    raise KeyError(f"table not found after {marker!r} requiring {required!r}")


def col_index(header: list[str], wanted: str) -> int:
    for i, c in enumerate(header):
        if c == wanted:
            return i
    raise KeyError(f"column {wanted!r} not in {header!r}")


def row_map(rows: list[list[str]]) -> dict[str, list[str]]:
    return {clean(r[0]): r for r in rows if r}


def field(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else None


def expected_or_none(row: dict[str, str], key: str) -> float | None:
    return field(row, key)


# --- sources --------------------------------------------------------------
conc_rows = read_csv(CONC / "vllm_concurrency.csv")
conc_by = {(r["case"], int(r["batch_size"])): r for r in conc_rows}

lat_rows = read_csv(COMMON / "vllm_latency.csv")
lat_by = {(r["case"], int(r["batch_size"])): r for r in lat_rows}

llama_rows = read_csv(COMMON / "llamacpp_model.csv")
llama_by = {r["case"]: r for r in llama_rows}

storage_rows = read_csv(COMMON / "vllm_model_storage.csv")
storage_by = {r["case"]: r for r in storage_rows}

theory_rows = read_csv(COMMON / "theoretical_roofline.csv")
theory_by = {r["precision"]: r for r in theory_rows}

hbm_rows = read_csv(COMMON / "hbm_bandwidth.csv")
hbm_by = {(int(r["size_mib"]), r["operation"]): r for r in hbm_rows}

small_rows = read_csv(SMALL_CONC)
small_by = {(r["case"], int(r["batch_size"])): r for r in small_rows}
mid_rows = read_csv(MID_CONC)
mid_by = {(r["case"], int(r["batch_size"])): r for r in mid_rows}
small_summary = json.loads(SMALL_SUMMARY.read_text(encoding="utf-8"))
mid_summary = json.loads(MID_SUMMARY.read_text(encoding="utf-8"))
conc_summary = json.loads(
    (CONC / "vllm_concurrency_summary.json").read_text(encoding="utf-8")
)

main = REPORT.read_text(encoding="utf-8")
conc = CONC_REPORT.read_text(encoding="utf-8")

case_for_label = {
    "FP16": "base_fp16",
    "GPTQ-INT4": "gptq_int4",
    "AWQ-INT4": "awq_int4",
    "GPTQ-INT8": "gptq_int8",
}
batch_cols = {"N=1": 1, "N=2": 2, "N=4": 4, "N=8": 8, "N=16": 16}
comparison_model_summaries = {
    "Qwen2.5-0.5B": small_summary,
    "Qwen3-4B": mid_summary,
    "Qwen2.5-7B": conc_summary,
}

# --- formula / model constants -------------------------------------------
for label, text in [("main-report", main), ("concurrency-report", conc)]:
    check(f"{label}: contains 7B logical parameter count",
          "7,070,285,824" in text)
    check(f"{label}: contains 7B token FLOP count",
          "14,140,571,648" in text)
    check(f"{label}: contains Q_HEADS*HEAD_DIM formula",
          "Q_HEADS*HEAD_DIM" in text)

# The reports intentionally mention the 4B comparison constants, but must not
# accidentally use the old 4B constants as the current model definition.
stale_tokens = [
    "LOGICAL_MATMUL_PARAMS = ... = 4,022,272,000",
    "TOKEN_MATMUL_FLOPS = 8,044,544,000",
]
for token in stale_tokens:
    check(f"reports: stale 4B-only formula absent {token!r}",
          token not in main and token not in conc)

# --- structural / source validity ----------------------------------------
check("concurrency CSV has 25 rows", len(conc_rows) == 25, str(len(conc_rows)))
check("concurrency CSV has 20 successful rows",
      sum(r["status"] == "ok" for r in conc_rows) == 20)
check("FP8 has five honest failures",
      all(conc_by[("fp8_dynamic", n)]["status"] == "failed_rc=1"
          for n in batch_cols.values()))
check("all vLLM measured batch1/batch8 rows are present",
      all((c, b) in lat_by
          for c in ("base_bf16", "base_fp16", "gptq_int4", "awq_int4",
                    "gptq_int8", "base_fp32", "fp8_dynamic")
          for b in (1, 8)))
check("llama.cpp model CSV has five formats", len(llama_rows) == 5)

# --- concurrency AI table in both reports --------------------------------
for label, text, marker in [
    ("main-report", main, "### 6.1 权重-only AI（OP/B）"),
    ("concurrency-report", conc, "### 2.1 权重逻辑访存比"),
]:
    header, rows = table_after(text, marker, ["N=16 达成度", "理论 ridge"])
    rmap = row_map(rows)
    for row_label, case in case_for_label.items():
        row = rmap[row_label]
        ridge = float(conc_by[(case, 16)]["theoretical_ridge_ops_per_byte"])
        for col, batch in batch_cols.items():
            got = number(row[col_index(header, col)])
            want = float(
                conc_by[(case, batch)]["weight_only_ai_ops_per_byte_batch_scaled"]
            )
            check_value(f"{label}: AI {row_label} N={batch}", got, want, 5e-4)
        attainment = number(row[col_index(header, "N=16 达成度")])
        want_att = 100.0 * float(
            conc_by[(case, 16)]["weight_only_ai_ops_per_byte_batch_scaled"]
        ) / ridge
        check_value(f"{label}: ridge attainment {row_label} N=16",
                    attainment, want_att, 6e-3)

# --- Byte/FLOP tables -----------------------------------------------------
for label, text, marker in [
    ("main-report", main, "### 6.2 Byte/FLOP（1/AI）"),
    ("concurrency-report", conc, "### 2.2 Byte/FLOP"),
]:
    header, rows = table_after(text, marker, ["N=1", "N=16"])
    rmap = row_map(rows)
    for row_label, case in case_for_label.items():
        for col, batch in batch_cols.items():
            got = number(rmap[row_label][col_index(header, col)])
            want = float(conc_by[(case, batch)]["weight_only_bytes_per_flop"])
            check_value(f"{label}: Byte/FLOP {row_label} N={batch}",
                        got, want, 6e-5)

# --- throughput / batch-efficiency tables --------------------------------
for label, text, marker in [
    ("main-report", main, "### 6.3 聚合吞吐"),
    ("concurrency-report", conc, "### 2.3 聚合吞吐"),
]:
    header, rows = table_after(text, marker, ["N=1", "N=16"])
    rmap = row_map(rows)
    scale_col = next(c for c in header if "N=16/N=1" in c)
    eff_col = next(c for c in header if "batch efficiency" in c)
    for row_label, case in case_for_label.items():
        for col, batch in batch_cols.items():
            got = number(rmap[row_label][col_index(header, col)])
            want = float(conc_by[(case, batch)]["total_output_tok_s"])
            check_value(f"{label}: throughput {row_label} N={batch}",
                        got, want, 6e-3)
        got_scale = number(rmap[row_label][col_index(header, scale_col)])
        want_scale = float(conc_by[(case, 16)]["throughput_scaling_vs_b1"])
        check_value(f"{label}: throughput scaling {row_label}",
                    got_scale, want_scale, 6e-3)
        got_eff = number(rmap[row_label][col_index(header, eff_col)])
        want_eff = float(conc_by[(case, 16)]["batch_efficiency_vs_b1"])
        check_value(f"{label}: batch efficiency {row_label}",
                    got_eff, want_eff, 6e-4)

# --- concurrency latency / bandwidth / ridge-attainment tables ------------
conc_detail_tables = [
    ("### 2.4 平均延迟", "avg_latency_s", 6e-4, None),
    ("### 2.5 等效权重读带宽", "effective_weight_read_gbps_equiv", 5e-2, None),
    ("### 2.6 相对理论 ridge", "weight_only_ai_ops_per_byte_batch_scaled",
     1e-2, "ridge_percent"),
]
for marker, csv_col, tol, mode in conc_detail_tables:
    header, rows = table_after(conc, marker, ["N=1", "N=16"])
    rmap = row_map(rows)
    for row_label, case in case_for_label.items():
        for col, batch in batch_cols.items():
            got = number(rmap[row_label][col_index(header, col)])
            if mode == "ridge_percent":
                want = 100.0 * float(conc_by[(case, batch)][csv_col]) / float(
                    conc_by[(case, batch)]["theoretical_ridge_ops_per_byte"]
                )
            else:
                want = float(conc_by[(case, batch)][csv_col])
            check_value(f"concurrency-report: {marker} {row_label} {col}",
                        got, want, tol)

# --- vLLM batch=1 / batch=8 tables in the main report --------------------
batch1_header, batch1_rows = table_after(
    main, "### 5.2 batch=1", ["权重-only AI"]
)
batch1_map = row_map(batch1_rows)
batch8_header, batch8_rows = table_after(
    main, "### 5.3 batch=8", ["batch-scaled AI"]
)
batch8_map = row_map(batch8_rows)

precision_label = {
    "BF16": ("base_bf16", "FP16"),
    "FP16": ("base_fp16", "FP16"),
    "FP32": ("base_fp32", "FP32"),
    "GPTQ-INT4": ("gptq_int4", "Q4_K"),
    "AWQ-INT4": ("awq_int4", "Q4_K"),
    "GPTQ-INT8": ("gptq_int8", "Q8_0"),
    "FP8-dynamic": ("fp8_dynamic", "FP8"),
}
batch1_fields = [
    ("每 token 权重 MiB", "per_token_weight_mib", 6e-3),
    ("延迟 s", "avg_latency_s", 6e-4),
    ("总 tok/s", "total_output_tok_s", 6e-3),
    ("权重-only AI", "weight_only_ai_ops_per_byte_batch_scaled", 6e-4),
    ("等效权重读带宽", "effective_weight_read_gbps_equiv", 6e-3),
    ("逻辑 TFLOPS", "attained_logical_tflops_output_equiv", 5e-5),
]
batch8_fields = [
    ("延迟 s", "avg_latency_s", 6e-4),
    ("总 tok/s", "total_output_tok_s", 6e-3),
    ("每请求 tok/s", "per_request_output_tok_s", 6e-3),
    ("batch-scaled AI", "weight_only_ai_ops_per_byte_batch_scaled", 6e-4),
    ("等效每请求权重带宽", "effective_weight_read_gbps_equiv", 5e-2),
    ("逻辑 TFLOPS", "attained_logical_tflops_output_equiv", 5e-5),
]
for row_label, (case, ridge_key) in precision_label.items():
    r1 = lat_by[(case, 1)]
    r8 = lat_by[(case, 8)]
    failed_row = "失败" in " ".join(batch1_map[row_label]) or (
        "失败" in " ".join(batch8_map[row_label])
    )
    if case in {"base_fp32", "fp8_dynamic"}:
        check(f"main-report: {row_label} batch=1 marked failed",
              "失败" in " ".join(batch1_map[row_label]))
        check(f"main-report: {row_label} batch=8 marked failed",
              "失败" in " ".join(batch8_map[row_label]))
        check(f"source: {row_label} batch=1 failed",
              r1["status"].startswith("failed"))
        check(f"source: {row_label} batch=8 failed",
              r8["status"].startswith("failed"))
        continue
    check(f"main-report: {row_label} is not a failed row", not failed_row)
    for col, csv_col, tol in batch1_fields:
        check_value(
            f"main-report: batch1 {row_label} {col}",
            number(batch1_map[row_label][col_index(batch1_header, col)]),
            float(r1[csv_col]),
            tol,
        )
    for col, csv_col, tol in batch8_fields:
        check_value(
            f"main-report: batch8 {row_label} {col}",
            number(batch8_map[row_label][col_index(batch8_header, col)]),
            float(r8[csv_col]),
            tol,
        )
    got_ar = number(batch8_map[row_label][col_index(batch8_header, "AI/ridge")])
    want_ar = 100.0 * float(
        r8["weight_only_ai_ops_per_byte_batch_scaled"]
    ) / float(theory_by[ridge_key]["ridge_point_ops_per_byte"])
    check_value(f"main-report: batch8 {row_label} AI/ridge",
                got_ar, want_ar, 1e-2)

# --- llama.cpp model table -------------------------------------------------
header, rows = table_after(
    main, "### 4.2 存储、算存比与吞吐", ["文件 MiB", "decode 128 tok/s"]
)
rmap = row_map(rows)
llama_storage_fields = [
    ("文件 MiB", "file_mib", 6e-3),
    ("每 token 权重 MiB", "per_token_weight_mib", 6e-3),
    ("实际 bit/逻辑参数", "bits_per_logical_parameter", 2e-3),
    ("权重-only AI", "weight_only_ai_ops_per_byte", 1e-4),
]
llama_ok_fields = [
    ("prompt 512 tok/s", "prompt_tok_s_median", 6e-3),
    ("decode 128 tok/s", "decode_tok_s_median", 6e-3),
    ("有效权重读带宽", "effective_weight_read_gbps_from_median", 6e-3),
    ("逻辑 TFLOPS", "attained_logical_tflops_from_median", 5e-5),
]
for row_label, csv in llama_by.items():
    row = rmap[row_label]
    for col, csv_col, tol in llama_storage_fields:
        check_value(
            f"main-report: llama.cpp {row_label} {col}",
            number(row[col_index(header, col)]),
            float(csv[csv_col]),
            tol,
        )
    if csv["status"] == "ok":
        for col, csv_col, tol in llama_ok_fields:
            check_value(
                f"main-report: llama.cpp {row_label} {col}",
                number(row[col_index(header, col)]),
                float(csv[csv_col]),
                tol,
            )
    else:
        check(f"main-report: llama.cpp {row_label} marked failed",
              "失败" in " ".join(row))

# --- theory and HBM tables -------------------------------------------------
header, rows = table_after(main, "## 2. 理论算力", ["理论 ridge"])
rmap = row_map(rows)
for row_label in ["FP32", "FP16", "BF16", "INT8 / Q8", "INT4 / Q4"]:
    precision = {"INT8 / Q8": "Q8_0", "INT4 / Q4": "Q4_K"}.get(
        row_label, row_label
    )
    check_value(
        f"main-report: theoretical ridge {row_label}",
        number(rmap[row_label][col_index(header, "理论 ridge")]),
        float(theory_by[precision]["ridge_point_ops_per_byte"]),
        6e-3,
    )

header, rows = table_after(main, "## 10. HBM", ["操作", "中位带宽"])
rmap = row_map(rows)
for operation in ["read_sum", "write_zero", "copy", "triad_add"]:
    check_value(
        f"main-report: HBM 1024 MiB {operation}",
        number(rmap[operation][col_index(header, "中位带宽")]),
        float(hbm_by[(1024, operation)]["median_gbps"]),
        6e-3,
    )

# --- per-token storage / equivalent bpw table -----------------------------
header, rows = table_after(
    conc, "### 1.3 权重字节的口径", ["权重字节", "等价位宽"]
)
rmap = row_map(rows)
storage_label_to_case = {
    "FP16": "base_fp16",
    "AWQ-INT4": "awq_int4",
    "GPTQ-INT4": "gptq_int4",
    "GPTQ-INT8": "gptq_int8",
    "FP8-dynamic": "fp8_dynamic",
}
for row_label, case in storage_label_to_case.items():
    row = rmap[row_label]
    expected_bytes = int(conc_summary["per_token_weight_bytes"][case])
    check(
        f"concurrency-report: storage bytes {row_label}",
        f"{expected_bytes:,} B" in row[col_index(header, "权重字节")],
        f"want {expected_bytes:,} B, got {row[col_index(header, '权重字节')]!r}",
    )
    check_value(
        f"concurrency-report: equivalent bpw {row_label}",
        number(row[col_index(header, "等价位宽")]),
        float(storage_by[case]["bits_per_logical_parameter"]),
        6e-5,
    )

# --- 7B vs 4B / 0.5B comparison table -------------------------------------
header, rows = table_after(
    conc, "### 4.1 直接对照表", ["Qwen2.5-0.5B", "Qwen3-4B"]
)
rmap = row_map(rows)

# Parameter and FLOP rows are exact model metadata.
model_meta = [
    ("逻辑矩阵参数量", "logical_matmul_params", 1e9, 1.0e-3),
    ("每 token 逻辑 FLOPs", "token_matmul_flops", 1e9, 1.0e-3),
]
for row_label, key, scale, tol in model_meta:
    for column, summary in comparison_model_summaries.items():
        got = number(rmap[row_label][col_index(header, column)])
        want = float(summary["logical_model"][key]) / scale
        check_value(f"concurrency-report: {row_label} {column}",
                    got, want, tol)

# FP16 per-token storage and batch-1 bandwidth.
for column, (source, _, _) in [
    ("Qwen2.5-0.5B", (small_by, None, small_summary)),
    ("Qwen3-4B", (mid_by, None, mid_summary)),
    ("Qwen2.5-7B", (conc_by, None, conc_summary)),
]:
    row = rmap["FP16 每 token 权重字节"]
    got = number(row[col_index(header, column)])
    want = float(source[("base_fp16", 1)]["per_token_weight_mib"])
    check_value(f"concurrency-report: FP16 per-token MiB {column}",
                got, want, 1.0)
    row = rmap["FP16 等效权重读带宽 @N=1"]
    got = number(row[col_index(header, column)])
    want = float(source[("base_fp16", 1)]["effective_weight_read_gbps_equiv"])
    check_value(f"concurrency-report: FP16 bandwidth {column}",
                got, want, 5e-2)

# Throughput rows.  The 0.5B report intentionally leaves N=1 blank.
throughput_rows = {
    "FP16 聚合吞吐 @N=1": "base_fp16",
    "FP16 聚合吞吐 @N=16": "base_fp16",
    "GPTQ-INT4 聚合吞吐 @N=16": "gptq_int4",
    "AWQ-INT4 聚合吞吐 @N=16": "awq_int4",
    "GPTQ-INT8 聚合吞吐 @N=16": "gptq_int8",
}
for row_label, case in throughput_rows.items():
    batch = 1 if row_label.endswith("@N=1") else 16
    for column, source in [
        ("Qwen2.5-0.5B", small_by),
        ("Qwen3-4B", mid_by),
        ("Qwen2.5-7B", conc_by),
    ]:
        cell = rmap[row_label][col_index(header, column)]
        if clean(cell) in {"-", "—", "失败", "不适用"}:
            expected_blank = (
                (column == "Qwen2.5-0.5B" and case == "base_fp16" and batch == 1)
                or (column == "Qwen2.5-0.5B" and case in {"awq_int4", "gptq_int8"} and batch == 16)
            )
            check(f"concurrency-report: {row_label} {column} intentionally blank",
                  expected_blank, f"unexpected blank cell {cell!r}")
            continue
        got = number(cell)
        want = float(source[(case, batch)]["total_output_tok_s"])
        check_value(f"concurrency-report: {row_label} {column}",
                    got, want, 0.5)

# AI rows use the 0.5B AWQ correction documented in the 4B report.
ai_rows = {
    "FP16 AI @N=1 / N=16": "base_fp16",
    "GPTQ-INT4 AI @N=1 / N=16": "gptq_int4",
    "AWQ-INT4 AI @N=1 / N=16": "awq_int4",
    "GPTQ-INT8 AI @N=1 / N=16": "gptq_int8",
}
for row_label, case in ai_rows.items():
    for column, source in [
        ("Qwen2.5-0.5B", small_by),
        ("Qwen3-4B", mid_by),
        ("Qwen2.5-7B", conc_by),
    ]:
        got_values = numbers(rmap[row_label][col_index(header, column)])
        check(f"concurrency-report: {row_label} {column} has two values",
              len(got_values) == 2, f"got {got_values!r}")
        if len(got_values) != 2:
            continue
        for idx, batch in enumerate((1, 16)):
            want = float(
                source[(case, batch)]["weight_only_ai_ops_per_byte_batch_scaled"]
            )
            if column == "Qwen2.5-0.5B" and case == "awq_int4":
                # 0.5B's original manifest missed the independent output head.
                want *= float(source[(case, batch)]["per_token_weight_mib"]) / 696.739013671875
            check_value(
                f"concurrency-report: {row_label} {column} N={batch}",
                got_values[idx], want, 6e-3,
            )

# Batch-efficiency rows.
batch_eff_rows = {
    "FP16 batch efficiency @N=16": "base_fp16",
    "GPTQ-INT4 batch efficiency @N=16": "gptq_int4",
}
for row_label, case in batch_eff_rows.items():
    for column, source in [
        ("Qwen2.5-0.5B", small_by),
        ("Qwen3-4B", mid_by),
        ("Qwen2.5-7B", conc_by),
    ]:
        cell = rmap[row_label][col_index(header, column)]
        if clean(cell) in {"-", "—", "失败", "不适用"}:
            check(f"concurrency-report: {row_label} {column} intentionally blank",
                  not (case == "base_fp16" and column == "Qwen2.5-0.5B"),
                  f"unexpected blank cell {cell!r}")
            continue
        check_value(
            f"concurrency-report: {row_label} {column}",
            number(cell),
            float(source[(case, 16)]["batch_efficiency_vs_b1"]),
            6e-4,
        )

# --- kernel comparison table ---------------------------------------------
header, rows = table_after(
    conc, "## 5. Kernel 行为", ["GPTQ-INT4", "AWQ-INT4", "倍率"]
)
rmap = row_map(rows)
kernel_rows = [
    ("per-token 权重 MiB", "per_token_weight_mib", 5e-2),
    ("N=1 AI", "weight_only_ai_ops_per_byte_batch_scaled", 6e-4),
    ("N=1 延迟 s", "avg_latency_s", 6e-4),
    ("N=1 聚合吞吐", "total_output_tok_s", 6e-3),
    ("N=16 聚合吞吐", "total_output_tok_s", 6e-3),
    ("N=16 等效权重读带宽", "effective_weight_read_gbps_equiv", 5e-2),
]
for row_label, csv_col, tol in kernel_rows:
    gptq = conc_by[("gptq_int4", 1)]
    awq = conc_by[("awq_int4", 1)]
    if row_label.startswith("N=16"):
        gptq = conc_by[("gptq_int4", 16)]
        awq = conc_by[("awq_int4", 16)]
    row = rmap[row_label]
    check_value(f"concurrency-report: kernel {row_label} GPTQ",
                number(row[col_index(header, "GPTQ-INT4")]),
                float(gptq[csv_col]), tol)
    check_value(f"concurrency-report: kernel {row_label} AWQ",
                number(row[col_index(header, "AWQ-INT4")]),
                float(awq[csv_col]), tol)

# --- archived copies are exact snapshots ----------------------------------
check("archive concurrency report exists",
      ARCHIVE_CONC_REPORT.exists(), str(ARCHIVE_CONC_REPORT))
check("archive concurrency report matches experiment copy",
      ARCHIVE_CONC_REPORT.exists()
      and ARCHIVE_CONC_REPORT.read_bytes() == CONC_REPORT.read_bytes())
check("archive main report exists",
      ARCHIVE_MAIN_REPORT.exists(), str(ARCHIVE_MAIN_REPORT))
check("archive main report matches experiment copy",
      ARCHIVE_MAIN_REPORT.exists()
      and ARCHIVE_MAIN_REPORT.read_bytes() == REPORT.read_bytes())

print()
print(f"report-vs-csv checks = {n_checks}, failures = {len(failures)}")
if failures:
    for item in failures:
        print("  FAILED:", item)
    sys.exit(1)
print("ALL REPORT NUMBERS CONSISTENT WITH CSV")
