#!/usr/bin/env python3
"""
MTP Side-Channel Experiment Driver (Contribution ①)
=====================================================
Runs mtp-probe across multiple prompt categories, collects per-step
MTP accept/reject statistics, and performs cross-domain analysis.

Usage: python3 run_mtp_experiment.py [--output results.json]
"""

import subprocess, json, re, sys, time, os, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# --- Config ---
MODEL = "/mnt/public/GGUF_models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
PROBE_BIN = "/mnt/luyuzhou/hpc/moecho/llama.cpp/build/bin/mtp-probe"
GPU_LAYERS = 99
N_TOKENS = 64
TEMP = 0.6
DRAFT_N_MAX = 2
CTX_SIZE = 256
BATCH_SIZE = 128
UBATCH_SIZE = 128

# --- Matrix overrides (CLI, defaults = v7 口径) ---
CFG = {}


def parse_overrides():
    ap = argparse.ArgumentParser(description="MTP probe matrix runner (defaults = v7 口径)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpus", default=None, help='CUDA_VISIBLE_DEVICES, e.g. "1" 或 "0,1"')
    ap.add_argument("--draft-max", type=int, default=DRAFT_N_MAX)
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--n-tokens", type=int, default=N_TOKENS)
    ap.add_argument("--split-mode", default=None, help="graph|tensor (默认 llama.cpp graph)")
    ap.add_argument("--tensor-split", default=None, help='TP 模式权重比, 如 "0.5/0.5"')
    ap.add_argument("--probe-extra", default="", help="追加到 mtp-probe 的原始参数")
    ap.add_argument("--output", default=None, help="输出 JSON 路径")
    return ap.parse_args()


# --- Prompt corpus ---
PROMPTS = {
    "medical": [
        "Common cold symptoms and treatment:",
        "Type 2 diabetes management guidelines:",
        "Early signs of heart disease:",
        "Lung cancer diagnosis and staging:",
        "COVID-19 antiviral treatment options:",
        "Depression screening and therapy:",
        "Hypertension medication side effects:",
        "Alzheimer's disease early detection:",
    ],
    "code": [
        "Python quicksort algorithm implementation:",
        "Rust async HTTP server example:",
        "SQL query to find duplicate records:",
        "JavaScript event loop explained:",
        "C memory management best practices:",
        "Docker compose multi-service setup:",
        "React useEffect hook patterns:",
        "Linux kernel module hello world:",
    ],
    "finance": [
        "Retirement investment strategy for 401k:",
        "Stock market index fund comparison:",
        "Real estate vs stock investment returns:",
        "Tax loss harvesting strategies:",
        "Cryptocurrency portfolio allocation:",
        "Bond yield curve inversion implications:",
        "Small business valuation methods:",
        "High frequency trading basics:",
    ],
    "general": [
        "France capital city:",
        "Italian carbonara pasta recipe:",
        "Tokyo must-visit attractions:",
        "Marathon training plan for beginners:",
        "French Revolution causes and effects:",
        "General relativity theory explained:",
        "Chess opening strategies for beginners:",
        "Classical music composers ranked:",
    ],
}


def run_probe(prompt: str) -> dict:
    """Run mtp-probe for a single prompt and parse output."""
    model = CFG.get("model", MODEL)
    cmd = [
        PROBE_BIN,
        "-m", model,
        "--gpu-layers", str(GPU_LAYERS),
        "-p", prompt,
        "-n", str(CFG.get("n_tokens", N_TOKENS)),
        "--temp", str(CFG.get("temp", TEMP)),
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", str(CFG.get("draft_max", DRAFT_N_MAX)),
        "--ctx-size", str(CTX_SIZE),
        "--batch-size", str(BATCH_SIZE),
        "--ubatch-size", str(UBATCH_SIZE),
    ]
    if CFG.get("split_mode"):
        cmd += ["--split-mode", CFG["split_mode"]]
    if CFG.get("tensor_split"):
        cmd += ["--tensor-split", CFG["tensor_split"]]
    if CFG.get("probe_extra"):
        cmd += CFG["probe_extra"].split()
    env = dict(os.environ)
    if CFG.get("gpus"):
        env["CUDA_VISIBLE_DEVICES"] = CFG["gpus"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        return _parse_output(result.stderr + "\n" + result.stdout, prompt)
    except subprocess.TimeoutExpired:
        return {"prompt": prompt, "error": "timeout"}
    except Exception as e:
        return {"prompt": prompt, "error": str(e)}


def _parse_output(text: str, prompt: str) -> dict:
    """Parse mtp-probe output into structured data."""
    record = {
        "prompt": prompt,
        "n_predict": None,
        "n_drafted": None,
        "n_accept": None,
        "accept_rate": None,
        "speed_tps": None,
        "total_time_s": None,
        "steps": [],
    }

    # Clean function: strip log timestamp prefix like "0.04.739.512 I "
    def _clean(line: str) -> str:
        return re.sub(r'^[\d.]+ [A-Z] ', '', line.strip())

    # Extract summary stats
    for line in text.split("\n"):
        cleaned = _clean(line)
        m = re.search(r"n_predict\s*=\s*(\d+)", cleaned)
        if m:
            record["n_predict"] = int(m.group(1))
        m = re.search(r"n_drafted\s*=\s*(\d+)", cleaned)
        if m:
            record["n_drafted"] = int(m.group(1))
        m = re.search(r"n_accept\s*=\s*(\d+)", cleaned)
        if m:
            record["n_accept"] = int(m.group(1))
        m = re.search(r"accept_rate\s*=\s*([\d.]+)%", cleaned)
        if m:
            record["accept_rate"] = float(m.group(1))
        m = re.search(r"speed\s*=\s*([\d.]+)\s*t/s", cleaned)
        if m:
            record["speed_tps"] = float(m.group(1))
        m = re.search(r"total_time\s*=\s*([\d.]+)\s*s", cleaned)
        if m:
            record["total_time_s"] = float(m.group(1))

    # Extract per-step CSV data (lines have log prefix: "timestamp I csv,data,...")
    in_csv = False
    for line in text.split("\n"):
        if "step,n_draft,n_accept,accept_rate,time_us" in line:
            in_csv = True
            continue
        if in_csv:
            # Strip log prefix: "0.04.242.160 I 0,2,0,0.0%,37601"
            stripped = re.sub(r"^[\d.]+ [A-Z] ", "", line.strip())
            parts = stripped.split(",")
            if len(parts) >= 5:
                try:
                    record["steps"].append({
                        "step": int(parts[0]),
                        "n_draft": int(parts[1]),
                        "n_accept": int(parts[2]),
                        "accept_rate": float(parts[3].replace("%", "")),
                        "time_us": float(parts[4]),
                    })
                except (ValueError, IndexError):
                    pass

    # Detect per-step timing pattern: first step vs steady state
    if record["steps"]:
        times = [s["time_us"] for s in record["steps"]]
        record["first_step_us"] = times[0]
        record["steady_step_us"] = np.median(times[3:]) if len(times) > 3 else times[-1]
        record["time_ratio"] = record["first_step_us"] / record["steady_step_us"] if record["steady_step_us"] > 0 else 1.0

    # Extract MoE expert routing data
    record["moe_layers"] = []
    record["moe_layers"] = []
    record["moe_n_layers"] = None
    record["moe_n_exp_used"] = None
    record["moe_n_tokens"] = None
    record["moe_n_expert_total"] = None
    record["per_expert_accept"] = []  # [layer][expert_idx] = {rate, accept, reject}

    # Parse MoE header and aggregated expert data
    for line in text.split("\n"):
        cleaned = _clean(line)
        m = re.search(r'n_moe_layers\s*=\s*(\d+)', cleaned)
        if m: record["moe_n_layers"] = int(m.group(1))
        m = re.search(r'n_exp_used\s*=\s*(\d+)', cleaned)
        if m: record["moe_n_exp_used"] = int(m.group(1))
        m = re.search(r'n_expert_total\s*=\s*(\d+)', cleaned)
        if m: record["moe_n_expert_total"] = int(m.group(1))

    # Parse MOE_RAW format (from mtp_probe's fprintf debug output)
    # Format: MOE_RAW n_layers=N n_exp=E n_tok=T
    #         MOE_L0 e1 e2 e3 ... (E*T integers)
    #         MOE_L1 e1 e2 e3 ...
    raw_moe_layers = []
    for line in text.split("\n"):
        m = re.match(r'^MOE_L(\d+)\s+(.+)', line)
        if m:
            layer_idx = int(m.group(1))
            indices = [int(x) for x in m.group(2).strip().split()]
            while len(raw_moe_layers) <= layer_idx:
                raw_moe_layers.append([])
            raw_moe_layers[layer_idx] = indices

    # Parse per-expert MTP accept rate (PEA_L format)
    # Format: PEA_L0 e1:0.0:0:2 e2:0.0:0:6 ... (rate:accept:reject)
    for line in text.split("\n"):
        m = re.match(r'^PEA_L(\d+)\s+(.+)', line)
        if m:
            layer_idx = int(m.group(1))
            while len(record["per_expert_accept"]) <= layer_idx:
                record["per_expert_accept"].append({})
            for token in m.group(2).split():
                em = re.match(r'e(\d+):([\d.]+):(\d+):(\d+)', token)
                if em:
                    record["per_expert_accept"][layer_idx][int(em.group(1))] = {
                        "rate": float(em.group(2)),
                        "accept": int(em.group(3)),
                        "reject": int(em.group(4)),
                    }

    if raw_moe_layers:
        # Convert raw indices to per-layer expert frequency counts
        n_exp = record.get("moe_n_exp_used", 8)
        for layer_idx, indices in enumerate(raw_moe_layers):
            expert_counts = {}
            if indices and n_exp > 0:
                n_tok = len(indices) // n_exp
                for t in range(n_tok):
                    for e in range(n_exp):
                        idx = t * n_exp + e
                        if idx < len(indices) and 0 <= indices[idx] < 256:
                            expert_counts[indices[idx]] = expert_counts.get(indices[idx], 0) + 1
            record["moe_layers"].append(expert_counts)
    else:
        # Fallback: parse Layer X: eY:count format
        for line in text.split("\n"):
            cleaned = _clean(line)
            layer_match = re.match(r'^Layer\s+(\d+):\s+(.+)', cleaned)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                expert_str = layer_match.group(2)
                expert_counts = {}
                for token in expert_str.split():
                    em = re.match(r'e(\d+):(\d+)', token)
                    if em:
                        expert_counts[int(em.group(1))] = int(em.group(2))
                while len(record["moe_layers"]) <= layer_idx:
                    record["moe_layers"].append({})
                record["moe_layers"][layer_idx] = expert_counts

    return record


def analyze_results(results: list) -> dict:
    """Compute cross-domain statistics."""
    by_category = defaultdict(list)
    for r in results:
        cat = _classify_prompt(r["prompt"])
        by_category[cat].append(r)

    analysis = {"per_category": {}, "cross_domain": {}, "per_prompt": results}

    for cat, records in by_category.items():
        accept_rates = [r["accept_rate"] for r in records if r.get("accept_rate") is not None]
        speeds = [r["speed_tps"] for r in records if r.get("speed_tps") is not None]
        time_ratios = [r["time_ratio"] for r in records if r.get("time_ratio") is not None]
        first_steps = [r["first_step_us"] for r in records if r.get("first_step_us") is not None]
        steady_steps = [r["steady_step_us"] for r in records if r.get("steady_step_us") is not None]

        analysis["per_category"][cat] = {
            "n_prompts": len(records),
            "n_errors": sum(1 for r in records if "error" in r),
            "accept_rate_mean": float(np.mean(accept_rates)) if accept_rates else None,
            "accept_rate_std": float(np.std(accept_rates)) if accept_rates else None,
            "speed_mean": float(np.mean(speeds)) if speeds else None,
            "speed_std": float(np.std(speeds)) if speeds else None,
            "time_ratio_mean": float(np.mean(time_ratios)) if time_ratios else None,
            "first_step_mean_us": float(np.mean(first_steps)) if first_steps else None,
            "steady_step_mean_us": float(np.mean(steady_steps)) if steady_steps else None,
        }

    # Cross-domain pairwise comparisons
    cats = sorted(by_category.keys())
    for i, c1 in enumerate(cats):
        for c2 in cats[i + 1 :]:
            ar1 = [r["accept_rate"] for r in by_category[c1] if r.get("accept_rate") is not None]
            ar2 = [r["accept_rate"] for r in by_category[c2] if r.get("accept_rate") is not None]
            if len(ar1) >= 2 and len(ar2) >= 2:
                from scipy import stats
                t_stat, p_val = stats.ttest_ind(ar1, ar2)
                analysis["cross_domain"][f"{c1}_vs_{c2}"] = {
                    "mean_diff": float(np.mean(ar1) - np.mean(ar2)),
                    "p_value": float(p_val),
                    "significant": p_val < 0.05,
                }

    return analysis


def _classify_prompt(prompt: str) -> str:
    for cat, prompts in PROMPTS.items():
        if prompt in prompts:
            return cat
    return "unknown"


def main():
    args = parse_overrides()
    global CFG
    CFG = {k: v for k, v in vars(args).items() if v is not None and v != ""}
    if args.output:
        output_path = args.output
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        output_path = sys.argv[1]  # 旧式: 位置参数
    else:
        output_path = "mtp_experiment_results.json"

    all_prompts = []
    for cat, prompts in PROMPTS.items():
        for p in prompts:
            all_prompts.append((cat, p))

    print(f"Running MTP probe on {len(all_prompts)} prompts across {len(PROMPTS)} categories...")
    print(f"Model: {CFG.get('model', MODEL)}")
    print(f"GPUs: {CFG.get('gpus', 'all')} | Temp: {CFG.get('temp', TEMP)}, "
          f"Draft max: {CFG.get('draft_max', DRAFT_N_MAX)}, n_tokens: {CFG.get('n_tokens', N_TOKENS)}"
          + (f" | split: {CFG['split_mode']} {CFG.get('tensor_split', '')}" if CFG.get("split_mode") else ""))
    print()

    results = []
    for i, (cat, prompt) in enumerate(all_prompts):
        print(f"[{i+1}/{len(all_prompts)}] [{cat}] {prompt[:60]}...", end=" ", flush=True)
        record = run_probe(prompt)
        record["category"] = cat

        if "error" in record:
            print(f"ERROR: {record['error']}")
        else:
            print(f"accept={record.get('accept_rate', 'N/A')}% "
                  f"speed={record.get('speed_tps', 'N/A')} t/s "
                  f"steps={len(record.get('steps', []))}")

        results.append(record)

    # Analyze
    print("\n--- Analysis ---")
    analysis = analyze_results(results)

    for cat, stats in analysis["per_category"].items():
        print(f"\n[{cat}] ({stats['n_prompts']} prompts, {stats['n_errors']} errors)")
        if stats["accept_rate_mean"] is not None:
            print(f"  Accept rate:  {stats['accept_rate_mean']:.1f}% ± {stats['accept_rate_std']:.1f}%")
        if stats["speed_mean"] is not None:
            print(f"  Speed:        {stats['speed_mean']:.1f} ± {stats['speed_std']:.1f} t/s")
        if stats["time_ratio_mean"] is not None:
            print(f"  Time ratio:   {stats['time_ratio_mean']:.2f}x")
        if stats["first_step_mean_us"] is not None:
            print(f"  First step:   {stats['first_step_mean_us']:.0f} us")
        if stats["steady_step_mean_us"] is not None:
            print(f"  Steady step:  {stats['steady_step_mean_us']:.0f} us")

    print("\n--- Cross-Domain Significance ---")
    for pair, stats in analysis["cross_domain"].items():
        sig = "SIGNIFICANT" if stats["significant"] else "not significant"
        print(f"  {pair}: Δ={stats['mean_diff']:.2f}%, p={stats['p_value']:.4f} ({sig})")

    # Save
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
