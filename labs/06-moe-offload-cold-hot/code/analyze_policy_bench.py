from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/policy-v1"
POLICY_ORDER = ["baseline", "noop", "top85", "top90", "floor", "combined", "layer"]


def load_jsonl(path: Path, source: str) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            item = json.loads(line)
            measured = item["measured"]
            rows.append(
                {
                    "source": source,
                    "policy": item["policy"],
                    "prompt_id": item["prompt_id"],
                    "repeat": int(item.get("repeat") or 0),
                    "ttft_ms": measured["ttft_ms"],
                    "decode_tok_s": measured["decode_tok_s"],
                    "ms_per_token": measured["ms_per_token"],
                    "text": measured["text"],
                    "text_sha1": measured["text_sha1"],
                    "completion_tokens": measured["completion_tokens"],
                    "hybrid_fetch": item.get("hybrid_fetch", -1),
                }
            )
    return pd.DataFrame(rows)


def common_prefix_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b), 1)


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_prompt = (
        frame.groupby(["source", "policy", "prompt_id"], as_index=False)
        .agg(
            decode_tok_s_mean=("decode_tok_s", "mean"),
            decode_tok_s_std=("decode_tok_s", "std"),
            ms_per_token_mean=("ms_per_token", "mean"),
            ms_per_token_std=("ms_per_token", "std"),
            ttft_ms_mean=("ttft_ms", "mean"),
            n=("decode_tok_s", "size"),
        )
    )
    summary_rows = []
    for source, source_frame in frame.groupby("source"):
        q = by_prompt[by_prompt.source == source]
        base = q[q.policy == "baseline"].set_index("prompt_id")
        base_raw = source_frame[source_frame.policy == "baseline"].set_index(["prompt_id", "repeat"])
        for policy, p in q.groupby("policy"):
            p = p.set_index("prompt_id")
            common = p.index.intersection(base.index)
            ratios = []
            for prompt_id in common:
                b = float(base.loc[prompt_id, "decode_tok_s_mean"])
                if b > 0:
                    ratios.append(float(p.loc[prompt_id, "decode_tok_s_mean"]) / b)
            policy_raw = source_frame[source_frame.policy == policy].set_index(["prompt_id", "repeat"])
            paired = (
                policy_raw["decode_tok_s"]
                .reindex(base_raw.index)
                .div(base_raw["decode_tok_s"])
                .dropna()
            )
            if len(paired) > 1 and paired.std(ddof=1) > 0:
                ci_lo, ci_hi = student_t.interval(
                    0.95,
                    df=len(paired) - 1,
                    loc=paired.mean(),
                    scale=paired.std(ddof=1) / len(paired) ** 0.5,
                )
            else:
                ci_lo = ci_hi = float(paired.mean()) if len(paired) else 1.0
            summary_rows.append(
                {
                    "source": source,
                    "policy": policy,
                    "decode_tok_s_mean": float(p.decode_tok_s_mean.mean()),
                    "decode_tok_s_std": float(source_frame[source_frame.policy == policy].decode_tok_s.std(ddof=1)),
                    "speedup_vs_baseline_mean": float(np.mean(ratios)) if ratios else 1.0,
                    "speedup_paired_mean": float(paired.mean()) if len(paired) else 1.0,
                    "speedup_paired_std": float(paired.std(ddof=1)) if len(paired) > 1 else 0.0,
                    "speedup_paired_n": int(len(paired)),
                    "speedup_paired_ci95_lo": float(ci_lo),
                    "speedup_paired_ci95_hi": float(ci_hi),
                    "ms_per_token_mean": float(p.ms_per_token_mean.mean()),
                    "ttft_ms_mean": float(p.ttft_ms_mean.mean()),
                    "n": int(p.n.sum()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary["policy"] = pd.Categorical(summary["policy"], POLICY_ORDER, ordered=True)
    summary = summary.sort_values(["source", "policy"]).reset_index(drop=True)

    divergence_rows = []
    for (source, prompt_id, repeat), q in frame.groupby(["source", "prompt_id", "repeat"]):
        base = q[q.policy == "baseline"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for _, row in q.iterrows():
            divergence_rows.append(
                {
                    "source": source,
                    "policy": row.policy,
                    "prompt_id": prompt_id,
                    "repeat": repeat,
                    "exact_text_match": bool(row.text_sha1 == base_row.text_sha1),
                    "prefix_ratio": common_prefix_ratio(row.text, base_row.text),
                    "len_ratio": len(row.text) / max(len(base_row.text), 1),
                }
            )
    divergence = pd.DataFrame(divergence_rows)
    return by_prompt, summary, divergence


def make_figure(summary: pd.DataFrame) -> None:
    q = summary[summary.source == "fused_r3"]
    if q.empty:
        return
    piv = q.pivot(index="policy", columns="source", values="decode_tok_s_mean")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    piv.plot(kind="bar", ax=ax, legend=False)
    ax.set_ylabel("decode tokens/s")
    ax.set_xlabel("policy")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(OUT / "policy_speed.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            f"fused_r3={OUT / 'policy_bench_fused_r3.jsonl'}",
            f"fused_r3={OUT / 'policy_bench_noop_r3.jsonl'}",
            f"fetch8_r3={OUT / 'baseline_fetch8_r3.jsonl'}",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for spec in args.inputs:
        source, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if path.exists():
            frames.append(load_jsonl(path, source))
    if not frames:
        raise SystemExit("no input JSONL files found")
    frame = pd.concat(frames, ignore_index=True)
    by_prompt, summary, divergence = summarize(frame)
    by_prompt.to_csv(args.output_dir / "policy_speed_by_prompt.csv", index=False)
    summary.to_csv(args.output_dir / "policy_speed_summary.csv", index=False)
    divergence.to_csv(args.output_dir / "policy_output_divergence.csv", index=False)
    make_figure(summary)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
