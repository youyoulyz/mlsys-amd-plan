from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/policy-v1"
POLICY_ORDER = ["baseline", "noop", "top85", "top90", "floor", "combined", "layer"]


def load(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            item = json.loads(line)
            rows.append(
                {
                    "policy": item["policy"],
                    "task_id": item["task_id"],
                    "family": item["family"],
                    "repeat": item["repeat"],
                    "expected": item["expected"],
                    "predicted": item["predicted"],
                    "answer_text": item["answer_text"],
                    "passed": bool(item["passed"]),
                    "completion_tokens": item["measured"]["completion_tokens"],
                    "text": item["measured"]["text"],
                }
            )
    return pd.DataFrame(rows)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=OUT / "policy_accuracy_all_r10.jsonl")
    parser.add_argument(
        "--stress-input",
        type=Path,
        nargs="*",
        default=[
            OUT / "policy_accuracy_math_total_r10.jsonl",
            OUT / "policy_accuracy_all_r10.jsonl",
        ],
    )
    parser.add_argument("--stress-task", default="math_expert_total")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    frame = load(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (policy, family), q in frame.groupby(["policy", "family"]):
        successes = int(q.passed.sum())
        total = len(q)
        lo, hi = wilson(successes, total)
        rows.append(
            {
                "policy": policy,
                "family": family,
                "passed": successes,
                "total": total,
                "accuracy": successes / total,
                "wilson_lo": lo,
                "wilson_hi": hi,
                "mean_completion_tokens": q.completion_tokens.mean(),
            }
        )
    family = pd.DataFrame(rows)
    family["policy"] = pd.Categorical(family.policy, POLICY_ORDER, ordered=True)
    family = family.sort_values(["family", "policy"]).reset_index(drop=True)

    rows = []
    for policy, q in frame.groupby("policy"):
        successes = int(q.passed.sum())
        total = len(q)
        lo, hi = wilson(successes, total)
        rows.append(
            {
                "policy": policy,
                "passed": successes,
                "total": total,
                "accuracy": successes / total,
                "wilson_lo": lo,
                "wilson_hi": hi,
                "mean_completion_tokens": q.completion_tokens.mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary["policy"] = pd.Categorical(summary.policy, POLICY_ORDER, ordered=True)
    summary = summary.sort_values("policy").reset_index(drop=True)

    baseline = summary[summary.policy == "baseline"].iloc[0]
    fisher_p = []
    for row in summary.itertuples():
        _, pvalue = fisher_exact(
            [[int(row.passed), int(row.total - row.passed)], [int(baseline.passed), int(baseline.total - baseline.passed)]],
            alternative="less",
        )
        fisher_p.append(1.0 if row.policy == "baseline" else float(pvalue))
    summary["fisher_p_vs_baseline"] = fisher_p

    task = (
        frame.groupby(["policy", "family", "task_id"], as_index=False)
        .agg(passed=("passed", "sum"), total=("passed", "size"))
    )
    task["accuracy"] = task.passed / task.total
    task["policy"] = pd.Categorical(task.policy, POLICY_ORDER, ordered=True)
    task = task.sort_values(["family", "task_id", "policy"]).reset_index(drop=True)
    family.to_csv(args.output_dir / "policy_accuracy_by_family.csv", index=False)
    task.to_csv(args.output_dir / "policy_accuracy_by_task.csv", index=False)
    summary.to_csv(args.output_dir / "policy_accuracy_summary.csv", index=False)
    print(summary.to_string(index=False))
    print()
    print(family.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    summary.plot(x="policy", y="accuracy", kind="bar", ax=axes[0], ylim=(0, 1.05), legend=False)
    axes[0].set_title("9-task exact-answer suite (10 repeats)")
    axes[0].set_ylabel("accuracy")
    axes[0].tick_params(axis="x", rotation=30)

    stress_paths = [path for path in args.stress_input if path.exists()]
    if stress_paths:
        stress = pd.concat([load(path) for path in stress_paths], ignore_index=True)
        stress = stress[stress.task_id == args.stress_task].copy()
        rows = []
        for policy, q in stress.groupby("policy"):
            successes = int(q.passed.sum())
            total = len(q)
            lo, hi = wilson(successes, total)
            rows.append(
                {
                    "policy": policy,
                    "task_id": q.task_id.iloc[0],
                    "passed": successes,
                    "total": total,
                    "accuracy": successes / total,
                    "wilson_lo": lo,
                    "wilson_hi": hi,
                }
            )
        stress_summary = pd.DataFrame(rows)
        stress_summary["policy"] = pd.Categorical(stress_summary.policy, POLICY_ORDER, ordered=True)
        stress_summary = stress_summary.sort_values("policy").reset_index(drop=True)
        stress_summary.to_csv(args.output_dir / "policy_accuracy_stress.csv", index=False)
        stress_summary.plot(
            x="policy", y="accuracy", kind="bar", ax=axes[1], ylim=(0, 1.05), legend=False
        )
        axes[1].set_title("focused stress task (two 10-repeat runs)")
        axes[1].set_ylabel("accuracy")
        axes[1].tick_params(axis="x", rotation=30)
        print()
        print(stress_summary.to_string(index=False))
    fig.tight_layout()
    fig.savefig(args.output_dir / "policy_accuracy.png", dpi=160)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
