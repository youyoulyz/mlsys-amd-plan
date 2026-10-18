#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${FREETOKEN_ANALYSIS_PYTHON:-/home/luyzh/.venv/bin/python3}"

"$PY" "$ROOT/scripts/analyze_experts.py"
"$PY" "$ROOT/scripts/analyze_clustering.py"
"$PY" "$ROOT/scripts/analyze_clustering_bootstrap.py"
"$PY" "$ROOT/scripts/analyze_policy.py"
"$PY" "$ROOT/scripts/analyze_policy_bench.py"
"$PY" "$ROOT/scripts/analyze_policy_accuracy.py" \
  --input "$ROOT/results/policy-v1/policy_accuracy_all_r10.jsonl"
