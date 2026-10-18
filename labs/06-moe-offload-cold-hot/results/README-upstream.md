# Qwen3.6-35B-A3B-NVFP4 MoE offload expert analysis

Self-contained probes, policy experiments, analysis scripts, raw traces, result
tables, figures, and reports for the single RTX 3080 20GB GPU1 run.

The run used:

- FreeToken commit: `54d767d`
- Model: `/home/luyzh/models/Qwen3.6-35B-A3B-NVFP4`
- GPU: GPU1, RTX 3080 20GB, PCIe 4.0 x8
- MoE strategy: `hybrid`
- Expert cache slots: `7200`
- Batch size: `1`
- Runtime: `/home/luyzh/.venvs/freetoken/bin/python`
- Analysis: `/home/luyzh/.venv/bin/python3`
- CUDA graph: disabled for probing, enabled for the real speed A/B

The model, virtual environments, and checkpoint are intentionally not copied.

## Layout

```text
.
├── probe/
│   ├── launch_probe.py            # server launch with the original run config
│   ├── policy_kernel.py           # fused top-p / low-floor routing kernel
│   └── sitecustomize.py           # runtime route probe and policy monkeypatch
├── scripts/
│   ├── analyze_experts.py         # decode/prefill cache and per-expert analysis
│   ├── analyze_clustering.py      # concentration and scenario fingerprint analysis
│   ├── analyze_clustering_bootstrap.py
│   ├── analyze_policy.py          # offline policy/route analysis
│   ├── analyze_policy_bench.py    # real speed A/B aggregation
│   ├── analyze_policy_accuracy.py # exact-answer accuracy aggregation
│   ├── check_policy_semantics.py  # synthetic kernel correctness checks
│   ├── probe_workloads.py         # prior math/python/moe workload probe
│   ├── run_policy_accuracy.py     # exact-answer policy A/B
│   ├── run_policy_bench.py        # real decode policy A/B
│   └── run_all.sh
├── traces/
│   ├── trace-410395.tsv           # original route trace
│   └── trace-555575.tsv           # trace with decode gate weights
├── results/
│   ├── analysis-v1/               # cache, fetch/CPU, per-layer and per-expert results
│   ├── clustering-v1/             # concentration, overlap, bootstrap and figures
│   └── policy-v1/                 # top-p, weight-floor and layer-policy results
└── reports/
    ├── expert-activation-distribution-zh.md
    ├── expert-clustering-report-zh.md
    ├── experiment-report-zh.md
    └── routing-truncation-verification-zh.md
```

## Probe

The probe is loaded through `sitecustomize.py`. Set `PYTHONPATH` before starting
Python so `sitecustomize` is imported at interpreter startup.

```bash
REPO=/mnt/luyuzhou/hpc/FreeToken
EXP="$REPO/experiments/qwen3.6-35b-a3b-nvfp4-moe-offload"

export FREETOKEN_MOE_PROBE=1
export FREETOKEN_PROBE_DIR="$EXP/traces"
export FREETOKEN_PROBE_TOPK=8
export PYTHONPATH="$EXP/probe:${PYTHONPATH:-}"

/home/luyzh/.venvs/freetoken/bin/python "$EXP/probe/launch_probe.py"
```

`FREETOKEN_MODEL` and `FREETOKEN_GPU` can override the default model path and
GPU id. The default GPU id is `1`.

The probe adds synchronization and disables CUDA graphs. Its decode speed is
therefore not a serving benchmark. The route ids, cache masks, fetch counts,
CPU fallback counts, gate weights and per-layer timing trace are taken from the
real engine execution path.

## Policy experiment

Supported policies are `baseline`, `noop`, `top85`, `top90`, `floor`,
`combined`, and `layer`. The fused policy kernel mutates route ids and gate
weights before the existing hybrid decode path; dropped routes are zero
weighted and redirected to a resident anchor.

Offline analysis:

```bash
/home/luyzh/.venv/bin/python3 "$EXP/scripts/analyze_policy.py"
```

Kernel semantics:

```bash
/home/luyzh/.venvs/freetoken/bin/python "$EXP/scripts/check_policy_semantics.py"
```

Real decode A/B:

```bash
/home/luyzh/.venvs/freetoken/bin/python "$EXP/scripts/run_policy_bench.py" \
  --policies baseline,noop,top85,top90,floor,combined,layer \
  --repeats 3 --decode 255 \
  --output "$EXP/results/policy-v1/policy_bench_fused_r3.jsonl"
```

Exact-answer accuracy A/B:

```bash
/home/luyzh/.venvs/freetoken/bin/python "$EXP/scripts/run_policy_accuracy.py" \
  --policies baseline,noop,top85,top90,floor,combined,layer \
  --repeats 10 --max-tokens 64 --no-thinking \
  --output "$EXP/results/policy-v1/policy_accuracy_all_r10.jsonl"
```

Aggregate reports:

```bash
/home/luyzh/.venv/bin/python3 "$EXP/scripts/analyze_policy_bench.py"
/home/luyzh/.venv/bin/python3 "$EXP/scripts/analyze_policy_accuracy.py" \
  --input "$EXP/results/policy-v1/policy_accuracy_all_r10.jsonl"
```

## Old analysis

The repository `.venv` is not the current runtime environment. Use
`/home/luyzh/.venvs/freetoken/bin/python` for FreeToken and
`/home/luyzh/.venv/bin/python3` for pandas, matplotlib and scipy.

```bash
EXP=/mnt/luyuzhou/hpc/FreeToken/experiments/qwen3.6-35b-a3b-nvfp4-moe-offload
PY=/home/luyzh/.venv/bin/python3

"$PY" "$EXP/scripts/analyze_experts.py"
"$PY" "$EXP/scripts/analyze_clustering.py"
"$PY" "$EXP/scripts/analyze_clustering_bootstrap.py"
```

Or:

```bash
FREETOKEN_ANALYSIS_PYTHON=/home/luyzh/.venv/bin/python3 \
  "$EXP/scripts/run_all.sh"
```

Analysis output paths are derived from the script location. No script depends
on the old external working directory after the move.

## Reports

- `reports/expert-activation-distribution-zh.md`: expert activation
  distribution, concentration, prefill/decode regimes, topic drift and cache
  implications.
- `reports/expert-clustering-report-zh.md`: clustering and topic separability.
- `reports/experiment-report-zh.md`: full offload experiment report.
- `reports/routing-truncation-verification-zh.md`: top-p, weight-floor and
  layer-policy speed/accuracy verification.

## Result entry points

| File | Content |
|---|---|
| `results/analysis-v1/decode_expert_stats.csv` | 30,720 rows of `(segment, layer, expert)` decode counts |
| `results/analysis-v1/prefill_expert_stats.csv` | 163,840 rows of `(chunk, layer, expert)` prefill counts |
| `results/analysis-v1/decode_per_layer.csv` | 40-layer decode cache/fetch summary |
| `results/analysis-v1/prefill_layer_chunk.csv` | per-prefill-chunk and per-layer summary |
| `results/clustering-v1/concentration_summary.csv` | top-k concentration, Gini and effective experts |
| `results/clustering-v1/pairwise_similarity.csv` | prefill/decode and topic fingerprint similarity |
| `results/clustering-v1/temporal_bootstrap_summary.csv` | temporal drift versus sampling baseline |
| `results/clustering-v1/top_experts_by_scenario_layer.csv` | top experts by scenario and layer |
| `results/policy-v1/offline/policy_summary.json` | projected route, mass and miss removal |
| `results/policy-v1/offline/policy_by_layer.csv` | per-layer policy effects |
| `results/policy-v1/offline/weight_floor_bound.csv` | high-weight floor impossibility check |
| `results/policy-v1/policy_speed_summary.csv` | real decode speed A/B |
| `results/policy-v1/policy_accuracy_summary.csv` | exact-answer accuracy A/B |
| `results/policy-v1/policy_accuracy_by_task.csv` | per-task accuracy |
| `results/policy-v1/policy_accuracy_stress.csv` | focused accuracy stress test |
