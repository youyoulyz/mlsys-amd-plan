#!/usr/bin/env bash
# HBM microbenchmark (torch copy/triad) inside the vLLM docker image with GPU access.
set -eu
EXP=/work/experiments/arith_intensity_7b_7900xtx_20260924
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --ipc=host \
  -e OUT="$EXP/results/common/hbm_bandwidth.json" \
  -v /mnt/luyuzhou/hpc/moecho:/work -w /work \
  free.hubfast.cn/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0 \
  bash -lc "python3 $EXP/scripts/bench_hbm_bandwidth.py > $EXP/logs/hbm_bandwidth.log 2>&1"
