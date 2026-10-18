#!/usr/bin/env bash
# 无 P2P 双卡 × Qwen3.6-35B-A3B-MTP × llama.cpp 全矩阵 (2026-08-09, stock 驱动)
# 用法: bash experiments/mtp_matrix_nop2p.sh [reps 默认3]
set -u
REPS=${1:-3}
RUNNER=/mnt/luyuzhou/hpc/moecho/experiments/run_mtp_experiment.py
Q3=/mnt/public/GGUF_models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
Q4=/mnt/public/GGUF_models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
OUT=/mnt/luyuzhou/hpc/moecho/results/data
mkdir -p "$OUT"

run_cell() {  # name model gpus dmax temp [extra...]
  local name=$1 model=$2 gpus=$3 dmax=$4 temp=$5; shift 5
  for r in $(seq 1 "$REPS"); do
    echo ">>> [$(date +%H:%M:%S)] CELL $name rep $r"
    python3 "$RUNNER" --model "$model" --gpus "$gpus" --draft-max "$dmax" --temp "$temp" \
      --output "$OUT/mtp_nop2p_${name}_rep${r}.json" "$@" \
      || echo "!!! CELL $name rep $r FAILED"
  done
}

echo "=========== S1 单卡 Q3_K_XL (GPU1, 基线锚 v7) ==========="
run_cell s1_single_q3 "$Q3" "1" 2 0.6

echo "=========== S2 双卡 PP Q3_K_XL (主实验) + dmon 采样 ==========="
nvidia-smi dmon -s mu -d 2 > "$OUT/mtp_nop2p_dmon_s2.csv" 2>&1 &
DMON_PID=$!
run_cell s2_pp_q3 "$Q3" "0,1" 2 0.6
kill "$DMON_PID" 2>/dev/null

echo "=========== S2x draft-max=4 子集 (1 rep) ==========="
REPS=1 run_cell s2x_dm4_q3 "$Q3" "0,1" 4 0.6
echo "=========== S2x temp=0.0 子集 (1 rep) ==========="
REPS=1 run_cell s2x_t0_q3 "$Q3" "0,1" 2 0.0

echo "=========== S3 双卡 TP=2 tensor (NCCL_P2P_DISABLE=1) ==========="
for r in $(seq 1 "$REPS"); do
  echo ">>> [$(date +%H:%M:%S)] CELL s3_tp2_q3 rep $r"
  NCCL_P2P_DISABLE=1 python3 "$RUNNER" --model "$Q3" --gpus "0,1" --draft-max 2 --temp 0.6 \
    --split-mode tensor --tensor-split 0.5/0.5 \
    --output "$OUT/mtp_nop2p_s3_tp2_q3_rep${r}.json" \
    || echo "!!! CELL s3_tp2_q3 rep $r FAILED"
done

echo "=========== S4 双卡 PP Q4_K_M ==========="
run_cell s4_pp_q4 "$Q4" "0,1" 2 0.6

echo "=========== ALL CELLS DONE $(date) ==========="
