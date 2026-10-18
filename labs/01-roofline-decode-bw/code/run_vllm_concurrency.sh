#!/usr/bin/env bash
# vLLM static-batch concurrency sweep for a ~4B model (Qwen2.5-7B) on RX 7900 XTX.
# Precision paths: FP16, AWQ-INT4, GPTQ-INT4, GPTQ-INT8, FP8-dynamic.
# Batch N in {1,2,4,8,16}; failures are recorded per N and never abort the matrix.
set -u
set -o pipefail

ROOT=/work
EXP="$ROOT/experiments/arith_intensity_7b_7900xtx_20260924"
OUT="$EXP/results/vllm_concurrency"
LOG="$EXP/logs/vllm_concurrency"
mkdir -p "$OUT" "$LOG"

INPUT_LEN=512
OUTPUT_LEN=128
NUM_ITERS=5
NUM_WARMUP=2
MAX_MODEL_LEN=4096
GPU_UTIL=0.85

run_case() {
  local name="$1" model="$2" dtype="$3" quantization="$4" batch="$5"
  local tag="${name}_b${batch}"
  local json="$OUT/${tag}.json"
  local log="$LOG/${tag}.log"
  local -a cmd=(vllm bench latency
    --model "$model"
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
    --batch-size "$batch"
    --num-iters "$NUM_ITERS"
    --num-iters-warmup "$NUM_WARMUP"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_UTIL"
    --enforce-eager
    --output-json "$json")
  if [[ "$dtype" != "auto" ]]; then
    cmd+=(--dtype "$dtype")
  fi
  if [[ "$quantization" != "auto" ]]; then
    cmd+=(--quantization "$quantization")
  fi

  echo "===== BEGIN $tag $(date -Is) ====="
  printf 'CMD:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  set +e
  "${cmd[@]}" >"$log" 2>&1
  local rc=$?
  set -e
  echo "RC=$rc"
  tail -n 14 "$log" || true
  printf '%s\n' "$rc" > "$OUT/${tag}.rc"
  echo "===== END $tag $(date -Is) ====="
}

set -e
CASES=(
  "base_fp16|/work/experiments/arith_intensity_7b_7900xtx_20260924/models/hf/Qwen2.5-7B-Instruct|float16|auto"
  "awq_int4|/work/experiments/arith_intensity_7b_7900xtx_20260924/models/hf/Qwen2.5-7B-Instruct-AWQ|float16|auto"
  "gptq_int4|/work/experiments/arith_intensity_7b_7900xtx_20260924/models/hf/Qwen2.5-7B-Instruct-GPTQ-Int4|float16|auto"
  "gptq_int8|/work/experiments/arith_intensity_7b_7900xtx_20260924/models/hf/Qwen2.5-7B-Instruct-GPTQ-Int8|float16|auto"
  "fp8_dynamic|/work/experiments/arith_intensity_7b_7900xtx_20260924/models/hf/Qwen2.5-7B-Instruct-FP8-dynamic|auto|auto"
)
BATCHES=(1 2 4 8 16)

for spec in "${CASES[@]}"; do
  IFS='|' read -r name model dtype quant <<<"$spec"
  for b in "${BATCHES[@]}"; do
    run_case "$name" "$model" "$dtype" "$quant" "$b"
  done
done

echo "===== MATRIX DONE $(date -Is) ====="
