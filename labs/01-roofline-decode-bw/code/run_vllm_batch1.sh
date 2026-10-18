#!/usr/bin/env bash
# vLLM batch=1 latency sweep for Qwen2.5-7B on RX 7900 XTX.
# Precision paths: BF16 / FP16 / FP32 / GPTQ-INT4 / AWQ-INT4 / GPTQ-INT8 / FP8-dynamic.
set -u
set -o pipefail

ROOT=/work
EXP="$ROOT/experiments/arith_intensity_7b_7900xtx_20260924"
OUT="$EXP/results/vllm_batch1"
LOG="$EXP/logs/vllm_batch1"
mkdir -p "$OUT" "$LOG"

run_case() {
  local name="$1" model="$2" dtype="$3" quantization="$4"
  local json="$OUT/${name}.json"
  local log="$LOG/${name}.log"
  local -a cmd=(vllm bench latency
    --model "$model"
    --input-len 512
    --output-len 128
    --batch-size 1
    --num-iters 5
    --num-iters-warmup 2
    --max-model-len 4096
    --gpu-memory-utilization 0.85
    --enforce-eager
    --output-json "$json")
  if [[ "$dtype" != "auto" ]]; then cmd+=(--dtype "$dtype"); fi
  if [[ "$quantization" != "auto" ]]; then cmd+=(--quantization "$quantization"); fi

  echo "===== BEGIN $name $(date -Is) ====="
  printf 'CMD:'; printf ' %q' "${cmd[@]}"; printf '\n'
  set +e
  "${cmd[@]}" >"$log" 2>&1
  local rc=$?
  set -e
  echo "RC=$rc"
  tail -n 14 "$log" || true
  printf '%s\n' "$rc" > "$OUT/${name}.rc"
  echo "===== END $name $(date -Is) ====="
}

set -e
run_case base_bf16   /work/experiments/arith_intensity_7b_7900xtx_20260924/models/hf/Qwen2.5-7B-Instruct             bfloat16 auto
run_case base_fp16   /work/experiments/arith_intensity_7b_7900xtx_20260924/models/hf/Qwen2.5-7B-Instruct             float16  auto
run_case base_fp32   /work/experiments/arith_intensity_7b_7900xtx_20260924/models/hf/Qwen2.5-7B-Instruct             float32  auto
run_case gptq_int4   /work/experiments/arith_intensity_7b_7900xtx_20260924/models/hf/Qwen2.5-7B-Instruct-GPTQ-Int4   float16  auto
run_case gptq_int8   /work/experiments/arith_intensity_7b_7900xtx_20260924/models/hf/Qwen2.5-7B-Instruct-GPTQ-Int8   float16  auto
run_case awq_int4    /work/experiments/arith_intensity_7b_7900xtx_20260924/models/hf/Qwen2.5-7B-Instruct-AWQ         float16  auto
run_case fp8_dynamic /work/experiments/arith_intensity_7b_7900xtx_20260924/models/hf/Qwen2.5-7B-Instruct-FP8-dynamic auto     auto
