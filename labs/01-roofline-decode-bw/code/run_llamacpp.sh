#!/usr/bin/env bash
# llama.cpp model-level decode/prompt benchmarks for the Qwen2.5-7B GGUFs.
set -u
set -o pipefail
BIN=/opt/llamacpp/llama.cpp/build-gfx1100-rocm10/bin
EXP="$(cd "$(dirname "$0")/.." && pwd)"
GGUF="$EXP/models/gguf"
OUT="$EXP/results/llamacpp"
LOG="$EXP/logs/llamacpp"
mkdir -p "$OUT" "$LOG"

for stem in fp16 bf16 f32 q8_0 q4_k_m; do
  m="$GGUF/qwen2.5-7b-instruct-${stem}.gguf"
  [[ -f "$m" ]] || { echo "SKIP missing $m"; continue; }
  echo "===== $stem $(date -Is) ====="
  "$BIN/llama-bench" -m "$m" -p 512 -n 128 -ngl 99 -r 5 -o json \
      > "$OUT/${stem}.json" 2> "$LOG/${stem}.log"
  echo "RC=$? -> $OUT/${stem}.json"
done
