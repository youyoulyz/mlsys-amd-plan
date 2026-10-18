#!/bin/bash
# MoE RPB + nwarps sweep for LFM2.5-8B-A1B on RX 7900 XTX
set -e

MODEL="/mnt/public/GGUF_models/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf"
BENCH="build-hip/bin/llama-bench"
OUT="results/moe_sweep_$(date +%Y%m%d_%H%M).txt"

mkdir -p results

{
echo "# MoE RPB/nwarps Sweep — $(date)"
echo "# Model: LFM2.5-8B-A1B-Q4_K_M"
echo "# GPU: $(rocminfo 2>/dev/null | grep "Marketing Name" | head -1)"
echo "# Args: -p 0 -n 64 -ngl 99 -t 1 -r 5"
echo ""

echo "## RPB Sweep (nwarps=default=8)"
for rpb in 1 2 4 8; do
    echo "--- RPB=$rpb ---"
    LLAMA_MMVQ_RPB=$rpb $BENCH -m "$MODEL" -p 0 -n 64 -ngl 99 -t 1 -r 5 2>&1 | grep "^|"
done

echo ""
echo "## NWARPS Sweep (rpb=default=1)"
for nw in 8 4 2 1; do
    echo "--- NWARPS=$nw ---"
    LLAMA_MMVQ_NWARPS=$nw $BENCH -m "$MODEL" -p 0 -n 64 -ngl 99 -t 1 -r 5 2>&1 | grep "^|"
done

echo ""
echo "## Combined Sweep"
for rpb in 1 2 4 8; do
    for nw in 8 4 2 1; do
        if [ $rpb -eq 1 ] && [ $nw -eq 8 ]; then continue; fi  # skip default
        LLAMA_MMVQ_RPB=$rpb LLAMA_MMVQ_NWARPS=$nw \
            $BENCH -m "$MODEL" -p 0 -n 64 -ngl 99 -t 1 -r 3 2>&1 | \
            grep "^|" | sed "s/^/RPB=${rpb}_NW=${nw} /"
    done
done

echo ""
echo "# Done at $(date)"
} | tee "$OUT"

echo "Results saved to $OUT"
