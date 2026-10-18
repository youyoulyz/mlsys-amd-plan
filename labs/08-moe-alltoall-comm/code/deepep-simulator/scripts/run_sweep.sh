#!/usr/bin/env bash
# Analytical sweep (no GPUs needed): predicts cost across a shape panel.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
source env.sh

"$DEEPEPSIM_PYTHON" -m deepep_sim.cli sweep \
    --config configs/sweep_v100.yaml \
    --tokens 1024 2048 4096 8192 \
    --hidden 1024 2048 4096 \
    --topk 1 2 4 8 \
    --distributions uniform hotspot rank_skew \
    --output results/sweep_v100.csv
