#!/usr/bin/env bash
# Run the distributed dispatch/combine benchmark on 2 GPUs.
#
#   scripts/run_2gpu.sh [config.yaml]
#
# Prerequisite: both GPUs must be free (check nvidia-smi).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
source env.sh

config="${1:-configs/default.yaml}"
nproc="${NPROC:-2}"

"$DEEPEPSIM_PYTHON" -m torch.distributed.run \
    --nproc_per_node="$nproc" \
    --standalone \
    -m deepep_sim.cli bench --config "$config"
