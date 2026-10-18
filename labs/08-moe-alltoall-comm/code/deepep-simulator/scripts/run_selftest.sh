#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
source env.sh
"$DEEPEPSIM_PYTHON" -m deepep_sim.cli selftest
PYTHONPATH=src "$DEEPEPSIM_PYTHON" tests/run_tests.py

# 2-rank distributed round-trip check over gloo (CPU; no GPU needed)
PYTHONPATH=src "$DEEPEPSIM_PYTHON" -m torch.distributed.run --nproc_per_node=2 --standalone \
    scripts/check_distributed.py --distribution hotspot --rank-skew 0.5
