#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# This host exposes exactly one AMD GPU (validated with `amd-smi list`).
# Do not set *_VISIBLE_DEVICES here: this ROCm 7.14 build maps those values
# to zero visible devices on the host.
python bench_decode.py "$@"
python plot_results.py --input results/decode_results.csv --output results/decode_throughput.png
