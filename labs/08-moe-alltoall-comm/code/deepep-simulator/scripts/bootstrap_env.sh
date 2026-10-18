#!/usr/bin/env bash
# OPTIONAL: create a dedicated venv for the simulator.
# This downloads a torch wheel (~GBs). Skip it if you already have a
# torch-enabled interpreter and are happy to point env.sh at it.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; falling back to python -m venv" >&2
    python3 -m venv .venv
    ./.venv/bin/pip install -U pip
    ./.venv/bin/pip install -r requirements.txt
else
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
fi

echo "Done. Use: source env.sh"
