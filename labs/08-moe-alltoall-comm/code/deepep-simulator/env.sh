#!/usr/bin/env bash
# Source this to get a working interpreter for the simulator:
#
#     source env.sh
#     python -m deepep_sim.cli selftest
#
# Resolution order:
#   1. $DEEPEPSIM_PYTHON if set
#   2. ./  .venv/bin/python (a dedicated venv, see scripts/bootstrap_env.sh)
#   3. a known torch-enabled interpreter on this machine
#
# NOTE: this box currently has no standalone torch install; the only torch
# build available reports sm_70 (V100) support, so we reuse it by default.

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [ -n "${DEEPEPSIM_PYTHON:-}" ] && [ -x "$DEEPEPSIM_PYTHON" ]; then
    _py="$DEEPEPSIM_PYTHON"
elif [ -x "$_here/.venv/bin/python" ]; then
    _py="$_here/.venv/bin/python"
else
    _py="/mnt/luyuzhou/hpc/1Cat-vLLM/.venv/bin/python"
fi

export DEEPEPSIM_PYTHON="$_py"
export DEEPEPSIM_ROOT="$_here"
export PYTHONPATH="$_here/src:${PYTHONPATH:-}"

echo "[env] DEEPEPSIM_PYTHON=$DEEPEPSIM_PYTHON"
echo "[env] PYTHONPATH=$PYTHONPATH"
