#!/usr/bin/env bash
# 本机验证矩阵脚本（2026-10-17 产出 results/local_run.log 的那台主机）
# 路径是本机专属（venv / LMCache 源码树 / 数据目录），其他机器请按 README「跑法」改；
# 数据目录必须在真实本地盘上（/tmp 若为 tmpfs、NAS/NFS 都不行）。
# CUDA_VISIBLE_DEVICES= : ROCm 上 torch.cuda.is_available() 恒为 True，会把 LMCache
# 导入带进编译版 c_ops 分支；本微基准纯 CPU，禁用后可走 non_cuda_equivalents 纯 Python 回退
set -u
PY=/home/luyzh/.venvs/lab11/bin/python
SCRIPT=/mnt/luyuzhou/hello-mlsys/mlsys-amd-plan/labs/11-kv-storage-tiering/code/storage_backend_io_benchmark.py
BASE=/home/luyzh/tmp_lab11
NUM_OPS=8192
export PYTHONPATH=/mnt/luyuzhou/hpc/LMCache
export CUDA_VISIBLE_DEVICES=

for CONC in 4 16 32; do
  for OD in pc od; do
    TAG="c${CONC}_${OD}"
    DIR="$BASE/bench_${TAG}"
    JSON="$BASE/bench_${TAG}.json"
    ODARGS=""
    [ "$OD" = "od" ] && ODARGS="--local-disk-odirect --raw-odirect"
    echo "=== RUN $TAG (num_ops=$NUM_OPS, concurrency=$CONC, odirect=$OD) $(date +%H:%M:%S) ==="
    timeout 600 $PY $SCRIPT \
      --num-ops $NUM_OPS --concurrency $CONC \
      --local-disk-dir "$DIR" --max-local-disk-gb 4 \
      --raw-device-size-gb 1 \
      $ODARGS \
      --output-json "$JSON" 2>&1 | tail -6
    exit_code=${PIPESTATUS[0]}   # 管道的 $? 是 tail 的退出码，取 PIPESTATUS[0] 才是 benchmark 的
    echo "=== END $TAG exit=$exit_code $(date +%H:%M:%S) ==="
  done
done
echo "ALL RUNS DONE"