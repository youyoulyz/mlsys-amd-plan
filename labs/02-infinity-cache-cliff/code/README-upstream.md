# scripts

- `bench_cache_sweep.py` – 实验 A：工作集扫描（read/copy/write/triad）。
- `sweep_llama_mulmat.py` – 实验 B 驱动：对 llama.cpp `test-backend-ops perf`
  跑 `MUL_MAT(m=4096, n∈{1,16}, k)` 权重体量扫描。
- `analyze_ifcache.py` – 汇总，生成 CSV/JSON/图。
- `test-backend-ops.cpp.orig` – 上游原始 `tests/test-backend-ops.cpp`（备份）。
- `test-backend-ops.orig.bin` – 上游原始 `test-backend-ops` 二进制（备份，已还原）。
- `test-backend-ops.patched.bin` – 打了形状补丁的二进制（供复现实验 B）。
- `perf_shapes.patch.cpp` – 新增到 `make_test_cases_perf()` 的形状代码块。

实验 B 复现步骤：
1. 把 `perf_shapes.patch.cpp` 的内容插入到
   `/opt/llamacpp/llama.cpp/tests/test-backend-ops.cpp` 的
   `make_test_cases_perf()` 中（锚点：`// qwen3-30b-a3b` 之前）。
2. `cd /opt/llamacpp/llama.cpp/build-gfx1100-rocm10 && make test-backend-ops -j8`
3. 运行 `sweep_llama_mulmat.py`。
4. 用 `test-backend-ops.cpp.orig` / `test-backend-ops.orig.bin` 还原。

> 实验完成后已经还原：`git status` 干净，`bin/test-backend-ops` 为原始二进制。
