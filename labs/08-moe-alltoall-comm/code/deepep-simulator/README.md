# deepep-sim: MoE EP dispatch/combine 模拟器（V100 双卡）

这是一个 **简化版 DeepEP 实验台**：用普通 PyTorch 重实现 MoE expert-parallel
的 `dispatch` / `combine` 语义，用来在双卡 V100 上研究 **通信量模型、负载不均衡、
permutation 开销、通信与 expert GEMM 的 overlap**。

> ⚠️ **定位**：这是 *方法原型与算法正确性* 工具，**不是 DeepEP 的移植或替代**。
> V100 上的绝对带宽、延迟、扩展性 **不能外推到 H20/H100**。真正的 DeepEP 性能、
> 扩展性问题必须回到 ≥8 卡 Hopper/Blackwell 环境验证。

为什么不在 V100 上"跑 DeepEP"：本机 DeepEP 是 V2（NCCL Gin backend，要求
NCCL ≥ 2.30.4，且 kernel 面向 Hopper/Blackwell 的 TMA 等特性）；本机 NCCL 为
2.27.5，V100 是 sm_70，既缺依赖又不匹配 kernel。所以正确的做法是——自己写一个
简化版，把**算法、代价模型、overlap 结构**研究清楚，之后原样搬到 8 卡现代 GPU。

---

## 目录结构

```
deepep-simulator/
├── README.md                 # 本文档
├── env.sh                    # source 一下即可获得可用解释器 + PYTHONPATH
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── default.yaml          # 单次实验配置
│   └── sweep_v100.yaml       # 参数扫描配置
├── src/deepep_sim/
│   ├── config.py             # SimConfig：形状 / 分布 / benchmark 参数
│   ├── routing.py            # top-k 路由 + 负载分布（uniform/hotspot/rank_skew）
│   ├── permutation.py        # token permutation / inverse / weighted scatter
│   ├── comm.py               # all_to_all_single 变长交换（NCCL/gloo 通用）
│   ├── gemm.py               # grouped GEMM（逐 expert 参考实现 + bmm 版本）
│   ├── dispatch_combine.py    # 端到端 dispatch / combine（真实进程组）
│   ├── reference.py          # 单进程 oracle（校验置换/散射数学）
│   ├── cost_model.py         # 解析代价模型（bytes / latency / imbalance）
│   ├── bench.py              # CUDA events 计时 harness（torchrun 内运行）
│   └── cli.py                # plan / sweep / bench / selftest 入口
├── tests/                    # 零依赖测试（pytest 兼容 + run_tests.py）
├── scripts/
│   ├── bootstrap_env.sh      # 可选：创建独立 venv
│   ├── run_selftest.sh
│   ├── run_2gpu.sh           # 双卡真实 benchmark
│   └── run_sweep.sh          # 解析扫描（不需要 GPU）
├── profiling/nsys_notes.md   # Nsight Systems / CUPTI 拆解方法
└── results/                  # benchmark / sweep 输出
```

---

## 快速开始

```bash
cd deepep-simulator
source env.sh                       # 选择 torch 解释器并设置 PYTHONPATH

# 1) 算法核心自检 + 2-rank 分布式 round-trip 校验（CPU/gloo，秒级）
scripts/run_selftest.sh
# 等价于手动执行：
python -m deepep_sim.cli selftest
PYTHONPATH=src python tests/run_tests.py
python -m torch.distributed.run --nproc_per_node=2 --standalone \
    scripts/check_distributed.py --distribution hotspot --rank-skew 0.5

# 2) 解析代价模型扫描（CPU，不需要 GPU）
scripts/run_sweep.sh

# 3) 双卡真实 dispatch/combine 计时（需要两张 V100 空闲）
nvidia-smi                          # 先确认 GPU 空闲
scripts/run_2gpu.sh configs/default.yaml
```

`bench` 的输出写入 `results/<tag>.csv`，每行一个 rank，包含实测
`dispatch_avg_us` / `combine_avg_us`、`imbalance_factor`，以及解析模型的预测值
`model_dispatch_us` / `model_combine_us`，便于直接对比实测与模型。

---

## 代码做了什么（对应你的 7 步计划）

| 你的计划 | 本仓库对应实现 |
| --- | --- |
| 1. 构造 token，按 top-k 路由到两个 rank | `routing.make_routing_logits` + `route_tokens` + `expert_to_rank` |
| 2. 统计每 rank/每 expert 的 token 数 | `routing.count_tokens_per_expert` / `imbalance_factor`、`reference.grouped_counts` |
| 3. token permutation | `permutation.group_by_expert` / `apply_perm` |
| 4. all_to_all / P2P 交换 token | `comm.all_to_all_variable`（`all_to_all_single` 变长分块） |
| 5. expert grouped GEMM | `gemm.grouped_gemm_ref`（逐 expert）/ `grouped_gemm_bmm` |
| 6. combine 回原 rank 并加权求和 | `dispatch_combine.SimEP.combine` + `permutation.scatter_back` |
| 7. 对比配置（分布/dtype/拓扑） | `cli sweep` + `configs/*.yaml`（rank_skew、hotspot、dtype） |

### dispatch 语义

每个 `(token, k)` 对打上 4 个 int64 标签 `(src_rank, token_row, k, expert_id)`，
按目标 rank 稳定排序后走变长 `all_to_all_single`；接收侧再按本地 expert 排序
（= permutation），得到 grouped GEMM 需要的连续布局。

### combine 语义

把 expert 输出按标签送回原 rank，原 rank 用本地 `topk_weights[row, k]` 做
`index_add_` 加权求和。**正确性不变量**：专家为恒等映射、权重按 token 归一化时，
`combine(dispatch(x)) == x`。这正是 `selftest` 与 `test_reference.py` 校验的点。

---

## 可以研究什么（与你的清单一致）

- **dispatch/combine 通信量模型**：实测 vs `cost_model` 预测（bytes 项 + latency 项）。
- **路由不均 → rank 间等待**：`rank_skew` / `hotspot` 拉高 `imbalance_factor`，
  观察最慢 rank 如何决定整体时间。
- **all-to-all 与分组 P2P 的延迟差异**：当前用 `all_to_all_single`；可加
  `send/recv` 版分组 P2P 做 A/B（`comm.py` 已预留接口风格）。
- **permutation / 反 permutation 开销**：`group_by_expert` 的排序 + `index_select`
  可单独计时（`bench._time_fn`）。
- **通信与 expert GEMM overlap**：用多 stream 把 `all_to_all` 与
  `grouped_gemm_ref` 交叠，扫流水深度。
- **量化 activation 对通信量的影响**：`dtype: fp8` 时 `cost_model` 按 1 byte/元素
  计通信量（真实 fp8 kernel 需另接量化）。
- **P2P 与 host staging 差异**：用 `cudaMemcpyPeer` / pinned host 中转做对照。

---

## 环境说明（重要）

- 本机可用的 torch 为 `2.10.0+cu128`，`arch_list` **包含 sm_70**，已在 V100 上
  验证可见 2 卡；NCCL 为 `2.27.5`。
- `env.sh` 按 `$DEEPEPSIM_PYTHON` → `./.venv` → 机器上已知的 torch 解释器 的顺序
  选择解释器。想完全隔离可跑 `scripts/bootstrap_env.sh`（会下载 torch）。
- `bench` 需要两张 GPU 空闲；若被占用会 OOM，先 `nvidia-smi` 确认。

---

## 设计边界（写论文时必须声明）

1. 通信走通用 `all_to_all_single`，**不含** DeepEP 的 NVLink/RDMA 专用 kernel、
   zero-SM、FP8 融合等优化；因此**只能比较相对趋势**，不能当作 DeepEP 基线。
2. V100 无 TMA、无 FP8 tensor core、NVLink 为 2.0；结论仅限"算法/正确性/模型原型"。
3. 建议：V100 只做 prototype；最终结论在 A100/H100/H20 或 ≥8 卡 NVLink 上复现后再下。

一句话：**双卡 V100 是学习 MoE EP 通信的实验台，不是 DeepEP 的研究平台。**
本仓库把"值得在 V100 上做的事"沉淀成可复用接口，换到 8 卡时直接复用。
