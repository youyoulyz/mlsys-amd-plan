# 路由权重条件截断的验证性实验

## 0. 结论摘要

本实验在单张 RTX 3080 20GB、GPU1、PCIe 4.0 x8 上，对 Qwen3.6-35B-A3B-NVFP4 的 hybrid MoE decode 路径验证了三类策略：

1. 在卡专家权重质量达到阈值后，删除所有未在卡 expert route，并重新归一化。
2. `w_i < 0.05` 的尾路由删除，`w_i > 0.30` 的非在卡路由禁止删除。
3. 浅层放宽、深层激进的逐层策略。

实测结论：

- Top-85 只删除 **3.71% route、2.55% gate mass**，消除 **27.0% miss**；Top-90 只消除 **16.6% miss**。
- `w_i > 0.30` 的兜底在 0.85 阈值下**不可能被触发**。若在卡质量已经达到 0.85，单个未在卡权重最多只能是 0.15。
- 真实 decode A/B 中，Top-85 仅 **+1.4%**，组合策略 **+2.0%**；95% 置信区间基本覆盖 1.0，不能证明稳定的端到端加速。
- 作为上限控制，`hybrid_fetch=8` 完全避免 CPU fallback 后达到 **+9.3%**，但这是 PCIe 全拉取的效果，不是权重截断本身。
- 9 个精确答案任务乘 10 次重复中，baseline/floor/combined 为 90/90，Top-85 为 83/90，Top-90 为 88/90，layer 为 82/90。
- 两个独立 server 的敏感算术任务压测合并后：baseline/floor 为 20/20，Top-85 为 9/20，Top-90 为 17/20，combined 为 18/20，layer 为 10/20。
- 因此当前实现存在**小的速度收益或噪声级收益，但已经观察到任务相关的精度风险**，不建议直接作为生产加速策略上线。

## 1. 实验环境

| 项目 | 配置 |
|---|---|
| 仓库 | `/mnt/luyuzhou/hpc/FreeToken` |
| 代码版本 | `54d767d` |
| 模型 | `/home/luyzh/models/Qwen3.6-35B-A3B-NVFP4` |
| GPU | RTX 3080 20GB，GPU1 |
| GPU 互联 | PCIe 4.0 x8 |
| MoE 策略 | hybrid |
| Expert cache | 7,200 slots |
| MoE 结构 | 40 layers，256 experts/layer，top-8 |
| Batch | 1 |
| CUDA graph | 开启 |
| hybrid auto fetch | 35.7% miss 走 PCIe，其余 CPU |
| 运行时 | `/home/luyzh/.venvs/freetoken/bin/python` |
| 分析环境 | `/home/luyzh/.venv/bin/python3` |

测试沿用之前的 math、Python list-vs-dict、MoE expert-cache 三类工作负载，不使用 GSM8K。

## 2. 策略实现

设当前层 top-8 在卡专家集合为 `R`，路由权重为 `w_i`：

```text
resident_mass = sum(w_i for i in R)

if resident_mass >= p:
    drop all i not in R
    w'_i = 0
    normalize(w')
else:
    keep all nonresident routes
```

实现上没有显式排序，但条件使用的是在卡专家总权重，累计和与排序无关。达到 `p` 后保留全部在卡专家，就保留了至少 `p` 的原质量。

本实验只截断 decode route。Prefill 仍走 full-layer streaming；prefill 每个 chunk 需要整层专家，按 route 丢权重不能消除整层 H2D 搬运，而且会直接改变 chunk 内所有 token 的表示。

被删除 route 的 expert id 被替换为一个在卡 anchor，权重置 0。这样不会改变引擎固定的 top-8 tensor shape，CPU/PCIe split 也无需改变。

策略变体：

| 名称 | 规则 |
|---|---|
| `top85` | 在卡质量 >= 0.85 时删除全部非在卡 route |
| `top90` | 在卡质量 >= 0.90 时删除全部非在卡 route |
| `floor` | 非在卡且 `w_i < 0.05` 时删除 |
| `combined` | `w_i < 0.05`，或在卡质量 >= 0.85 且 `w_i <= 0.30` 时删除 |
| `layer` | Layer 0-4 只做受限 low-floor；Layer 20/38 使用 Top-85；其余层使用 Top-90 + low-floor |
| `noop` | 经过同一 Triton 内核，但不删除 route，用于扣除内核开销 |

实现位于：

- `probe/policy_kernel.py`
- `probe/sitecustomize.py`
- `scripts/check_policy_semantics.py`

内核级合成测试已验证：低权重删除、高权重保留、在卡质量不足时不触发 Top-p 删除。

## 3. 离线路由分析

数据源为 `traces/trace-555575.tsv`，包含 3 x 255 个 decode step、40 层，共 31,320 个 token-layer 和 250,560 条 route。

### 3.1 总体结果

| 策略 | route 删除率 | 删除 gate mass | 消除 miss 比例 | `w_i > 0.30` 被误删 |
|---|---:|---:|---:|---:|
| top85 | 3.71% | 2.55% | 27.01% | 0 |
| top90 | 2.27% | 1.37% | 16.57% | 0 |
| floor | 0.48% | 0.16% | 3.49% | 0 |
| combined | 3.96% | 2.63% | 28.87% | 0 |
| layer | 2.33% | 1.36% | 16.95% | 0 |

baseline 在这 31,320 个 token-layer 中共有 34,401 条非在卡 route，平均 **1.098 miss/token-layer**。

Top-85 删除了其中 27%，剩下约 0.802 miss/token-layer。即使这些删除全部线性转化为 PCIe/CPU 工作减少，理论上也只能解释约 2.5% 的端到端收益；实测 +1.4% 与之量级一致。

### 3.2 不同工作负载

| Segment | baseline miss/token-layer | 在卡质量均值 | Top-85 删除/token-layer | miss 删除比例 | 删除质量均值 |
|---|---:|---:|---:|---:|---:|
| math | 2.045 | 0.770 | 0.287 | 14.0% | 2.54% |
| python | 0.782 | 0.917 | 0.341 | 43.6% | 2.89% |
| moe | 0.469 | 0.950 | 0.262 | 56.0% | 2.22% |

math 的 cache 最冷、路由最分散，因此 Top-85 能删除的比例最低。python 和 moe 进入热 cache 后更容易满足阈值。

### 3.3 高权重兜底不会触发

完整 trace 中有：

- 34,401 条非在卡 route。
- 479 条非在卡 `w_i > 0.30`。
- 这 479 条全部满足 `resident_mass < 0.85`，所以 Top-p 本来就不会删除它们。
- 在 `resident_mass >= 0.85` 的样本中，非在卡最大权重只有 **0.14991**。
- `resident_mass >= 0.85` 且 `w_i > 0.30` 的次数为 **0**。

这是数学必然：若在卡质量和至少 0.85，所有未在卡权重之和最多 0.15，单个未在卡权重不可能超过 0.15。

因此：

- `w_i > 0.30` 的保护规则在 Top-85 下是冗余的。
- `combined` 相对 `top85` 多删除的 route 几乎全部来自 `w_i < 0.05`，不是高权重保护产生的作用。
- 若希望高权重保护真正参与决策，需要降低 Top-p 阈值，或使用不同 threshold/floor，而不是保持 0.85。

### 3.4 层级差异

Top-85 相关逐层统计：

| Layer | 在卡质量均值 | baseline miss/token-layer | Top-85 删除/token-layer |
|---:|---:|---:|---:|
| 0 | 0.828 | 1.464 | 0.327 |
| 1 | 0.843 | 1.333 | 0.264 |
| 2 | 0.843 | 1.369 | 0.314 |
| 3 | 0.854 | 1.250 | 0.277 |
| 4 | 0.874 | 1.189 | 0.313 |
| 20 | 0.905 | 0.951 | 0.352 |
| 38 | 0.887 | 1.029 | 0.273 |
| 39 | 0.868 | 1.142 | 0.275 |

浅层 0-4 平均删除 0.299 route/token-layer，Layer 20/38 平均删除 0.313。差距很小。

`layer` 策略按预期形成了强层级差异：

- Layer 0-4：仅删除 0.033 route/token-layer。
- Layer 20/38：删除 0.313 route/token-layer，是浅层的约 9.4 倍。

但这个更激进的深层策略并没有转化为端到端速度收益，实测反而为 -0.9%。

## 4. 真实 decode 速度 A/B

每个策略启动独立 server，每个 workload 先 warm 一次，再计时 3 次。报告值来自客户端 token stream，共 9 个测量值；速度提升使用按 `(prompt, repeat)` 配对的 baseline ratio。

| 策略 | decode tok/s | 配对加速比 | 95% CI |
|---|---:|---:|---:|
| baseline | 105.76 +/- 3.51 | 1.000 | - |
| noop kernel | 105.01 +/- 3.79 | 0.994 | 0.959 - 1.028 |
| top85 | 107.14 +/- 4.88 | 1.014 | 0.971 - 1.057 |
| top90 | 104.42 +/- 4.92 | 0.988 | 0.958 - 1.017 |
| floor | 102.36 +/- 4.04 | 0.969 | 0.927 - 1.011 |
| combined | 107.83 +/- 3.97 | 1.020 | 0.997 - 1.043 |
| layer | 104.75 +/- 3.51 | 0.992 | 0.950 - 1.034 |

TTFT 始终约 1.35-1.37 秒，策略只影响 decode 路径。

去掉策略计算开销后的 `noop` 为 -0.6%，说明 Triton policy kernel 自身不是主要问题。Top-85 和 combined 的点估计只有 1.4%-2.0%，且波动为 3%-5%，无法支撑稳定加速结论。

### 4.1 CPU fallback 消除上限

`baseline_auto_r3` 与 `baseline_fetch8_r3` 的配对结果为：

| 控制项 | decode tok/s | 相对 auto |
|---|---:|---:|
| hybrid auto，35.7% miss 走 PCIe | 105.57 +/- 2.31 | 1.000 |
| `hybrid_fetch=8`，所有 miss 都走 PCIe，无 CPU fallback | 115.22 +/- 3.92 | **+9.26% [4.87%, 13.65%]** |

所以在这台机器上有一定加速空间，但它来自**彻底移除 CPU fallback**。当前权重截断只移除 17%-29% miss，远小于这个控制项。

另一个限制是：被删除 route 仍占据固定 top-8 GPU GEMM shape。策略消除了部分权重搬运和 CPU 工作，却没有减少 GPU 端的专家计算槽位数，因此不能指望接近纯 fetch8 的收益。

## 5. 精度验证

255-token 的原始长回答没有在 255 tokens 内给出最终答案，而且 baseline 自己重复运行时文本 hash 也不完全相同。因此长回答只用于速度测试，不用于判断精度。

精度采用 9 个短、可自动判分的任务，覆盖 math、Python、MoE 三类：

- 精确整数答案。
- `dict` 与 `list` 选择。
- LRU 与 random 选择。
- Top-8 路由计数。
- 在卡质量 0.85 时非在卡单权重能否大于 0.30。

每个任务、每个策略重复 10 次，关闭 thinking，`max_tokens=64`。baseline、floor、Top-85、Top-90、combined、layer 各自使用独立 server。noop 另有 3-repeat 27/27 和 focused 10/10 的控制结果。

### 5.1 总体精度

| 策略 | 正确/总数 | accuracy | Wilson 95% CI |
|---|---:|---:|---:|
| baseline | 90/90 | 100% | 95.9% - 100% |
| top85 | 83/90 | 92.2% | 84.8% - 96.2% |
| top90 | 88/90 | 97.8% | 92.3% - 99.4% |
| floor | 90/90 | 100% | 95.9% - 100% |
| combined | 90/90 | 100% | 95.9% - 100% |
| layer | 82/90 | 91.1% | 83.4% - 95.4% |

唯一出现错误的任务是 `40 x 256` expert instance 计数题。Top-85、Top-90、combined 和 layer 在这个任务上输出了 `40`、`4` 或空答案。

若把 90 次请求近似看作独立二项样本，相对于 baseline 的单侧 Fisher exact test 为：Top-85 `p=0.0069`，layer `p=0.0033`，Top-90 `p=0.249`。这个检验没有建模同一任务的重复相关性，只作为误差规模的辅助证据。

### 5.2 重点压力测试

为了确认它不是一个孤立请求，对 `40 x 256` 任务做了两个独立 server 的 10-repeat 压测，共 20 次：

| 策略 | 正确/总数 | accuracy | Wilson 95% CI |
|---|---:|---:|---:|
| baseline | 20/20 | 100% | 83.9% - 100% |
| noop | 10/10 | 100% | 72.2% - 100% |
| top85 | 9/20 | 45.0% | 25.8% - 65.8% |
| top90 | 17/20 | 85.0% | 64.0% - 94.8% |
| floor | 20/20 | 100% | 83.9% - 100% |
| combined | 18/20 | 90.0% | 69.9% - 97.2% |
| layer | 10/20 | 50.0% | 29.9% - 70.1% |

Top-85 两个独立 server 分别为 6/10 和 3/10；combined 分别为 8/10 和 10/10；layer 分别为 8/10 和 2/10。说明策略确实会改变部分请求的答案，同时不同 server 生命周期的 cache 和数值轨迹也会改变错误率。

作为对照，另一个更直接的 `40 x 8 = 320` 任务重复 10 次，baseline、Top-85、Top-90、combined、layer 全部 10/10。因此不能声称所有任务都退化，但可以确认：**当前截断幅度足以在部分数学计数任务上稳定地改变最终答案。** noop、floor 和 baseline 的全对结果排除了单纯 Triton 内核开销以及 `w_i < 0.05` 长尾删除作为主因。

这个精度测试仍然是定向小集，不能替代完整评测集。它支持的结论是“已经确认存在任务相关精度风险”，不是“所有任务平均精度下降固定百分点”。

## 6. 总体判断

### 加速比

- Top-85：约 +1.4%，95% CI 跨过 1.0。
- combined：约 +2.0%，95% CI 基本围绕 1.0。
- Top-90、floor、layer：在噪声范围内变慢。
- 真正彻底消除 CPU fallback 的 fetch8 控制：约 +9.3%。
- 当前截断策略只移除 17%-29% 的 miss，并保留固定 top-8 GPU GEMM，因此收益上限很低。

### 精度损失

- 9-task、每策略 10 次的总体结果为：Top-85 92.2%、Top-90 97.8%、layer 91.1%，baseline/floor/combined 100%。
- 敏感数学计数任务的两个独立压力测试合并后，Top-85 为 45%、layer 为 50%，正常路径为 100%。
- 小样本无法给出可信的平均精度损失，但已经足以否定“完全无损”的强结论。

### 建议

1. 不将当前 Top-p 权重置零策略直接作为生产加速上线。
2. 若目标是降低 CPU fallback，优先评估显式 PCIe fetch/admission 策略；本机 fetch8 给出了约 9% 的上限。
3. 若继续研究权重截断，需要同时减少 GPU 端固定 top-8 GEMM 的工作量，否则主要收益仍停留在少量 CPU/PCIe route 消除。
4. 高权重 `>0.30` 的兜底在 0.85 阈值下没有实际作用；应重新设计阈值或保护规则。
5. 精度评测必须扩大到更多任务和多次重复，尤其覆盖短答案、数学计数和格式敏感任务。

## 7. 复现命令

运行离线分析：

```bash
/home/luyzh/.venv/bin/python3 \
  experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/scripts/analyze_policy.py
```

运行策略内核语义检查：

```bash
/home/luyzh/.venvs/freetoken/bin/python \
  experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/scripts/check_policy_semantics.py
```

运行真实速度 A/B：

```bash
/home/luyzh/.venvs/freetoken/bin/python \
  experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/scripts/run_policy_bench.py \
  --policies baseline,noop,top85,top90,floor,combined,layer \
  --repeats 3 --decode 255 \
  --output experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/results/policy-v1/policy_bench_fused_r3.jsonl
```

运行精确答案精度集：

```bash
/home/luyzh/.venvs/freetoken/bin/python \
  experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/scripts/run_policy_accuracy.py \
  --policies baseline,noop,top85,top90,floor,combined,layer \
  --repeats 10 --max-tokens 64 --no-thinking \
  --output experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/results/policy-v1/policy_accuracy_all_r10.jsonl
```

汇总速度与精度：

```bash
/home/luyzh/.venv/bin/python3 \
  experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/scripts/analyze_policy_bench.py

/home/luyzh/.venv/bin/python3 \
  experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/scripts/analyze_policy_accuracy.py \
  --input experiments/qwen3.6-35b-a3b-nvfp4-moe-offload/results/policy-v1/policy_accuracy_all_r10.jsonl
```

## 8. 结果文件

| 文件 | 内容 |
|---|---|
| `results/policy-v1/offline/policy_summary.json` | 离线策略总表 |
| `results/policy-v1/offline/policy_by_layer.csv` | 逐层删除、质量和 miss |
| `results/policy-v1/offline/weight_floor_bound.csv` | `w_i > 0.30` 与在卡质量的关系 |
| `results/policy-v1/policy_bench_fused_r3.jsonl` | Top-p/floor 策略真实 A/B 原始结果 |
| `results/policy-v1/policy_bench_noop_r3.jsonl` | noop 内核控制 |
| `results/policy-v1/baseline_fetch8_r3.jsonl` | 完全消除 CPU fallback 的 fetch8 控制 |
| `results/policy-v1/policy_accuracy_all_r10.jsonl` | 9-task x 10-repeat 精确答案结果 |
| `results/policy-v1/policy_accuracy.jsonl` | 首轮 9-task x 3-repeat 控制结果 |
| `results/policy-v1/policy_accuracy_math_total_r10.jsonl` | 敏感算术任务第一轮 10-repeat 压测 |
| `results/policy-v1/policy_accuracy_moe_routes_r10.jsonl` | 另一项 10-repeat 精度对照 |
| `results/policy-v1/policy_accuracy_by_task.csv` | 逐任务精度汇总 |
| `results/policy-v1/policy_accuracy_stress.csv` | 精度压测汇总 |
| `results/policy-v1/policy_speed_summary.csv` | 速度 A/B 汇总 |
| `results/policy-v1/policy_speed.png` | 速度图 |
| `results/policy-v1/policy_accuracy.png` | 精度图 |
