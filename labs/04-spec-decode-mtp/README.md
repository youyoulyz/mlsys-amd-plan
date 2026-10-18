# 04 · 两张卡合跑 35B 大模型：单用户快 30%，一上并发加卡白加（瓶颈在 CPU）

> **一句话在做什么**：35B MoE 模型（每字只激活 3B）开了"一次猜多字"加速后单张 20 GB 卡装不下，拆到两张卡上推理。两种拆法（按层拆 / 按权重拆）× 两种模式（普通 / 猜多字）逐一比较。单用户：按层拆 + 猜多字 = **+26~34%**，按权重拆只有 +15%。8 个用户同时用：**单卡 = 双卡 = 双卡+猜多字，加卡的收益完全消失**——profiler 数出来：8.6 秒里 CPU 串行给 GPU 派了 71 万个计算任务，派不过来。

## 实验导读

模型比单卡显存大时的标准做法是拆到多卡上。拆法有两种：按层拆（前 20 层在卡 A、后 20 层在卡 B，数据逐层传递）和按权重拆（每层的权重两卡各存一半、每层算完合并结果）。这个实验测哪种拆法对"猜多字"更友好，以及并发上量后加卡还划不划算。

跑完（或读完）你会知道：

- 为什么"猜多字"加速在按层拆下收益是按权重拆的两倍——按权重拆时"猜"那一步也要两卡互相等（每层 2 次同步 × 40 层）；
- 为什么并发一高"双卡 = 单卡"：瓶颈从 GPU 算力转移到 CPU 侧的任务派发（8.6 秒里串行派发 71 万次），GPU 再加几张也等不到活；
- "单用户最优配置 ≠ 并发最优配置"——同一套参数，场景一变结论反手；
- 两张卡没有直连线（数据走 CPU 中转，~6 GB/s）时，跨卡同步的"税"大概是多少（一次 4 KB 同步 ~10–30 微秒）。

| 参数 | 人话解释 | 本实验取值 |
| --- | --- | --- |
| 2× 20 GB 卡 | 两张卡，无直连（走 CPU，~6 GB/s） | RTX 3080 × 2 |
| 模型 | 总参数 35B、每字激活 3B（MoE），3 bit 量化 | Qwen3.6-35B-A3B |
| 按层拆（PP） | 卡 A 算前半层、卡 B 算后半层 | 层数对半分 |
| 按权重拆（TP） | 每层权重两卡各一半、每层算完合并 | 0.5 / 0.5 |
| 并发 4 / 8 | 同时服务的用户数 | 2 档 |

**主题**：Spec-Decode / 扩展与通信
**硬件**：2× RTX 3080 20GB（**无卡间直连**，走 CPU 中转 ~6.1 GB/s）+ Ryzen 3950X — plan **T2** 档（双卡）
**强化**：第 4 周（服务）、第 5 周（扩展、成本与能耗）
**状态**：实测已有（含 profile），官方 llama.cpp 构建可复跑

## 你会怎么猜

两件事：

1. **PP vs TP**：两张卡做投机解码，张量并行（TP）把每层权重切开、all-reduce 合并，看起来比流水线并行（PP）更"对称"、更该快。
2. **加卡**：单卡算力翻倍 → 吞吐应该接近翻倍，尤其开了 MTP 这种强优化后。

## 实测会看到

**单流（batch=1）——MTP 收益取决于跨卡通信，PP 完胜 TP：**

| 配置 | 无 MTP | 开 MTP | MTP 增量 |
| --- | --- | --- | --- |
| 单卡 | 134.1 t/s | 装不下（20GB OOM） | — |
| **PP**（layer split） | 135.2 | **170–182 t/s**（均值 ~176） | **+26~34%** |
| **TP**（0.5/0.5，关 P2P） | 129.2 | 149.0 | **+15.3%** |

**反直觉 1**：MTP 的收益 PP 是 TP 的**两倍**。机制——TP 下 MTP 的 **draft 步也要跨卡 all-reduce**（draft ctx 权重分片，每步 ~2 次 all-reduce × 40 层），这张"跨卡税"（4KB all-reduce ~10–30µs × 80 次/步 ≈ 1–2 ms，对 7–8 ms 步长是 15–25%）正好吃掉 MTP 的增量收益；PP 下 draft 步全在最后一张卡，**零跨卡**，MTP 收益 = 纯计算收益。

**并发（req 打满）——双卡收益归零：**

| 配置 | 4 并发 | 8 并发 |
| --- | --- | --- |
| 单卡 | 223.0 t/s | 260.4 t/s |
| 双卡 PP | 226.6 t/s | 257.7 t/s |
| 双卡 PP + MTP | 226.4 t/s | — |

**反直觉 2**：并发打满后 **单卡 = 双卡 = 双卡+MTP**，双卡算力翻倍完全没兑现，MTP 增量也消失（接受率仍正常 64–72%）。

**瓶颈定位（nsys profile，8 并发 8.6s 压测段）**：**712,990 次 cudaLaunchKernel**，CPU launch 总耗时 ~3.5s ≈ GPU 空洞 3.0s——**ggml 逐节点串行 launch 是共享瓶颈，与 GPU 数量无关**。跨卡 `cudaMemcpyPeerAsync` 只有 348 次，根本不是瓶颈。

**一句话结论**：单流上的最优（PP + MTP）到了并发场景直接失效，因为瓶颈从 GPU 算力/跨卡通信转移到了 **CPU 侧的 kernel 串行 launch**。加卡救不了 CPU-bound 的调度。

## 为什么会这样

decode 是"每步很多小 kernel"的形状，每个 token 步要 launch 几百个 kernel。单流时 GPU 有空等 CPU launch，所以加卡/投机解码能填上这些空洞、拿到收益；并发一多，GPU 队列被填满，瓶颈变成 CPU 以 ~5µs/次的速度串行 launch——再加 GPU 也没用，因为 CPU 供不上活。这是"单流最优配置 ≠ 并发最优配置"的教科书案例。

## 用了什么工具，为什么

- **为什么用 llama.cpp**：这个实验要同时控制"拆法（PP/TP）"和"猜多字（MTP）"两个变量，llama.cpp 的命令行参数两者都直接暴露，且这台双卡机（无卡间直连）的构建是现成的。
- **MTP / 投机解码是什么**：模型里额外的"猜"层先快速猜 k 个字，主模型一次验证——猜对了就一次出 k 个字，摊薄"每字读一遍权重"的成本。lab 05 是它的"猜多长自适应"版。
- **nsys 是什么**：NVIDIA 官方的 profiler（本实验的卡是 RTX 3080）。"瓶颈是不是 CPU"这个问题的答案只能靠它数出来：8.6 秒压测段里 712,990 次计算任务派发、CPU 侧派发总耗时 ~3.5 s ≈ GPU 空闲时间。
- **为什么两张卡"无直连"**：这台机器的板卡没给两张卡配直连线，卡间数据只能经 CPU 内存中转（~6 GB/s）。这是"最差情况"设定——它让跨卡同步的代价最大化，正好用来隔离"拆法"这个变量。

## 跑法

本地 `code/` 是矩阵扫描脚本（来自 mtp_nop2p_matrix 实验；需官方 llama.cpp NCCL 构建 + 双卡无 P2P）：

```bash
# 主扫描：单卡 / PP / TP × 开/关 MTP（NCCL_P2P_DISABLE=1）
bash code/mtp_matrix_nop2p.sh

# 并发（4/8 req）测试：
python code/mtp_concurrency_test.py

# MTP draft/verify 探针（llama.cpp 编译扩展，逐步输出探针数据）：
#   code/mtp_probe.cpp 编译后替换 src/llama-context.cpp 的对应探针点
# 分析与出图：
python code/run_mtp_experiment.py
python code/analyze_mtp_matrix.py
```
再用 nsys profile 并发段，数 `cudaLaunchKernel` 次数与 CPU launch 总耗时（定位"瓶颈是不是 CPU"的关键）。
完整报告在本地 `results/`：`report-mtp-pp-tp-recheck.md`（核心）、`report-mtp-policy.md`、`report-llamacpp-concurrent.md`。

## 来源

- 报告：`/mnt/luyuzhou/hpc/moecho/results/reports/EXP_MTP_PP_TP_RECHECK.md`
- 配套：`EXP_MTP_POLICY_20260813.md`、`EXP_LLAMACPP_CONCURRENT_DECODE_20260908.md`
- 实验仓库：`/mnt/luyuzhou/hpc/moecho/`（Theme A：MTP 投机解码）
- 自 fork：`/mnt/luyuzhou/hpc/moecho/llama.cpp/`（`LLAMA_SPEC_JSONL` spec-decode trace 输出、MTP expert-selection 探针）

## 延伸

- 第 4 周阅读：Leviathan 等，"Fast Inference via Speculative Decoding"（2023）。
- 投机解码的"负载漂移"版本（自适应 vs 静态）见 [05 · 自适应投机解码](05-adaptive-spec-decode/)。
- MoE 专家并行的通信账见 [08 · MoE 通信](08-moe-alltoall-comm/)。