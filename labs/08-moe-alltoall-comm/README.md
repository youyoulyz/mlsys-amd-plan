# 08 · MoE 多卡的"快递费"：通信成本 = token 数 × 宽度 × 专家数，线性涨

> **一句话在做什么**：MoE 拆多卡时，每个字要先"寄"到有它所需专家的那张卡上（集合通信叫 all-to-all），算完再寄回来。这个实验用模拟器把这笔"快递费"算出来：**时间 ≈ 启动开销 + 字节数 ÷ 带宽，字节数 = token 数 × 隐藏层宽度 × 每字用的专家数**。2×V100 实测：2048 个 token 寄一次 90 微秒，宽度翻倍 + 专家数 4 倍 → 680 微秒——三个参数乘在一起，通信成本线性涨，迟早和专家计算抢时间。

## 实验导读

"专家并行"是 MoE 多卡扩展的标准方案：每张卡放一部分专家，字按路由寄过去、算完寄回来。直觉上这是"白拿"的并行——算得开了，通信只是搬点数据。这个实验算的就是"快递费"什么时候超过"计算费"。

跑完（或读完）你会知道：

- 通信成本 = α（启动开销）+ β × 字节数 ÷ 带宽，字节数 = tokens × hidden × top-k——三个参数任何一个加倍，通信就加倍；
- 为什么 decode（每步 token 少、专家计算短）是通信占比最高的场景：计算分母短，固定启动开销的占比就大；
- "加卡何时变慢"的判据：通信/计算比 > 1 时，多一张卡多一轮 all-to-all 反而净亏；
- 为什么这套数字换硬件要重新校准（V100 校出来的 α/β 不能直接搬到 MI300）——带宽和延迟都是硬件参数。

| 参数 | 人话解释 | 本实验取值 |
| --- | --- | --- |
| tokens | 一步要寄多少个 token | 2048 / 4096 |
| hidden | 每个 token 的向量宽度（2048 维 ≈ 每个 token 4 KB） | 2048 / 4096 |
| top-k | 每个 token 寄给几个专家 | 2 / 8 |
| 分布 | token 在专家间怎么分 | uniform / hotspot |

**主题**：MoE / 扩展与通信
**硬件**：模拟器（任意 CPU 主机可跑）+ 2× V100 实测校准 — plan **T4** 档多卡（通信账通用）
**强化**：第 5 周（扩展、成本与能耗）
**状态**：✅ 本机已验证（2026-10-17，selftest 通过 + mini sweep 复现线性缩放，日志在 `results/local_run_*.log`）

## 你会怎么猜

专家并行 = 每卡放一部分 expert，token 按路由 all-to-all 过去、算完再 all-to-all 回来。你的直觉：

- expert 计算并行掉了，通信只是"搬运 token 的 hidden state"，量不大。
- 加卡 = expert 分布更开 = 吞吐线性涨。

## 实测会看到

DeepEP 的 dispatch/combine 语义有一个**可运行的 PyTorch 模拟器 + 解析成本模型**，并在 2× V100 上跑了 36 行实测 sweep（`results/sweep_v100.csv`）。节选（uniform 分布）：

| num_tokens | hidden | top-k | dispatch | combine | 每 rank 字节 |
| --- | --- | --- | --- | --- | --- |
| 2048 | 2048 | 2 | **89.5 µs** | 89.5 µs | 8.45 MB |
| 2048 | 4096 | 8 | **678.7 µs** | 343.2 µs | 67.4 MB |
| 4096 | 4096 | 8 | ~1.35 ms | ~0.68 ms | ~134 MB |

规律很直白：**dispatch/combine 时间和"每 rank 要搬运的字节数"近似线性**，而字节数 = `num_tokens × hidden × top-k`（再乘精度字节）。token 数、hidden、top-k 三个乘在一起，通信成本就指数感地涨。

**反直觉**：通信不是固定小量，它和**你并行掉的那部分 expert 计算**在同一量级上竞争。当 expert 计算本身很短（小 batch、decode 阶段），all-to-all 的 α（启动延迟）+ β×bytes（带宽项）会占掉很大比例——这时"加卡"反而因为多一轮 all-to-all 而变慢。

**一句话结论**：专家并行不是白拿的。通信成本随 `tokens × hidden × top-k` 线性涨，在 decode 这种 expert 计算短的形态下，all-to-all 常常和 expert FLOPs 抢时间——什么时候加卡不划算，要看通信/计算比。

## 为什么会这样

all-to-all 的成本 = α（每对卡的启动延迟）+ β × 搬运字节 / 带宽。token 派发的字节量正比于 `tokens × hidden × top-k`：top-k 从 2 到 8、hidden 从 2048 到 4096，字节数翻 16 倍，dispatch 从 90µs 涨到 ~700µs。expert 计算在 decode（小 batch）下很短，通信的相对占比就高。这正是第 5 周"α–β 模型 + 扩展效率：卡加到一定数量后反而变慢"的 MoE 版本。

## 用了什么工具，为什么

- **为什么用自研模拟器而不是直接跑 DeepEP**：DeepEP（NVIDIA 官方 MoE 通信库）是编译型 C++ 库、只在特定卡上跑。模拟器用纯 PyTorch 实现了同样的"寄出/收回"语义（可自检：round-trip 误差 ≈5e-7），任何 CPU 机器都能跑，而且成本模型的每个预测值都能单独打印——这个实验要的是"这笔账怎么算"，不是"这个库怎么装"。
- **为什么还要 2×V100 实测**：模拟器给的是公式，公式里的 α/β 是硬件常数，必须在真机上量一轮 36 行扫描才能校准（`results/sweep_v100.csv`）。模拟器自己警告：数字不要外推到没校准过的硬件。
- **all-to-all / dispatch / combine 是什么**：多卡集合通信里"每卡给每卡各发一份"的原语。MoE 里分两步：dispatch（把 token 寄到目标专家所在的卡）和 combine（把算完的 token 寄回原卡）。

## 用了什么工具，为什么

- **为什么写模拟器而不是真起多卡**：本 lab 的论点（通信 ∝ tokens × hidden × top-k）是个解析式结论，用模拟器可以在任何 CPU 主机上几秒内把 36 格扫描跑完、随时改参数；真机多卡用来校准公式里的 α（启动延迟）/β（带宽项）两个常数，而不是用来"验证结论"。分工：模拟器算账，真机定常数。
- **DeepEP 是什么**：DeepSeek 开源的 MoE 通信库，all-to-all 的"寄 token / 收 token"语义以它为准——模拟器按它的 dispatch/combine 语义实现，账才算得对。
- **为什么数字只敢用到 V100 量级**：α/β 是硬件常数，在 2×V100（36 行实测）上校准，换到 MI300/H100 必须重测——所以本 lab 的结论给的是"形状"（线性涨、何时超过计算），具体微秒数只在 V100 上成立。

## 跑法

本地 `code/deepep-simulator/` 是完整模拟器（src + tests + 实测校准），本机已验证可跑：

```bash
cd code/deepep-simulator
# 自检（identity round-trip + 不平衡因子），本机已跑通：
PYTHONPATH=src python3 -m deepep_sim.cli selftest
# 扫 tokens/hidden/top-k/分布，输出成本模型预测：
PYTHONPATH=src python3 -m deepep_sim.cli sweep --tokens 1024 4096 --hidden 2048 \
  --topk 2 8 --distributions uniform
# CLI 支持 plan / sweep / bench / selftest
# 双卡分布式校验（需 2 卡）：
python -m torch.distributed.run --nproc_per_node=2 --standalone \
    scripts/check_distributed.py --distribution hotspot --rank-skew 0.5
# 成本模型：src/deepep_sim/cost_model.py
# 实测校准（2× V100，36 行）：results/sweep_v100.csv（lab 的 results/ 也有一份）
```
本机运行输出（2026-10-17）在 `results/local_run_selftest.log` 与 `results/local_run_sweep.log`——mini sweep 直观复现了本 lab 的论点：byte 数 4.2→16.9→67.6 MB（tokens 1024→4096、top-k 2→8），dispatch 时间 47→174→681 µs 同比例涨。
配套单卡 MoE demo（理解 expert 路由与负载均衡，不涉及 offload）在本地 `code/`：

```bash
# DeepSeek 风格细粒度 MoE：32 routed (top-4) + 2 shared，看路由/负载均衡/延迟
python code/fine_grained_moe_single_gpu.py
```
多卡 all-to-all 版 `fine_grained_moe_demo.py`（`torchrun --nproc_per_node=8`）未复制，需要 8 卡环境，见来源原仓库。

## 来源

- 模拟器 + 成本模型 + 实测：`/mnt/luyuzhou/hpc/deepep/deepep-simulator/`（`src/deepep_sim/`、`results/sweep_v100.csv`、`tests/`）
- 单卡/多卡 MoE demo：`/mnt/luyuzhou/hpc/dsv4/fine_grained_moe_single_gpu.py`、`fine_grained_moe_demo.py`
- 注：模拟器明确警告**不要外推到 H20/H100**（V100 校准）；在 T4（MI300）上需用 RCCL 的 all-to-all 实测重新校准 α/β

## 延伸

- 第 5 周阅读：Megatron-LM（2019）、RCCL 文档与 rccl-tests；α–β 模型。
- MoE 单卡 decode 的形状/调参见 [07 · MoE decode 调参](07-moe-decode-tune/)。