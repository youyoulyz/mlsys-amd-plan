# 03 · 一张卡同时服务多少人最划算：3B 模型 128 人见顶，再加人反而慢

> **一句话在做什么**：一张显卡跑 3B 模型做推理服务，把"同时服务的用户数"从 1 一路加到 256，每一档测总产出（每秒生成多少字，即 tokens/s，下文"字"均指 token）。实测：**128 人时最高 ~5703 字/秒，加到 256 人反而掉到 ~5387 字/秒**——离显存爆掉还远着呢。

## 实验导读

推理服务的核心经济账：多个用户共享一次模型权重读取（读一次权重算 32 个人），所以直觉上"同时服务的人越多越划算"。这个实验测的就是"划算到什么程度"，以及拐点在哪。

跑完（或读完）你会知道：

- "吞吐随人数单调上升直到爆显存"是错的——曲线有拐点，过顶之后加人反而降吞吐；
- 拐点为什么存在：加人的边际收益（摊薄权重读取）被 attention 计算、每人的"记忆"占用、调度开销吃掉；
- 拐点位置是容量信号：它告诉你"这张卡 + 这个模型 + 这个上下文长度"最多该喂多少人——压测要在拐点附近细扫，而不是只测大 batch；
- 为什么换更强的卡（T4/MI300）拐点会后移——显存容量和带宽都变了。

| 参数 | 人话解释 | 本实验取值 |
| --- | --- | --- |
| 3B 模型 | 参数量 30 亿（约 6 GB 显存） | Qwen2.5-3B-Instruct |
| batch 1→256 | 同时服务的用户数 | 9 档（1,2,4,8,16,32,64,128,256） |
| 每请求 128 字 | 每个用户要生成的长度 | 128 tokens |
| 贪心解码 | 每步选最可能的字（temperature=0，结果可复现） | on |

**主题**：Batch-Decode / 推理服务
**硬件**：单卡 ROCm（Qwen2.5-3B-Instruct，MTP 关闭）— plan **T2/T4** 档可移植
**强化**：第 4 周（推理服务就是排队）
**状态**：自包含脚本 + 原始数据，可复跑

## 你会怎么猜

批量解码把多个请求共享一次权重读取，所以直觉上：

- batch 越大 → 权重读取摊得越薄 → 总吞吐（output tokens/s）单调上升。
- 直到显存装不下为止。

所以你会猜：**吞吐随 batch 单调递增，直到 OOM。**

## 实测会看到

单卡 vLLM（贪心解码 temperature=0，warmup 后计时）扫 batch = 1, 2, 4, 8, 16, 32, 64, 128, 256：

- **batch 128 达到最高 ~5703 output tok/s**
- **batch 256 回落到 ~5387 output tok/s** ← 掉头了

也就是说吞吐在 batch 128 附近见顶，再加 batch 反而下降——离 OOM 还远着呢。

**一句话结论**：批量解码的吞吐-批量曲线是**有拐点的**，不是单调的。拐点处，权重共享的收益已经被 attention/KV cache、调度和固定开销吃掉。

## 为什么会这样

加 batch 的边际收益 = 多摊薄一次权重读取；边际成本 = attention 计算（O(batch·seq)）、KV cache 占用、调度与 kernel 启动的固定开销。小 batch 时收益 >> 成本，吞吐涨；大 batch 时 KV 和 attention 主导、权重摊薄接近饱和，成本超过收益，吞吐掉头。这和 lab 01 里"N=8→16 batch efficiency 掉到 0.41–0.66"是同一个现象在吞吐曲线上的体现。

对第 4 周的意义：**拐点位置就是"这条 GPU 在给定模型/序列下能喂多大批量"的容量信号**，压测时要在拐点附近细扫，而不是只测大 batch。

## 用了什么工具，为什么

- **为什么用 vLLM 而不是 llama.cpp**：vLLM 是工业部署大模型的主力引擎，它的"连续批处理 + 显存分页（PagedAttention）"就是为"同时服务很多人"设计的。这个实验测的是"服务系统能扛多少人"，必须用真正的 serving 引擎；llama.cpp 的批量路径和它不是一回事。
- **为什么用贪心解码（temperature=0）而不是随机采样**：要测的是系统吞吐，不是生成质量。贪心解码让每次运行结果一致，重复 3 次的数字才可比。
- **MTP（多 token 预测）是什么、为什么关着**：一种"一次猜多个字"的加速技术（见 lab 04/05）。本实验只测"批量"这一个变量，把它关死，避免两个因素搅在一起。

## 跑法

本地 `code/` 完整自包含（run.sh + bench_decode.py + plot_results.py），本次实验的实测结果就在 `results/`：

```bash
bash code/run.sh --model /path/to/Qwen2.5-3B-Instruct \
  --batches 1,2,4,8,16,32,64,128,256 --max-new-tokens 128 --repeats 2

# 或直接用 ROCm vLLM 容器（单卡；注意该构建设 *_VISIBLE_DEVICES=0 会零设备，故不设）：
# docker run ... rocm/vllm:rocm7.14.0_... python code/bench_decode.py --model /model \
#   --batches 1,2,4,8,16,32,64,128,256 --max-new-tokens 128 --repeats 2
```
本地 `results/` 已含实测：`decode_results.csv`（原始）、`decode_throughput.png`（曲线）、`run_config.json`（含运行前 `amd-smi` 状态）。

在 T4（MI300）上重跑：拐点会因 HBM 容量/带宽不同而后移，对比两条拐点曲线就是一份"硬件档位如何移动容量拐点"的实测。

## 来源

- 实验目录（README + run.sh + bench_decode.py + plot_results.py + results/）：
  `/mnt/luyuzhou/hpc/moecho/experiments/vllm_batch_decode_nomtp_20260913/`
- 环境：ROCm vLLM 容器 `rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0`；`amd-smi list` 确认单卡

## 延伸

- 第 4 周阅读：Orca（2022）、Sarathi-Serve（2024）——连续批处理与分块 prefill 如何移动这个拐点。
- 相关：为什么 decode 阶段算力大量闲置？见 [01 · Roofline](01-roofline-decode-bw/)。