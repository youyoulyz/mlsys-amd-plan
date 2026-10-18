# 05 · "一次猜几个字"写死成常数行不行：负载一切换就输

> **一句话在做什么**：起两个推理服务——A 的"猜字长度"写死，B 的随负载自动调——用三种负载（稳态容易猜 / 稳态难猜 / 两种交替切换）打过去，比吞吐和延迟。预期：稳态下两者差不多，**切换负载下 B 明显赢**——"一个长度打天下"只在稳态成立。

## 实验导读

lab 04 的"一次猜多字"（投机解码）有个超参：一次猜几个字。猜多了猜错浪费，猜少了收益不够。这个实验问：这个长度能不能写死成一个全局常数？

跑完（或读完）你会知道：

- 最优"猜字长度"取决于猜对率，而猜对率随负载类型漂移——数学题和自由写诗的猜对率完全不同；
- 静态配置在稳态负载上未必输，在"切换"负载上系统性吃亏（刚切到容易猜时还在保守猜 2 个字，刚切到难猜时还在猛猜 8 个）；
- "为什么需要自适应"的证据长什么样：只在切换负载上稳定拉开的差距；
- 本 lab 目前**还没有实测数字**——脚本已就位，起两个服务跑一轮 A/B 是第一件要做的事（也是本课程"自己补一组数据"的示范作业）。

| 参数 | 人话解释 | 本实验取值 |
| --- | --- | --- |
| 三种负载 | 低猜对率（自由写作）/ 高猜对率（重复性生成）/ 两者交替 | low / high / transition |
| 猜字长度 | 一次让"猜"模型出几个字待验证 | A 写死 / B 自动调 |
| 指标 | 每秒生成字数、每请求延迟、实际猜对字数 | 3 项 |

**主题**：Spec-Decode / 推理服务
**硬件**：需起 sglang 推理服务（自适应模式 + 静态基线）— plan **T2/T4** 档
**强化**：第 4 周（推理服务）
**状态**：基准脚本已就绪，需起服务后跑 A/B

## 你会怎么猜

投机解码的"投机长度"（一次 draft 几个 token）是固定超参。你的直觉：

- 选一个"平均最好"的长度，对所有请求都用它，就够用了。
- 反正接受率低时多 draft 的 token 会被拒掉，浪费有限。

## 实测会看到

这个基准把**同一负载**分别打到 *一个 adaptive server* 和 *一个/多个 static server*，比较吞吐、延迟、接受长度。三种负载：

- **low**：稳态低接受率（自由写作：诗、传记、长文）
- **high**：稳态高接受率（重复性强的生成）
- **transition**：low/high 交替切换，专门压**运行时切换**

预期看到的反直觉：

- **static 的"最优点"随负载变**：对 high 负载最优的长度，丢到 low 负载上就是在多 draft 一堆会被拒的 token——烧算力还拉高延迟。
- **transition 是照妖镜**：静态配置在切换点上有滞后（要么刚切到 high 还用着保守长度、要么刚切到 low 还在猛 draft），adaptive 在这里把平均接受长度和延迟都拉下来。
- 单看 high 或单看 low，static 未必输；**只有 transition 才稳定拉开差距**——这正是"为什么需要自适应"的证据。

**一句话结论**：投机长度不该是全局常数。接受率随负载漂移时，"一个长度打天下"的静态配置在切换负载上系统性吃亏；adaptive 的价值在 transition，不在稳态。

## 为什么会这样

投机解码的每步成本 = draft 成本 + 一次 verify（verify 的 batch 随 draft 长度变大，能摊薄）。最优 draft 长度取决于**接受率**：接受率高就多 draft 划算，接受率低就少 draft 划算。静态配置用一个常数近似，稳态还行，负载切换时必然有一段"配错"。adaptive 在线估计接受率、随负载调长度，专治切换点。

## 用了什么工具，为什么

- **为什么用 sglang**：工业推理引擎之一，"自适应投机解码"是它的官方内建模式（在线估计猜对率、自动调猜字长度），不用自己实现调度逻辑——本实验比较的就是"内建自适应 vs 写死常数"，用它的官方模式才公平。
- **为什么必须起真实服务而不是脚本里直接调**：自适应调整发生在服务运行时（每个请求都在估猜对率），离线脚本复现不了"运行时切换"这个核心场景。
- **猜对率（接受率）是什么**：猜的 k 个字里平均有几个被主模型验证通过。它是决定"猜多长划算"的唯一输入——猜对率 90% 时猜 8 个赚，30% 时猜 8 个纯烧算力。

## 跑法

```bash
# 起一个 adaptive sglang server + 一个（或多个不同投机长度的）static sglang server
# 然后（本地已复制基准脚本，Apache-2.0，来自 sglang 上游）：
python code/bench_adaptive_speculative.py --workloads low,high,transition ...
# 脚本按 (prompt, repeat) 配对统计吞吐 / 延迟 / 接受长度
# 需先按上游 sglang 仓库的 docker/rocm.Dockerfile 起 ROCm 环境的服务
```
> 本 lab 目前**还没有本地实测结果**——上游脚本已完整复制，在 T2/T4 上跑一轮把结果落进 `results/` 是第一件要做的事。

## 来源

- 基准脚本：`/mnt/luyuzhou/hpc/sglang/benchmark/bench_adaptive_speculative.py`
- 文档：`/mnt/luyuzhou/hpc/sglang/docs/docs/advanced_features/adaptive_speculative_decoding.mdx`、`speculative_decoding.mdx`
- ROCm 环境：`/mnt/luyuzhou/hpc/sglang/docker/rocm.Dockerfile`、`rocm-gfx1151.Dockerfile`
- 调参参考：`docs/docs/advanced_features/hyperparameter_tuning.mdx`（chunked-prefill、mem-fraction、CUDA graph）

## 延伸

- 单流 / 双卡 + CPU 调度瓶颈版本的投机解码见 [04 · MTP 投机解码](04-spec-decode-mtp/)。
- 第 4 周阅读：Leviathan 等（2023）投机解码原典。