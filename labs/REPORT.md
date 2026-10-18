# 实验盒（labs/）建设与代码/结果搬运报告

日期：2026-10-16
范围：`mlsys-amd-plan` 仓库新增 `labs/`（11 个实验 + 索引 + 许可证说明），主 `README.md` 加指针。

## 1. 目标

按 plan 要求，往 mlsys-amd-plan 加"很好玩的实验性内容"（KVCache / MoE offload / batch-decode / spec-decode），从 `/mnt/luyuzhou/hpc` 找现成实验来支撑，寓教于乐。要求：

1. 建 `labs/` 目录（按 plan 规划的结构），精选 hero labs；
2. 每个 lab 把**关键代码**、**实测结果**搬进 lab 目录，**启动命令**指向本地副本。

## 2. 完成内容

### 2.1 11 个 hero lab（全部中文，统一"猜 → 测 → 解释"结构）

| # | Lab | 主题 | 反直觉钩子（实测数字均回源核实） |
| --- | --- | --- | --- |
| 01 | 一张卡跑 7B 大模型生成文字：算力只用 1–3% | Batch-Decode | 7B decode AI 仅 0.5–3.24 OP/B，离 ridge 差 40–250×，与模型大小无关 |
| 02 | 实测读数据比官方带宽快 3 倍：藏着 96 MB 超快缓存 | 校准 | ≤96MiB 1.8–2.6 TB/s，过线断崖到 ~800 GB/s；纯 HBM 模型错 2–2.5× |
| 03 | 一张卡同时服务多少人最划算：128 人见顶 | Batch-Decode | batch 128 见顶 5703 tok/s，256 回落到 5387 |
| 04 | 两卡合跑 35B：单用户快 30%，并发下加卡白加 | Spec-Decode | PP+MTP +26~34% vs TP+15.3%；并发后单卡=双卡=双卡+MTP，瓶颈是 CPU 串行 launch（8.6s 内 71 万次 kernel launch） |
| 05 | "一次猜几个字"写死成常数行不行 | Spec-Decode | 静态投机长度只在 transition 负载上露馅 |
| 06 | 裁掉 MoE 里"少用的"专家路：速度没涨、精度掉 8 点 | MoE Offload | 冷 22.09 vs 热 5.85 experts/token；top85 速度 +1.4%（CI 跨 1.0）但精度 90/90→83/90（p=0.0069） |
| 07 | 单卡跑 8B MoE：调两个内部参数 +69% | MoE | RPB=8/nw=1 最优 363.3 t/s（+69%），最优在"一最大一最小"角落 |
| 08 | MoE 多卡"快递费"：通信成本线性涨 | MoE/扩展 | 通信 ∝ tokens×hidden×top-k，decode 下吃掉并行收益 |
| 09 | 100 万字上下文"记忆"占显存压到 0.1 MB | KVCache | 1M 上下文 KV 549MB→30MB→0.1MB（~5000×），纯 PyTorch 可 CPU 跑 |
| 10 | 4 GB 对话记忆：搬回 GPU 还是重算？ | KVCache/存储 | 4GB KV 走 NVMe 慢端（2 GB/s）比 GPU 重算还慢；SSD 卸载 GPU 空转占 72.3% |
| 11 | 写盘：同一块盘两种写法差 1000 倍 | KVCache/存储 | 标准文件写 1000 ops/s vs 预分配块写 137 万 ops/s（3–4 个数量级）；O_DIRECT 高并发下反慢 1.5–1.6× |

每个 lab 的目录结构：`README.md`（实验导读【人话目标 + 参数表】+ 猜/测/解释 + 「用了什么工具，为什么」+ 跑法 + 来源）+ `code/`（关键脚本）+ `results/`（实测数据/报告/图）。开头口径参照 hello-gpu 的写法：大标题给"做什么 + 能到多少"的常人可懂参数，专业名词推迟到细则里并解释为什么用。

### 2.2 代码与结果搬运（本次重点）

各 lab 从来源仓库复制的关键内容（总量 **3.2 MB**）：

| Lab | code/ | results/ | 未搬（原因） |
| --- | --- | --- | --- |
| 01 | HBM 微基准 + llama.cpp/vLLM 运行脚本 + 分析脚本（9 个） | 完整原始数据树 + 438 行报告 | 101 GB 模型权重 |
| 02 | 3 个微基准/复现脚本 | 完整数据（CSV/JSON）+ 3 张图 + 完整报告 | 2.8 MB llama.cpp 二进制、516 KB .cpp.orig |
| 03 | bench_decode.py + run.sh + 绘图（完整自包含） | 完整（CSV + 图 + 运行配置） | 无 |
| 04 | MTP 矩阵扫描脚本 + probe 源码（6 个） | 3 份完整报告 | llama.cpp fork 本体（体积大，MIT） |
| 05 | 自适应投机基准脚本（sglang，Apache-2.0） | 空（待首跑，README 已注明） | 上游 sglang 引擎本体 |
| 06 | 探针 3 个 + 运行/分析脚本 6 个 | 4 份中文报告 + 关键 CSV/JSON/PNG + 2 个 trace 样本 | 13 MB 完整 traces、25 MB 原始 jsonl |
| 07 | run_moe_sweep.sh | 完整 sweep 结果表 | llama-bench 二进制（需自构建） |
| 08 | deepep-simulator 完整（src/tests/results）+ 单卡 MoE demo | sweep_v100.csv（36 行实测） | 8 卡 all-to-all demo |
| 09 | KV 效率基准 + 可视化（可 CPU 跑） | 结果表 + 6 张图 | CUDA 门控的第二脚本 |
| 10 | **新写** `kv_transfer_vs_recompute.py`（决策边界计算器，已跑通） | smartdag 调研 + moecho 长上下文报告 | 无 |
| 11 | LMCache 存储 I/O 微基准（Apache-2.0） | Valkey 分级参考数据 | LMCache 包本体（需 pip 安装） |

- 启动命令：11 个 lab 的「跑法」节全部改为指向本地 `code/`，不再引用 `/mnt/luyuzhou/hpc` 绝对路径。
- 许可证：新增 `labs/LICENSES.md`，逐 lab 标注来源与许可证（自有 / MIT / Apache-2.0），注明未搬内容。

### 2.4 本机验证（2026-10-17）

在 RX 7900 XTX 主机上实际跑了可 CPU/GPU 执行的 lab，确认启动命令有效、输出与上游一致：

| Lab | 本地运行 | 结果 |
| --- | --- | --- |
| 09 | `python3 code/benchmark_kv_cache_efficiency.py`（在 7900 XTX 上跑） | ✅ 数字与上游完全一致（4096: MLA 5.15e10 → CSA 23.99× / HCA 5461×；1M 上下文 549.32→30.52→0.12 MB）；日志存 `results/local_run.log` |
| 08 | `PYTHONPATH=src python3 -m deepep_sim.cli selftest` + mini sweep | ✅ round-trip 误差 4.8e-7；sweep 复现线性缩放（byte 4.2→16.9→67.6 MB，dispatch 47→174→681 µs）；日志存 `results/local_run_*.log` |
| 10 | `python3 code/kv_transfer_vs_recompute.py` | ✅ 跑通，PCIe 153.5ms vs 重算 1.185s、NVMe 慢端 2.098s 比重算慢（1.77×），与文档结论一致 |
| 11 | 独立 venv（`--system-site-packages`）+ LMCache 源码树 + cargo 编译 rust 扩展，12 行矩阵（并发 4/16/32 × buffered/O_DIRECT，num_ops=8192） | ✅ 跑通；本地盘 Micron M600 SATA SSD：local_disk 503–1021 ops/s vs rust_raw_block 612K–1.37M ops/s（差 3–4 个数量级）；O_DIRECT 高并发下 local_disk 反慢 1.5–1.6×；POSIX 路径 4 线程即峰值。日志 `results/local_run.log`、数据 `results/local_run_matrix.json` |

ROCm 坑（lab 11 踩到，已写进 lab README）：ROCm 上 `torch.cuda.is_available()` 恒为 True，LMCache 导入会走编译版 `c_ops` 分支（CUDA 扩展，未构建）；本微基准纯 CPU，`CUDA_VISIBLE_DEVICES=` 置空即走 `non_cuda_equivalents` 纯 Python 回退。另注：rust_raw_block 的 ops/s 是提交/受理口径（buffered=页缓存、O_DIRECT=设备 DRAM 写缓存受理，NAND 刷盘异步；本盘 fsync 刷盘实测 116 MB/s），且 slot 复用使 8192 op 唯一落盘 ≈15 MB，不是介质带宽——lab README 已注明口径。

不可本机验证的 lab（01/02/03/04/06/07 需对应模型/框架/双卡环境；05 需起 sglang server）其命令均按来源 README 核实过，但尚未执行。

### 2.3 质量保证

- **数字回源**：所有关键数字（797.6 GB/s、2093→812 GB/s、363.3 t/s、90/90→83/90、160×、72.3% 等）均对照原始报告/数据文件逐一核实，未照搬二手摘要。
- **一致性**：lab 10 首跑时发现默认中值下 NVMe 判定与文档叙述矛盾，已修正脚本（按带宽区间两端展示）与 README 表述。
- **校验通过**：全部 .py 通过 `py_compile`，全部 .sh 通过 `bash -n`，README 引用的本地文件无缺失，索引链接与目录一致。

## 3. 已知限制 / TODO

1. **没有 T4（MI300）实测**：所有现成数字在消费级卡（RX 7900 XTX / RTX 3080 / 2×V100 校准）上。脚本可移植，在 Instinct 上补跑是第 2 周"校准"的天然作业（README 已设计此闭环）。
2. **lab 05 无本地结果**：需要起 sglang server 才能跑 A/B，`results/` 待首跑填充。
3. **lab 10 数字来自文献/评测**（smartdag 调研 + Tutti/EfficientAgent 2026 评测），自写脚本只是把公式可执行化；在自己机器上实测 PCIe/NVMe 带宽后应更新数字。
4. **本机可跑的 lab（08/09/10/11）已验证**，但其余 lab 的命令只做了静态核对（语法 + 路径），未实际执行。
4. **主 README 英文版未同步**（按约定只写中文）；`README_en.md` 尚无 labs 对应内容。
5. **未提交**：改动都在工作区（`M README.md` + 新增 `labs/` + `labs/REPORT.md`）。

## 4. 建议的下一步

- 在 T4 云端补跑 2–3 个 lab（02 缓存层次、03 吞吐拐点、09 KV 体积），产出首批 MI300 校准数据。
- 跑 lab 05（adaptive spec）填充第一个"我们自己的"spec-decode 实测。
- 开源前：把 `/mnt/luyuzhou/hpc` 来源路径替换为公开仓库链接（或保留为本课程私有实验数据，看定位）。
- 可选扩展主题：prefix cache / RadixAttention、P/D 分离、量化对 decode 的影响（`/mnt/luyuzhou/hpc/phone-low-quant` 可能还有料）。