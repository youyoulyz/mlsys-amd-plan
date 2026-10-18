# 代码与数据的来源与许可证

`labs/` 下每个实验的 `code/` 与 `results/` 是从研究/实验仓库复制来的关键代码与实测结果。来源与许可证如下：

| Lab | 代码来源 | 许可证 | 备注 |
| --- | --- | --- | --- |
| 01 | moecho（自有实验仓库） | 无（自有，随本仓库） | `results/report-arith-intensity-7b.md` 为完整报告 |
| 02 | moecho（自有） | 无（自有） | 未复制 2.8 MB 的 llama.cpp 二进制与 516 KB 的 .cpp.orig 备份 |
| 03 | moecho（自有） | 无（自有） | 完整自包含 |
| 04 | moecho（自有）+ llama.cpp 官方构建 | llama.cpp 为 MIT | 未复制 llama.cpp fork 本体（体积大）；`mtp_probe.cpp` 是探针源码 |
| 05 | [sgl-project/sglang](https://github.com/sgl-project/sglang) `benchmark/` | **Apache-2.0** | 单文件复制，保留原版权 |
| 06 | [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) 实验目录 | **Apache-2.0** | 未复制 13 MB 完整 route traces 与 25 MB 原始 jsonl；`results/traces-samples/` 有 2 个小样本 |
| 07 | llama.cpp（MIT）+ 自有 sweep 脚本 | MIT | 脚本调用 `llama-bench` 二进制（需自行构建） |
| 08 | deepep-simulator（自有重实现，上游 [DeepEP](https://github.com/deepseek-ai/DeepEP) 为 MIT）+ dsv4 demo（自有） | 上游 MIT | 模拟器完整复制（src/tests/results） |
| 09 | dsv4（自有分析脚本） | 无（自有） | 图片为脚本生成 |
| 10 | smartdag 调研（自有）+ moecho 报告（自有）+ 本 lab 自写计算器 | 无（自有） | `code/kv_transfer_vs_recompute.py` 为本课程新写 |
| 11 | [LMCache](https://github.com/lmcache/lmcache) `benchmarks/` | **Apache-2.0** | 需安装 LMCache 包才能跑（`pip install lmcache`） |

> 复制自第三方仓库的文件保留其原有许可证义务（Apache-2.0 / MIT 均允许再分发与修改，保留版权声明即可）。
> 未复制的内容（模型权重、完整 trace、大体积二进制）在各 lab 的 README「来源」一节有说明。