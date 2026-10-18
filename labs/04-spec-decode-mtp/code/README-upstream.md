# README — mtp_nop2p_matrix_20260809

> MTP 无 P2P 双卡全矩阵（探针版）运行器 + mtp-probe 探针。
> 实验日：2026-08-09（归档提交 08-11）| 状态：⚠️ 探针版结论作废，以官方路径复检为准
> 报告：[EXP_MTP_NOP2P_DUALGPU.md](../../results/reports/EXP_MTP_NOP2P_DUALGPU.md)、[EXP_MTP_PP_TP_RECHECK.md](../../results/reports/EXP_MTP_PP_TP_RECHECK.md)

## 清单

| 文件 | 说明 |
|---|---|
| `mtp_probe.cpp` / `mtp_probe_noprobe.cpp` | MTP 探针（draft-mtp 插桩）/ 无探针对照 |
| `run_mtp_experiment.py` | 矩阵运行器（S1-S4 配置参数化）|
| `mtp_matrix_nop2p.sh` | 一键矩阵脚本（S1-S4 × reps=3，输出 `results/data/mtp_nop2p_*.json`）|
| `analyze_mtp_matrix.py` | 矩阵分析（输出 `results/data/mtp_matrix_analysis.json`）|
| `mtp_concurrency_test.py` | 并发口径测试（后转 mtp_causal）|

## 日志

见 [LOG.md](LOG.md)。原始数据在 `results/data/`（`mtp_nop2p_*`、`mtp_q3/q4_pp_mtp.json`、`mtp_matrix_analysis.json`、`mtp_nop2p_dmon_s2.csv`）。