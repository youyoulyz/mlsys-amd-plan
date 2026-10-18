#!/usr/bin/env python3
"""KV cache 换入 vs 重算：决策边界计算器。

决策规则（只有满足它，从慢速介质搬 KV 回 GPU 才有正收益）：

    N_c * mu / B_H2D + tau_load   <   N_c * t_pf
    （搬运字节 / 有效带宽 + 启动开销  <  原地重算耗时）

其中：
    N_c     缺失前缀 token 数
    mu      每 token KV 字节数
    B_H2D   主机/盘 -> GPU 的有效带宽
    tau     传输启动固定开销
    t_pf    当前硬件在该前缀长度下的每 token prefill 耗时

默认数字来自已发表的实测/评测（见 results/ 下来源）：
  - 32B GQA FP8 模型，32K 上下文，KV ≈ 128 KB/token -> 4 GB
  - PCIe 4.0 H2D 有效带宽 25-30 GB/s
  - NVMe SSD 有效带宽 2-6.8 GB/s
  - 现代卡长前缀 FP8 prefill 吞吐 20k-35k tokens/s

用法：
    python kv_transfer_vs_recompute.py
    python kv_transfer_vs_recompute.py --tokens 16384 --kv-per-token-kb 64 \
        --bw-pcie 27.5 --bw-nvme 4.0 --prefill-tok-s 27000
"""
import argparse


def fmt_ms(x: float) -> str:
    return f"{x * 1000:.1f} ms" if x < 1 else f"{x:.3f} s"


def evaluate(
    n_c: int,
    mu_bytes: float,
    bw: float,
    tau: float,
    t_pf: float,
    label: str,
) -> bool:
    transfer_s = n_c * mu_bytes / bw + tau
    recompute_s = n_c * t_pf
    win = transfer_s < recompute_s
    ratio = transfer_s / recompute_s if recompute_s > 0 else float("inf")
    print(
        f"  {label:<28} 搬运 {fmt_ms(transfer_s):>9}   "
        f"重算 {fmt_ms(recompute_s):>9}   "
        f"搬运/重算 = {ratio:6.2f}x   ->  {'值得搬' if win else '重算更快'}"
    )
    return win


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=32_000, help="缺失前缀 token 数 N_c（默认 32K）")
    p.add_argument("--kv-per-token-kb", type=float, default=128.0, help="每 token KV 字节 /1024（默认 128 KB）")
    p.add_argument("--bw-pcie", type=float, default=27.5, help="PCIe H2D 有效带宽 GB/s（默认 27.5，即 25-30 中值）")
    p.add_argument("--bw-nvme", type=float, default=4.4, help="NVMe 有效带宽 GB/s（默认 4.4，即 2-6.8 中值）")
    p.add_argument("--prefill-tok-s", type=float, default=27_000, help="GPU prefill 吞吐 tokens/s（默认 27k，即 20k-35k 中值）")
    p.add_argument("--tau-ms", type=float, default=1.0, help="传输启动固定开销 ms（默认 1）")
    args = p.parse_args()

    mu = args.kv_per_token_kb * 1024.0
    t_pf = 1.0 / args.prefill_tok_s
    tau = args.tau_ms / 1000.0
    kv_gb = args.tokens * mu / 1e9

    print(f"前缀 {args.tokens} tokens x {args.kv_per_token_kb:.0f} KB/token = {kv_gb:.2f} GB KV")
    print(f"PCIe  {args.bw_pcie:.1f} GB/s | NVMe {args.bw_nvme:.1f} GB/s | prefill {args.prefill_tok_s/1000:.1f}k tok/s\n")

    print("== 场景一：完整缺失前缀（N_c = 全部前缀）")
    evaluate(args.tokens, mu, args.bw_pcie * 1e9, tau, t_pf, "PCIe 4.0 (H2D)")
    # NVMe 按区间两端 + 中值展示：结论对带宽区间敏感
    evaluate(args.tokens, mu, 2.0e9, tau, t_pf, "NVMe 慢端 (2 GB/s)")
    evaluate(args.tokens, mu, args.bw_nvme * 1e9, tau, t_pf, "NVMe 中值")
    evaluate(args.tokens, mu, 6.8e9, tau, t_pf, "NVMe 快端 (6.8 GB/s)")

    # 倒挂阈值：搬 N 个 token 恰好等于重算 N 个 token 时的 N
    # N*mu/B + tau = N/t_pf  ->  N = tau / (1/t_pf - mu/B)
    print("\n== 场景二：倒挂阈值（搬 N 个 token 恰好 = 重算 N 个 token）")
    for name, bw in (("PCIe 4.0", args.bw_pcie * 1e9), ("NVMe SSD", args.bw_nvme * 1e9)):
        denom = t_pf - mu / bw
        if denom <= 0:
            print(f"  {name:<14} 每 token 搬运 {mu/bw*1e3:.2f} ms vs 重算 {t_pf*1e3:.2f} ms -> 任何前缀长度都该重算")
        else:
            n_star = tau / denom
            print(f"  {name:<14} 前缀 > {n_star:8.0f} tokens 时搬运开始划算（< {n_star:.0f} 时重算更快）")

    print("""
== 怎么读这张表
- PCIe 行：4 GB 搬回约 130-160 ms，比全量重算（~0.9-1.5 s）快 6-10 倍 -> 该搬。
- NVMe 行：结论对带宽区间**敏感**——慢端（2 GB/s）搬 4 GB 要 ~2.0 s，比全量重算还慢，
  且搬运期间 GPU 张量核空等（实测 SSD 卸载时 GPU 空转气泡可占推理总耗时 72.3%）；
  快端（6.8 GB/s）只要 ~0.6 s，又开始划算。"存盘省 prefill"不是无脑赚，
  必须看你的盘落在区间的哪一端 + 只搬增量而不是整块前缀。
- 倒挂阈值解释了决策边界：介质越慢、前缀越短、GPU 越强，重算越可能赢。
  调度器必须运行时采样 B_H2D 与 prefill 吞吐来仲裁，总线一堵就触发原位重算。
""")


if __name__ == "__main__":
    main()