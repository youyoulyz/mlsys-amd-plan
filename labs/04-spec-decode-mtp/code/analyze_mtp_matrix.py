#!/usr/bin/env python3
"""
MTP 无 P2P 双卡矩阵分析 (2026-08-09)
聚合 mtp_nop2p_*.json → 每 cell 每域统计 + cell 间对比 + draft 窗口信号 + 专家图对比
用法: python3 analyze_mtp_matrix.py [结果目录] [输出json]
"""
import json, sys, re, glob
from collections import defaultdict, Counter
import numpy as np

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/luyuzhou/hpc/moecho/results/data"
OUTJSON = sys.argv[2] if len(sys.argv) > 2 else f"{OUTDIR}/mtp_matrix_analysis.json"

CELLS = ["s1_single_q3", "s2_pp_q3", "s2x_dm4_q3", "s2x_t0_q3", "s3_tp2_q3", "s4_pp_q4"]


def load_cell(prefix):
    recs = []
    for f in sorted(glob.glob(f"{OUTDIR}/mtp_nop2p_{prefix}_rep*.json")):
        try:
            data = json.load(open(f))
            recs.extend(data.get("per_prompt", []))
        except Exception as e:
            print(f"  !! {f}: {e}")
    return recs


def cell_stats(recs, key):
    vals = [r[key] for r in recs if r.get(key) is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def window_features(recs):
    """draft 窗口序列特征 (RQ4 profiling 信号): 每步 n_draft 序列"""
    all_windows = []
    for r in recs:
        if r.get("steps"):
            all_windows.append([s["n_draft"] for s in r["steps"]])
    if not all_windows:
        return {}
    flat = [w for ws in all_windows for w in ws]
    lens = [len(ws) for ws in all_windows]
    # 窗口序列的相邻差分熵近似: 用窗口长度的转移矩阵熵 (简化: 均值/熵)
    from collections import Counter
    trans = Counter(zip(flat[:-1], flat[1:]))
    tot = sum(trans.values())
    ent = -sum((c / tot) * np.log2(c / tot) for c in trans.values()) if tot else 0
    return {
        "mean_window": float(np.mean(flat)),
        "std_window": float(np.std(flat)),
        "mean_steps": float(np.mean(lens)),
        "transfer_entropy_bits": float(ent),
        "window_histogram": dict(sorted(Counter(flat).items())),
    }


def expert_topk_overlap(recs_a, recs_b, k=20):
    """跨 cell 的每层 top-k 专家集合 Jaccard 重叠 (H2 地图稳定性)"""
    def layer_topk(recs):
        agg = defaultdict(Counter)
        for r in recs:
            for li, layer in enumerate(r.get("moe_layers", [])):
                for e, c in layer.items():
                    agg[li][e] += c
        return {li: set(c.most_common(k) and [e for e, _ in c.most_common(k)]) for li, c in agg.items()}
    ta, tb = layer_topk(recs_a), layer_topk(recs_b)
    common = sorted(set(ta) & set(tb))
    js = {}
    for li in common:
        a, b = ta[li], tb[li]
        js[li] = len(a & b) / len(a | b) if a | b else 1.0
    return {"n_layers": len(common), "jaccard_mean": float(np.mean(list(js.values()))) if js else None,
            "jaccard_by_layer": {str(k): round(v, 3) for k, v in js.items()}}


def main():
    cells = {}
    for name in CELLS:
        cells[name] = load_cell(name)
        print(f"{name}: {len(cells[name])} records")

    out = {"cells": {}, "comparisons": {}}
    KEYS = ["accept_rate", "speed_tps", "time_ratio", "first_step_us", "steady_step_us"]

    for name, recs in cells.items():
        if not recs:
            out["cells"][name] = {"error": "no data"}
            continue
        by_cat = defaultdict(list)
        for r in recs:
            by_cat[r.get("category", "unknown")].append(r)
        cstats = {}
        for cat, cr in by_cat.items():
            cstats[cat] = {k: cell_stats(cr, k) for k in KEYS}
        allstats = {k: cell_stats(recs, k) for k in KEYS}
        out["cells"][name] = {"n_prompts": len(recs), "per_category": cstats,
                              "all": allstats, "draft_window": window_features(recs)}

    # 对比: 单卡 vs 双卡 PP vs TP (主 RQ1/RQ2)
    def cmp_pair(a, b, key="steady_step_us"):
        sa = out["cells"][a]["all"].get(key)
        sb = out["cells"][b]["all"].get(key)
        if not sa or not sb:
            return None
        return {"a": a, "b": b, "key": key,
                "delta_pct": (sb["mean"] - sa["mean"]) / sa["mean"] * 100 if sa["mean"] else None,
                "a_val": sa, "b_val": sb}

    pairs = [
        ("s1_single_q3", "s2_pp_q3"),   # RQ1: 双卡 PP 代价
        ("s1_single_q3", "s3_tp2_q3"),  # RQ2: TP 代价 (E 对照: -14.4%)
        ("s2_pp_q3", "s3_tp2_q3"),      # RQ2: PP vs TP 交叉
        ("s2_pp_q3", "s4_pp_q4"),       # 量化维度
        ("s2_pp_q3", "s2x_dm4_q3"),     # draft 窗口维度
        ("s2_pp_q3", "s2x_t0_q3"),      # 温度维度
    ]
    for a, b in pairs:
        if a in out["cells"] and b in out["cells"]:
            out["comparisons"][f"{a}_vs_{b}"] = {k: cmp_pair(a, b, k) for k in KEYS}

    # 专家图: S1 vs S2 top-k 重叠 (H2)
    if out["cells"].get("s1_single_q3") and out["cells"].get("s2_pp_q3"):
        out["expert_overlap_s1_s2"] = expert_topk_overlap(cells["s1_single_q3"], cells["s2_pp_q3"])
        out["expert_overlap_s2_s3"] = expert_topk_overlap(cells["s2_pp_q3"], cells["s3_tp2_q3"])

    json.dump(out, open(OUTJSON, "w"), indent=2, default=str)
    print(f"\nSaved: {OUTJSON}")

    # 打印主表
    print("\n=== 主对比表 (steady_step_us) ===")
    for k, v in out["comparisons"].items():
        d = v.get("steady_step_us")
        if d and d.get("delta_pct") is not None:
            print(f"{k}: {d['a_val']['mean']:.1f} vs {d['b_val']['mean']:.1f} us  Δ={d['delta_pct']:+.1f}%")
    print("\n=== speed_tps ===")
    for k, v in out["comparisons"].items():
        d = v.get("speed_tps")
        if d and d.get("delta_pct") is not None:
            print(f"{k}: {d['a_val']['mean']:.1f} vs {d['b_val']['mean']:.1f} t/s  Δ={d['delta_pct']:+.1f}%")


if __name__ == "__main__":
    main()
