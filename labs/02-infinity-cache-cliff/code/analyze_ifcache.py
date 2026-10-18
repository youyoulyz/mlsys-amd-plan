#!/usr/bin/env python3
"""Analyse the Infinity-Cache experiments and emit tables + figures."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

IF_MIB = 96  # Infinity Cache capacity

# ---- model configs (params) -------------------------------------------------
MODELS = {
    "Qwen3-4B": dict(H=2560, I=9728, L=36, V=151936, nq=32, nkv=8, D=128, tied=True),
    "Qwen2.5-7B": dict(H=3584, I=18944, L=28, V=152064, nq=28, nkv=4, D=128, tied=False),
}


def op_params(c):
    h, i, v, nq, nkv, d = c["H"], c["I"], c["V"], c["nq"], c["nkv"], c["D"]
    return {
        "qkv_proj": (nq + 2 * nkv) * d * h,
        "o_proj": h * (nq * d),
        "gate_up_proj": (2 * i) * h,
        "down_proj": h * i,
        "lm_head": v * h,
    }


def per_op_table():
    rows = []
    for name, c in MODELS.items():
        p = op_params(c)
        for op, n in p.items():
            for prec, bpe in (("BF16", 2.0), ("Q4_K", 0.5625)):
                mib = n * bpe / 1024 / 1024
                rows.append(dict(model=name, op=op, params=n, precision=prec,
                                 mib=mib, fits_if=mib <= IF_MIB, layers=c["L"]))
    return rows


# ---- figures ----------------------------------------------------------------
def fig_cache_sweep():
    d = json.load(open(os.path.join(RES, "cache_sweep.json")))
    rows = d["results"]
    ops = ["read_sum", "write_zero", "copy", "triad_add"]
    colors = dict(zip(ops, ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for op in ops:
        pts = sorted([(r["size_mib"], r["effective_gbps"]["median"])
                      for r in rows if r["operation"] == op])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=op,
                color=colors[op], ms=4)
    ax.axvline(IF_MIB, color="k", ls="--", lw=1.2)
    ax.text(IF_MIB * 1.03, 2500, "96 MiB Infinity Cache", fontsize=9, rotation=90,
            va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("working-set size per operand (MiB)")
    ax.set_ylabel("effective logical bandwidth (GB/s)")
    ax.set_title("7900 XTX: bandwidth vs working-set size\n"
                 "(copy/triad touch 2 operands -> knee at ~48/32 MiB)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cache_sweep.png"), dpi=140)
    plt.close(fig)


def fig_mulmat():
    d = json.load(open(os.path.join(RES, "llama_mulmat_sweep.json")))
    rows = [r for r in d["results"] if "weight_gbps" in r]
    types = ["f16", "q8_0", "q4_K", "f32"]
    colors = dict(zip(types, ["#1f77b4", "#2ca02c", "#d62728", "#7f7f7f"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, n in zip(axes, (1, 16)):
        for t in types:
            pts = sorted([(r["weight_mib"], r["weight_gbps"])
                          for r in rows if r["type_a"] == t and r["n"] == n])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=t,
                    color=colors[t], ms=4)
        ax.axvline(IF_MIB, color="k", ls="--", lw=1.2)
        ax.axhline(960, color="b", ls=":", lw=1.0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64, 96, 128, 256, 512])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("weight footprint |W| (MiB)")
        ax.set_title(f"llama.cpp MUL_MAT, n={n} token(s), m=4096")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("effective weight bandwidth (GB/s)")
    axes[1].text(IF_MIB * 1.05, 120, "96 MiB IF", rotation=90, fontsize=8, va="bottom")
    axes[0].text(600, 980, "960 GB/s nominal HBM", fontsize=8, color="b")
    axes[0].legend()
    fig.suptitle("7900 XTX: decode-style GEMV weight bandwidth vs weight size")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "llama_mulmat_sweep.png"), dpi=140)
    plt.close(fig)


def fig_footprint():
    rows = per_op_table()
    models = list(MODELS)
    ops = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for ax, model in zip(axes, models):
        r = [x for x in rows if x["model"] == model and x["precision"] == "BF16"]
        x = range(len(ops))
        vals = [next(y["mib"] for y in r if y["op"] == o) for o in ops]
        bars = ax.bar(x, vals, color=["#1f77b4" if v <= IF_MIB else "#d62728"
                                      for v in vals])
        for xi, v in zip(x, vals):
            ax.text(xi, v * 1.02, f"{v:.1f}", ha="center", fontsize=8)
        ax.axhline(IF_MIB, color="k", ls="--", lw=1.2)
        ax.set_xticks(list(x))
        ax.set_xticklabels(ops, rotation=20, ha="right")
        ax.set_yscale("log")
        ax.set_title(f"{model} (BF16), L={MODELS[model]['L']} layers")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("per-layer weight footprint (MiB)")
    axes[1].text(4.4, IF_MIB, "96 MiB IF", fontsize=8, va="bottom")
    fig.suptitle("Per-op weight footprint vs 96 MiB Infinity Cache")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "weight_footprint.png"), dpi=140)
    plt.close(fig)


# ---- tables -----------------------------------------------------------------
def write_csvs():
    d = json.load(open(os.path.join(RES, "cache_sweep.json")))
    with open(os.path.join(RES, "cache_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size_mib", "operation", "logical_bytes_per_iter",
                    "inner_iters", "median_gbps", "p10_gbps", "p90_gbps"])
        for r in sorted(d["results"], key=lambda r: (r["size_mib"], r["operation"])):
            g = r["effective_gbps"]
            w.writerow([r["size_mib"], r["operation"], r["logical_bytes_per_iter"],
                        r["inner_iters"], round(g["median"], 2),
                        round(g["p10"], 2), round(g["p90"], 2)])

    d = json.load(open(os.path.join(RES, "llama_mulmat_sweep.json")))
    with open(os.path.join(RES, "llama_mulmat_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type_a", "n", "m", "k", "weight_mib", "us_per_run",
                    "weight_gbps", "tflops", "fits_if"])
        for r in d["results"]:
            if "weight_gbps" not in r:
                continue
            w.writerow([r["type_a"], r["n"], r["m"], r["k"],
                        round(r["weight_mib"], 2), round(r["us_per_run"], 3),
                        round(r["weight_gbps"], 1), round(r["tflops"], 3),
                        r["weight_mib"] <= IF_MIB])

    with open(os.path.join(RES, "weight_footprint.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "op", "params", "precision", "mib", "fits_if",
                    "layers", "fc_mib"])
        for r in per_op_table():
            w.writerow([r["model"], r["op"], r["params"], r["precision"],
                        round(r["mib"], 2), r["fits_if"], r["layers"],
                        round(r["mib"] * r["layers"], 1)])


def summarise():
    d = json.load(open(os.path.join(RES, "llama_mulmat_sweep.json")))
    rows = [r for r in d["results"] if "weight_gbps" in r]
    s = {"if_cache_mib": IF_MIB, "per_type": {}}
    for t in ("f16", "q8_0", "q4_K", "f32"):
        for n in (1, 16):
            rr = sorted([r for r in rows if r["type_a"] == t and r["n"] == n],
                        key=lambda r: r["weight_mib"])
            # peak within IF, plateau beyond 128 MiB
            inside = [r for r in rr if r["weight_mib"] <= 88]
            outside = [r for r in rr if r["weight_mib"] >= 128]
            if not inside or not outside:
                continue
            peak = max(inside, key=lambda r: r["weight_gbps"])
            hbm = sorted(r["weight_gbps"] for r in outside)
            med_out = hbm[len(hbm) // 2]
            s["per_type"][f"{t}_n{n}"] = {
                "peak_inside_gbps": round(peak["weight_gbps"], 1),
                "peak_at_mib": round(peak["weight_mib"], 1),
                "outside_median_gbps": round(med_out, 1),
                "ratio": round(peak["weight_gbps"] / med_out, 2),
            }
    # cache sweep read_sum regime means
    cs = json.load(open(os.path.join(RES, "cache_sweep.json")))
    rs = [r for r in cs["results"] if r["operation"] == "read_sum"]
    ins = [r["effective_gbps"]["median"] for r in rs if r["size_mib"] <= 96]
    outs = [r["effective_gbps"]["median"] for r in rs if r["size_mib"] >= 112]
    s["read_sum"] = {
        "inside_le96_median_gbps": round(sorted(ins)[len(ins) // 2], 1),
        "inside_le96_max_gbps": round(max(ins), 1),
        "outside_ge112_median_gbps": round(sorted(outs)[len(outs) // 2], 1),
    }
    json.dump(s, open(os.path.join(RES, "ifcache_summary.json"), "w"),
              indent=2, ensure_ascii=False)
    return s


if __name__ == "__main__":
    fig_cache_sweep()
    fig_mulmat()
    fig_footprint()
    write_csvs()
    s = summarise()
    print(json.dumps(s, indent=2, ensure_ascii=False))
    print("\nfigures:", sorted(os.listdir(FIG)))
