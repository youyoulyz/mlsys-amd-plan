from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "traces/trace-410395.tsv"
OUT = ROOT / "results/analysis-v1"
OUT.mkdir(parents=True, exist_ok=True)

NUM_LAYERS = 40
NUM_EXPERTS = 256
TOPK = 8
EXPERT_BYTES = 1_775_616
FULL_PASS_GIB = EXPERT_BYTES * NUM_EXPERTS * NUM_LAYERS / 2**30

DECODE_SEGMENTS = {
    "math": (21, 276),
    "python": (276, 531),
    "moe": (531, 786),
}
PREFILL_SEGMENTS = {
    "prefill_7k": [0],
    "prefill_32k": [1, 2, 3, 4],
    "prefill_60k": [5, 6, 7, 8, 9, 10, 11, 12],
    "math_prefill": [13],
    "python_prefill": [14],
    "moe_prefill": [15],
}


def mask_from_hex(text: str) -> np.ndarray:
    return np.unpackbits(
        np.frombuffer(bytes.fromhex(text), dtype=np.uint8), bitorder="big"
    )[:NUM_EXPERTS].astype(bool)


def read_trace(path: Path):
    decode = []
    prefill = []
    prefill_after = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            x = line.rstrip("\n").split("\t")
            event = x[0]
            if event == "D":
                decode.append(
                    {
                        "step": int(x[1]),
                        "layer": int(x[2]),
                        "batch": int(x[3]),
                        "raw": np.fromstring(x[4], sep=",", dtype=np.int16),
                        "pre": np.fromstring(x[5], sep=",", dtype=np.int8).astype(bool),
                        "post": np.fromstring(x[6], sep=",", dtype=np.int32),
                        "fetched": int(x[7]),
                        "cpu": int(x[8]),
                        "missing": int(x[9]),
                        "resident_count": int(x[10]),
                        "resident": mask_from_hex(x[11]),
                        "dt_ms": float(x[16]),
                    }
                )
            elif event == "P":
                prefill.append(
                    {
                        "chunk": int(x[1]),
                        "layer": int(x[2]),
                        "batch": int(x[3]),
                        "resident_count": int(x[10]),
                        "resident": mask_from_hex(x[11]),
                        "route_counts": np.fromstring(x[12], sep=",", dtype=np.int32),
                        "active_unique": int(x[13]),
                        "resident_unique": int(x[14]),
                        "missing_unique": int(x[15]),
                        "dt_ms": float(x[16]),
                    }
                )
            elif event == "P_after":
                prefill_after.append(
                    {
                        "chunk": int(x[1]),
                        "layer": int(x[2]),
                        "resident_count": int(x[10]),
                        "resident": mask_from_hex(x[11]),
                    }
                )
    return header, decode, prefill, prefill_after


HEADER, decode_rows, prefill_rows, prefill_after_rows = read_trace(TRACE)
print("header", HEADER)
print("rows", len(decode_rows), len(prefill_rows), len(prefill_after_rows))

# Dense decode tensors: layer x step x route/expert.
raw = np.full((NUM_LAYERS, 765, TOPK), -1, dtype=np.int16)
pre = np.zeros((NUM_LAYERS, 765, TOPK), dtype=bool)
post = np.full((NUM_LAYERS, 765, TOPK), -1, dtype=np.int32)
resident = np.zeros((NUM_LAYERS, 765, NUM_EXPERTS), dtype=bool)
decode_resident_count = np.zeros((NUM_LAYERS, 765), dtype=np.int16)
decode_fetched = np.zeros((NUM_LAYERS, 765), dtype=np.int16)
decode_cpu = np.zeros((NUM_LAYERS, 765), dtype=np.int16)
decode_missing = np.zeros((NUM_LAYERS, 765), dtype=np.int16)
decode_dt = np.zeros((NUM_LAYERS, 765), dtype=np.float32)
for row in decode_rows:
    step = row["step"]
    if step < 21:
        continue
    j = step - 21
    layer = row["layer"]
    raw[layer, j] = row["raw"]
    pre[layer, j] = row["pre"]
    post[layer, j] = row["post"]
    resident[layer, j] = row["resident"]
    decode_resident_count[layer, j] = row["resident_count"]
    decode_fetched[layer, j] = row["fetched"]
    decode_cpu[layer, j] = row["cpu"]
    decode_missing[layer, j] = row["missing"]
    decode_dt[layer, j] = row["dt_ms"]

step_table = pd.DataFrame(
    {
        "step": np.arange(21, 786),
        "resident_total": decode_resident_count.sum(axis=0),
        "fetched": decode_fetched.sum(axis=0),
        "cpu": decode_cpu.sum(axis=0),
        "missing": decode_missing.sum(axis=0),
        "probe_dt_ms": decode_dt.sum(axis=0),
    }
)
step_table["hits"] = NUM_LAYERS * TOPK - step_table["missing"]
step_table["hit_rate"] = step_table["hits"] / (NUM_LAYERS * TOPK)
step_table["segment"] = "unassigned"
for name, (start, end) in DECODE_SEGMENTS.items():
    step_table.loc[(step_table.step >= start) & (step_table.step < end), "segment"] = name

results = {
    "trace": str(TRACE),
    "model": {
        "layers": NUM_LAYERS,
        "experts_per_layer": NUM_EXPERTS,
        "topk": TOPK,
        "expert_bytes": EXPERT_BYTES,
        "full_layer_stream_gib": FULL_PASS_GIB / NUM_LAYERS,
        "full_model_pass_gib": FULL_PASS_GIB,
    },
    "decode_segments": {},
    "prefill_segments": {},
    "notes": {
        "fetched": "expert rows assigned a GPU slot and copied over PCIe",
        "cpu": "missed expert rows left non-resident and computed on host CPU",
        "prefill_stream": "full 256-expert layer is copied into volatile double buffers for each chunk; prefill_hit_d2d was disabled",
    },
}

for name, (start, end) in DECODE_SEGMENTS.items():
    q = step_table[(step_table.step >= start) & (step_table.step < end)]
    fetched = q.fetched.to_numpy(float)
    cpu = q.cpu.to_numpy(float)
    missing = q.missing.to_numpy(float)
    resident_total = q.resident_total.to_numpy(float)
    windows = {
        "first_8": [0, 8],
        "first_32": [0, 32],
        "last_128": [len(q) - 128, len(q)],
    }
    win = {}
    for wn, (a, b) in windows.items():
        z = q.iloc[a:b]
        win[wn] = {
            "tokens": len(z),
            "fetched_per_token": float(z.fetched.mean()),
            "cpu_per_token": float(z.cpu.mean()),
            "missing_per_token": float(z.missing.mean()),
            "hit_rate": float(z.hit_rate.mean()),
            "resident_total_mean": float(z.resident_total.mean()),
        }
    rolling32 = pd.Series(fetched).rolling(32, min_periods=1).mean().to_numpy()
    results["decode_segments"][name] = {
        "steps": len(q),
        "route_exposures": int(len(q) * NUM_LAYERS * TOPK),
        "hit_rate": float(q.hits.sum() / (len(q) * NUM_LAYERS * TOPK)),
        "missing_per_token": float(missing.mean()),
        "fetched_per_token": float(fetched.mean()),
        "cpu_per_token": float(cpu.mean()),
        "fetched_fraction_of_misses": float(fetched.sum() / missing.sum()) if missing.sum() else None,
        "resident_total_mean": float(resident_total.mean()),
        "resident_total_p5": float(np.percentile(resident_total, 5)),
        "resident_total_p50": float(np.percentile(resident_total, 50)),
        "resident_total_p95": float(np.percentile(resident_total, 95)),
        "fetched_p50": float(np.percentile(fetched, 50)),
        "fetched_p95": float(np.percentile(fetched, 95)),
        "fetched_p99": float(np.percentile(fetched, 99)),
        "missing_p50": float(np.percentile(missing, 50)),
        "missing_p95": float(np.percentile(missing, 95)),
        "cpu_p95": float(np.percentile(cpu, 95)),
        "fetched_bytes_mib_per_token": float(fetched.mean() * EXPERT_BYTES / 2**20),
        "cpu_bytes_mib_per_token": float(cpu.mean() * EXPERT_BYTES / 2**20),
        "windows": win,
        "rolling32_fetched_min": float(rolling32.min()),
        "rolling32_fetched_max": float(rolling32.max()),
        "fetched_std": float(fetched.std()),
        "fetched_cv": float(fetched.std() / fetched.mean()) if fetched.mean() else None,
        "fetched_lag1_corr": float(np.corrcoef(fetched[:-1], fetched[1:])[0, 1]),
    }

per_layer_rows = []
for layer in range(NUM_LAYERS):
    m = decode_missing[layer]
    f = decode_fetched[layer]
    c = decode_cpu[layer]
    r = decode_resident_count[layer]
    per_layer_rows.append(
        {
            "layer": layer,
            "resident_mean": r.mean(),
            "resident_p5": np.percentile(r, 5),
            "resident_p95": np.percentile(r, 95),
            "missing_per_token": m.mean(),
            "fetched_per_token": f.mean(),
            "cpu_per_token": c.mean(),
            "hit_rate": 1 - m.sum() / (len(m) * TOPK),
            "miss_p95": np.percentile(m, 95),
            "fetch_p95": np.percentile(f, 95),
            "probe_dt_ms_mean": decode_dt[layer].mean(),
        }
    )
per_layer = pd.DataFrame(per_layer_rows)
per_layer.to_csv(OUT / "decode_per_layer.csv", index=False)

expert_rows = []
for name, (start, end) in DECODE_SEGMENTS.items():
    for layer in range(NUM_LAYERS):
        ids = raw[layer, start - 21:end - 21].reshape(-1).astype(np.int32)
        pre_flat = pre[layer, start - 21:end - 21].reshape(-1)
        post_flat = post[layer, start - 21:end - 21].reshape(-1)
        miss_route = ~pre_flat
        fetched_route = miss_route & (post_flat >= 0)
        cpu_route = post_flat < 0
        freq = np.bincount(ids, minlength=NUM_EXPERTS)
        miss_counts = np.bincount(ids[miss_route], minlength=NUM_EXPERTS)
        fetched_counts = np.bincount(ids[fetched_route], minlength=NUM_EXPERTS)
        cpu_counts = np.bincount(ids[cpu_route], minlength=NUM_EXPERTS)
        resident_steps = resident[layer, start - 21:end - 21].sum(axis=0)
        for expert in range(NUM_EXPERTS):
            expert_rows.append(
                {
                    "segment": name,
                    "layer": layer,
                    "expert": expert,
                    "routes": int(freq[expert]),
                    "misses": int(miss_counts[expert]),
                    "fetched": int(fetched_counts[expert]),
                    "cpu": int(cpu_counts[expert]),
                    "resident_steps": int(resident_steps[expert]),
                    "resident_rate": float(resident_steps[expert] / (end - start)),
                }
            )
expert_stats = pd.DataFrame(expert_rows)
expert_stats.to_csv(OUT / "decode_expert_stats.csv", index=False)
results["decode_expert_summary"] = {
    name: {
        "layer_experts": int(len(expert_stats[expert_stats.segment == name])),
        "never_resident": int(
            (expert_stats[expert_stats.segment == name].resident_steps == 0).sum()
        ),
        "top_fetched": expert_stats[expert_stats.segment == name]
        .nlargest(20, "fetched")[
            ["layer", "expert", "routes", "misses", "fetched", "cpu", "resident_rate"]
        ]
        .to_dict(orient="records"),
        "top_routed": expert_stats[expert_stats.segment == name]
        .nlargest(20, "routes")[
            ["layer", "expert", "routes", "misses", "fetched", "cpu", "resident_rate"]
        ]
        .to_dict(orient="records"),
    }
    for name in DECODE_SEGMENTS
}

# Prefill arrays: chunk x layer x expert.
prefill_route_counts = np.zeros((16, NUM_LAYERS, NUM_EXPERTS), dtype=np.int32)
prefill_resident = np.zeros((16, NUM_LAYERS, NUM_EXPERTS), dtype=bool)
prefill_after = np.zeros((16, NUM_LAYERS, NUM_EXPERTS), dtype=bool)
prefill_layer_rows = []
for row in prefill_rows:
    chunk = row["chunk"]
    layer = row["layer"]
    counts = row["route_counts"]
    mask = row["resident"]
    prefill_route_counts[chunk, layer] = counts
    prefill_resident[chunk, layer] = mask
    prefill_layer_rows.append(
        {
            "chunk": chunk,
            "layer": layer,
            "batch_tokens": row["batch"],
            "resident_count": row["resident_count"],
            "active_unique": row["active_unique"],
            "resident_unique": row["resident_unique"],
            "missing_unique": row["missing_unique"],
            "route_hits": int(counts[mask].sum()),
            "route_misses": int(counts[~mask].sum()),
            "dt_ms": row["dt_ms"],
        }
    )
for row in prefill_after_rows:
    prefill_after[row["chunk"], row["layer"]] = row["resident"]
prefill_layer_df = pd.DataFrame(prefill_layer_rows)
prefill_layer_df.to_csv(OUT / "prefill_layer_chunk.csv", index=False)

prefill_expert_rows = []
for chunk in range(16):
    for layer in range(NUM_LAYERS):
        for expert in range(NUM_EXPERTS):
            prefill_expert_rows.append(
                {
                    "chunk": chunk,
                    "layer": layer,
                    "expert": expert,
                    "routes": int(prefill_route_counts[chunk, layer, expert]),
                    "resident_before": int(prefill_resident[chunk, layer, expert]),
                }
            )
prefill_expert = pd.DataFrame(prefill_expert_rows)
prefill_expert.to_csv(OUT / "prefill_expert_stats.csv", index=False)

for name, chunks in PREFILL_SEGMENTS.items():
    q = prefill_layer_df[prefill_layer_df.chunk.isin(chunks)]
    counts = prefill_route_counts[chunks].sum(axis=0)
    masks = prefill_resident[chunks]
    route_hits = 0
    route_total = 0
    for chunk in chunks:
        route_hits += int(prefill_route_counts[chunk][prefill_resident[chunk]].sum())
        route_total += int(prefill_route_counts[chunk].sum())
    results["prefill_segments"][name] = {
        "chunks": chunks,
        "tokens": int(q.groupby("chunk").batch_tokens.first().sum()),
        "resident_per_layer_mean": float(q.resident_count.mean()),
        "resident_per_layer_min": int(q.resident_count.min()),
        "resident_per_layer_max": int(q.resident_count.max()),
        "active_unique_mean": float(q.active_unique.mean()),
        "active_unique_min": int(q.active_unique.min()),
        "active_unique_max": int(q.active_unique.max()),
        "active_resident_unique_mean": float(q.resident_unique.mean()),
        "route_hit_fraction": float(route_hits / route_total),
        "streamed_experts_per_layer_chunk": NUM_EXPERTS,
        "streamed_expert_bytes_per_token_mean": float(
            q.groupby("chunk")
            .apply(
                lambda z: NUM_EXPERTS * EXPERT_BYTES / z.batch_tokens.iloc[0],
                include_groups=False,
            )
            .mean()
        ),
        "top_routed_expert_ids": (
            pd.DataFrame({"expert": np.arange(NUM_EXPERTS), "routes": counts.sum(axis=0)})
            .nlargest(20, "routes")
            .to_dict(orient="records")
        ),
        "top_missing_expert_ids": (
            pd.DataFrame(
                {
                    "expert": np.arange(NUM_EXPERTS),
                    "misses": (counts * ~masks).sum(axis=(0, 1)),
                }
            )
            .nlargest(20, "misses")
            .to_dict(orient="records")
        ),
    }

# Time-series plots.
colors = {"math": "#d62728", "python": "#1f77b4", "moe": "#2ca02c"}
fig, axes = plt.subplots(3, len(DECODE_SEGMENTS), figsize=(20, 12), sharex="col")
for col, name in enumerate(DECODE_SEGMENTS):
    start, end = DECODE_SEGMENTS[name]
    q = step_table.iloc[start - 21:end - 21]
    x = np.arange(len(q))
    ax = axes[0, col]
    ax.plot(x, q.fetched, color=colors[name], lw=1, label="PCIe fetched")
    ax.plot(x, q.cpu, color="#666666", lw=1, label="CPU computed")
    ax.plot(x, q.fetched.rolling(16, min_periods=1).mean(), color="black", lw=2, label="16-token fetch mean")
    ax.set_title(f"{name}: per-token miss split")
    ax.set_ylabel("expert rows / token")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax = axes[1, col]
    ax.plot(x, q.missing, color="#9467bd", lw=1, label="misses")
    ax.plot(x, q.missing.rolling(16, min_periods=1).mean(), color="black", lw=2, label="16-token mean")
    ax.set_ylabel("misses / token")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax = axes[2, col]
    ax.plot(x, q.resident_total, color="#ff7f0e", lw=1.5)
    ax.axhline(7200, color="black", ls="--", lw=1, label="cache capacity")
    ax.set_ylabel("resident experts (40 layers)")
    ax.set_xlabel("decode token")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "decode_timeseries.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].bar(per_layer.layer, per_layer.resident_mean, color="#1f77b4")
axes[0].set_ylabel("mean resident experts")
axes[0].set_title("Long decode: per-layer resident experts")
axes[0].grid(axis="y", alpha=0.25)
axes[1].bar(per_layer.layer, per_layer.missing_per_token, color="#d62728", label="misses")
axes[1].bar(per_layer.layer, per_layer.fetched_per_token, color="#2ca02c", label="PCIe fetched")
axes[1].set_xlabel("layer")
axes[1].set_ylabel("expert rows / token")
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "decode_per_layer.png", dpi=150)
plt.close(fig)

combined_fetch = (
    expert_stats.groupby(["layer", "expert"])["fetched"].sum().unstack(fill_value=0).to_numpy()
)
combined_resident = (
    expert_stats.groupby(["layer", "expert"])["resident_rate"].mean().unstack(fill_value=0).to_numpy()
)
fig, axes = plt.subplots(2, 1, figsize=(16, 8))
im0 = axes[0].imshow(np.log1p(combined_fetch), aspect="auto", cmap="magma")
axes[0].set_title("Long decode: per (layer, expert) PCIe fetch count (log1p)")
axes[0].set_ylabel("layer")
fig.colorbar(im0, ax=axes[0], shrink=0.8)
im1 = axes[1].imshow(combined_resident, aspect="auto", cmap="viridis", vmin=0, vmax=1)
axes[1].set_title("Long decode: per (layer, expert) resident fraction before routing")
axes[1].set_xlabel("expert id")
axes[1].set_ylabel("layer")
fig.colorbar(im1, ax=axes[1], shrink=0.8)
fig.tight_layout()
fig.savefig(OUT / "decode_expert_heatmap.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(16, 8))
im0 = axes[0].imshow(prefill_layer_df.pivot(index="layer", columns="chunk", values="resident_count"), aspect="auto", cmap="viridis")
axes[0].set_title("Prefill: stable cached experts before each layer/chunk")
axes[0].set_ylabel("layer")
fig.colorbar(im0, ax=axes[0], shrink=0.8)
im1 = axes[1].imshow(prefill_layer_df.pivot(index="layer", columns="chunk", values="resident_unique"), aspect="auto", cmap="magma")
axes[1].set_title("Prefill: active experts already in stable cache")
axes[1].set_xlabel("prefill chunk")
axes[1].set_ylabel("layer")
fig.colorbar(im1, ax=axes[1], shrink=0.8)
fig.tight_layout()
fig.savefig(OUT / "prefill_layer_chunk.png", dpi=150)
plt.close(fig)

with (OUT / "results.json").open("w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
