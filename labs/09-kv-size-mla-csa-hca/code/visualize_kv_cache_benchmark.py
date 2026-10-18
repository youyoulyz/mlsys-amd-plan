"""
Visualization script for DeepSeek-V4 KV Cache Efficiency Benchmark
Generates publication-quality plots comparing V3 MLA vs V4 CSA+HCA
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Benchmark results from the run
seq_lengths = [256, 512, 1024, 2048, 4096]

# KV Cache Size (MB)
mla_kv_mb = [0.14, 0.28, 0.56, 1.12, 2.25]
csa_kv_mb = [0.03, 0.06, 0.12, 0.25, 0.50]
hca_kv_mb = [0.00, 0.00, 0.01, 0.02, 0.03]  # Very small, shown as ~0

# Theoretical FLOPs
mla_flops = [2.01e8, 8.05e8, 3.22e9, 1.29e10, 5.15e10]
csa_flops = [3.36e7, 1.34e8, 5.37e8, 1.07e9, 2.15e9]
hca_flops = [9.83e4, 2.62e5, 7.86e5, 2.62e6, 9.44e6]

# Elements per token
mla_elem = [288.0] * 5
csa_elem = [64.0] * 5
hca_elem = [4.0] * 5

# Create output directory
output_dir = Path('/mnt/luyuzhou/hpc/dsv4/images')
output_dir.mkdir(exist_ok=True)

# ============================================================================
# Figure 1: KV Cache Size Comparison (Log Scale)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(seq_lengths))
width = 0.25

# Use log scale for better visualization
bars1 = ax.bar(x - width, mla_kv_mb, width, label='V3 MLA', color='#2166AC', alpha=0.85, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, csa_kv_mb, width, label='V4 CSA (m=4)', color='#D6604D', alpha=0.85, edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, hca_kv_mb, width, label='V4 HCA (m\'=64)', color='#66C2A5', alpha=0.85, edgecolor='white', linewidth=0.5)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)

ax.set_ylabel('KV Cache Size (MB, FP16)')
ax.set_title('DeepSeek-V3 vs V4: KV Cache Size Comparison')
ax.set_xticks(x)
ax.set_xticklabels([f'{s}' for s in seq_lengths])
ax.set_xlabel('Sequence Length')
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Set log scale
ax.set_yscale('log')
ax.set_ylim(0.001, 10)

plt.tight_layout()
plt.savefig(output_dir / 'v4-kv-cache-size.png', bbox_inches='tight')
plt.close()

print("✓ Saved: v4-kv-cache-size.png")

# ============================================================================
# Figure 2: KV Cache Reduction Percentage
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# CSA Reduction
reduction_csa = [(1 - csa / mla) * 100 for csa, mla in zip(csa_kv_mb, mla_kv_mb)]
reduction_hca = [(1 - hca / mla) * 100 for hca, mla in zip(hca_kv_mb, mla_kv_mb)]

ax1.bar(['V4 CSA\n(m=4)', 'V4 HCA\n(m\'=64)'], 
        [reduction_csa[2], reduction_hca[2]],  # Use 1024 seq_len
        color=['#D6604D', '#66C2A5'], 
        alpha=0.85,
        edgecolor='white',
        linewidth=0.5)

for i, v in enumerate([reduction_csa[2], reduction_hca[2]]):
    ax1.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax1.set_ylabel('KV Cache Reduction (%)')
ax1.set_title(f'KV Cache Reduction at Seq Len=1024\n(vs DeepSeek-V3 MLA)')
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# FLOPs Speedup
speedup_csa = [mla / csa for mla, csa in zip(mla_flops, csa_flops)]
speedup_hca = [mla / hca for mla, hca in zip(mla_flops, hca_flops)]

x = np.arange(len(seq_lengths))
width = 0.35

bars1 = ax2.bar(x - width/2, speedup_csa, width, label='V4 CSA', color='#D6604D', alpha=0.85)
bars2 = ax2.bar(x + width/2, speedup_hca, width, label='V4 HCA', color='#66C2A5', alpha=0.85)

ax2.set_ylabel('FLOPs Speedup (×)')
ax2.set_xlabel('Sequence Length')
ax2.set_title('Attention FLOPs Reduction\n(vs DeepSeek-V3 MLA)')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{s}' for s in seq_lengths])
ax2.legend()
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}×',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7)

plt.tight_layout()
plt.savefig(output_dir / 'v4-kv-cache-reduction.png', bbox_inches='tight')
plt.close()

print("✓ Saved: v4-kv-cache-reduction.png")

# ============================================================================
# Figure 3: Elements per Token Comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

# Horizontal bar chart
categories = ['V3 MLA', 'V4 CSA', 'V4 HCA']
values = [mla_elem[0], csa_elem[0], hca_elem[0]]
colors = ['#2166AC', '#D6604D', '#66C2A5']

bars = ax.barh(categories, values, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, 
            f'{val:.0f} elements/token',
            ha='left', va='center', fontweight='bold', fontsize=11)

ax.set_xlabel('KV Cache Elements per Token')
ax.set_title('KV Cache Efficiency: Elements per Token\n(Lower is Better)')
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add annotation
ax.annotate('V4 HCA achieves 72× reduction\nvs V3 MLA',
           xy=(200, 1.5),
           xytext=(400, 1.5),
           arrowprops=dict(arrowstyle='->', color='black'),
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / 'v4-elements-per-token.png', bbox_inches='tight')
plt.close()

print("✓ Saved: v4-elements-per-token.png")

# ============================================================================
# Figure 4: Extrapolation to 1M Context
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Extrapolated values (from benchmark output)
context_sizes = ['128K', '256K', '512K', '1M']
mla_1m = [68.67, 137.33, 274.66, 549.32]  # MB
csa_1m = [3.81, 7.63, 15.26, 30.52]  # MB
hca_1m = [0.015, 0.03, 0.06, 0.12]  # MB

x = np.arange(len(context_sizes))
width = 0.25

bars1 = ax.bar(x - width, mla_1m, width, label='V3 MLA', color='#2166AC', alpha=0.85)
bars2 = ax.bar(x, csa_1m, width, label='V4 CSA (m=4)', color='#D6604D', alpha=0.85)
bars3 = ax.bar(x + width, hca_1m, width, label='V4 HCA (m\'=64)', color='#66C2A5', alpha=0.85)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height >= 1:
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
        else:
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)

ax.set_ylabel('KV Cache Size (MB)')
ax.set_title('Extrapolated KV Cache Size at Long Contexts\n(Based on Benchmark Results)')
ax.set_xticks(x)
ax.set_xticklabels(context_sizes)
ax.set_xlabel('Context Length')
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add key finding box
textstr = '\n'.join((
    f'At 1M Context:',
    f'V3 MLA: {mla_1m[-1]:.0f} MB',
    f'V4 CSA: {csa_1m[-1]:.1f} MB (94.4% reduction)',
    f'V4 HCA: {hca_1m[-1]:.2f} MB (>99.9% reduction)'
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(output_dir / 'v4-1m-context-extrapolation.png', bbox_inches='tight')
plt.close()

print("✓ Saved: v4-1m-context-extrapolation.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("Visualization Summary")
print("="*80)
print(f"\nGenerated {4} plots in {output_dir}:")
print("  1. v4-kv-cache-size.png - KV cache size comparison (log scale)")
print("  2. v4-kv-cache-reduction.png - Reduction percentage and FLOPs speedup")
print("  3. v4-elements-per-token.png - Elements per token comparison")
print("  4. v4-1m-context-extrapolation.png - Extrapolation to 1M context")
print("\nKey Findings:")
print("  • V4 CSA reduces KV cache by 77.8% vs V3 MLA")
print("  • V4 HCA reduces KV cache by 98.6% vs V3 MLA")
print("  • V4 HCA achieves up to 5461× FLOPs reduction at seq_len=4096")
print("  • At 1M context, V4 CSA uses only 30.52 MB vs V3's 549.32 MB")
print("="*80)
