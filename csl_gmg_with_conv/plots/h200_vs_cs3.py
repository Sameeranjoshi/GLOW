#!/usr/bin/env python3
"""
Plot HPGMG benchmark speedup (WSE-3 over H200) from gpu_numbers.txt and wse_numbers.txt.
Reads the "WITH convergence check" tables; outputs publication-quality figures.
"""

import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless/paper generation
import matplotlib.pyplot as plt
import numpy as np


# Publication-quality parameters (larger scale/text for paper readability)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def _parse_table(filepath):
    """Parse the WITH convergence check table from gpu_numbers.txt or wse_numbers.txt.

    Returns {config: {grid_size: {'tts': float, 'iter': int, 'cycle': float}}}
    where config is e.g. '6/6/100' and grid_size is e.g. 16.
    """
    with open(filepath) as f:
        text = f.read()

    # Extract the WITH convergence section (up to the next section or EOF)
    m = re.search(r'WITH convergence check\s*\n={3,}\n(.*?)(?:\n={3,}\n\s*(?:WITHOUT|Shallow)|$)',
                  text, re.DOTALL)
    if not m:
        raise ValueError(f"No 'WITH convergence check' section found in {filepath}")
    section = m.group(1)

    # Parse config names from header line:  "GH200(6/6/100)" or "WSE3(4/4/6)"
    configs = []
    for line in section.splitlines():
        cfg_matches = re.findall(r'\w+\(([\d/]+)\)', line)
        if cfg_matches:
            configs = cfg_matches
            break

    if not configs:
        raise ValueError(f"No config headers found in {filepath}")

    # Parse data rows:  "  512^3 (9) |  0.040568     2   0.020284 | ..."
    data = {c: {} for c in configs}
    for line in section.splitlines():
        row_m = re.match(r'\s*(\d+)\^3\s*\(\w*(\d+)\)', line)
        if not row_m:
            continue
        grid = int(row_m.group(1))
        # Split by '|' and take the config blocks (skip first which is the grid label)
        parts = line.split('|')[1:]
        for i, cfg in enumerate(configs):
            if i >= len(parts):
                break
            nums = parts[i].strip().split()
            if len(nums) >= 3:
                tts = float(nums[0])
                iters = int(nums[1])
                cycle = float(nums[2])
                data[cfg][grid] = {'tts': tts, 'iter': iters, 'cycle': cycle}

    # Parse shallow V-cycle table if present
    shallow_m = re.search(r'Shallow V-cycle.*?\n-+\n(.*?)(?:\n={3,}|$)', text, re.DOTALL)
    if shallow_m:
        data['6/6/6(Shallow)'] = {}
        for line in shallow_m.group(1).splitlines():
            row_m = re.match(r'\s*(\d+)\^3\s*\(L\d+\)', line)
            if not row_m:
                continue
            grid = int(row_m.group(1))
            parts = line.split('|')[1:]
            if parts:
                nums = parts[0].strip().split()
                if len(nums) >= 3:
                    tts = float(nums[0])
                    iters = int(nums[1])
                    cycle = float(nums[2])
                    data['6/6/6(Shallow)'][grid] = {'tts': tts, 'iter': iters, 'cycle': cycle}

    return data


def parse_speedups(gpu_path, wse_path):
    """Compute per-V-cycle speedup (GH200 / WSE-3) for each config and grid size.

    Returns (grid_labels, speedup_dict) where grid_labels is e.g. ['16x16x16', ...]
    and speedup_dict maps config -> list of speedups aligned with grid_labels.
    """
    gpu = _parse_table(gpu_path)
    wse = _parse_table(wse_path)

    configs = ['6/6/100', '4/4/100', '4/4/6', '6/6/6']
    if '6/6/6(Shallow)' in wse:
        configs.append('6/6/6(Shallow)')

    # Grid sizes present in both GPU and WSE for the base configs
    common_grids = None
    for cfg in ['6/6/100', '4/4/100', '4/4/6', '6/6/6']:
        gpu_grids = set(gpu.get(cfg, {}).keys())
        wse_grids = set(wse.get(cfg, {}).keys())
        both = gpu_grids & wse_grids
        common_grids = both if common_grids is None else common_grids & both
    grids = sorted(common_grids)

    grid_labels = [f"{g}x{g}" for g in grids]
    speedups = {}
    for cfg in configs:
        vals = []
        for g in grids:
            # Shallow uses GPU 6/6/6 as baseline (it's a WSE optimization of 6/6/6)
            gpu_cfg = '6/6/6' if cfg == '6/6/6(Shallow)' else cfg
            gpu_cycle = gpu.get(gpu_cfg, {}).get(g, {}).get('cycle')
            wse_cycle = wse.get(cfg, {}).get(g, {}).get('cycle')
            if gpu_cycle and wse_cycle and wse_cycle > 0:
                vals.append(gpu_cycle / wse_cycle)
            else:
                vals.append(None)
        if any(v is not None for v in vals):
            speedups[cfg] = vals

    return grid_labels, speedups


def plot_hpgmg_speedup_bar(gpu_path: str, wse_path: str,
                           out_path: str = 'hpgmg_speedup_barplot.pdf'):
    grid_sizes, data = parse_speedups(gpu_path, wse_path)
    if not grid_sizes:
        raise ValueError("No valid speedup data found")

    x = np.arange(len(grid_sizes))
    bar_configs = ['6/6/100', '4/4/100', '4/4/6', '6/6/6', '6/6/6(Shallow)']
    n_bars = sum(1 for c in bar_configs if data.get(c))
    width = 0.2 if n_bars <= 4 else 0.14  # narrower when 5 configs
    offset_span = 1.5 if n_bars <= 4 else 2.0  # wider spread when 5 so bars don't overlap
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5th color for Shallow
    offsets = np.linspace(-offset_span * width, offset_span * width, len(bar_configs))


    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    handles = []
    labels = []

    # Bar groups for the actual configs
    for i, config in enumerate(bar_configs):
        values = data.get(config, [])
        if values:
            offset = offsets[i]
            ax.bar(x + offset, values, width, color=colors[i],
                   edgecolor='black', alpha=0.85, linewidth=0.5)
            # Color patch without box (edgecolor='none') for legend
            handles.append(Patch(facecolor=colors[i], edgecolor='none', alpha=0.85))
            labels.append(config)

    # Vertical arrow from red (6/6/6) to purple (6/6/6 Shallow); keep % text on top of purple bar
    if data.get('6/6/6') and data.get('6/6/6(Shallow)'):
        idx_666 = bar_configs.index('6/6/6')
        idx_shallow = bar_configs.index('6/6/6(Shallow)')
        for i in range(len(grid_sizes)):
            v_666 = data['6/6/6'][i]
            v_shallow = data['6/6/6(Shallow)'][i]
            if v_666 > 0:
                pct = ((v_shallow - v_666) / v_666) * 100
                x_shallow = i + offsets[idx_shallow]
                # Text on top of purple bar
                ax.text(x_shallow, v_shallow + 0.3, f'+{pct:.0f}%',
                        ha='center', va='bottom', fontsize=12, color='#232323')
            if v_666 > 0 and v_shallow > v_666:
                pct = ((v_shallow - v_666) / v_666) * 100
                xi = i + 0.5 * (offsets[idx_666] + offsets[idx_shallow])
                y_from = v_666
                y_to = v_shallow
                # Vertical arrow from red bar top to purple bar top
                ax.annotate(
                    '',
                    xy=(xi, y_to),
                    xytext=(xi, y_from),
                    arrowprops=dict(
                        arrowstyle='->',
                        color='#333',
                        lw=1.5,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                )

    # Add GH200 baseline and --- line for legend
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    handles.append(Line2D([0], [0], color= "black", linestyle='--', linewidth=1))
    labels.append('Baseline (1.0×)')

    ax.set_xlabel('Grid sizes', fontsize=18)
    ax.set_ylabel('Relative speedup over GH200', fontsize=18)
    ax.set_title('GLOW VS HPGMG', fontsize=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(grid_sizes, fontsize=14)
    ax.legend(
        handles=handles,
        labels=labels,
        title='Configuration (pre/post/coarse)',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        frameon=True,
        fancybox=True,
        ncol=1
    )
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Ensure y-axis starts at 0 for clearer bar comparison
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    default_gpu = os.path.join(_script_dir, 'gpu_numbers.txt')
    default_wse = os.path.join(_script_dir, 'wse_numbers.txt')
    ap = argparse.ArgumentParser(description='Plot HPGMG speedup (WSE-3 over GH200)')
    ap.add_argument('--gpu', default=default_gpu, help='GPU numbers file')
    ap.add_argument('--wse', default=default_wse, help='WSE numbers file')
    ap.add_argument('-o', '--output', default='hpgmg_speedup_barplot.pdf',
                    help='Output figure path (default: hpgmg_speedup_barplot.pdf)')
    args = ap.parse_args()
    plot_hpgmg_speedup_bar(args.gpu, args.wse, args.output)


if __name__ == '__main__':
    main()
