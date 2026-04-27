#!/usr/bin/env python3
"""
Roofline FLOP Analysis for GMG V-Cycle on WSE-3.

Parses roofline counter data from response.txt files (after recompilation
with FLOP counters enabled in kernel_gmg_vcycle.csl).

Follows Ruichisai et al. (Table V) methodology for per-operation FLOP
counting on Cerebras WSE.

Usage:
  # After recompiling with counters and running:
  python roofline_analysis.py <response.txt or all_responses.txt>

FLOPs per operation type:
  FSUB: 1 FLOP,  2 loads, 1 store
  FMAC: 2 FLOPs, 3 loads, 1 store  (fused multiply-add)
  FMUL: 1 FLOP,  2 loads, 1 store
  FADD: 1 FLOP,  2 loads, 1 store
  FNEG: 1 FLOP,  1 load,  1 store
  FMOV: 0 FLOPs, 0-1 loads, 1 store
  FMAX: 1 FLOP,  1 load,  1 store
"""

import argparse
import re
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# FLOP weight per operation type (matching ruichisai Table V)
FLOP_PER_OP = {
    'fsub': 1,
    'fmac': 2,        # fused multiply-add = 2 FLOPs
    'fmul': 1,
    'fadd': 1,
    'fneg': 1,
    'fmov_mem': 0,    # @fmovs mem→mem: 0 FLOPs
    'fmov_zero': 0,   # @fmovs zero-fill: 0 FLOPs
    'fmax': 1,
}

# Memory traffic per operation (loads + stores) in units of f32 words
MEM_TRAFFIC = {
    'fsub': {'loads': 2, 'stores': 1},
    'fmac': {'loads': 3, 'stores': 1},
    'fmul': {'loads': 2, 'stores': 1},
    'fadd': {'loads': 2, 'stores': 1},
    'fneg': {'loads': 1, 'stores': 1},
    'fmov_mem': {'loads': 1, 'stores': 1},   # SRAM read + SRAM write = 8 B
    'fmov_zero': {'loads': 0, 'stores': 1},  # no read, only SRAM write = 4 B
    'fmax': {'loads': 1, 'stores': 1},
}

# WSE-3 machine parameters (CS-3 golden data)
# Per-PE: 875 MHz clock, 1 FMAC/cycle = 2 FLOPs/cycle = 1.75 GFLOPS/PE
# Full wafer: 893,064 PEs × 1.75 GFLOPS = 1.563 PFLOPS peak (f32)
# Memory BW: 25 PB/s (full wafer), 14 GB/s read + 14 GB/s write per PE
PE_PEAK_GFLOPS = 1.75     # GFLOPS per PE: 2 FLOPs/cycle (FMAC) × 875 MHz
WSE3_FULL_WAFER_PES = 893_064
WSE3_PEAK_FLOPS = WSE3_FULL_WAFER_PES * PE_PEAK_GFLOPS * 1e9  # ~1.56 PFLOPS
WSE3_MEM_BW = 25e15       # 25 PB/s memory bandwidth (full wafer)
# Fabric BW: router receives 1 wavelet/cycle per direction from 4 neighbors
# Fabric BW per PE = 4 directions × 4 bytes/wavelet × 875 MHz
# System = num_PEs × 4 dirs × 4 bytes × frequency
WSE3_CLOCK = 875e6  # 875 MHz (same clock for CE and fabric)
WSE3_FABRIC_BW_PER_PE = 4 * 4 * WSE3_CLOCK  # 14.0 GB/s per PE
WSE3_FABRIC_BW = WSE3_FULL_WAFER_PES * 4 * 4 * WSE3_CLOCK  # ~12.5 PB/s system


def parse_device_counters(filepath):
    """Parse FLOP counter data from response.txt / all_responses.txt.

    Looks for the roofline table output from run_gmg_vcycle.py:
      Level N:
      FSUB      count  flop/op  total_flops  mem_traffic
      ...

    Returns dict: grid_label -> {level -> {op -> count}}
    """
    results = {}
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return results

    with open(filepath, 'r') as f:
        content = f.read()

    blocks = re.split(r'^Output directory: ', content, flags=re.MULTILINE)

    for block in blocks[1:] if len(blocks) > 1 else [content]:
        # Get grid size
        m = re.match(r'(?:shallow_)?out_dir_S(\d+)x_', block)
        if m:
            size = int(m.group(1))
            grid_label = f'{size}x{size}'
        else:
            grid_label = 'unknown'

        # Parse roofline table
        level_counts = {}
        level_pattern = re.compile(
            r'^Level (\d+)[^\n]*:\s*\n'
            r'Operation.*\n-+\n'
            r'((?:(?:FSUB|FMAC|FMUL|FADD|FNEG|FMOV_MEM|FMOV_ZERO|FMAX|FMOV32)\s+[\d,]+.*\n)*)',
            re.MULTILINE
        )

        for match in level_pattern.finditer(block):
            level = int(match.group(1))
            ops_text = match.group(2)
            ops = {}
            for line in ops_text.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 2:
                    op_name = parts[0].lower()
                    count = int(parts[1].replace(',', ''))
                    ops[op_name] = count
            level_counts[level] = ops

        # Parse iteration count
        m_iters = re.search(r'Device iterations(?:\s+to\s+converge)?\s*:\s*(\d+)', block)
        device_iterations = int(m_iters.group(1)) if m_iters else 1

        # Parse grand total (PE FLOPs line from roofline summary)
        m_flops = re.search(
            r'(?:Per PE\(0,0\), one V-cycle — FLOPs:\s*([\d,]+)|'
            r'(?:Total FLOPs|FLOPs per PE\s*\(1st V-cycle\))[^:]*:\s*([\d,]+))',
            block,
        )
        m_time = re.search(
            r'(?:Total solver wall time|Total Solver time)[^\d]*([\d.]+)\s*us',
            block,
        )
        m_pes = re.search(
            r'Full fine grid — allocated PEs:\s*([\d,]+)',
            block,
        ) or re.search(r'Active PEs[^:]*:\s+([\d,]+)', block)
        m_achieved = re.search(
            r'(?:Achieved performance|PE\(0,0\) achieved):\s+([\d.e+\-]+)\s+FLOP/s',
            block,
        )
        m_achieved_gf = re.search(
            r'Achieved at PE\(0,0\) vs peak:\s*([\d.]+)\s*GFLOP/s',
            block,
        )

        # Parse per-level timing from "Time per operation and level" table
        # Format: | level | smooth | residual | restriction | interpolation | setup | convergence | total |
        # Parse both total (with convergence) and vcycle-only (without convergence)
        level_times = {}       # total including convergence
        level_conv_times = {}  # convergence time only (non-zero only at L0)
        # First timing table only (stop before "7-pt Stencil" / next section — avoids
        # matching later tables whose rows don't have the same column layout).
        time_table_match = re.search(
            r'Time per operation and level[^\n]*\n(.*?)(?=\n\s*7-pt Stencil)',
            block,
            re.DOTALL,
        )
        if time_table_match:
            table_text = time_table_match.group(1)
            for line in table_text.split('\n'):
                if '|' not in line or not line.strip().startswith('|'):
                    continue
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if len(cells) >= 8:  # level + 6 ops + total
                    try:
                        lv = int(cells[0])
                        total_match = re.search(r'([\d.]+)us', cells[-1])
                        conv_match = re.search(r'([\d.]+)us', cells[-2])  # convergence
                        total_us = float(total_match.group(1)) if total_match else 0
                        conv_us = float(conv_match.group(1)) if conv_match else 0
                        level_times[lv] = total_us
                        level_conv_times[lv] = conv_us
                    except ValueError:
                        pass

        # FMOV32 counts are now parsed as part of level_counts (in the Table V section)

        if level_counts:
            total_time_us = float(m_time.group(1)) if m_time else 0

            results[grid_label] = {
                'levels': level_counts,
                'level_times': level_times,
                'level_conv_times': level_conv_times,
                'device_iterations': device_iterations,
                'total_time_us': total_time_us,
            }

    return results


def print_table_v(counts, total_time_us=None, device_iterations=1):
    """Print Table V style output (like Ruichisai et al. / Jacquelin et al.).

    counts: dict of {level -> {op_type -> count}}
    FMOV32 counts are in counts[level]['fmov32'].
    Counters are TOTALS across all V-cycle iterations. Convergence-check
    operators (the extra SpMV/residual/allreduce MAX) are excluded in the kernel
    via the `in_convergence` flag. Achieved GFLOP/s uses total_flops/total_time
    (ratio is iteration-invariant, so no explicit division needed).
    """
    all_ops = ['fsub', 'fmac', 'fmul', 'fadd', 'fneg', 'fmov_mem', 'fmov_zero', 'fmax']

    print("\n" + "=" * 130)
    print("TABLE V: FLOP & Traffic Counts per Operation Type per Level (per PE)")
    print("=" * 130)

    for level in sorted(counts.keys()):
        c = counts[level]
        fab_loads = c.get('fmov32', 0)

        print(f"\nLevel {level}:")
        print(f"{'Operation':<10} {'Count':>12} {'FLOP/op':>8} {'Total FLOPs':>12} "
              f"{'Mem Loads':>10} {'Mem Stores':>10} {'Mem Traffic(B)':>14} {'Fab Loads':>10} {'Fab Traffic(B)':>14}")
        print("-" * 105)

        level_total_flops = 0
        level_total_mem = 0

        for op in all_ops:
            count = c.get(op, 0)
            flop = FLOP_PER_OP[op]
            total_flops = count * flop
            loads = MEM_TRAFFIC[op]['loads'] * count
            stores = MEM_TRAFFIC[op]['stores'] * count
            mem_bytes = (loads + stores) * 4

            level_total_flops += total_flops
            level_total_mem += mem_bytes

            if count > 0:
                print(f"{op.upper():<10} {count:>12,} {flop:>8} {total_flops:>12,} "
                      f"{loads:>10,} {stores:>10,} {mem_bytes:>14,} {'0':>10} {'0':>14}")

        fab_bytes = fab_loads * 4
        fab_mem_store = fab_loads * 4  # 1 SRAM store per fabric receive
        level_total_mem += fab_mem_store
        mem_ai = level_total_flops / level_total_mem if level_total_mem > 0 else 0
        fab_ai = level_total_flops / fab_bytes if fab_bytes > 0 else 0

        # FMOV32 row: 0 FLOPs, 1 mem store (fabric data lands in SRAM), 1 fabric load
        print(f"{'FMOV32':<10} {fab_loads:>12,} {'0':>8} {'0':>12} "
              f"{'0':>10} {fab_loads:>10,} {fab_mem_store:>14,} {fab_loads:>10,} {fab_bytes:>14,}")

        print("-" * 105)
        print(f"{'TOTAL':<10} {'':>12} {'':>8} {level_total_flops:>12,} "
              f"{'':>10} {'':>10} {level_total_mem:>14,} {fab_loads:>10,} {fab_bytes:>14,}")
        print(f"  Memory AI: {mem_ai:.4f} FLOP/Byte    Fabric AI: {fab_ai:.4f} FLOP/Byte")

    # Grand total
    print("\n" + "=" * 130)
    print("GRAND TOTAL (all levels)")
    print("=" * 130)

    grand_counts = {op: 0 for op in all_ops}
    for level_c in counts.values():
        for op in all_ops:
            grand_counts[op] += level_c.get(op, 0)

    grand_total_flops = 0
    grand_total_mem = 0
    grand_fab_loads = sum(c.get('fmov32', 0) for c in counts.values())
    if grand_fab_loads == 0:
        grand_fab_loads = sum(int(c.get('fmov32', 0)) for c in counts.values())
    grand_fab_bytes = grand_fab_loads * 4

    print(f"{'Operation':<10} {'Count':>12} {'FLOP/op':>8} {'Total FLOPs':>12} "
          f"{'Mem Traffic(B)':>14} {'Fab Traffic(B)':>14}")
    print("-" * 80)

    for op in all_ops:
        count = grand_counts[op]
        flop = FLOP_PER_OP[op]
        total_flops = count * flop
        mem_bytes = (MEM_TRAFFIC[op]['loads'] + MEM_TRAFFIC[op]['stores']) * count * 4
        grand_total_flops += total_flops
        grand_total_mem += mem_bytes

        if count > 0:
            print(f"{op.upper():<10} {count:>12,} {flop:>8} {total_flops:>12,} "
                  f"{mem_bytes:>14,} {'0':>14}")

    grand_fmov_mem = grand_fab_loads * 4  # SRAM store per fabric wavelet (matches run_gmg_vcycle.py)
    grand_total_mem += grand_fmov_mem

    print(f"{'FMOV32':<10} {grand_fab_loads:>12,} {'0':>8} {'0':>12} "
          f"{grand_fmov_mem:>14,} {grand_fab_bytes:>14,}")

    grand_mem_ai = grand_total_flops / grand_total_mem if grand_total_mem > 0 else 0
    grand_fab_ai = grand_total_flops / grand_fab_bytes if grand_fab_bytes > 0 else 0
    print("-" * 80)
    print(f"{'TOTAL':<10} {'':>12} {'':>8} {grand_total_flops:>12,} "
          f"{grand_total_mem:>14,} {grand_fab_bytes:>14,}")
    print(f"\nTotal FLOPs per PE:             {grand_total_flops:,}")
    print(f"Total Memory Traffic per PE:    {grand_total_mem:,} bytes")
    print(f"Total Fabric Traffic per PE:    {grand_fab_bytes:,} bytes  ({grand_fab_loads:,} wavelets)")
    print(f"Memory Arithmetic Intensity:    {grand_mem_ai:.4f} FLOP/Byte")
    print(f"Fabric Arithmetic Intensity:    {grand_fab_ai:.4f} FLOP/Byte")
    print(f"Fabric BW ceiling:              {WSE3_FABRIC_BW_PER_PE/1e9:.1f} GB/s per PE "
          f"({WSE3_FABRIC_BW/1e15:.2f} PB/s system)  "
          f"[4 dirs × 4 B × {WSE3_CLOCK/1e6:.0f} MHz × {WSE3_FULL_WAFER_PES:,} PEs]")

    if total_time_us:
        print(f"\nTotal solver wall time:         {total_time_us:.3f} us ({device_iterations} iterations)")
    print(f"(Achieved FLOP/s computed in summary table below using per-level times)")



def compute_summary(all_results):
    """Compute per-grid summary for roofline plots and tables.

    FLOP/memory/fabric *counters* in the log are TOTALS across all V-cycles.
    Convergence-check operators (extra SpMV + residual + allreduce MAX) are
    gated out by the `in_convergence` flag in the kernel, so the counters
    represent only the pure V-cycle operators. Per-level *timings* are also
    totals; pure-operators time = level_time - conv_time (conv only at L0).
    GFLOP/s uses total_flops/total_time → ratio is iteration-invariant.

    PE(0,0) is active at every level — its counters are the reference PE.
    """
    summary = {}
    all_ops = ['fsub', 'fmac', 'fmul', 'fadd', 'fneg', 'fmov_mem', 'fmov_zero', 'fmax']

    for grid_label, data in all_results.items():
        size = int(grid_label.split('x')[0])
        num_levels = len(data['levels'])
        iters = data.get('device_iterations', 1)
        total_time_us = data.get('total_time_us', 0)

        pe00_total_flops = 0
        pe00_total_mem = 0
        level_details = []

        for level in sorted(data['levels'].keys()):
            level_c = data['levels'][level]
            active_pes = (size // (2 ** level)) ** 2
            nz = size >> level

            # Counters are TOTALS across all V-cycles (convergence ops excluded via
            # in_convergence flag in kernel). GFLOP/s uses flops/time (same scaling).
            level_flops = 0
            level_mem = 0
            for op in all_ops:
                count = level_c.get(op, 0)
                level_flops += count * FLOP_PER_OP[op]
                level_mem += count * (MEM_TRAFFIC[op]['loads'] + MEM_TRAFFIC[op]['stores']) * 4

            # FMOV32: fabric receive → 1 SRAM store (memory) + 1 fabric load
            fab_load = level_c.get('fmov32', 0)
            level_mem += fab_load * 4  # 1 SRAM store per fabric receive = 4 bytes

            pe00_total_flops += level_flops
            pe00_total_mem += level_mem

            level_ai = level_flops / level_mem if level_mem > 0 else 0
            # Per-level time: TOTAL across all V-cycles (accumulated by kernel).
            # level_time_us includes convergence (only L0 has conv_us > 0).
            level_time_us = data.get('level_times', {}).get(level, 0)
            level_conv_us = data.get('level_conv_times', {}).get(level, 0)
            level_vcycle_us = level_time_us - level_conv_us
            level_time_s = level_time_us * 1e-6 if level_time_us else 0
            level_vcycle_s = level_vcycle_us * 1e-6 if level_vcycle_us else 0
            level_achieved = level_flops / level_time_s if level_time_s > 0 else 0
            level_achieved_vcycle = level_flops / level_vcycle_s if level_vcycle_s > 0 else 0
            fabric_bytes = fab_load * 4
            fabric_ai = level_flops / fabric_bytes if fabric_bytes > 0 else 0

            level_details.append({
                'level': level, 'nz': nz, 'active_pes': active_pes,
                'flops': level_flops, 'mem': level_mem, 'ai': level_ai,
                'fabric_bytes': fabric_bytes, 'fabric_ai': fabric_ai,
                'time_us': level_time_us, 'conv_us': level_conv_us,
                'vcycle_us': level_vcycle_us,
                'achieved': level_achieved,           # with convergence
                'achieved_vcycle': level_achieved_vcycle,  # V-cycle only (no convergence)
            })

        pe00_ai = pe00_total_flops / pe00_total_mem if pe00_total_mem > 0 else 0
        # 1V time = sum of per-level times (directly measured for V-cycle 1)
        sum_level_time_us = sum(ld['time_us'] for ld in level_details)
        sum_vcycle_time_us = sum(ld['vcycle_us'] for ld in level_details)
        sum_vcycle_s = sum_vcycle_time_us * 1e-6 if sum_vcycle_time_us > 0 else 0
        pe00_achieved = pe00_total_flops / sum_vcycle_s if sum_vcycle_s > 0 else 0

        summary[grid_label] = {
            'size': size,
            'num_levels': num_levels,
            'device_iterations': iters,
            'total_solver_time_us': total_time_us,
            'pe00_flops': pe00_total_flops,
            'pe00_mem': pe00_total_mem,
            'pe00_ai': pe00_ai,
            'time_us': sum_level_time_us,       # 1V total (with convergence)
            'vcycle_us': sum_vcycle_time_us,     # 1V V-cycle only (no convergence)
            'pe00_achieved': pe00_achieved,      # uses vcycle time
            'pct_pe_peak': 100 * pe00_achieved / (PE_PEAK_GFLOPS * 1e9) if pe00_achieved > 0 else 0,
            'level_details': level_details,
        }
    return summary


def _setup_paper_style():
    """Configure matplotlib for paper-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 11,
        'lines.linewidth': 2.0,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.8,
        'grid.linewidth': 0.6,
    })


def _draw_roofline_ceiling(ax, peak_flops, mem_bw, label_peak, ai_range,
                           fabric_bw=None, color='black'):
    """Draw roofline ceiling lines on an axis, with optional fabric ceiling."""
    mem_ceiling = mem_bw * ai_range
    compute_ceiling = np.full_like(ai_range, peak_flops)
    roofline = np.minimum(mem_ceiling, compute_ceiling)
    ax.loglog(ai_range, roofline, '-', color=color, linewidth=3, label='Memory Roofline')
    ax.axhline(y=peak_flops, color='firebrick', linestyle='--', linewidth=2, alpha=0.6,
               label=f'Peak: {label_peak}')
    if fabric_bw is not None:
        fab_ceiling = fabric_bw * ai_range
        fab_roofline = np.minimum(fab_ceiling, compute_ceiling)
        ax.loglog(ai_range, fab_roofline, '-', color='darkorange', linewidth=2.5, alpha=0.8,
                  label='Fabric Roofline')


def _style_axis(ax, xlabel, ylabel, title, xlim=(0.01, 10), ylim_lo=None, ylim_hi=None):
    """Apply consistent styling to a roofline axis."""
    ax.set_xlabel(xlabel, fontsize=15, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=15, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=8)
    ax.grid(False)
    ax.set_xlim(xlim)
    if ylim_lo is not None and ylim_hi is not None:
        ax.set_ylim(ylim_lo, ylim_hi)
    ax.tick_params(axis='both', which='major', labelsize=13, width=1.0, length=5)
    ax.tick_params(axis='both', which='minor', width=0.6, length=3)


def plot_roofline(summary, output_dir):
    """Plot 2-panel roofline for paper.

    (a) Per PE(0,0) across levels. Ceiling = 1 PE peak.
    (b) Active-PE system with per-level ceilings.
    """
    _setup_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.subplots_adjust(wspace=0.22, left=0.09, right=0.99, top=0.86, bottom=0.15)

    sorted_grids = sorted(summary.keys(), key=lambda g: int(g.split('x')[0]))
    largest_grid = sorted_grids[-1] if sorted_grids else None
    if not largest_grid:
        return

    s = summary[largest_grid]
    levels = s['level_details']
    n_levels = len(levels)

    PE_MEM_BW = WSE3_MEM_BW / WSE3_FULL_WAFER_PES
    PE_FABRIC_BW = WSE3_FABRIC_BW_PER_PE
    PE_PEAK = PE_PEAK_GFLOPS * 1e9
    ai_range = np.logspace(-3, 2, 500)
    colors_level = plt.cm.plasma(np.linspace(0.15, 0.85, n_levels))
    grid_size = s['size']
    fine_pes = grid_size * grid_size
    fine_peak = fine_pes * PE_PEAK
    fine_bw = fine_pes * PE_MEM_BW
    fine_fab_bw = fine_pes * PE_FABRIC_BW
    MARKER_SIZE = 9
    MARKER_EDGE = 1.2

    # Compute y-axis lower bound from data: one decade below the weakest level
    pe_perfs = [ld['achieved_vcycle'] for ld in levels if ld['achieved_vcycle'] > 0]
    min_pe_perf = min(pe_perfs) if pe_perfs else 1e3
    pe_ylim_lo = 10 ** (np.floor(np.log10(min_pe_perf)) - 1)

    sys_perfs = []
    for ld in levels:
        active = ld['active_pes']
        vcycle_s = ld['vcycle_us'] * 1e-6 if ld.get('vcycle_us', 0) > 0 else 0
        sp = (ld['flops'] * active) / vcycle_s if vcycle_s > 0 else 0
        if sp > 0:
            sys_perfs.append(sp)
    min_sys_perf = min(sys_perfs) if sys_perfs else 1e3
    sys_ylim_lo = 10 ** (np.floor(np.log10(min_sys_perf)) - 1)

    # Config text (top center)
    cfg_text = f"{grid_size}\u00b3  |  {n_levels} levels  |  {fine_pes:,} PEs"
    fig.text(0.5, 0.96, cfg_text, ha='center', va='top', fontsize=13,
             fontstyle='italic', color='#333333')

    # ===================================================================
    # Panel (a): Per PE(0,0) — across levels
    # ===================================================================
    ax1 = axes[0]
    _draw_roofline_ceiling(ax1, PE_PEAK, PE_MEM_BW,
                           f'{PE_PEAK_GFLOPS} GFLOP/s', ai_range,
                           fabric_bw=PE_FABRIC_BW)

    for i, (ld, color) in enumerate(zip(levels, colors_level)):
        perf = ld['achieved_vcycle']
        pct = 100 * perf / PE_PEAK if perf > 0 else 0
        if perf > 0 and ld['ai'] > 0:
            # Memory AI dot (square)
            ax1.plot(ld['ai'], perf, 's', color=color, markersize=MARKER_SIZE,
                     markeredgecolor='black', markeredgewidth=MARKER_EDGE, zorder=5)
            # Alternate labels left/right to avoid vertical overlap
            if i % 2 == 0:
                x_off, ha = (16, 'left')
            else:
                x_off, ha = (-16, 'right')
            ax1.annotate(f"L{ld['level']} ({pct:.0f}%)",
                         (ld['ai'], perf),
                         textcoords="offset points", xytext=(x_off, 0), fontsize=10,
                         color=color, fontweight='bold', va='center', ha=ha)
            # Fabric AI dot (circle)
            fab_ai = ld.get('fabric_ai', 0)
            if fab_ai > 0:
                ax1.plot(fab_ai, perf, 'o', color=color, markersize=MARKER_SIZE - 2,
                         markeredgecolor='black', markeredgewidth=MARKER_EDGE, zorder=5,
                         alpha=0.8)

    ax1.plot([], [], 's', color='gray', markersize=8, markeredgecolor='black', label='Memory AI')
    ax1.plot([], [], 'o', color='gray', markersize=8, markeredgecolor='black', label='Fabric AI')
    ax1.legend(fontsize=11, loc='lower right', framealpha=0.9)
    _style_axis(ax1, 'Arithmetic Intensity (FLOP/Byte)', 'Performance (FLOP/s)',
                f'(a) Per PE(0,0)',
                ylim_lo=pe_ylim_lo, ylim_hi=PE_PEAK * 10)

    # ===================================================================
    # Panel (b): Active-PE system — per-level ceilings
    # ===================================================================
    ax2 = axes[1]

    for i, (ld, color) in enumerate(zip(levels, colors_level)):
        active = ld['active_pes']
        active_peak = active * PE_PEAK
        active_bw = active * PE_MEM_BW

        level_roofline = np.minimum(active_bw * ai_range,
                                    np.full_like(ai_range, active_peak))
        ax2.loglog(ai_range, level_roofline, '-', color=color,
                   linewidth=2.5, alpha=0.75)

        sys_flops = ld['flops'] * active
        sys_mem = ld['mem'] * active
        sys_ai = sys_flops / sys_mem if sys_mem > 0 else 0
        vcycle_s = ld['vcycle_us'] * 1e-6 if ld.get('vcycle_us', 0) > 0 else 0
        sys_perf = sys_flops / vcycle_s if vcycle_s > 0 else 0
        pct = 100 * sys_perf / active_peak if active_peak > 0 and sys_perf > 0 else 0
        peak_str = f'{active_peak/1e9:.0f}G' if active_peak < 1e12 else f'{active_peak/1e12:.1f}T'

        if sys_perf > 0 and sys_ai > 0:
            ax2.plot(sys_ai, sys_perf, 's', color=color, markersize=MARKER_SIZE,
                     markeredgecolor='black', markeredgewidth=MARKER_EDGE, zorder=5)
            ax2.annotate(f"L{ld['level']}  pk={peak_str}  ({pct:.0f}%)",
                         (sys_ai, sys_perf),
                         textcoords="offset points", xytext=(12, 0), fontsize=10,
                         color=color, fontweight='bold', va='center',
                         bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                   edgecolor='none', alpha=0.85),
                         zorder=6)

    _style_axis(ax2, 'Arithmetic Intensity (FLOP/Byte)', '',
                f'(b) Active-PE System (per-level peak)',
                ylim_lo=sys_ylim_lo, ylim_hi=fine_peak * 10)

    outpath_png = os.path.join(output_dir, 'roofline_plot.png')
    outpath_pdf = os.path.join(output_dir, 'roofline_plot.pdf')
    plt.savefig(outpath_png, dpi=300, bbox_inches='tight')
    plt.savefig(outpath_pdf, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {outpath_png}")
    print(f"Saved: {outpath_pdf}")


def _fmt_bw(val):
    """Format bandwidth value with appropriate unit."""
    if val >= 1e15:
        return f"{val/1e15:.2f} PB/s"
    elif val >= 1e12:
        return f"{val/1e12:.2f} TB/s"
    elif val >= 1e9:
        return f"{val/1e9:.2f} GB/s"
    else:
        return f"{val/1e6:.2f} MB/s"


def _fmt_flops(val):
    """Format FLOP/s value with appropriate unit."""
    if val >= 1e15:
        return f"{val/1e15:.2f} PFLOP/s"
    elif val >= 1e12:
        return f"{val/1e12:.2f} TFLOP/s"
    elif val >= 1e9:
        return f"{val/1e9:.2f} GFLOP/s"
    else:
        return f"{val/1e6:.2f} MFLOP/s"


def print_machine_peaks(grid_size, levels_details):
    """Print machine peak table for compute, memory BW, and fabric BW
    at three scales: per PE, per active PEs (each level), and full grid."""
    PE_PEAK = PE_PEAK_GFLOPS * 1e9
    PE_MEM_BW = WSE3_MEM_BW / WSE3_FULL_WAFER_PES
    PE_FAB_BW = WSE3_FABRIC_BW_PER_PE
    fine_pes = grid_size * grid_size

    print("\n" + "=" * 120)
    print("MACHINE PEAKS: Compute, Memory BW, Fabric BW")
    print("=" * 120)

    # Per-PE peaks
    print(f"\n  Per-PE peaks (1 PE):")
    print(f"    Compute:   {_fmt_flops(PE_PEAK):>15}   (1 FMAC/cycle × 2 FLOPs × 875 MHz)")
    print(f"    Memory BW: {_fmt_bw(PE_MEM_BW):>15}   (14 GB/s read + 14 GB/s write)")
    print(f"    Fabric BW: {_fmt_bw(PE_FAB_BW):>15}   (4 dirs × 4 B × {WSE3_CLOCK/1e6:.0f} MHz)")
    print(f"    Mem ridge:    {PE_PEAK / PE_MEM_BW:.4f} FLOP/Byte")
    print(f"    Fab ridge:    {PE_PEAK / PE_FAB_BW:.4f} FLOP/Byte")

    # Per-level peaks (active PEs at each level)
    print(f"\n  Per-level peaks (active PEs at each multigrid level):")
    print(f"  {'Level':>6} {'Active PEs':>12} {'Compute Peak':>16} {'Memory BW':>16} {'Fabric BW':>16} "
          f"{'Mem Ridge':>12} {'Fab Ridge':>12}")
    print("  " + "-" * 95)
    for ld in levels_details:
        n_pes = ld['active_pes']
        lvl_compute = n_pes * PE_PEAK
        lvl_mem = n_pes * PE_MEM_BW
        lvl_fab = n_pes * PE_FAB_BW
        print(f"  L{ld['level']:>5} {n_pes:>12,} {_fmt_flops(lvl_compute):>16} "
              f"{_fmt_bw(lvl_mem):>16} {_fmt_bw(lvl_fab):>16} "
              f"{PE_PEAK / PE_MEM_BW:>11.4f} {PE_PEAK / PE_FAB_BW:>11.4f}")

    # Full grid (allocated PEs = fine level)
    grid_compute = fine_pes * PE_PEAK
    grid_mem = fine_pes * PE_MEM_BW
    grid_fab = fine_pes * PE_FAB_BW
    print(f"\n  Full grid peaks ({fine_pes:,} PEs = {grid_size}x{grid_size}):")
    print(f"    Compute:   {_fmt_flops(grid_compute):>15}")
    print(f"    Memory BW: {_fmt_bw(grid_mem):>15}")
    print(f"    Fabric BW: {_fmt_bw(grid_fab):>15}")

    # Full wafer for reference
    wafer_compute = WSE3_FULL_WAFER_PES * PE_PEAK
    print(f"\n  Full wafer peaks ({WSE3_FULL_WAFER_PES:,} PEs = 762×1172):")
    print(f"    Compute:   {_fmt_flops(wafer_compute):>15}")
    print(f"    Memory BW: {_fmt_bw(WSE3_MEM_BW):>15}")
    print(f"    Fabric BW: {_fmt_bw(WSE3_FABRIC_BW):>15}")
    print("=" * 120)


def print_summary_table(summary):
    """Print PE(0,0) roofline summary per 1 V-cycle across all grid sizes.
    Counts/times are totals; GFLOP/s = total_flops/total_time is iter-invariant.
    Convergence ops excluded in kernel; conv time subtracted for *_vcycle columns.
    """

    sorted_grids = sorted(summary.keys(), key=lambda g: int(g.split('x')[0]))
    for g in sorted_grids:
        s = summary[g]

        # Print machine peaks first
        print_machine_peaks(s['size'], s['level_details'])

        # Per-grid summary
        PE_PEAK = PE_PEAK_GFLOPS * 1e9
        total_fab_bytes = sum(ld.get('fabric_bytes', 0) for ld in s['level_details'])
        total_fab_ai = s['pe00_flops'] / total_fab_bytes if total_fab_bytes > 0 else 0

        print("\n" + "=" * 170)
        print("PE(0,0) ROOFLINE SUMMARY — PER 1 V-CYCLE")
        print(f"Per-PE peak: {PE_PEAK_GFLOPS} GFLOP/s at 875 MHz (FMAC = 2 FLOPs/cycle)")
        print(f"Convention: Time includes convergence check. V-cycle-only % shown separately for L0.")
        print("=" * 170)

        # Per-level detail
        print(f"\n  {'Level':>6} {'nz':>6} {'Active PEs':>12} {'PE FLOPs':>12} "
              f"{'Mem AI':>8} {'Fab AI':>8} "
              f"{'Time(us)':>10} {'Conv(us)':>9} {'Vcyc(us)':>9} "
              f"{'GFLOP/s':>9} {'%pk':>6} {'GFLOP/s*':>9} {'%pk*':>6} "
              f"{'Sys GFLOP/s':>12} {'Sys Peak':>12} {'%Sys':>6}")
        print(f"  {'':>6} {'':>6} {'':>12} {'':>12} "
              f"{'':>8} {'':>8} "
              f"{'(total)':>10} {'(diag)':>9} {'(V-only)':>9} "
              f"{'(total)':>9} {'':>6} {'(V-only)':>9} {'':>6} "
              f"{'':>12} {'':>12} {'':>6}")
        print("  " + "-" * 160)
        sys_total_flops = 0
        sys_total_time = 0
        for ld in s['level_details']:
            pct_pe = 100 * ld['achieved'] / PE_PEAK if ld['achieved'] > 0 else 0
            pct_pe_vc = 100 * ld['achieved_vcycle'] / PE_PEAK if ld['achieved_vcycle'] > 0 else 0
            sys_flops = ld['flops'] * ld['active_pes']
            sys_peak = ld['active_pes'] * PE_PEAK
            sys_achieved = sys_flops / (ld['time_us'] * 1e-6) if ld['time_us'] > 0 else 0
            pct_sys = 100 * sys_achieved / sys_peak if sys_peak > 0 and sys_achieved > 0 else 0
            sys_total_flops += sys_flops
            sys_total_time += ld['time_us']
            fab_ai = ld.get('fabric_ai', 0)
            conv_us = ld.get('conv_us', 0)
            vcycle_us = ld.get('vcycle_us', 0)
            print(f"  {ld['level']:>6} {ld['nz']:>6} {ld['active_pes']:>12,} "
                  f"{ld['flops']:>12,.0f} "
                  f"{ld['ai']:>8.4f} {fab_ai:>8.4f} "
                  f"{ld['time_us']:>10.2f} {conv_us:>9.2f} {vcycle_us:>9.2f} "
                  f"{ld['achieved']/1e9:>9.3f} {pct_pe:>5.1f}% "
                  f"{ld['achieved_vcycle']/1e9:>9.3f} {pct_pe_vc:>5.1f}% "
                  f"{sys_achieved/1e9:>12.1f} {sys_peak/1e9:>12.1f} {pct_sys:>5.1f}%")

        # Total row
        alloc_peak = (s['size'] ** 2) * PE_PEAK
        sys_achieved_total = sys_total_flops / (s['time_us'] * 1e-6) if s['time_us'] > 0 else 0
        pct_alloc = 100 * sys_achieved_total / alloc_peak if alloc_peak > 0 else 0
        total_conv = sum(ld.get('conv_us', 0) for ld in s['level_details'])
        total_vcycle = s['time_us'] - total_conv
        total_vc_achieved = s['pe00_flops'] / (total_vcycle * 1e-6) if total_vcycle > 0 else 0
        pct_vc = 100 * total_vc_achieved / PE_PEAK if total_vc_achieved > 0 else 0
        print(f"  {'Total':>6} {'':>6} {s['size']**2:>12,} "
              f"{s['pe00_flops']:>12,.0f} "
              f"{s['pe00_ai']:>8.4f} {total_fab_ai:>8.4f} "
              f"{s['time_us']:>10.2f} {total_conv:>9.2f} {total_vcycle:>9.2f} "
              f"{s['pe00_achieved']/1e9:>9.3f} {s['pct_pe_peak']:>5.1f}% "
              f"{total_vc_achieved/1e9:>9.3f} {pct_vc:>5.1f}% "
              f"{sys_achieved_total/1e9:>12.1f} {alloc_peak/1e9:>12.1f} {pct_alloc:>5.1f}%")

        print(f"\n  * GFLOP/s* and %pk* = V-cycle only (excluding convergence diagnostic)")
        print(f"    Convergence check = SpMV + residual + allreduce MAX at L0 after each V-cycle")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', help='response.txt or all_responses.txt with device counter data')
    args = parser.parse_args()

    print("=" * 120)
    print("ROOFLINE FLOP ANALYSIS: GMG V-Cycle on WSE-3")
    print("Following Ruichisai et al. Table V methodology")
    print("=" * 120)

    if args.input:
        all_results = parse_device_counters(args.input)
        if all_results:
            for grid_label, data in sorted(all_results.items(),
                                            key=lambda x: int(x[0].split('x')[0])):
                print(f"\n{'#' * 120}")
                print(f"# Grid: {grid_label}")
                print(f"{'#' * 120}")
                # Counters are TOTALS across all V-cycles (conv ops excluded by kernel)
                print_table_v(
                    data['levels'],
                    total_time_us=data.get('total_time_us'),
                    device_iterations=data.get('device_iterations', 1),
                )

            summary = compute_summary(all_results)
            print_summary_table(summary)
            plot_roofline(summary, SCRIPT_DIR)
        else:
            print("No roofline counter data found in the file.")
            print("Make sure the file contains output from a run with FLOP counters enabled.")
    else:
        print("\nUsage: python roofline_analysis.py <response.txt or all_responses.txt>")
        print("\nPrerequisites:")
        print("  1. Recompile (kernel_gmg_vcycle.csl already instrumented with FLOP counters)")
        print("  2. Run on device: python compile_and_run_wse3.py --only-device")
        print("  3. Pass response.txt to this script")


if __name__ == '__main__':
    main()
