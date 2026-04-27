#!/usr/bin/env python3
"""
Plot performance data from GMG timing experiments.
Extracts Communication Time, Compute Time, and V-cycle times from output files.
ls -d out_dir_S*x* | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_6_6_6.txt
"""

import re
import sys
from typing import List, Dict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
    # Publication-quality defaults for paper figures (readable at single-column/full width)
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })
    PAPER_DPI = 300
except ImportError:
    HAS_MATPLOTLIB = False
    PAPER_DPI = 150
    print("Warning: matplotlib/numpy not found. Install with: pip install matplotlib numpy")


def parse_configuration_summary(text: str) -> List[Dict]:
    """
    Parse Configuration Summary sections to extract all configuration parameters.
    Uses run blocks (Output directory: out_dir_*) so each problem size appears once.
    """
    results = []
    blocks = split_into_run_blocks(text)

    for grid_size, block_text in blocks:
        run_text = block_text
        # Find Configuration Summary in this run
        config_match = re.search(
            r'Configuration Summary\s*============================================================\s*(.*?)(?=\n\n|\nProcessing on host|\nCompile command|\Z)',
            run_text,
            re.DOTALL
        )
        
        if not config_match:
            continue

        section = config_match.group(1)
        data = {}
        # Use grid_size from run block (one per problem size)
        size_val = int(grid_size.split('x')[0])
        data['grid_size'] = grid_size
        data['total_grid_size'] = size_val * size_val * size_val
        data['pe_tiles'] = grid_size
        
        # Extract all configuration parameters
        levels_match = re.search(r'Levels\s*:\s*(\d+)', section)
        if levels_match:
            data['levels'] = int(levels_match.group(1))
        
        max_iter_match = re.search(r'Max iterations\s*:\s*(\d+)', section)
        if max_iter_match:
            data['max_iterations'] = int(max_iter_match.group(1))
        
        tol_abs_match = re.search(r'Tolerance \(abs\)\s*:\s*([\d.eE+-]+)', section)
        if tol_abs_match:
            data['tolerance_abs'] = tol_abs_match.group(1)
        
        tol_rel_match = re.search(r'Tolerance \(rel\)\s*:\s*([\d.eE+-]+)', section)
        if tol_rel_match:
            data['tolerance_rel'] = tol_rel_match.group(1)
        
        pre_post_bottom_match = re.search(r'Pre/Post/Bottom iter\s*:\s*(\d+)/(\d+)/(\d+)', section)
        if pre_post_bottom_match:
            data['pre_iter'] = int(pre_post_bottom_match.group(1))
            data['post_iter'] = int(pre_post_bottom_match.group(2))
            data['bottom_iter'] = int(pre_post_bottom_match.group(3))
            data['pre_post_bottom_iter'] = f"{data['pre_iter']}/{data['post_iter']}/{data['bottom_iter']}"
        
        datatype_match = re.search(r'Datatype\s*:\s*(\w+)', section)
        if datatype_match:
            data['datatype'] = datatype_match.group(1)
        
        jacobi_omega_match = re.search(r'Jacobi omega\s*:\s*([\d.]+)', section)
        if jacobi_omega_match:
            data['jacobi_omega'] = float(jacobi_omega_match.group(1))
        
        stencil_match = re.search(r'Stencil alpha/beta\s*:\s*([\d.-]+)/([\d.]+)', section)
        if stencil_match:
            data['stencil_alpha'] = stencil_match.group(1)
            data['stencil_beta'] = stencil_match.group(2)
            data['stencil_alpha_beta'] = f"{data['stencil_alpha']}/{data['stencil_beta']}"
        
        block_size_match = re.search(r'Block size\s*:\s*(\d+)', section)
        if block_size_match:
            data['block_size'] = int(block_size_match.group(1))
        
        # Extract device iterations
        iter_match = re.search(r'Device iterations\s*:\s*(\d+)', section)
        if iter_match:
            data['device_iterations'] = int(iter_match.group(1))
        
        # Support legacy and current V-cycle time labels:
        #   "Wall time per V-cycle (total / iterations): X us"  (legacy)
        #   "1-V cycle time(Average) (us[cycles]): X us"         (mid)
        #   "1st V-cycle time (measured)     :    X us"          (old — misleading label)
        #   "1st V-cycle time (sum of per-level timers, ...)"    (old — misleading label)
        #   "Avg V-cycle time (no conv)      :    X us"          (current — standard methodology)
        vcycle_avg_match = re.search(
            r'(?:Wall time per V-cycle \(total / iterations\)'
            r'|1-V cycle time\(Average\)\s*\(us\[cycles\]\)'
            r'|1st V-cycle time \(measured\)'
            r'|1st V-cycle time \(sum of per-level timers, directly measured\)'
            r'|Avg V-cycle time \(no conv\))'
            r'\s*:\s*([\d.]+)\s*us',
            section,
        )
        if vcycle_avg_match:
            data['vcycle_avg_time_us'] = float(vcycle_avg_match.group(1))
        else:
            data['vcycle_avg_time_us'] = None
        
        # Extract device final |rho|_inf
        rho_match = re.search(r'Device final \|rho\|_inf\s*:\s*([\d.eE+-]+)', section)
        if rho_match:
            data['device_rho_inf'] = float(rho_match.group(1))
        
        # Find converged status in this device run (before Configuration Summary)
        # Pattern: "Converged: Yes" or "Converged: No" appears after rho values
        converged_match = re.search(r'Converged:\s*(Yes|No)', run_text, re.IGNORECASE)
        if converged_match:
            data['converged'] = converged_match.group(1).lower() == 'yes'
        else:
            data['converged'] = None
        
        # Extract compile time and run time (appear after Configuration Summary)
        compile_match = re.search(r'Compile time \(s\):\s*([\d.]+)', run_text, re.IGNORECASE)
        if not compile_match:
            compile_match = re.search(r'COMPILE TIME:\s*([\d.]+)\s*seconds', run_text, re.IGNORECASE)
        if compile_match:
            data['compile_time_s'] = float(compile_match.group(1))
        else:
            data['compile_time_s'] = None
        
        run_match = re.search(r'Run time \(s\):\s*([\d.]+)', run_text, re.IGNORECASE)
        if not run_match:
            run_match = re.search(r'RUN TIME:\s*([\d.]+)\s*seconds', run_text, re.IGNORECASE)
        if run_match:
            data['run_time_s'] = float(run_match.group(1))
        else:
            data['run_time_s'] = None

        results.append(data)
    return results


def split_into_run_blocks(text: str) -> List[tuple]:
    """
    Split file into one block per run using 'Output directory: out_dir_'.
    Returns list of (grid_size, block_text) where grid_size is e.g. '4x4x4'.
    Ensures exactly one record per problem size (no duplicates).
    """
    blocks = []
    # Split by run header; keep delimiter with following content
    parts = re.split(r'(Output directory: out_dir_S\d+x[^\n]*)', text)
    # parts[0] is preamble, then [delim, content, delim, content, ...]
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        header = parts[i]  # e.g. "Output directory: out_dir_S4x_L2_M100_P6_P6_B6"
        block_text = parts[i + 1]
        # Get size from next few lines: "Parameters: size=4, levels=2, ..."
        size_match = re.search(r'Parameters:\s*size=(\d+)', block_text)
        if size_match:
            size = int(size_match.group(1))
            grid_size = f"{size}x{size}x{size}"
            blocks.append((grid_size, block_text))
    return blocks


# SPMV table header strings (output format may vary: old "Time per Level" vs new "+ Smoothing (us[cycles]) per Level")
SPMV_HEADER_OLD = '7-pt Stencil Compute vs Communication Time per Level'
SPMV_HEADER_MID = '7-pt Stencil Compute vs Communication + Smoothing (us[cycles]) per Level'
SPMV_HEADER_NEW = '7-pt Stencil: SpMV vs smoothing (us[cycles]), per level, scaled to one V-cycle:'

# Interpolation micro-benchmark header (case changed over time)
INTERP_HEADERS = ('Interpolation micro-benchmark', 'Interpolation Micro-Benchmark')


def _find_interp_header(text: str):
    """Return the first matching interpolation header found in text, or None."""
    for header in INTERP_HEADERS:
        if header in text:
            return header
    return None


def _find_spmv_section(block_text: str):
    """Return section after SPMV header, or None if not found."""
    for header in (SPMV_HEADER_NEW, SPMV_HEADER_MID, SPMV_HEADER_OLD):
        if header in block_text:
            return block_text.split(header, 1)[-1]
    return None


def parse_spmv_totals(text: str) -> List[Dict]:
    """
    Parse SPMV Compute vs Communication Time tables from the output file.
    Returns list of dictionaries with grid_size, total_spmv_us, comm_time_us, compute_time_us.
    Uses run blocks to avoid duplicate problem sizes.
    """
    results = []
    blocks = split_into_run_blocks(text)
    for grid_size, block_text in blocks:
        section = _find_spmv_section(block_text)
        if section is None:
            continue
        # Extract total row: | total | TotalSpMV | Comm | Compute | ...
        total_match = re.search(
            r'\|\s*total\s*\|\s*([\d.]+)us.*?\|\s*([\d.]+)us.*?\|\s*([\d.]+)us',
            section,
            re.DOTALL
        )
        if total_match:
            total_spmv_us = float(total_match.group(1))
            comm_time_us = float(total_match.group(2))
            compute_time_us = float(total_match.group(3))
            results.append({
                'grid_size': grid_size,
                'total_spmv_us': total_spmv_us,
                'comm_time_us': comm_time_us,
                'compute_time_us': compute_time_us,
            })
    return results


def parse_spmv_per_level(text: str) -> List[Dict]:
    """
    Parse 7-pt Stencil Compute vs Communication table (all level rows) per run block.
    Returns list of {grid_size, levels: [ {level, total_spmv_us, comm_time_us, compute_time_us}, ... ] }.
    """
    results = []
    blocks = split_into_run_blocks(text)
    # Data row: | level | Total SpMV Time | Communication Time | Compute Time | ...
    row_re = re.compile(
        r'\|\s*(\d+)\s*\|'  # level
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'  # Total SpMV
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'  # Communication
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'   # Compute
    )
    for grid_size, block_text in blocks:
        section = _find_spmv_section(block_text)
        if section is None:
            continue
        # Only parse the 7-pt Stencil table; stop at Interpolation micro-benchmark (same regex would match its rows)
        interp_hdr = _find_interp_header(section)
        if interp_hdr:
            section = section.split(interp_hdr, 1)[0]
        levels = []
        for line in section.split('\n'):
            if re.search(r'\|\s*total\s*\|', line, re.IGNORECASE):
                continue
            match = row_re.search(line)
            if match:
                level = int(match.group(1))
                levels.append({
                    'level': level,
                    'total_spmv_us': float(match.group(2)),
                    'comm_time_us': float(match.group(3)),
                    'compute_time_us': float(match.group(4)),
                })
        if levels:
            results.append({'grid_size': grid_size, 'levels': levels})
    return results


def parse_interpolation_totals(text: str) -> List[Dict]:
    """
    Parse Interpolation Micro-Benchmark table total row from each run block.
    Returns list of dicts with grid_size, expand_z_T1, reset_routes_T21,
    send_data_T22, interp_add_T3, interp_total (us).
    Note: state_machine (T2.0) was removed to save timer memory.
    """
    results = []
    blocks = split_into_run_blocks(text)
    # Total row: | total | expand_z(T1) | bcast_total(T2) | reset_routes(T2.1) | send_data(T2.2) | interp_add(T3) | interp_total |
    total_row_re = re.compile(
        r'\|\s*total\s*\|\s*([\d.]+)us\s*\([^)]+\)\s*\|'   # T1 expand_z
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'                 # T2 bcast_total
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'                 # T2.1 reset_routes
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'                 # T2.2 send_data
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'                 # T3 interp_add
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'                  # interp_total
    )
    for grid_size, block_text in blocks:
        interp_hdr = _find_interp_header(block_text)
        if not interp_hdr:
            continue
        section = block_text.split(interp_hdr, 1)[-1]
        match = total_row_re.search(section)
        if match:
            results.append({
                'grid_size': grid_size,
                'expand_z_T1': float(match.group(1)),
                'reset_routes_T21': float(match.group(3)),
                'send_data_T22': float(match.group(4)),
                'interp_add_T3': float(match.group(5)),
                'interp_total': float(match.group(6)),
            })
    return results


def parse_interpolation_per_level(text: str) -> List[Dict]:
    """
    Parse full Interpolation Micro-Benchmark table (all level rows) per run block.
    Returns list of {grid_size, levels: [ {level, expand_z_T1, bcast_T2,
    reset_routes_T21, send_data_T22, interp_add_T3, interp_total}, ... ] }.
    Note: state_machine (T2.0) was removed to save timer memory.
    """
    results = []
    blocks = split_into_run_blocks(text)
    # Data row: | level | T1 | T2 | T2.1 | T2.2 | T3 | interp_total |
    row_re = re.compile(
        r'\|\s*(\d+)\s*\|'  # level
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'  # T1 expand_z
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'  # T2 bcast_total
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'  # T2.1 reset_routes
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'  # T2.2 send_data
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'  # T3 interp_add
        r'\s*([\d.]+)us\s*\([^)]+\)\s*\|'   # interp_total
    )
    for grid_size, block_text in blocks:
        interp_hdr = _find_interp_header(block_text)
        if not interp_hdr:
            continue
        section = block_text.split(interp_hdr, 1)[-1]
        levels = []
        for line in section.split('\n'):
            if re.search(r'\|\s*total\s*\|', line, re.IGNORECASE):
                continue
            match = row_re.search(line)
            if match:
                level = int(match.group(1))
                levels.append({
                    'level': level,
                    'expand_z_T1': float(match.group(2)),
                    'bcast_T2': float(match.group(3)),
                    'reset_routes_T21': float(match.group(4)),
                    'send_data_T22': float(match.group(5)),
                    'interp_add_T3': float(match.group(6)),
                    'interp_total': float(match.group(7)),
                })
        if levels:
            results.append({'grid_size': grid_size, 'levels': levels})
    return results


def parse_memory_usage(text: str) -> List[Dict]:
    """
    Parse Memory usage check section from the output file (one per run block).
    Extracts Code (FUNC symbols) and Data (OBJECT symbols) sizes in bytes.
    Uses the last occurrence in each block when multiple memory check sections exist.
    """
    results = []
    blocks = split_into_run_blocks(text)
    code_re = re.compile(r'Code \(FUNC symbols\):\s*(\d+)\s*bytes')
    data_re = re.compile(r'Data \(OBJECT symbols\):\s*(\d+)\s*bytes')
    for grid_size, block_text in blocks:
        code_matches = code_re.findall(block_text)
        data_matches = data_re.findall(block_text)
        code_bytes = int(code_matches[-1]) if code_matches else None
        data_bytes = int(data_matches[-1]) if data_matches else None
        results.append({
            'grid_size': grid_size,
            'code_bytes': code_bytes,
            'data_bytes': data_bytes,
        })
    return results


def parse_vcycle_times(text: str) -> List[Dict]:
    """
    Parse V-cycle time from output file (one per run block).
    Supports legacy, old, and current formats:
      - "Total V-cycle time (Kernel Launch + V-cycle time): X us"       (legacy)
      - "1st V-cycle time (measured)     :    X us"                      (old — misleading)
      - "1st V-cycle time (sum of per-level timers, directly measured)"  (old — misleading)
      - "Avg V-cycle time (no conv)      :    X us"                      (current)
    """
    results = []
    blocks = split_into_run_blocks(text)
    # Try legacy format first (has the extra "upper bound" line)
    legacy_pattern = re.compile(
        r'Total Upper bound V-cycle time \(sum of operations\):\s*([\d.]+)\s*us.*?\n'
        r'Total V-cycle time \(Kernel Launch \+ V-cycle time\):\s*([\d.]+)\s*us',
        re.DOTALL
    )
    current_pattern = re.compile(
        r'(?:1st V-cycle time \(measured\)'
        r'|1st V-cycle time \(sum of per-level timers, directly measured\)'
        r'|Avg V-cycle time \(no conv\))'
        r'\s*:\s*([\d.]+)\s*us'
    )
    for grid_size, block_text in blocks:
        match = legacy_pattern.search(block_text)
        if match:
            results.append({'grid_size': grid_size, 'vcycle_time_us': float(match.group(2))})
            continue
        match = current_pattern.search(block_text)
        if match:
            results.append({'grid_size': grid_size, 'vcycle_time_us': float(match.group(1))})
    return results


def parse_all_data(text: str) -> List[Dict]:
    """
    Parse all data from the output file and combine results.
    Returns sorted list of combined data dictionaries.
    """
    # Step 1: Parse all data sources
    spmv_data = parse_spmv_totals(text)
    vcycle_data = parse_vcycle_times(text)
    config_data = parse_configuration_summary(text)
    memory_data = parse_memory_usage(text)
    
    # Step 2: Combine data by grid size
    combined = {}
    
    # Add SPMV data
    for item in spmv_data:
        grid = item['grid_size']
        combined[grid] = {
            'pe_tiles': grid,
            'grid_size': grid,
            'comm_time_us': item['comm_time_us'],
            'compute_time_us': item['compute_time_us'],
            'total_spmv_us': item.get('total_spmv_us'),
        }
    
    # Add V-cycle data
    for item in vcycle_data:
        grid = item['grid_size']
        if grid in combined:
            combined[grid]['vcycle_time_us'] = item['vcycle_time_us']
        else:
            combined[grid] = {
                'pe_tiles': grid,
                'grid_size': grid,
                'vcycle_time_us': item['vcycle_time_us']
            }
    
    # Add configuration data
    for item in config_data:
        grid = item['grid_size']
        if grid in combined:
            combined[grid].update({
                'total_grid_size': item.get('total_grid_size', 0),
                'device_iterations': item.get('device_iterations', None),
                'vcycle_avg_time_us': item.get('vcycle_avg_time_us', None),
                'device_rho_inf': item.get('device_rho_inf', None),
                'converged': item.get('converged', None),
                'compile_time_s': item.get('compile_time_s', None),
                'run_time_s': item.get('run_time_s', None)
            })
        else:
            combined[grid] = {
                'pe_tiles': item.get('pe_tiles', grid),
                'grid_size': grid,
                'total_grid_size': item.get('total_grid_size', 0),
                'device_iterations': item.get('device_iterations', None),
                'vcycle_avg_time_us': item.get('vcycle_avg_time_us', None),
                'device_rho_inf': item.get('device_rho_inf', None),
                'converged': item.get('converged', None),
                'compile_time_s': item.get('compile_time_s', None),
                'run_time_s': item.get('run_time_s', None)
            }

    # Add memory usage data (code/data size per PE from ELF)
    for item in memory_data:
        grid = item['grid_size']
        if grid in combined:
            combined[grid]['code_bytes'] = item.get('code_bytes')
            combined[grid]['data_bytes'] = item.get('data_bytes')
        else:
            combined[grid] = {
                'pe_tiles': grid,
                'grid_size': grid,
                'code_bytes': item.get('code_bytes'),
                'data_bytes': item.get('data_bytes'),
            }
    
    # Step 3: Convert to sorted list (by total_grid_size)
    sorted_data = sorted(combined.values(), key=lambda x: x.get('total_grid_size', 0))
    
    return sorted_data


def plot_comm_vs_compute(data: List[Dict], output_file: str = 'comm_vs_compute_time.png'):
    """
    Create plot showing Communication/Compute ratio to demonstrate relative scaling.
    Shows how communication overhead scales relative to compute time.
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not available. Cannot create plots.")
        return None
    
    grid_sizes = [d['grid_size'] for d in data]
    comm_times = [d['comm_time_us'] for d in data]  # Keep in microseconds
    compute_times = [d['compute_time_us'] for d in data]  # Keep in microseconds
    
    # Extract grid dimension (assuming cubic grids like 4x4x4 -> 4)
    grid_dims = []
    for grid in grid_sizes:
        dim = int(grid.split('x')[0])
        grid_dims.append(dim)
    
    # Calculate ratios (comm/compute)
    ratios = []
    for comm, comp in zip(comm_times, compute_times):
        if comp > 0:
            ratios.append(comm / comp)
        else:
            ratios.append(0)
    
    x_labels = [f"{dim}³" for dim in grid_dims]
    x_pos = np.arange(len(grid_sizes))
    
    # Create figure with two subplots: ratio plot and normalized stacked bars (paper-sized)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Top plot: Communication/Compute Ratio
    bars = ax1.bar(x_pos, ratios, color='#D62728', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Grid Size (Subdomain Dimension)', fontsize=18)
    ax1.set_ylabel('Communication / Compute Ratio', fontsize=18)
    ax1.set_title('Laplacian Communication Overhead Relative to Compute (Comm/Compute Ratio)', 
                  fontsize=20, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=16)
    # ax1.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    ax1.grid(False)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='1:1 ratio')
    
    # Add value labels on top of bars
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.1f}x', ha='center', va='bottom', fontsize=15, fontweight='bold')
    
    ax1.legend(fontsize=15)
    ax1.tick_params(axis='both', labelsize=15)
    
    # Bottom plot: Normalized stacked bars (each bar normalized to 100%)
    # This shows the relative proportion without absolute time differences
    comm_percentages = []
    comp_percentages = []
    for comm, comp in zip(comm_times, compute_times):
        total = comm + comp
        if total > 0:
            comm_percentages.append((comm / total) * 100)
            comp_percentages.append((comp / total) * 100)
        else:
            comm_percentages.append(0)
            comp_percentages.append(0)
    
    bars1 = ax2.bar(x_pos, comm_percentages, label='Communication (%)', color='#FF7F0E', alpha=0.8)
    bars2 = ax2.bar(x_pos, comp_percentages, bottom=comm_percentages, label='Compute (%)', 
                    color='#1F77B4', alpha=0.8)
    
    # Add percentage labels
    for i, (comm_pct, comp_pct) in enumerate(zip(comm_percentages, comp_percentages)):
        # Label communication percentage in the middle of its segment
        if comm_pct > 5:  # Only label if segment is large enough
            ax2.text(i, comm_pct / 2, f'{comm_pct:.0f}%', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
        # Label compute percentage at the top of the bar (100% line) - like the sample image
        if comp_pct > 0:
            ax2.text(i, 100, f'{int(comp_pct)}%', ha='center', va='bottom',
                    fontsize=15, fontweight='bold', color='black')
    
    ax2.set_xlabel('Grid Size (Subdomain Dimension)', fontsize=18)
    ax2.set_ylabel('Relative Proportion (%)', fontsize=18)
    ax2.set_title('Laplacian Normalized Time Distribution', 
                  fontsize=20, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=16)
    ax2.set_ylim(0, 130)  # Add extra space at top for labels
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=15)
    ax2.tick_params(axis='both', labelsize=15)
    # ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=PAPER_DPI, bbox_inches='tight')
    print(f"Saved: {output_file}")
    return fig


def plot_spmv_internal(spmv_per_level_data: List[Dict], output_file: str = 'spmv_internal.png'):
    """
    Plot 7-pt Stencil per level for 3 problem sizes (128³, 256³, 512³) in one figure.
    One subplot per problem size; each subplot: level vs time (µs), Communication (solid),
    Compute (solid), Total SpMV Time (dotted).
    """
    if not HAS_MATPLOTLIB or not spmv_per_level_data:
        return None
    want_order = ['128x128x128', '256x256x256', '512x512x512']
    by_size = {d['grid_size']: d for d in spmv_per_level_data}
    spmv_per_level_data = [by_size[g] for g in want_order if g in by_size]
    if not spmv_per_level_data:
        return None
    n_plots = len(spmv_per_level_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5))
    axes = np.atleast_1d(axes)
    for idx, block in enumerate(spmv_per_level_data):
        ax = axes.flat[idx]
        grid_size = block['grid_size']
        levels_data = sorted(block['levels'], key=lambda x: x['level'])
        levels = [r['level'] for r in levels_data]
        ax.plot(levels, [r['comm_time_us'] for r in levels_data], 'o-', label='Communication Time', linewidth=2.5, markersize=8)
        ax.plot(levels, [r['compute_time_us'] for r in levels_data], 's-', label='Compute Time', linewidth=2.5, markersize=8)
        ax.plot(levels, [r['total_spmv_us'] for r in levels_data], '^--', label='Total Time', linewidth=3, markersize=9)
        dim = grid_size.split('x')[0]
        ax.set_title(f'Grid {dim}³', fontsize=18, fontweight='bold')
        ax.set_xlabel('Level', fontsize=16)
        ax.set_ylabel('Time (µs)', fontsize=16)
        ax.set_xticks(levels)
        ax.legend(loc='best', framealpha=0.9, fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
    fig.suptitle('7-pt Stencil: Communication vs Compute vs Total Time (config: 6/6/6)', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=PAPER_DPI, bbox_inches='tight')
    print(f"Saved: {output_file}")
    return fig


def plot_interpolation_internal(interp_per_level_data: List[Dict], output_file: str = 'interpolation_internal.png'):
    """
    Plot Interpolation Micro-Benchmark table per level for 3 problem sizes (128³, 256³, 512³).
    One subplot per problem size. Each subplot shows level vs time (µs) for T1, T2.1,
    T2.2, T3 (solid) and interpolation_total (dotted; data from bcast_total).
    Y-axis in log scale. All 3 in a single PNG.
    """
    if not HAS_MATPLOTLIB or not interp_per_level_data:
        return None
    want_order = ['128x128x128', '256x256x256', '512x512x512']
    by_size = {d['grid_size']: d for d in interp_per_level_data}
    interp_per_level_data = [by_size[g] for g in want_order if g in by_size]
    if not interp_per_level_data:
        return None
    n_plots = len(interp_per_level_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5))
    axes = np.atleast_1d(axes)  # shape (3,) or (1,) so axes.flat[i] is the i-th Axes
    for idx, block in enumerate(interp_per_level_data):
        ax = axes.flat[idx]
        grid_size = block['grid_size']
        levels_data = sorted(block['levels'], key=lambda x: x['level'])
        # Only use expand_z_T1, reset_routes_T21, send_data_T22, interp_add_T3, bcast_T2 (labeled as interpolation_total)
        def has_positive(row):
            return (
                row['expand_z_T1'] or
                row['reset_routes_T21'] or
                row['send_data_T22'] or
                row['interp_add_T3'] or
                row['bcast_T2']
            ) > 0
        plot_data = [r for r in levels_data if has_positive(r)]
        if not plot_data:
            plot_data = levels_data
        levels = [r['level'] for r in plot_data]
        eps = 1e-3
        def _y(v):
            return v if v > 0 else eps
        ax.plot(levels, [_y(r['expand_z_T1']) for r in plot_data], 'o-', label='expand_z (T1)', linewidth=2.5, markersize=8)
        ax.plot(levels, [_y(r['reset_routes_T21']) for r in plot_data], '^-', label='reset_routes (T2.1)', linewidth=2.5, markersize=8)
        ax.plot(levels, [_y(r['send_data_T22']) for r in plot_data], 'd-', label='send_data (T2.2)', linewidth=2.5, markersize=8)
        ax.plot(levels, [_y(r['interp_add_T3']) for r in plot_data], 'p-', label='interp_add (T3)', linewidth=2.5, markersize=8)
        ax.plot(levels, [_y(r['bcast_T2']) for r in plot_data], '*-', label='interpolation_total', linewidth=3, markersize=9, linestyle='--')
        dim = grid_size.split('x')[0]
        ax.set_title(f'Grid {dim}³', fontsize=18, fontweight='bold')
        ax.set_xlabel('Level', fontsize=16)
        ax.set_ylabel('Time (µs, log)', fontsize=16)
        ax.set_yscale('log')
        ax.set_xticks(levels)
        ax.legend(loc='best', framealpha=0.9, fontsize=13)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.grid(True, which='minor', linestyle=':', alpha=0.15)
    fig.suptitle('Interpolation Micro-Benchmark per Level(config: 6/6/6)', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=PAPER_DPI, bbox_inches='tight')
    print(f"Saved: {output_file}")
    return fig


def parse_per_operation_timing(text: str) -> List[Dict]:
    """
    Parse "Time per operation and level" tables from the output file.
    Uses run blocks so each problem size appears once.
    Supports both legacy (5 columns) and current (7 columns) formats.
    """
    results = []
    blocks = split_into_run_blocks(text)
    # Match the first 5 numeric columns (level, smooth, residual, restriction, interpolation).
    # Current format has 7 columns (adds setup, convergence) but we only need the first 5.
    row_pattern = re.compile(
        r'\|\s*(\d+)\s*\|'
        r'\s*([\d.]+)us\([^)]+\)\s*\|'
        r'\s*([\d.]+)us\([^)]+\)\s*\|'
        r'\s*([\d.]+)us\([^)]+\)\s*\|'
        r'\s*([\d.]+)us\([^)]+\)\s*\|'
    )
    # Support both headers: legacy "Time per operation and level (us[cycles]):"
    # and current "Time per operation and level (us[cycles]), first V-cycle:"
    legacy_header = 'Time per operation and level (us[cycles]):'
    current_header = 'Time per operation and level (us[cycles]), first V-cycle:'
    for grid_size, block_text in blocks:
        if current_header in block_text:
            section = block_text.split(current_header, 1)[-1]
        elif legacy_header in block_text:
            section = block_text.split(legacy_header, 1)[-1]
        else:
            continue
        # Only parse the first table; stop at 7-pt Stencil (any variant)
        for stencil_hdr in (SPMV_HEADER_NEW, SPMV_HEADER_MID, SPMV_HEADER_OLD, '7-pt Stencil'):
            if stencil_hdr in section:
                section = section.split(stencil_hdr, 1)[0]
                break
        rows = []
        for line in section.split('\n'):
            if re.search(r'\|\s*total\s*\|', line, re.IGNORECASE):
                continue
            match = row_pattern.search(line)
            if match:
                level = int(match.group(1))
                smooth = float(match.group(2))
                residual = float(match.group(3))
                restriction = float(match.group(4))
                interpolation = float(match.group(5))
                rows.append({
                    'level': level,
                    'smooth': smooth,
                    'residual': residual,
                    'restriction': restriction,
                    'interpolation': interpolation,
                    'setup_init': 0.0,
                })
        if rows:
            results.append({'grid_size': grid_size, 'levels': rows})
    return results


def plot_per_operation_timing(timing_data: Dict, output_file: str = None):
    """
    Plot per-operation timing by level for a single grid size.
    Simple and clean visualization similar to pp.py style.
    
    Args:
        timing_data: Dictionary with 'grid_size' and 'levels' (list of level data)
        output_file: Optional output filename. If None, auto-generates from grid_size.
    
    Returns:
        Figure object or None
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not available. Cannot create plots.")
        return None
    
    if not timing_data or 'levels' not in timing_data:
        print("Warning: No timing data provided")
        return None
    
    grid_size = timing_data['grid_size']
    levels_data = timing_data['levels']
    
    if not levels_data:
        print(f"Warning: No level data for grid {grid_size}")
        return None
    
    # Sort levels by level number
    levels_data = sorted(levels_data, key=lambda x: x['level'])
    
    # Extract data - simple and clean
    levels = [d['level'] for d in levels_data]
    smooth = [d['smooth'] for d in levels_data]
    residual = [d['residual'] for d in levels_data]
    restriction = [d['restriction'] for d in levels_data]
    interpolation = [d['interpolation'] for d in levels_data]
    
    # Find last non-zero index for residual, restriction, and interpolation
    # (they have 0 at the last level, we don't want to plot those)
    def find_last_nonzero_index(values):
        """Find the last index with non-zero value."""
        for i in range(len(values) - 1, -1, -1):
            if values[i] > 0:
                return i
        return -1
    
    residual_end = find_last_nonzero_index(residual)
    restriction_end = find_last_nonzero_index(restriction)
    interpolation_end = find_last_nonzero_index(interpolation)
    
    # Create figure (paper-sized for single-column or full width)
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot each operator - truncate residual, restriction, interpolation at last non-zero
    # Add markers to show exact data points
    ax.plot(levels, smooth, 'o-', label='smooth', linewidth=3, markersize=9)
    if residual_end >= 0:
        ax.plot(levels[:residual_end+1], residual[:residual_end+1], '^-', label='residual', linewidth=3, markersize=9)
    if restriction_end >= 0:
        ax.plot(levels[:restriction_end+1], restriction[:restriction_end+1], 'v-', label='restriction', linewidth=3, markersize=9)
    if interpolation_end >= 0:
        ax.plot(levels[:interpolation_end+1], interpolation[:interpolation_end+1], 'd-', label='interpolation', linewidth=3, markersize=9)
    
    # Simple labels (larger for paper readability)
    ax.set_xlabel('Level', fontsize=16)
    ax.set_ylabel('Time (µs, log scale)', fontsize=16)
    ax.set_title(f'Timing Breakdown per Level - Grid: {grid_size}', fontsize=20, fontweight='bold')
    ax.set_yscale('log')  # Log scale to handle wide range of values
    ax.set_xticks(levels)  # Set x-axis ticks to integer levels only
    ax.legend(loc='best', framealpha=0.9, fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, which='major', linestyle='-', alpha=0.15)
    ax.grid(False, which='minor')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=PAPER_DPI, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig


def plot_all_per_operation_timing(all_timing_data: List[Dict], output_dir: str = ''):
    """
    Plot per-operation timing for all grid sizes.
    Creates separate plots for each grid size.
    
    Args:
        all_timing_data: List of timing data dictionaries (one per grid size)
        output_dir: Directory to save plots (default: current directory)
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not available. Cannot create plots.")
        return
    
    if not all_timing_data:
        print("Warning: No timing data provided")
        return
    
    prefix = f"{output_dir}/" if output_dir else ""
    
    for timing_data in all_timing_data:
        grid_size = timing_data['grid_size']
        output_file = f"{prefix}per_operation_timing_{grid_size}.png"
        plot_per_operation_timing(timing_data, output_file)
    
    print(f"\n✓ Generated {len(all_timing_data)} per-operation timing plots")


def print_per_operation_timing_tables(all_timing_data: List[Dict]):
    """
    Print per-operation timing tables for all grid sizes.
    Shows time per operation and level in a readable format.
    """
    for timing_data in all_timing_data:
        grid_size = timing_data['grid_size']
        levels_data = timing_data['levels']
        
        if not levels_data:
            continue
        
        # Sort levels by level number
        levels_data = sorted(levels_data, key=lambda x: x['level'])
        
        # Extract grid dimension for subdomain size labels
        grid_dim = int(grid_size.split('x')[0])
        
        print(f"\n{'='*100}")
        print(f"Per-Operation Timing: Grid {grid_size}")
        print(f"{'='*100}")
        
        # Header
        print(f"{'Level':<8} {'Subdomain':<12} {'smooth (us)':<15} {'residual (us)':<15} "
              f"{'restriction (us)':<18} {'interpolation (us)':<20} {'total (us)':<18}")
        print("-"*100)
        
        # Print each level
        for level_data in levels_data:
            level = level_data['level']
            subdomain_size = grid_dim // (2 ** level)
            smooth = level_data['smooth']
            residual = level_data['residual']
            restriction = level_data['restriction']
            interpolation = level_data['interpolation']
            setup_init = level_data.get('setup_init', 0.0)  # May not be present in new format
            # Calculate total if setup_init is 0 (new format)
            total = smooth + residual + restriction + interpolation + setup_init if setup_init > 0 else smooth + residual + restriction + interpolation
            
            print(f"{level:<8} {subdomain_size}³{'':<8} {smooth:<15.2f} {residual:<15.2f} "
                  f"{restriction:<18.2f} {interpolation:<20.2f} {total:<18.2f}")
        
        print(f"{'='*100}\n")


def print_spmv_per_level_tables(spmv_per_level_data: List[Dict]):
    """
    Print 7-pt Stencil per-level tables for problem sizes 128³, 256³, 512³ (same data as spmv_internal.png).
    """
    want_order = ['128x128x128', '256x256x256', '512x512x512']
    by_size = {d['grid_size']: d for d in spmv_per_level_data}
    for grid_size in want_order:
        if grid_size not in by_size:
            continue
        block = by_size[grid_size]
        levels_data = sorted(block['levels'], key=lambda x: x['level'])
        if not levels_data:
            continue
        grid_dim = int(grid_size.split('x')[0])
        print(f"\n{'='*100}")
        print(f"7-pt Stencil (SpMV) per Level: Grid {grid_size}")
        print(f"{'='*100}")
        print(f"{'Level':<8} {'Subdomain':<12} {'Total SpMV (us)':<18} {'Communication (us)':<20} {'Compute (us)':<15}")
        print("-"*100)
        for r in levels_data:
            level = r['level']
            subdomain = grid_dim // (2 ** level)
            print(f"{level:<8} {subdomain}³{'':<8} {r['total_spmv_us']:<18.2f} {r['comm_time_us']:<20.2f} {r['compute_time_us']:<15.2f}")
        print(f"{'='*100}\n")


def print_interpolation_per_level_tables(interp_per_level_data: List[Dict]):
    """
    Print Interpolation Micro-Benchmark per-level tables for 128³, 256³, 512³ (same data as interpolation_internal.png).
    """
    want_order = ['128x128x128', '256x256x256', '512x512x512']
    by_size = {d['grid_size']: d for d in interp_per_level_data}
    for grid_size in want_order:
        if grid_size not in by_size:
            continue
        block = by_size[grid_size]
        levels_data = sorted(block['levels'], key=lambda x: x['level'])
        if not levels_data:
            continue
        grid_dim = int(grid_size.split('x')[0])
        print(f"\n{'='*120}")
        print(f"Interpolation Micro-Benchmark per Level: Grid {grid_size}")
        print(f"{'='*120}")
        print(f"{'Level':<6} {'Sub':<6} {'expand_z(T1)':<14} {'interpolation_total_time':<22} {'reset(T2.1)':<12} {'send(T2.2)':<12} {'interp(T3)':<12}")
        print("-"*100)
        for r in levels_data:
            level = r['level']
            subdomain = grid_dim // (2 ** level)
            print(f"{level:<6} {subdomain}³{'':<3} {r['expand_z_T1']:<14.2f} {r['bcast_T2']:<22.2f} "
                  f"{r['reset_routes_T21']:<12.2f} {r['send_data_T22']:<12.2f} {r['interp_add_T3']:<12.2f}")
        print(f"{'='*120}\n")


def print_configuration_table(config_data: List[Dict]):
    """
    Print configuration parameters table with problem sizes as columns and config parameters as rows.
    """
    if not config_data:
        return
    
    # Sort by grid size
    sorted_config = sorted(config_data, key=lambda x: x.get('total_grid_size', 0))
    
    # Define all configuration parameters to display
    config_params = [
        ('HeightxWidthxZDim', 'grid_size', None),
        ('Levels', 'levels', None),
        ('Max iterations', 'max_iterations', None),
        ('Tolerance (abs)', 'tolerance_abs', None),
        ('Tolerance (rel)', 'tolerance_rel', None),
        ('Pre/Post/Bottom iter', 'pre_post_bottom_iter', None),
        ('Datatype', 'datatype', None),
        ('Jacobi omega', 'jacobi_omega', lambda x: f"{x:.6f}" if x is not None else 'N/A'),
        ('Stencil alpha/beta', 'stencil_alpha_beta', None),
        ('Block size', 'block_size', None),
    ]
    
    # Get grid sizes (columns)
    grid_sizes = [d.get('grid_size', 'N/A') for d in sorted_config]
    
    print("\n" + "="*150)
    print("Configuration Parameters by Problem Size")
    print("="*150)
    
    # Print header
    header = f"{'Parameter':<30}"
    for grid_size in grid_sizes:
        header += f" {grid_size:<15}"
    print(header)
    print("-"*150)
    
    # Print each parameter row
    for param_name, param_key, formatter in config_params:
        row = f"{param_name:<30}"
        for config in sorted_config:
            value = config.get(param_key, None)
            if formatter:
                value_str = formatter(value)
            elif value is None:
                value_str = 'N/A'
            else:
                value_str = str(value)
            row += f" {value_str:<15}"
        print(row)
    
    print("="*150 + "\n")


def print_summary_table(data: List[Dict]):
    """
    Print a summary table of the parsed data with updated column names and units.
    """
    print("\n" + "="*180)
    print("Performance Data Summary")
    print("="*180)
    header = (f"{'PE tiles':<15} {'Total Grid size':<18} {'Comm Time(spmv) (us)':<25} "
              f"{'Compute Time(spmv) (us)':<28} {'V-cycle Time (us)':<20} "
              f"{'Iterations':<12} {'1-V cycle time(Average) (us)':<30} {'|rho|_inf':<15} {'Converged':<10} "
              f"{'Compile Time (s)':<18} {'Run Time (s)':<15} {'Code (bytes)':<14} {'Data (bytes)':<14}")
    print(header)
    print("-"*180)
    
    for d in data:
        pe_tiles = d.get('pe_tiles', 'N/A')
        total_grid = d.get('total_grid_size', 0)
        comm = d.get('comm_time_us', 0)
        comp = d.get('compute_time_us', 0)
        vcycle = d.get('vcycle_time_us', 0)
        iterations = d.get('device_iterations', None)
        vcycle_avg = d.get('vcycle_avg_time_us', None)
        rho_inf = d.get('device_rho_inf', None)
        converged = d.get('converged', None)
        compile_time = d.get('compile_time_s', None)
        run_time = d.get('run_time_s', None)
        code_bytes = d.get('code_bytes', None)
        data_bytes = d.get('data_bytes', None)
        
        iter_str = str(iterations) if iterations is not None else 'N/A'
        vcycle_avg_str = f"{vcycle_avg:.3f}" if vcycle_avg is not None else 'N/A'
        rho_str = f"{rho_inf:.3e}" if rho_inf is not None else 'N/A'
        conv_str = 'Yes' if converged else 'No' if converged is False else 'N/A'
        compile_str = f"{compile_time:.3f}" if compile_time is not None else 'N/A'
        run_str = f"{run_time:.3f}" if run_time is not None else 'N/A'
        code_str = f"{code_bytes:,}" if code_bytes is not None else 'N/A'
        data_str = f"{data_bytes:,}" if data_bytes is not None else 'N/A'
        
        print(f"{pe_tiles:<15} {total_grid:<18,} {comm:<25.2f} {comp:<28.2f} "
              f"{vcycle:<20.2f} {iter_str:<12} {vcycle_avg_str:<30} {rho_str:<15} {conv_str:<10} "
              f"{compile_str:<18} {run_str:<15} {code_str:<14} {data_str:<14}")
    
    print("="*180 + "\n")


def main():
    """
    Main function: Read file, parse data, print summary, generate plots.
    """
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <gmg_output_file.txt>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Step 1: Read file
    print(f"Reading: {input_file}")
    with open(input_file, 'r') as f:
        text = f.read()
    
    # Step 2: Parse all data
    print("Parsing data...")
    data = parse_all_data(text)
    
    if not data:
        print("Error: No data found in file")
        sys.exit(1)
    
    # Step 2.5: Parse and print configuration table
    print("Parsing configuration data...")
    config_data = parse_configuration_summary(text)
    if config_data:
        print_configuration_table(config_data)
    
    # Step 3: Print summary table
    print_summary_table(data)
    
    # Step 4: Parse per-operation timing data
    print("Parsing per-operation timing data...")
    all_timing_data = parse_per_operation_timing(text)
    
    # Step 5: Print per-operation timing tables
    if all_timing_data:
        print_per_operation_timing_tables(all_timing_data)
    
    # Step 6: Parse per-level data for 3-subplot figures (128³, 256³, 512³)
    interp_per_level = parse_interpolation_per_level(text)
    spmv_per_level = parse_spmv_per_level(text)

    # Step 6b: Print SPMV and Interpolation tables (same data we plot)
    if spmv_per_level:
        print_spmv_per_level_tables(spmv_per_level)
    if interp_per_level:
        print_interpolation_per_level_tables(interp_per_level)

    # Step 7: Generate plots if matplotlib is available
    # Per-operation timing: only plot 128³, 256³, 512³ (same as spmv/interpolation internal)
    PLOT_GRID_SIZES = ['512x512x512']   # add more if needed.
    if HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        # plot_comm_vs_compute(data, 'comm_vs_compute_time.png')
        if spmv_per_level:
            plot_spmv_internal(spmv_per_level, 'spmv_internal.png')
        if interp_per_level:
            plot_interpolation_internal(interp_per_level, 'interpolation_internal.png')
        timing_to_plot = [d for d in (all_timing_data or []) if d['grid_size'] in PLOT_GRID_SIZES]
        if timing_to_plot:
            plot_all_per_operation_timing(timing_to_plot)
        print("\n✓ All plots generated successfully!")
    else:
        print("\n⚠ Plots not generated. Install matplotlib and numpy to create visualizations:")
        print("   pip install matplotlib numpy")


if __name__ == "__main__":
    main()
