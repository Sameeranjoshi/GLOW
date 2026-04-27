#!/usr/bin/env python3
"""
V-cycle vs W-cycle comparison for GMG solver on WSE-3.

Reads response.txt files from V-cycle (build/) and W-cycle (../w_cycle/)
for the same problem size and produces:
  1. Per-level timing comparison table (smooth time per iteration)
  2. High-level summary table (iterations, cycle time, TTS, convergence rate)

Usage:
  python v_vs_w_cycle.py                           # defaults: 256^3, 6/6/6
  python v_vs_w_cycle.py --size 128                # different size
  python v_vs_w_cycle.py --v-response <path> --w-response <path>  # explicit paths
"""

import os
import re
import sys
import math
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..')


def find_response_paths(size, levels, pre, post, bottom, max_ite=100):
    """Find V-cycle and W-cycle response.txt paths for given config."""
    v_dir = f"out_dir_S{size}x_L{levels}_M{max_ite}_P{pre}_P{post}_B{bottom}"
    w_dir = f"w_out_dir_S{size}x_L{levels}_M{max_ite}_P{pre}_P{post}_B{bottom}"
    v_path = os.path.join(BASE_DIR, 'build', v_dir, 'response.txt')
    w_path = os.path.join(BASE_DIR, '..', 'w_cycle', w_dir, 'response.txt')
    return v_path, w_path


def parse_response(path):
    """Parse a response.txt and extract timing/convergence data."""
    with open(path) as f:
        content = f.read()

    data = {}

    # Iterations
    m = re.search(r'Total iterations performed:\s*(\d+)', content)
    data['iterations'] = int(m.group(1)) if m else None

    # Rho values per iteration
    rho_vals = []
    for m in re.finditer(r'^\s*(\d+)\s+([\d.eE+-]+)\s*$', content, re.MULTILINE):
        rho_vals.append(float(m.group(2)))
    data['rho_history'] = rho_vals

    # Final rho - try "Device final |rho|_inf : X" first, then "|b-A*x|_inf = X"
    m = re.search(r'Device final \|rho\|_inf\s*:\s*([\d.eE+-]+)', content)
    if not m:
        m = re.search(r'\|b-A\*x\|_inf\s*=\s*([\d.eE+-]+)', content)
    data['final_rho'] = float(m.group(1)) if m else None

    # Converged
    m = re.search(r'Converged:\s*(Yes|No)', content)
    data['converged'] = m.group(1) if m else '?'

    # Avg cycle time — match any "Avg V-cycle time" or "Average V-cycle time" variant
    m = re.search(r'Avg (?:V|W)-cycle time.*?:\s*([\d.]+)\s*us', content)
    if not m:
        m = re.search(r'Average (?:V|W)-cycle time.*?:\s*([\d.]+)\s*us', content)
    data['avg_cycle_us'] = float(m.group(1)) if m else None

    # Total solver wall time
    m = re.search(r'Total solver wall time.*?:\s*([\d.]+)\s*us', content)
    data['total_wall_us'] = float(m.group(1)) if m else None

    # Convergence diagnostic time
    m = re.search(r'Convergence diagnostic time.*?:\s*([\d.]+)\s*us', content)
    data['conv_diag_us'] = float(m.group(1)) if m else None

    # Per-level timing from first table: "Time per operation and level"
    # Extract only the first table section (before "7-pt Stencil" second table)
    level_data = {}
    first_table_match = re.search(
        r'Time per operation and level.*?\n(.*?)(?=\n\n7-pt Stencil|\n\n\n)',
        content, re.DOTALL
    )
    if first_table_match:
        table_text = first_table_match.group(1)
        row_pattern = re.compile(
            r'\|\s*(\d+)\s*\|'            # level
            r'\s*([\d.]+)us\([^)]*\)\s*\|' # smooth
            r'\s*([\d.]+)us\([^)]*\)\s*\|' # residual
            r'\s*([\d.]+)us\([^)]*\)\s*\|' # restriction
            r'\s*([\d.]+)us\([^)]*\)\s*\|' # interpolation
            r'\s*([\d.]+)us\([^)]*\)\s*\|' # setup
            r'\s*([\d.]+)us\([^)]*\)\s*\|' # convergence
            r'\s*([\d.]+)us\([^)]*\)\s*\|' # total
        )
        for m in row_pattern.finditer(table_text):
            lvl = int(m.group(1))
            level_data[lvl] = {
                'smooth': float(m.group(2)),
                'residual': float(m.group(3)),
                'restriction': float(m.group(4)),
                'interpolation': float(m.group(5)),
                'setup': float(m.group(6)),
                'convergence': float(m.group(7)),
                'total': float(m.group(8)),
            }
    data['levels'] = level_data

    # Size/levels from config
    m = re.search(r'Parameters:\s*size=(\d+),\s*levels=(\d+)', content)
    if m:
        data['size'] = int(m.group(1))
        data['num_levels'] = int(m.group(2))

    # Pre/post/bottom
    m = re.search(r'Pre/Post/Bottom iter\s*:\s*(\d+)/(\d+)/(\d+)', content)
    if m:
        data['pre'] = int(m.group(1))
        data['post'] = int(m.group(2))
        data['bottom'] = int(m.group(3))

    return data


def convergence_rate(rho_history):
    """Compute average convergence rate in decades per iteration."""
    if len(rho_history) < 2 or rho_history[0] <= 0 or rho_history[-1] <= 0:
        return 0.0
    decades = math.log10(rho_history[0] / rho_history[-1])
    intervals = len(rho_history) - 1
    return decades / intervals


def print_comparison(v_data, w_data):
    """Print paper-ready V-cycle vs W-cycle comparison table."""
    v_rate = convergence_rate(v_data['rho_history'])
    w_rate = convergence_rate(w_data['rho_history'])
    v_tts = v_data['total_wall_us']
    w_tts = w_data['total_wall_us']
    v_it, w_it = v_data['iterations'], w_data['iterations']
    v_avg, w_avg = v_data['avg_cycle_us'], w_data['avg_cycle_us']

    size = v_data.get('size', '?')
    levels = v_data.get('num_levels', '?')
    pre = v_data.get('pre', '?')
    post = v_data.get('post', '?')
    bottom = v_data.get('bottom', '?')

    w = 75
    print()
    print('=' * w)
    print(f'  V-cycle vs W-cycle  |  {size}^3, {levels} levels, {pre}/{post}/{bottom}')
    print('=' * w)
    print(f'  {"Metric":<38} {"V-cycle":>14} {"W-cycle":>14} {"Ratio":>7}')
    print('-' * w)

    def row(label, v_val, w_val, fmt=',.1f', ratio_fmt='.1f', unit=''):
        v_str = f'{v_val:{fmt}}{unit}'
        w_str = f'{w_val:{fmt}}{unit}'
        r = w_val / v_val if v_val else 0
        r_str = f'{r:{ratio_fmt}}x'
        print(f'  {label:<38} {v_str:>14} {w_str:>14} {r_str:>7}')

    row('Iterations to converge', v_it, w_it, fmt=',d', ratio_fmt='.1f')
    row('Avg cycle time (us)', v_avg, w_avg, fmt=',.1f')
    row('Time to solution (us)', v_tts, w_tts, fmt=',.1f')
    row('Convergence rate (decades/iter)', v_rate, w_rate, fmt='.2f')

    # Final rho — no ratio
    print(f'  {"Final |rho|_inf":<38} {v_data["final_rho"]:>14.2e} {w_data["final_rho"]:>14.2e}')

    print('=' * w)


def main():
    parser = argparse.ArgumentParser(description='V-cycle vs W-cycle comparison')
    parser.add_argument('--size', type=int, default=256, help='Grid size (default: 256)')
    parser.add_argument('--levels', type=int, default=None, help='Number of levels (auto from size)')
    parser.add_argument('--pre', type=int, default=6)
    parser.add_argument('--post', type=int, default=6)
    parser.add_argument('--bottom', type=int, default=6)
    parser.add_argument('--max-ite', type=int, default=100)
    parser.add_argument('--v-response', type=str, default=None, help='Path to V-cycle response.txt')
    parser.add_argument('--w-response', type=str, default=None, help='Path to W-cycle response.txt')
    args = parser.parse_args()

    # Auto-compute levels: size=256 -> levels=8, size=4 -> levels=2
    if args.levels is None:
        args.levels = int(math.log2(args.size))

    if args.v_response and args.w_response:
        v_path, w_path = args.v_response, args.w_response
    else:
        v_path, w_path = find_response_paths(
            args.size, args.levels, args.pre, args.post, args.bottom, args.max_ite
        )

    # Validate
    for label, path in [('V-cycle', v_path), ('W-cycle', w_path)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} response not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"V-cycle: {os.path.relpath(v_path, SCRIPT_DIR)}")
    print(f"W-cycle: {os.path.relpath(w_path, SCRIPT_DIR)}")

    v = parse_response(v_path)
    w = parse_response(w_path)

    print(f"\nProblem: {v.get('size', args.size)}^3, {v.get('num_levels', args.levels)} levels, "
          f"{v.get('pre', args.pre)}/{v.get('post', args.post)}/{v.get('bottom', args.bottom)}")

    print_comparison(v, w)


if __name__ == '__main__':
    main()
