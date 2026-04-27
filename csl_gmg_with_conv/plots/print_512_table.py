#!/usr/bin/env python3
"""Parse GPU and WSE number files and print 512^3 comparison table (WITH convergence)."""

import re
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_FILE = os.path.join(SCRIPT_DIR, "gpu_numbers.txt")
WSE_FILE = os.path.join(SCRIPT_DIR, "wse_numbers.txt")
TARGET_GRID = 512


def parse_main_table(lines):
    """Parse a standard multi-config table block.
    Returns (platform, configs_list, {config: {grid_size: {TTS, Iter, 1-cycle}}})
    """
    platform = None
    configs = []
    data = {}

    for line in lines:
        # Detect config header — e.g. "GH200(6/6/100)" or "WSE3(4/4/6)"
        cfg_matches = re.findall(r'(\w+)\(([\d/]+)\)', line)
        if cfg_matches and 'TTS' not in line and 'Grid' not in line:
            platform = cfg_matches[0][0]
            configs = [m[1] for m in cfg_matches]
            continue

        # Parse data row — e.g. "  512^3 (9)" or "  512^3 (L7)"
        grid_match = re.match(r'\s*(\d+)\^3\s*\([L\d]+\)', line)
        if grid_match and configs:
            grid_size = int(grid_match.group(1))
            parts = line.split('|')[1:]
            for i, part in enumerate(parts):
                if i >= len(configs):
                    break
                nums = part.strip().split()
                if len(nums) >= 3:
                    cfg = configs[i]
                    data.setdefault(cfg, {})[grid_size] = {
                        'TTS': float(nums[0]),
                        'Iter': int(nums[1]),
                        '1-cycle': float(nums[2]),
                    }

    return platform, configs, data


def parse_shallow_table(lines):
    """Parse shallow V-cycle table block (single config in title).
    Returns (config_name, {grid_size: {TTS, Iter, 1-cycle}})
    """
    config = None
    data = {}

    for line in lines:
        cfg_match = re.search(r'Shallow.*?\(([\d/]+)', line)
        if cfg_match:
            config = cfg_match.group(1)
            continue

        grid_match = re.match(r'\s*(\d+)\^3\s*\(L\d+\)', line)
        if grid_match:
            grid_size = int(grid_match.group(1))
            parts = line.split('|')[1:]
            for part in parts:
                nums = part.strip().split()
                if len(nums) >= 3:
                    data[grid_size] = {
                        'TTS': float(nums[0]),
                        'Iter': int(nums[1]),
                        '1-cycle': float(nums[2]),
                    }
                    break

    return config, data


def parse_file(filepath):
    """Parse a numbers file.
    Returns (platform, main_data, shallow_config, shallow_data).
    """
    with open(filepath) as f:
        content = f.read()

    lines = content.split('\n')

    # Find the WITH convergence section boundaries
    with_start = None
    with_end = len(lines)
    for i, line in enumerate(lines):
        if 'WITH convergence' in line and with_start is None:
            with_start = i
        elif 'WITHOUT convergence' in line and with_start is not None:
            with_end = i
            break

    if with_start is None:
        return None, {}, None, {}

    with_lines = lines[with_start:with_end]

    # Split off shallow section if present
    shallow_start = None
    for i, line in enumerate(with_lines):
        if 'Shallow' in line:
            shallow_start = i
            break

    if shallow_start is not None:
        main_lines = with_lines[:shallow_start]
        shallow_lines = with_lines[shallow_start:]
        shallow_cfg, shallow_data = parse_shallow_table(shallow_lines)
    else:
        main_lines = with_lines
        shallow_cfg, shallow_data = None, {}

    platform, _, main_data = parse_main_table(main_lines)
    return platform, main_data, shallow_cfg, shallow_data


def fmt_data(d, sep=" |"):
    """Format a data dict as a table cell."""
    return f"{d['TTS']:>10.6f}{d['Iter']:>8d}{d['1-cycle']:>10.6f} {sep}"


def fmt_empty(sep=" |"):
    return f"{'--':>10}{'--':>8}{'--':>10} {sep}"


def fmt_speedup(gd, wd, sep=" |"):
    tts_sp = gd['TTS'] / wd['TTS']
    cyc_sp = gd['1-cycle'] / wd['1-cycle']
    return f"{tts_sp:>9.1f}x{'--':>8}{cyc_sp:>8.1f}x {sep}"


def print_table(gpu_file, wse_file, target_grid):
    gpu_platform, gpu_data, _, _ = parse_file(gpu_file)
    wse_platform, wse_data, shallow_cfg, shallow_data = parse_file(wse_file)

    # Collect all configs across both files, sorted
    all_configs = sorted(
        set(gpu_data.keys()) | set(wse_data.keys()),
        key=lambda c: tuple(int(x) for x in c.split('/'))
    )

    # Insert shallow right after its matching config (e.g. 6/6/6)
    has_shallow = shallow_data and target_grid in shallow_data
    # Build column list: each entry is (label, config_key, is_shallow)
    columns = []
    for cfg in all_configs:
        columns.append((f"({cfg})", cfg, False))
        if has_shallow and cfg == shallow_cfg:
            columns.append((f"Shallow({shallow_cfg})", shallow_cfg, True))

    # Determine separator after each column:
    # Use "||" between the config and its shallow variant, "|" elsewhere
    separators = []
    for i, (_, _, is_shallow) in enumerate(columns):
        next_is_shallow = (i + 1 < len(columns) and columns[i + 1][2])
        separators.append("||" if next_is_shallow else " |")

    col_w = 28
    label_w = 16

    # --- Header ---
    # Compute total width from a sample row
    sample = f"{'':>{label_w}} |"
    for i, (lbl, _, _) in enumerate(columns):
        sample += f"{lbl:^{col_w}}{separators[i]}"
    total_w = len(sample)

    print("=" * total_w)
    print(f"  {target_grid}^3 Problem Size  (convergence included, #Cycles = V-cycles required for inf-norm to reach 1e-2)")
    print("=" * total_w)

    # Config names
    hdr = f"{'Platform':>{label_w}} |"
    for i, (lbl, _, _) in enumerate(columns):
        hdr += f"{lbl:^{col_w}}{separators[i]}"
    print(hdr)

    # Sub-header
    sub = f"{'':>{label_w}} |"
    for i in range(len(columns)):
        sub += f"{'TTS':>10}{'#Cycles':>8}{'1-cycle':>10} {separators[i]}"
    print(sub)

    print("-" * total_w)

    # --- GPU row: for shallow column, use the matching GPU config (6/6/6) ---
    if gpu_platform:
        row = f"{gpu_platform:>{label_w}} |"
        for i, (_, cfg, is_shallow) in enumerate(columns):
            d = gpu_data.get(cfg, {}).get(target_grid)
            row += fmt_data(d, separators[i]) if d else fmt_empty(separators[i])
        print(row)

    # --- WSE row: for shallow column, use shallow data ---
    if wse_platform:
        row = f"{wse_platform:>{label_w}} |"
        for i, (_, cfg, is_shallow) in enumerate(columns):
            if is_shallow:
                row += fmt_data(shallow_data[target_grid], separators[i])
            else:
                d = wse_data.get(cfg, {}).get(target_grid)
                row += fmt_data(d, separators[i]) if d else fmt_empty(separators[i])
        print(row)

    # --- Speedup row ---
    print("-" * total_w)
    if gpu_platform and wse_platform:
        row = f"{'Speedup':>{label_w}} |"
        for i, (_, cfg, is_shallow) in enumerate(columns):
            gd = gpu_data.get(cfg, {}).get(target_grid)
            if is_shallow:
                wd = shallow_data[target_grid]
            else:
                wd = wse_data.get(cfg, {}).get(target_grid)
            row += fmt_speedup(gd, wd, separators[i]) if (gd and wd) else fmt_empty(separators[i])
        print(row)

    print("=" * total_w)


if __name__ == '__main__':
    print_table(GPU_FILE, WSE_FILE, TARGET_GRID)
