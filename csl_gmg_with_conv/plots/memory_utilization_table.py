#!/usr/bin/env python3
"""Generate Code/Data/PE utilization table from plot_gmg_performance output."""

import re
import sys

REF_48KB = 49152
WAFER_PES = 893_064  # WSE-3: 762 x 1172

def main(filepath):
    with open(filepath) as f:
        content = f.read()

    pattern = re.compile(
        r"(\d+)x\1x\1\s+[\d,]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+\S+\s+"
        r"[\d.e+-]+\s+\w+\s+[\d.]+\s+[\d.]+\s+([\d,]+)\s+([\d,]+)"
    )

    rows = []
    for m in pattern.finditer(content):
        n = int(m.group(1))
        code = int(m.group(2).replace(",", ""))
        data = int(m.group(3).replace(",", ""))
        total = code + data
        active_pes = n * n
        rows.append((n, code, data, total, active_pes))

    # Header
    print("{:>8}  {:>18}  {:>18}  {:>22}  {:>14}".format(
        "Problem", "Code (B)", "Data (C)", "Total (D=B+C) (48KB)", "PE Util."))
    print("-" * 86)

    for n, code, data, total, active_pes in rows:
        code_pct = (code / REF_48KB) * 100
        data_pct = (data / REF_48KB) * 100
        total_pct = (total / REF_48KB) * 100
        pe_pct = (active_pes / WAFER_PES) * 100
        print("{:>5}\u00b3  {:>8,} ({:>5.1f}%)  {:>8,} ({:>5.1f}%)  {:>12,} ({:>5.1f}%)  {:>11.3f}%".format(
            n, code, code_pct, data, data_pct, total, total_pct, pe_pct))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python memory_utilization_table.py <out_X_X_X.txt>")
        sys.exit(1)
    main(sys.argv[1])
