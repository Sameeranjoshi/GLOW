#!/usr/bin/env python3
"""Print TTS / 1-V-cycle summary table across all configs and problem sizes.

Usage:
    python print_results_table.py
    python print_results_table.py --build-dir build   # default
"""

import argparse
import re
import os
import glob


FULL_DEPTH = {4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8, 512: 9}


def parse_response(filepath):
    """Return (tts_us, iters, vcycle_us) or None.

    Parses the configuration summary fields:
      Total solver wall time (us[cycles]):  10834.675us(   9480341)
      Device iterations               : 12
      Avg V-cycle time (inc. conv)    :    902.890us
    """
    text = open(filepath).read()
    tts_m = re.search(r"Total solver wall time \(us\[cycles\]\):\s*([\d.]+)\s*us", text)
    iter_m = re.search(r"Device iterations\s*:\s*(\d+)", text)
    vcycle_m = re.search(r"Avg V-cycle time \(inc\. conv\)\s*:\s*([\d.]+)\s*us", text)
    if not all([tts_m, iter_m, vcycle_m]):
        return None
    return float(tts_m.group(1)), int(iter_m.group(1)), float(vcycle_m.group(1))


def main():
    parser = argparse.ArgumentParser(description="Print results table")
    parser.add_argument("--build-dir", default="build", help="Directory with out_dir_* folders")
    args = parser.parse_args()

    regular = {}   # config -> {size: (tts_us, iters, vcycle_us, levels)}
    shallow = {}   # {size: (tts_us, iters, vcycle_us, levels)}

    patterns = [
        os.path.join(args.build_dir, "out_dir_S*/"),
        os.path.join(args.build_dir, "shallow_out_dir_S*/"),
    ]
    dirs = []
    for pat in patterns:
        dirs.extend(glob.glob(pat))

    for d in sorted(dirs):
        f = os.path.join(d, "response.txt")
        if not os.path.exists(f) or os.path.getsize(f) < 2000:
            continue
        name = os.path.basename(d.rstrip("/"))
        m = re.search(r"S(\d+)x_L(\d+)_M\d+_P(\d+)_P(\d+)_B(\d+)", name)
        if not m:
            continue
        size = int(m.group(1))
        levels = int(m.group(2))
        pre = int(m.group(3))
        post = int(m.group(4))
        bottom = int(m.group(5))

        result = parse_response(f)
        if result is None:
            continue
        tts_us, iters, vcycle_us = result

        is_shallow = (size in FULL_DEPTH and levels < FULL_DEPTH[size])
        if is_shallow:
            shallow[size] = (tts_us, iters, vcycle_us, levels)
        else:
            config = "{}/{}/{}".format(pre, post, bottom)
            regular.setdefault(config, {})[size] = (tts_us, iters, vcycle_us, levels)

    # --- Regular configs table ---
    configs = sorted(regular.keys())
    all_sizes = sorted(set(s for cfg in regular.values() for s in cfg))

    col_w = 30
    sep = "=" * (14 + len(configs) * (col_w + 3))
    print(sep)
    hdr = "             |"
    for c in configs:
        hdr += "{:>30} |".format("WSE3({})".format(c))
    print(hdr)

    sub = "        Grid |"
    for c in configs:
        sub += "{:>10}{:>6}{:>14} |".format("TTS(s)", "Iter", "1-cycle(s)")
    print(sub)
    print("-" * (14 + len(configs) * (col_w + 3)))

    for size in all_sizes:
        lvl = FULL_DEPTH.get(size, "?")
        label = "{}^3 ({})".format(size, lvl)
        row = "{:>12} |".format(label)
        for c in configs:
            if size in regular.get(c, {}):
                tts, it, vc, _ = regular[c][size]
                row += "{:>10.6f}{:>6}{:>14.6f} |".format(tts / 1e6, it, vc / 1e6)
            else:
                row += "{:>10}{:>6}{:>14} |".format("", "", "")
        print(row)

    print(sep)

    # --- Shallow table ---
    if shallow:
        print("")
        print("Shallow V-cycle (6/6/6, reduced levels):")
        print("=" * 55)
        print("{:>12} |{:>10}{:>6}{:>14} |".format("Grid", "TTS(s)", "Iter", "1-cycle(s)"))
        print("-" * 55)
        for size in sorted(shallow):
            tts, it, vc, lvl = shallow[size]
            label = "{}^3 (L{})".format(size, lvl)
            print("{:>12} |{:>10.6f}{:>6}{:>14.6f} |".format(label, tts / 1e6, it, vc / 1e6))
        print("=" * 55)


if __name__ == "__main__":
    main()
