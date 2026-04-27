#!/usr/bin/env bash
#
# run_all.sh — drive GLOW end-to-end across both repos:
#   1. csl_gmg_with_conv  : compile + CS-3 device run (V-cycle sweep)
#   2. w_cycle            : compile + CS-3 device run (W-cycle, 256^3)
#   3. csl_gmg_with_conv/plots : aggregate + regenerate all figures/tables
#
# Usage:
#   ./run_all.sh                # full pipeline
#   ./run_all.sh --skip-vcycle  # skip V-cycle device runs (use cached build/)
#   ./run_all.sh --skip-wcycle  # skip W-cycle device runs
#   ./run_all.sh --plots-only   # just (re)generate plots
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VCYCLE_DIR="${ROOT_DIR}/csl_gmg_with_conv"
WCYCLE_DIR="${ROOT_DIR}/w_cycle"
PLOTS_DIR="${VCYCLE_DIR}/plots"

SKIP_V=0
SKIP_W=0
for arg in "$@"; do
    case "${arg}" in
        --skip-vcycle) SKIP_V=1 ;;
        --skip-wcycle) SKIP_W=1 ;;
        --plots-only)  SKIP_V=1; SKIP_W=1 ;;
        -h|--help)
            sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown arg: ${arg}"; exit 2 ;;
    esac
done

log() { printf '\n========== %s ==========\n' "$*"; }

# Required GPU baseline numbers — without it Fig. 5 and Table 4 are skipped silently.
if [ ! -f "${PLOTS_DIR}/gpu_numbers.txt" ]; then
    echo "ERROR: ${PLOTS_DIR}/gpu_numbers.txt not found." >&2
    echo "Copy gpu_numbers.txt into ${PLOTS_DIR}/ before running." >&2
    exit 1
fi

if [ "${SKIP_V}" -eq 0 ]; then
    log "Step 1/3: V-cycle compile + device run (csl_gmg_with_conv)"
    cd "${VCYCLE_DIR}"
    python compile_and_run_wse3.py --only-device
fi

if [ "${SKIP_W}" -eq 0 ]; then
    log "Step 2/3: W-cycle compile + device run (w_cycle)"
    cd "${WCYCLE_DIR}"
    python compile_and_run_wse3.py --only-device
fi

log "Step 3/3: Regenerate all figures (plots/GENERATEFIGURES.sh)"
cd "${PLOTS_DIR}"
bash GENERATEFIGURES.sh
