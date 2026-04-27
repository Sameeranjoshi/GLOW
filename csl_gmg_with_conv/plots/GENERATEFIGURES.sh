#!/bin/bash
# GENERATEFIGURES.sh
#
# Single entry point to regenerate ALL plots for the GMG V-cycle paper.
# Run from this directory (plots/):  ./GENERATEFIGURES.sh
#
# Pipeline:
#   0. Append memory usage data to response.txt files (from .tar.gz ELFs)
#   1. Aggregate response.txt files from out_dir_*/ into all_responses_*.txt
#   2. Parse response data with plot_gmg_performance.py -> out_*.txt + internal plots
#   3. Analysis/comparison: h200_vs_cs3.py, memory_utilization_table.py, print_512_table.py
#   4. Standalone plots: v_vs_w_cycle.py
#   5. Roofline analysis: roofline_analysis.py
#   6. Results tables: print_results_table.py

set -u  # unset variable -> error (but continue on errors: no -e)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CSL_GMG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${CSL_GMG_DIR}" || exit 1

# Single log file for all script output
LOGFILE="${SCRIPT_DIR}/GENERATEFIGURES.log"
> "${LOGFILE}"  # truncate

# ----------------------------------------------------------------------------
# Step 0: append memory usage data to response.txt files (from .tar.gz ELFs)
# ----------------------------------------------------------------------------
echo "=========================================="
echo "Step 0: Appending memory usage to response files"
echo "=========================================="

CHECK_SCRIPT="${CSL_GMG_DIR}/check_memory_usage.sh"
if [ -x "${CHECK_SCRIPT}" ]; then
    for out_dir in "${CSL_GMG_DIR}"/build/out_dir_S*; do
        [ -d "${out_dir}" ] || continue
        response="${out_dir}/response.txt"
        [ -f "${response}" ] || continue

        # Skip if memory data already appended
        if grep -q "Code (FUNC symbols)" "${response}" 2>/dev/null; then
            echo "  SKIP $(basename ${out_dir}): memory data already present"
            continue
        fi

        # Find the .tar.gz artifact
        tarball=$(ls "${out_dir}"/*.tar.gz 2>/dev/null | head -1)
        if [ -z "${tarball}" ]; then
            echo "  SKIP $(basename ${out_dir}): no .tar.gz found"
            continue
        fi

        # Extract to temp dir, run check, append, clean up
        tmpdir=$(mktemp -d -p "${CSL_GMG_DIR}/build" cslgmg_mem_XXXXXX)
        tar xzf "${tarball}" -C "${tmpdir}" 2>/dev/null

        # Find the directory containing bin/*.elf (may be nested 2+ levels)
        elf_dir=""
        for cand in "${tmpdir}" "${tmpdir}"/* "${tmpdir}"/*/*; do
            if ls "${cand}"/bin/*.elf &>/dev/null; then
                elf_dir="${cand}"
                break
            fi
        done

        if [ -z "${elf_dir}" ]; then
            echo "  SKIP $(basename ${out_dir}): no bin/*.elf in archive"
            rm -rf "${tmpdir}"
            continue
        fi

        output=$(bash "${CHECK_SCRIPT}" "${elf_dir}" 2>&1)
        {
            echo ""
            echo "======================================================================"
            echo "Memory usage check:"
            echo "======================================================================"
            echo "${output}"
            echo ""
        } >> "${response}"
        echo "  $(basename ${out_dir}): appended memory data"
        rm -rf "${tmpdir}"
    done
else
    echo "  SKIP: check_memory_usage.sh not found"
fi

# ----------------------------------------------------------------------------
# Step 1: aggregate response.txt files into all_responses_*.txt
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 1: Aggregating response files"
echo "=========================================="

# All aggregated all_responses_*.txt files are written into build/ (alongside
# the out_dir_*/ they were aggregated from). Plot scripts read them from build/.
mkdir -p build
collect() {
    local pattern="$1"
    local outfile="build/$2"
    if ls -d ${pattern} 2>/dev/null | head -1 > /dev/null; then
        ls -d ${pattern} \
          | sed 's/.*S\([0-9]*\)x.*/\1 &/' \
          | sort -n | cut -d' ' -f2- \
          | xargs -I{} cat {}/response.txt > "${outfile}" 2>/dev/null
        echo "  -> ${outfile} ($(wc -l < ${outfile}) lines)"
    else
        echo "  SKIP ${outfile}: no matching dirs for pattern '${pattern}'"
    fi
}

# NOTE: out_dir_* and shallow_* directories now live under build/ (see
# compile_and_run_wse3.py's BUILD_DIR). The collect globs reflect that.
collect 'build/out_dir_S*x*_P6_P6_B100' all_responses_6_6_100.txt
collect 'build/out_dir_S*x*_P4_P4_B100' all_responses_4_4_100.txt
collect 'build/out_dir_S*x*_P4_P4_B6'   all_responses_4_4_6.txt
collect 'build/out_dir_S*x*_P6_P6_B6'   all_responses_6_6_6.txt
collect 'build/shallow_*'               all_responses_6_6_6_shallow.txt
collect 'build/out_dir_S*unoptimized*'  all_responses_6_6_6_unoptimized.txt

# ----------------------------------------------------------------------------
# Step 2: generate out_*.txt + internal plots from each aggregated file
#   plot_gmg_performance.py produces:
#     stdout -> out_*.txt (tee'd)
#     spmv_internal.png, interpolation_internal.png, per_operation_timing_*.png
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 2: Running plot_gmg_performance.py"
echo "=========================================="
cd "${SCRIPT_DIR}"

run_perf() {
    local input="$1"
    local output="$2"
    # Inputs (all_responses_*.txt) live under build/ now.
    if [ -s "../build/${input}" ]; then
        echo "  ${input} -> ${output}"
        python plot_gmg_performance.py "../build/${input}" > "${output}" 2>&1
    else
        echo "  SKIP ${input}: file missing or empty"
    fi
}

run_perf all_responses_6_6_100.txt        out_6_6_100.txt
run_perf all_responses_4_4_100.txt        out_4_4_100.txt
run_perf all_responses_4_4_6.txt          out_4_4_6.txt
run_perf all_responses_6_6_6_shallow.txt  out_6_6_6_shallow.txt
run_perf all_responses_6_6_6_unoptimized.txt out_6_6_6_unoptimized.txt
# 6/6/6 last so its plots (spmv_internal, interpolation_internal, per_operation_timing_512)
# are the ones kept on disk — they are the paper's primary figures.
run_perf all_responses_6_6_6.txt          out_6_6_6.txt

# ----------------------------------------------------------------------------
# Step 2.5: results tables (must run before Step 3, which consumes wse_numbers.txt)
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 2.5: Results tables"
echo "=========================================="

echo "  print_results_table.py -> wse_numbers.txt"
python print_results_table.py --build-dir ../build/ > wse_numbers.txt 2>> "${LOGFILE}"
{ echo "--- print_results_table.py (wrote wse_numbers.txt) ---"; echo; } >> "${LOGFILE}"

# ----------------------------------------------------------------------------
# Step 3: analysis/comparison scripts that depend on out_*.txt
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 3: Comparison/analysis plots"
echo "=========================================="

if [ -f gpu_numbers.txt ] && [ -f wse_numbers.txt ]; then
    echo "  h200_vs_cs3.py -> hpgmg_speedup_barplot.pdf"
    { echo "--- h200_vs_cs3.py ---"; python h200_vs_cs3.py 2>&1; echo; } | tee -a "${LOGFILE}"
else
    echo "  SKIP h200_vs_cs3.py: missing gpu_numbers.txt or wse_numbers.txt"
fi

if [ -s out_6_6_6.txt ]; then
    echo "  memory_utilization_table.py <- out_6_6_6.txt"
    { echo "--- memory_utilization_table.py ---"; python memory_utilization_table.py out_6_6_6.txt 2>&1; echo; } | tee -a "${LOGFILE}"
else
    echo "  SKIP memory_utilization_table.py: out_6_6_6.txt missing or empty"
fi

if [ -f gpu_numbers.txt ] && [ -f wse_numbers.txt ]; then
    echo "  print_512_table.py -> 512^3 comparison table"
    { echo "--- print_512_table.py ---"; python print_512_table.py 2>&1; echo; } | tee -a "${LOGFILE}"
else
    echo "  SKIP print_512_table.py: missing gpu_numbers.txt or wse_numbers.txt"
fi

# ----------------------------------------------------------------------------
# Step 4: standalone plot scripts (read response.txt or aggregates)
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 4: Standalone plots"
echo "=========================================="

echo "  v_vs_w_cycle.py -> W vs V cycle plot"
{ echo "--- v_vs_w_cycle.py ---"; python v_vs_w_cycle.py 2>&1; echo; } | tee -a "${LOGFILE}"

# ----------------------------------------------------------------------------
# Step 5: roofline analysis (one per sample problem)
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Step 5: Roofline analysis"
echo "=========================================="

ROOFLINE_SAMPLE="${CSL_GMG_DIR}/build/out_dir_S512x_L9_M100_P6_P6_B6/response.txt"
if [ -s "${ROOFLINE_SAMPLE}" ]; then
    echo "  roofline_analysis.py <- 512³ 6/6/6 -> roofline_plot.png"
    { echo "--- roofline_analysis.py ---"; python roofline_analysis.py "${ROOFLINE_SAMPLE}" 2>&1; echo; } | tee -a "${LOGFILE}"
else
    echo "  SKIP roofline_analysis.py: no 512³ 6/6/6 response.txt"
fi

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Done — generated figures:"
echo "=========================================="
ls -1 *.png *.pdf 2>/dev/null | sort
