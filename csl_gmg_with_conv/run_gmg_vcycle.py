#!/usr/bin/env python3
"""
Geometric Multigrid (GMG) V-cycle solver using CSL with state machine
Runs complete V-cycle on device
"""

import math
from pathlib import Path
import shutil
import copy
import numpy as np
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', "python_gmg"))
# from python_gmg.gmgpython import SimpleGMG as SimpleGMGPYTHON
from gmgpython import SimpleGMG as SimpleGMGPYTHON
from cmd_parser import parse_args, print_arguments
from util import hwl_2_oned_colmajor, oned_to_hwl_colmajor
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyOrder, MemcpyDataType

DTYPE = np.float32    
WORDS_PER_TIMESTAMP = 3
WORDS_PER_START_END = WORDS_PER_TIMESTAMP * 2

def get_exponent(A: int) -> int:
    return int(math.log2(A))

def make_u48(words):
    """Convert three u16 words to 48-bit timestamp"""
    return words[0] + (words[1] << 16) + (words[2] << 32)

def print_2d(matrix):
    for row in matrix:
        print(" | ".join(f"{v:8.2f}" for v in row))

def copy_data_d2h(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u, symbol_r, symbol_rho, symbol_rho_history, args):
    pass
    # u_wse_1d = np.zeros(height*width*zDim, DTYPE)
    # r_wse_1d = np.zeros(height*width*zDim, DTYPE)
    # rho_wse = np.zeros(1, np.float32)
    # simulator.memcpy_d2h(u_wse_1d, symbol_u, 0, 0, width, height, zDim,
    #                     streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    # simulator.memcpy_d2h(r_wse_1d, symbol_r, 0, 0, width, height, zDim,
    #                     streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    # simulator.memcpy_d2h(rho_wse, symbol_rho, 0, 0, 1, 1, 1, 
    #                     streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=False)
    # # rho_history will be copied later after counter is available
    # rho_history_array = np.array([], dtype=np.float32)
    # total_bytes = u_wse_1d.nbytes + r_wse_1d.nbytes + rho_wse.nbytes
    # return u_wse_1d, r_wse_1d, rho_wse[0], rho_history_array, total_bytes

def copy_data_h2d(height, width, zDim, memcpy_dtype, memcpy_order, simulator, symbol_u, symbol_f, symbol_hx_array, symbol_jacobi_coeff_array, device_solver, args):
    """
    Copies problem and grid spacing data to device.
    Isotropic grid: hx=hy=hz, so only hx_array is sent (hy/hz removed from kernel).
    """
    # Prepare grid spacing arrays for one PE (vector length = args.levels)
    # Device expects 1/hx (reciprocal) to avoid float division on device
    inv_hx_base = np.array([1.0 / device_solver.grids[i]['hx'] for i in range(args.levels)], dtype=DTYPE)

    # Prepare Jacobi coefficient array for one PE (isotropic: hx=hy=hz)
    jacobi_coeff_base = np.zeros(args.levels, dtype=DTYPE)
    omega = device_solver.omega
    for level in range(args.levels):
        hx = device_solver.grids[level]['hx']
        jacobi_coeff_base[level] = -omega / (2.0 * (3.0/(hx*hx)))

    # Repeat these arrays so shape is (height, width, levels)
    hx_array = np.tile(inv_hx_base, (height, width, 1))  # stores 1/hx
    jacobi_coeff_array = np.tile(jacobi_coeff_base, (height, width, 1))

    # Prepare u/f arrays
    u_hwl = device_solver.grids[0]['u']
    f_hwl = device_solver.grids[0]['f']
    u_1d = hwl_2_oned_colmajor(height, width, zDim, u_hwl, DTYPE)
    f_1d = hwl_2_oned_colmajor(height, width, zDim, f_hwl, DTYPE)
    # Debug: Print what we're sending
    print(f"Grid spacing and Jacobi coefficients:")
    for level in range(args.levels):
        hx = device_solver.grids[level]['hx']
        print(f"  Level {level}: hx={hx:.6f}, 1/hx={inv_hx_base[level]:.6f}, jacobi={jacobi_coeff_base[level]:.6e}")

    # Copy spacing/jacobi arrays: flatten in COLUMN-MAJOR order to match memcpy
    hx_flat = hwl_2_oned_colmajor(height, width, args.levels, hx_array, DTYPE)
    jacobi_flat = hwl_2_oned_colmajor(height, width, args.levels, jacobi_coeff_array, DTYPE)

    # Copy u and f arrays
    simulator.memcpy_h2d(symbol_u, u_1d, 0, 0, width, height, zDim,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=True)
    simulator.memcpy_h2d(symbol_f, f_1d, 0, 0, width, height, zDim,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=True)
    simulator.memcpy_h2d(symbol_hx_array, hx_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=True)
    simulator.memcpy_h2d(symbol_jacobi_coeff_array, jacobi_flat, 0, 0, width, height, args.levels,
                         streaming=False, data_type=memcpy_dtype, order=memcpy_order, nonblock=True)
    # simulator.launch("f_toc_h2d", nonblock=False)
    total_bytes = (
        u_1d.nbytes
        + f_1d.nbytes
        + hx_flat.nbytes
        + jacobi_flat.nbytes
    )

    return total_bytes

def copy_timing_make_48bit(height, width, levels, simulator, symbol_timing, words_per_entry, operation_name):
    """
    Copy timing data from device and extract per-level 48-bit timestamps (PE 0,0 only).
    """
    timing_size = words_per_entry * levels
    timing_1d = np.zeros(height*width*timing_size, np.uint32)
    simulator.memcpy_d2h(
        timing_1d, symbol_timing, 0, 0, width, height, timing_size,
        streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.COL_MAJOR, nonblock=False
    )
    # Repack to work with oned_to_hwl_colmajor as before, using uint16 version
    timing_hwl = oned_to_hwl_colmajor(width, height, timing_size, timing_1d, np.uint16)
    per_pe_time_levels = timing_hwl[0, 0, :]  # All levels' words packed for PE(0,0)
    # if operation_name == "restriction":
    #     print("restriction PE0,0 = per_pe_time_levels", per_pe_time_levels)
    timing_per_level = []
    for level in range(levels):
        base = level * words_per_entry
        level_time = per_pe_time_levels[base:base+3]
        if len(level_time) < 3:
            continue
        cycles_send = make_u48(level_time)
        time_send = (cycles_send / 0.875) * 1.e-3
        timing_per_level.append({
            'level': level,
            'operation': operation_name,
            'cycles_send': cycles_send,
            'time_send': time_send
        })
    return timing_per_level

def copy_counters(height, width, levels, simulator, symbol_counter):
    """Copy operation counter data from device - optimized to copy only minimal region (2x1) since only PE(0,0) is used"""

    counter_1d = np.zeros(1*1*levels, dtype=np.uint32)
    simulator.memcpy_d2h(counter_1d, symbol_counter, 0, 0, 1, 1, levels,
                        streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, 
                        order=MemcpyOrder.COL_MAJOR, nonblock=False)
    counter_hwl = oned_to_hwl_colmajor(1, 1, levels, counter_1d, np.uint16)
    per_pe_time_levels = counter_hwl[0, 0, :]  # All levels' words packed for PE(0,0)                        

    return per_pe_time_levels

def copy_rho_history(height, width, memcpy_dtype, memcpy_order, simulator, symbol_rho_history, counter_rho_check, args):
    """Copy rho_history from device using counter to determine how many elements to copy"""
    
    # actual_iterations = int(counter_rho_check[0, 0]) if np.ndim(counter_rho_check) > 0 and counter_rho_check.size > 0 else 0
    actual_iterations = int(counter_rho_check[0])
    
    rho_history_1d = np.zeros(1 * 1 * actual_iterations, np.float32)
    simulator.memcpy_d2h(rho_history_1d, symbol_rho_history, 0, 0, 1, 1, actual_iterations,
                        streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

    return rho_history_1d, actual_iterations

def print_configuration_summary(
    args,
    device_solver,
    device_rho,
    counter_rho_check,
    timing_total_start_end_data,
    first_vcycle_time_us=0,
    ):
    dtype_name = DTYPE.__name__ if hasattr(DTYPE, "__name__") else str(DTYPE)
    # counter_rho_check has shape (1, levels), so access [0, 0] for first level's count
    # device_iterations = int(counter_rho_check[0, 0]) if np.ndim(counter_rho_check) > 0 and counter_rho_check.size > 0 else 0
    # print("ndim", np.ndim(counter_rho_check))
    device_iterations = max(int(counter_rho_check[0]), 1)
    t_wall = timing_total_start_end_data[0]

    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)

    config_items = [
        # problem/PE size.
        ("HeightxWidthxZDim", f"{args.m}x{args.n}x{args.zDim}"),
        ("Levels", args.levels),
        ("Max iterations", args.max_ite),
        ("Tolerance (abs)", f"{device_solver.abs_tolerance:.2e}"),
        ("Tolerance (rel)", f"{device_solver.rel_tolerance:.2e}"),
        ("Pre/Post/Bottom iter", f"{args.pre_iter}/{args.post_iter}/{args.bottom_iter}"),
        ("Datatype", dtype_name),
        ("Jacobi omega", f"{device_solver.omega:.6f}"),
        ("Stencil alpha/beta", f"{device_solver.ALPHA}/{device_solver.BETA}"),
        ("Channels", args.channels),
        ("Block size", args.blockSize),
        ("West/East buffer width", f"{args.width_west_buf}/{args.width_east_buf}"),
        ("Total solver wall time (us[cycles])", f"{t_wall['time_send']:10.3f}us({t_wall['cycles_send']:10.0f})"),
        ("Device iterations", int(counter_rho_check[0])),
        ("Avg V-cycle time (inc. conv)", f"{first_vcycle_time_us:10.3f}us"),
        ("Device final |rho|_inf", f"{device_rho:.3e}"),
    ]

    key_width = 32
    for label, value in config_items:
        print(f"{label:<{key_width}}: {value}")

def profiling(
        args,
        WORDS_PER_TIMESTAMP,
        device_solver, device_rh,
        counter_rho_check,
        simulator,
        timing_smooth_data, timing_residual_data, timing_restrict_data, timing_interp_data,
        timing_smooth_apply_data, timing_smooth_update_data,
        timing_interp_expand_z_data, timing_interp_bcast_data, timing_bcast_configure_data, timing_bcast_to_all_data, timing_interp_add_data,
        timing_spmv_total_data, timing_spmv_communication_data, timing_spmv_compute_data,
        timing_setup_data, timing_convergence_data,
        timing_total_start_end_data
    ):
    print("\n" + "="*60)
    print("Performance Timing")
    print("="*60)
    print(
        "PE(0,0) only. Phase timers (smooth, residual, …) ACCUMULATE across all V-cycles.\n"
        "Convergence (L0 only) is separated; see summary block below the tables for per-V-cycle averages."
    )

    ############################################################
    # Time per operation and level (us[cycles]):
    ############################################################
    
    print("\nTime per operation and level (us[cycles]), first V-cycle:")
    operators = [
        ("smooth", timing_smooth_data),
        ("residual", timing_residual_data),
        ("restriction", timing_restrict_data),
        ("interpolation", timing_interp_data),
        ("setup", timing_setup_data),
        ("convergence", timing_convergence_data),
    ]

    header_cols = ["level"] + [op[0] for op in operators] + ["total"]
    col_width = 20

    def format_entry(timing_entry):
        return f"{timing_entry['time_send']:6.3f}us({timing_entry['cycles_send']:7.0f})"

    def build_divider(columns, char="-"):
        return "+" + "+".join(char * col_width for _ in columns) + "+"

    print(build_divider(header_cols, "="))
    print("|" + "|".join(f"{name:^{col_width}}" for name in header_cols) + "|")
    print(build_divider(header_cols))

    level_totals_cycles = []
    level_totals_time = []

    for level in range(args.levels):
        row_entries = [f"{level:^{col_width}}"]
        total_cycles = 0
        total_time = 0.0

        for _, timing_list in operators:
            entry = timing_list[level]
            row_entries.append(f"{format_entry(entry):^{col_width}}")
            total_cycles += entry["cycles_send"]
            total_time += entry["time_send"]

        level_totals_cycles.append(total_cycles)
        level_totals_time.append(total_time)

        total_str = f"{total_time:6.3f}us({total_cycles:7.0f})"
        row_entries.append(f"{total_str:^{col_width}}")
        print("|" + "|".join(row_entries) + "|")
        print(build_divider(header_cols))

    total_row_entries = [f"{'total':^{col_width}}"]
    for _, timing_list in operators:
        cycles_sum = sum(entry["cycles_send"] for entry in timing_list)
        time_sum = sum(entry["time_send"] for entry in timing_list)
        total_row_entries.append(f"{time_sum:6.3f}us({cycles_sum:7.0f})".center(col_width))

    grand_total_cycles = sum(level_totals_cycles)
    grand_total_time = sum(level_totals_time)
    total_row_entries.append(f"{grand_total_time:6.3f}us({grand_total_cycles:7.0f})".center(col_width))
    print("|" + "|".join(total_row_entries) + "|")
    print(build_divider(header_cols, "="))
    total_time_us = grand_total_time
    total_time_cycles = grand_total_cycles

    # ------------------------------------------------------------
    # 7-pt Stencil: SpMV vs smoothing (PE 0,0)
    # Stencil SpMV timers are *not* gated by roofline_measure and accumulate over
    # every V-cycle; scale by device_iterations so columns match one V-cycle.
    # Smooth sub-timers use save_timestamp_* and match the first V-cycle only.
    # ------------------------------------------------------------
    device_iterations = max(int(counter_rho_check[0]), 1)
    print("\n7-pt Stencil: SpMV vs smoothing (us[cycles]), per level, scaled to one V-cycle:")
    print(
        f"  SpMV Total / Communication / Compute ÷ {device_iterations} (iterations); "
        "smooth_* from first V-cycle only."
    )
    compute_comm_header = [
        "level", "Total SpMV Time", "Communication Time", "Compute Time",
        "smooth_apply", "smooth_update", "smooth_total", "smooth_overhead"
    ]
    print(build_divider(compute_comm_header, "="))
    print("|" + "|".join(f"{name:^{col_width}}" for name in compute_comm_header) + "|")
    print(build_divider(compute_comm_header))

    sum_spmv_total_time = 0.0
    sum_spmv_communication_time = 0.0
    sum_spmv_compute_time = 0.0
    sum_spmv_total_cycles = 0
    sum_spmv_communication_cycles = 0
    sum_spmv_compute_cycles = 0
    sum_smooth_apply = 0
    sum_smooth_update = 0
    sum_smooth_total = 0

    for level in range(args.levels):
        spmv_total_entry = timing_spmv_total_data[level]
        spmv_communication_entry = timing_spmv_communication_data[level]
        spmv_compute_entry = timing_spmv_compute_data[level]
        smooth_apply_entry = timing_smooth_apply_data[level]
        smooth_update_entry = timing_smooth_update_data[level]
        smooth_total_entry = timing_smooth_data[level]

        # SpMV micro-timers are inside the stencil module (NOT gated by roofline_measure)
        # They accumulate across all iterations, so divide by iterations to get per-1V values
        spmv_total_time_level = spmv_total_entry["time_send"] / device_iterations
        spmv_communication_time_level = spmv_communication_entry["time_send"] / device_iterations
        spmv_compute_time_level = spmv_compute_entry["time_send"] / device_iterations

        spmv_total_cycles_level = spmv_total_entry["cycles_send"] / device_iterations
        spmv_communication_cycles_level = spmv_communication_entry["cycles_send"] / device_iterations
        spmv_compute_cycles_level = spmv_compute_entry["cycles_send"] / device_iterations

        # smooth_overhead = smooth_total - (smooth_apply + smooth_update)
        smooth_overhead_us = smooth_total_entry['time_send'] - (smooth_apply_entry['time_send'] + smooth_update_entry['time_send'])
        smooth_overhead_cyc = smooth_total_entry['cycles_send'] - (smooth_apply_entry['cycles_send'] + smooth_update_entry['cycles_send'])
        row = [
            f"{level:^{col_width}}",
            f"{spmv_total_time_level:6.3f}us({spmv_total_cycles_level:7.0f})".center(col_width),
            f"{spmv_communication_time_level:6.3f}us({spmv_communication_cycles_level:7.0f})".center(col_width),
            f"{spmv_compute_time_level:6.3f}us({spmv_compute_cycles_level:7.0f})".center(col_width),
            f"{smooth_apply_entry['time_send']:6.3f}us({smooth_apply_entry['cycles_send']:7.0f})".center(col_width),
            f"{smooth_update_entry['time_send']:6.3f}us({smooth_update_entry['cycles_send']:7.0f})".center(col_width),
            f"{smooth_total_entry['time_send']:6.3f}us({smooth_total_entry['cycles_send']:7.0f})".center(col_width),
            f"{smooth_overhead_us:6.3f}us({smooth_overhead_cyc:7.0f})".center(col_width),
        ]
        print("|" + "|".join(row) + "|")
        print(build_divider(compute_comm_header))

        sum_spmv_total_time += spmv_total_time_level  # already per one V-cycle
        sum_spmv_communication_time += spmv_communication_time_level
        sum_spmv_compute_time += spmv_compute_time_level
        sum_spmv_total_cycles += spmv_total_cycles_level
        sum_spmv_communication_cycles += spmv_communication_cycles_level
        sum_spmv_compute_cycles += spmv_compute_cycles_level
        sum_smooth_apply += smooth_apply_entry["cycles_send"]
        sum_smooth_update += smooth_update_entry["cycles_send"]
        sum_smooth_total += smooth_total_entry["cycles_send"]

    sum_smooth_apply_time = sum(e["time_send"] for e in timing_smooth_apply_data)
    sum_smooth_update_time = sum(e["time_send"] for e in timing_smooth_update_data)
    sum_smooth_total_time = sum(e["time_send"] for e in timing_smooth_data)
    sum_smooth_overhead_time = sum_smooth_total_time - (sum_smooth_apply_time + sum_smooth_update_time)
    sum_smooth_overhead_cyc = sum_smooth_total - (sum_smooth_apply + sum_smooth_update)
    total_row = [
        f"{'total':^{col_width}}",
        f"{sum_spmv_total_time:6.3f}us({sum_spmv_total_cycles:7.0f})".center(col_width),
        f"{sum_spmv_communication_time:6.3f}us({sum_spmv_communication_cycles:7.0f})".center(col_width),
        f"{sum_spmv_compute_time:6.3f}us({sum_spmv_compute_cycles:7.0f})".center(col_width),
        f"{sum_smooth_apply_time:6.3f}us({sum_smooth_apply:7.0f})".center(col_width),
        f"{sum_smooth_update_time:6.3f}us({sum_smooth_update:7.0f})".center(col_width),
        f"{sum_smooth_total_time:6.3f}us({sum_smooth_total:7.0f})".center(col_width),
        f"{sum_smooth_overhead_time:6.3f}us({sum_smooth_overhead_cyc:7.0f})".center(col_width),
    ]
    print("|" + "|".join(total_row) + "|")
    print(build_divider(compute_comm_header, "="))

    # ------------------------------------------------------------
    # Interpolation Micro-Benchmark (f_interpolation_expand_z, f_bcast_from_top_left, f_interpolation_add)
    # bcast breakdown: bcast_configure | bcast_to_all | remaining(fabric+TD)
    # ------------------------------------------------------------
    print("\nInterpolation micro-benchmark (us[cycles]), PE(0,0), first V-cycle only:")
    interp_micro_header = ["level", "expand_z(T1)", "bcast_total(T2)", "reset_routes(T2.1)", "send_data(T2.2)", "interp_add(T3)", "interp_total(T1+T2+T3)"]
    print(build_divider(interp_micro_header, "="))
    print("|" + "|".join(f"{name:^{col_width}}" for name in interp_micro_header) + "|")
    print(build_divider(interp_micro_header))

    for level in range(args.levels):
        expand_z_entry = timing_interp_expand_z_data[level]
        bcast_entry = timing_interp_bcast_data[level]
        bcast_configure_entry = timing_bcast_configure_data[level]
        bcast_to_all_entry = timing_bcast_to_all_data[level]
        interp_add_entry = timing_interp_add_data[level]
        interp_total_entry = timing_interp_data[level]

        row = [
            f"{level:^{col_width}}",
            f"{expand_z_entry['time_send']:6.3f}us({expand_z_entry['cycles_send']:7.0f})".center(col_width),
            f"{bcast_entry['time_send']:6.3f}us({bcast_entry['cycles_send']:7.0f})".center(col_width),
            f"{bcast_configure_entry['time_send']:6.3f}us({bcast_configure_entry['cycles_send']:7.0f})".center(col_width),
            f"{bcast_to_all_entry['time_send']:6.3f}us({bcast_to_all_entry['cycles_send']:7.0f})".center(col_width),
            f"{interp_add_entry['time_send']:6.3f}us({interp_add_entry['cycles_send']:7.0f})".center(col_width),
            f"{interp_total_entry['time_send']:6.3f}us({interp_total_entry['cycles_send']:7.0f})".center(col_width),
        ]
        print("|" + "|".join(row) + "|")
        print(build_divider(interp_micro_header))

    sum_expand_z = sum(e["cycles_send"] for e in timing_interp_expand_z_data)
    sum_bcast = sum(e["cycles_send"] for e in timing_interp_bcast_data)
    sum_bcast_configure = sum(e["cycles_send"] for e in timing_bcast_configure_data)
    sum_bcast_to_all = sum(e["cycles_send"] for e in timing_bcast_to_all_data)
    sum_interp_add_cycles = sum(e["cycles_send"] for e in timing_interp_add_data)
    sum_interp_total_cycles = sum(e["cycles_send"] for e in timing_interp_data)
    interp_micro_total_row = [
        f"{'total':^{col_width}}",
        f"{sum(e['time_send'] for e in timing_interp_expand_z_data):6.3f}us({sum_expand_z:7.0f})".center(col_width),
        f"{sum(e['time_send'] for e in timing_interp_bcast_data):6.3f}us({sum_bcast:7.0f})".center(col_width),
        f"{sum(e['time_send'] for e in timing_bcast_configure_data):6.3f}us({sum_bcast_configure:7.0f})".center(col_width),
        f"{sum(e['time_send'] for e in timing_bcast_to_all_data):6.3f}us({sum_bcast_to_all:7.0f})".center(col_width),
        f"{sum(e['time_send'] for e in timing_interp_add_data):6.3f}us({sum_interp_add_cycles:7.0f})".center(col_width),
        f"{sum(e['time_send'] for e in timing_interp_data):6.3f}us({sum_interp_total_cycles:7.0f})".center(col_width),
    ]
    print("|" + "|".join(interp_micro_total_row) + "|")
    print(build_divider(interp_micro_header, "="))

    ############################################################
    # V-cycle time summary (Time to Solution methodology)
    #   TTS (wall_total) = solver time, all V-cycles (includes convergence check per V-cycle)
    #   avg_vcycle       = wall_total / iterations  (one V-cycle, includes convergence)
    #   conv_total       = sum of convergence checks (only L0), for reference
    ############################################################
    di = max(int(counter_rho_check[0]), 1)
    wall_total_us = timing_total_start_end_data[0]["time_send"]
    wall_total_cyc = timing_total_start_end_data[0]["cycles_send"]
    # Convergence accumulates only at L0 (only L0 has conv check)
    conv_total_us = timing_convergence_data[0]["time_send"]
    conv_total_cyc = timing_convergence_data[0]["cycles_send"]
    avg_vcycle_us = wall_total_us / di
    avg_vcycle_cyc = wall_total_cyc / di
    print("=" * 100)
    print(
        f"Time to Solution (TTS, all {di} V-cycles, inc. convergence):               "
        f"{wall_total_us:10.3f} us ({wall_total_cyc:10.0f} cycles)"
    )
    print(
        f"Convergence diagnostic time (total, {di} checks, only at L0):               "
        f"{conv_total_us:10.3f} us ({conv_total_cyc:10.0f} cycles)"
    )
    print(
        f"Avg V-cycle time (TTS / {di} iters, inc. convergence):                      "
        f"{avg_vcycle_us:10.3f} us ({avg_vcycle_cyc:10.0f} cycles)"
    )
    print("=" * 100)
    return avg_vcycle_us  # avg V-cycle time including convergence


def main():
    """Main function"""
    np.random.seed(2)

    ############################################################
    # Parameters
    ############################################################
    args, logs_dir = parse_args()
    
    # Problem parameters
    height = args.m
    width = args.n
    pe_length = args.k
    zDim = args.zDim
    
    # Validation
    assert pe_length >= 2, "pe_length must be >= 2"
    assert zDim >= 2, "zDim must be >= 2"
    assert zDim <= pe_length, "zDim must be <= pe_length"
    
    max_possible_levels = get_exponent(height)
    if args.levels > max_possible_levels:
        raise ValueError(f"levels ({args.levels}) > max_possible_levels ({max_possible_levels})")

    ############################################################
    # Host
    ############################################################
    # Create reference solver on host
    print("\n" + "="*60)
    print("Creating reference solver on host...")
    print("="*60)
    host_solver = SimpleGMGPYTHON(width, height, zDim, args.levels, args.verbose,
                                 args.tolerance, args.pre_iter, args.post_iter, args.bottom_iter)
    device_solver = copy.deepcopy(host_solver)

    # Run reference on host
    # host_residual, host_iterations = host_solver.solve_iterative(args.max_ite)

    ############################################################
    # Device
    ############################################################
    # Initialize device
    print("\n" + "="*60)
    print("Device calculations...")
    print("="*60)
    
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    memcpy_order = MemcpyOrder.COL_MAJOR
    simulator = SdkRuntime(logs_dir, cmaddr=args.cmaddr, simfab_numthreads=64)
    
    symbol_u = simulator.get_id("u")
    symbol_f = simulator.get_id("f")
    symbol_r = simulator.get_id("r")
    symbol_hx_array = simulator.get_id("hx_array")
    # hy_array, hz_array removed — isotropic grid uses hx_array only
    symbol_jacobi_coeff_array = simulator.get_id("jacobi_coeff_array")
    symbol_timing_smooth = simulator.get_id("timing_smooth")
    symbol_timing_smooth_apply = simulator.get_id("timing_smooth_apply")
    symbol_timing_smooth_update = simulator.get_id("timing_smooth_update")
    symbol_timing_residual = simulator.get_id("timing_residual")
    symbol_timing_restrict = simulator.get_id("timing_restrict")
    symbol_timing_interp = simulator.get_id("timing_interp")
    symbol_timing_interp_expand_z = simulator.get_id("timing_interp_expand_z")
    symbol_timing_interp_bcast = simulator.get_id("timing_interp_bcast")
    symbol_timing_bcast_configure = simulator.get_id("timing_bcast_configure")
    symbol_timing_bcast_to_all = simulator.get_id("timing_bcast_to_all")
    symbol_timing_interp_add = simulator.get_id("timing_interp_add")
    symbol_time_total_start_end = simulator.get_id("time_total_start_end")
    symbol_timing_spmv_total = simulator.get_id("timing_spmv_total")
    symbol_timing_spmv_communication = simulator.get_id("timing_spmv_communication")
    symbol_timing_spmv_compute = simulator.get_id("timing_spmv_compute")
    symbol_timing_setup = simulator.get_id("timing_setup")
    symbol_timing_convergence = simulator.get_id("timing_convergence")
    symbol_counter_rho_check = simulator.get_id("counter_rho_check")
    symbol_count_fmov32 = simulator.get_id("count_fmov32")
    symbol_rho_history = simulator.get_id("rho_history")
    # Roofline FLOP counter symbols
    symbol_count_fsub = simulator.get_id("count_fsub")
    symbol_count_fmac = simulator.get_id("count_fmac")
    symbol_count_fmul = simulator.get_id("count_fmul")
    symbol_count_fadd = simulator.get_id("count_fadd")
    symbol_count_fneg = simulator.get_id("count_fneg")
    symbol_count_fmov_mem = simulator.get_id("count_fmov_mem")
    symbol_count_fmov_zero = simulator.get_id("count_fmov_zero")
    symbol_count_fmax = simulator.get_id("count_fmax")
    
    simulator.load()
    simulator.run()

    ############################################################
    # Warmup + reset
    print("1. Warmup(3 iterations) + reset...")
    ############################################################
    warmup_iterations = 3
    simulator.launch("f_gmg_vcycle", 
                    # np.int16(zDim), 
                    np.int16(args.levels),
                    np.int16(args.pre_iter),
                    np.int16(args.post_iter),
                    np.int16(args.bottom_iter),
                    np.int16(warmup_iterations),  # max_iter parameter
                    np.float32(device_solver.rel_tolerance),  # tolerance parameter (relative tolerance, will be squared in kernel)
                    nonblock=False)
    simulator.launch("f_reset", nonblock=False)
    
    ############################################################
    # Copy data to device
    print("2. Copying initial data to device...")
    ############################################################
   # Timing measures inside the function
    copy_data_h2d(height, width, zDim, memcpy_dtype, memcpy_order, simulator, 
                 symbol_u, symbol_f, symbol_hx_array, symbol_jacobi_coeff_array, device_solver, args)

    ############################################################
    # Kernel launch
    ############################################################
    # Enable timing and synchronize PEs
    print("3. Enabling timer...")
    simulator.launch("f_enable_timer", nonblock=False)
    print("4. Synchronizing PEs...")
    simulator.launch("f_sync", nonblock=False)
    print("5. Tic total...")
    simulator.launch("f_tic_total", nonblock=True)
    print(f"6. Running GMG V-cycle(max_iter={args.max_ite}, levels={args.levels}) on device...")
    simulator.launch("f_gmg_vcycle", 
                    np.int16(args.levels),
                    np.int16(args.pre_iter),
                    np.int16(args.post_iter),
                    np.int16(args.bottom_iter),
                    np.int16(args.max_ite),  # max_iter parameter
                    np.float32(device_solver.rel_tolerance),  # tolerance parameter (relative tolerance, will be squared in kernel)
                    nonblock=False)
    print("7. Toc total...")
    simulator.launch("f_toc_total", nonblock=False)

    ############################################################
    # Copy timers and counters
    print("8. Copying timers and counters from device...")
    ############################################################
    timing_smooth_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_smooth, WORDS_PER_TIMESTAMP, "smooth")
    timing_smooth_apply_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_smooth_apply, WORDS_PER_TIMESTAMP, "smooth_apply")
    timing_smooth_update_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_smooth_update, WORDS_PER_TIMESTAMP, "smooth_update")
    timing_residual_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_residual, WORDS_PER_TIMESTAMP, "residual")
    timing_restrict_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_restrict, WORDS_PER_TIMESTAMP, "restriction")
    timing_interp_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_interp, WORDS_PER_TIMESTAMP, "interpolation")
    timing_interp_expand_z_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_interp_expand_z, WORDS_PER_TIMESTAMP, "interp_expand_z")
    timing_interp_bcast_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_interp_bcast, WORDS_PER_TIMESTAMP, "interp_bcast")
    timing_bcast_configure_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_bcast_configure, WORDS_PER_TIMESTAMP, "bcast_configure")
    timing_bcast_to_all_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_bcast_to_all, WORDS_PER_TIMESTAMP, "bcast_to_all")
    timing_interp_add_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_interp_add, WORDS_PER_TIMESTAMP, "interp_add")
    timing_spmv_total_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_spmv_total, WORDS_PER_TIMESTAMP, "spmv_total")
    timing_spmv_communication_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_spmv_communication, WORDS_PER_TIMESTAMP, "spmv_communication")
    timing_spmv_compute_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_spmv_compute, WORDS_PER_TIMESTAMP, "spmv_compute")
    timing_setup_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_setup, WORDS_PER_TIMESTAMP, "setup")
    timing_convergence_data = copy_timing_make_48bit(height, width, args.levels, simulator, symbol_timing_convergence, WORDS_PER_TIMESTAMP, "convergence")
    timing_total_start_end_data = copy_timing_make_48bit(height, width,1, simulator, symbol_time_total_start_end, WORDS_PER_TIMESTAMP, "total")

    counter_rho_check = copy_counters(height, width, args.levels, simulator, symbol_counter_rho_check)
    # Read roofline counters using MEMCPY_16BIT with width×height region
    # (u32 read as 2×u16 words — avoids MEMCPY_32BIT call limit on small grids)
    roofline_counters = {}
    for name, sym in [('fsub', symbol_count_fsub), ('fmac', symbol_count_fmac),
                      ('fmul', symbol_count_fmul), ('fadd', symbol_count_fadd),
                      ('fneg', symbol_count_fneg), ('fmov_mem', symbol_count_fmov_mem),
                      ('fmov_zero', symbol_count_fmov_zero),
                      ('fmax', symbol_count_fmax),
                      ('fmov32', symbol_count_fmov32)]:
        n_words = args.levels * 2  # 2 u16 words per u32 element
        buf = np.zeros(height * width * n_words, dtype=np.uint32)
        simulator.memcpy_d2h(buf, sym, 0, 0, width, height, n_words,
                            streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                            order=MemcpyOrder.COL_MAJOR, nonblock=False)
        hwl = oned_to_hwl_colmajor(width, height, n_words, buf, np.uint16)
        pe00 = hwl[0, 0, :]
        vals = np.zeros(args.levels, dtype=np.uint32)
        for lv in range(args.levels):
            lo = int(pe00[lv * 2])
            hi = int(pe00[lv * 2 + 1])
            vals[lv] = lo | (hi << 16)
        roofline_counters[name] = vals

    ###########################################################
    # Convergence
    print("9. Checking convergence...")
    ###########################################################
    # Copy rho_history after counter is available to minimize data transfer
    rho_history_array, actual_iterations = copy_rho_history(height, width, memcpy_dtype, memcpy_order, simulator, symbol_rho_history, counter_rho_check, args)
    if actual_iterations > 0:
        device_solver.grids[0]['rho_up'] = rho_history_array[actual_iterations - 1]
    else:
        device_solver.grids[0]['rho_up'] = 0.0

    device_rh = device_solver.grids[0]['rho_up']

    # # Print rho values after each iteration
    print("\n" + "="*60)
    print("Rho values after each iteration")
    print("="*60)
    if len(rho_history_array) > 0:
        if actual_iterations > 0:
            print(f"Total iterations performed: {actual_iterations}")
            print(f"{'Iteration':<12} {'|rho|_max':<20}")
            print("-" * 60)
            for i in range(actual_iterations):
                rho_val = rho_history_array[i]
                print(f"{i+1:<12} {rho_val:>19.6e}")
        else:
            print("No iterations were performed.")
    else:
        print("No iterations were performed.")

    print(f"[GMG] rho = |b-A*x|_inf = {device_rh:.6e}")
    # Use rel_tolerance^2 for convergence check (matching solve_iterative pattern)
    tolerance = device_solver.rel_tolerance
    print(f"  Tolerance = {tolerance:.6e}")
    converged = device_rh <= tolerance
    print(f"  Converged: {'Yes' if converged else 'No'}")

    ############################################################
    # Profiling
    print("10. Profiling...")
    ############################################################

    # counter_rho_check = 0
    # device_rh = 0.0
    first_vcycle_time_us = profiling(args,
        WORDS_PER_TIMESTAMP,
        device_solver, device_rh,
        counter_rho_check,
        simulator,
        timing_smooth_data, timing_residual_data, timing_restrict_data, timing_interp_data,
        timing_smooth_apply_data, timing_smooth_update_data,
        timing_interp_expand_z_data, timing_interp_bcast_data, timing_bcast_configure_data, timing_bcast_to_all_data, timing_interp_add_data,
        timing_spmv_total_data, timing_spmv_communication_data, timing_spmv_compute_data,
        timing_setup_data, timing_convergence_data,
        timing_total_start_end_data
        )  
    
    ###########################################################
    # Roofline — PE(0,0) op/traffic counts, accumulated across ALL V-cycles.
    # Convergence-check operators are EXCLUDED (gated by in_convergence flag
    # in kernel). Divide by device_iterations in analysis to get per-V-cycle.
    ###########################################################
    print("\n" + "=" * 120)
    print("ROOFLINE — PE(0,0) only; totals accumulated across ALL V-cycles (convergence-check ops excluded)")
    print("  Per-level 'active PEs' is the 2D PE count used at that multigrid level (for scaling).")
    print("  FLOP/Mem/Fab rows below are not summed over the wafer — they are one PE's counts.")
    print(f"  Divide counts by 'Device iterations to converge' ({max(int(counter_rho_check[0]), 1)}) to get per-V-cycle values.")
    print("=" * 120)

    # Raw counter dump — roofline computation is done by plots/roofline_analysis.py
    all_ops = ['fsub', 'fmac', 'fmul', 'fadd', 'fneg', 'fmov_mem', 'fmov_zero', 'fmax']
    size = args.zDim
    device_iterations = max(int(counter_rho_check[0]), 1)
    vcycle_time_us = timing_total_start_end_data[0]['time_send'] if timing_total_start_end_data else 0

    for level in range(args.levels):
        n_z = size >> level
        active_pes = (size // (2 ** level)) ** 2
        print(f"\nLevel {level} (nz={n_z}, active PEs: {active_pes:,}, stride: {2**level}):")
        print(f"{'Operation':<12} {'Count':>10}")
        print("-" * 25)
        for op in all_ops:
            count = int(roofline_counters[op][level])
            if count > 0:
                print(f"{op.upper():<12} {count:>10,}")
        fab_load = int(roofline_counters['fmov32'][level])
        if fab_load > 0:
            print(f"{'FMOV32':<12} {fab_load:>10,}")

    print(f"\nDevice iterations to converge: {device_iterations}")
    print(f"Total solver time (all iters): {vcycle_time_us:.3f} us")
    print(f"\nRun: python plots/roofline_analysis.py <response.txt> for full roofline analysis and plots.")
    print("=" * 120)

    print_configuration_summary(
        args,
        device_solver,
        device_rh,
        counter_rho_check,
        timing_total_start_end_data,
        first_vcycle_time_us=first_vcycle_time_us,
    )

    ###########################################################
    # Clean up
    print("11. Stopping simulator...")
    ###########################################################
    simulator.stop()
    if args.cmaddr is None:
        # Move simulation logs
        dst_log = Path(f"{logs_dir}/sim.log")
        src_log = Path("sim.log")
        if src_log.exists():
            shutil.move(src_log, dst_log)
        
        dst_trace = Path(f"{logs_dir}/simfab_traces")
        src_trace = Path("simfab_traces")
        if dst_trace.exists():
            shutil.rmtree(dst_trace)
        if src_trace.exists():
            shutil.move(src_trace, dst_trace)

if __name__ == "__main__":
    main()

