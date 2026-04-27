#!/usr/bin/env python3
"""
Benchmark script for the GMG solver
Demonstrates the Python implementation with various problem sizes
"""

import argparse
# from gmg import SimpleGMG
from gmgpython import SimpleGMG as SimpleGMGPYTHON
import time


def benchmark_problem(nx, ny, nz, num_levels, verbose, abs_tolerance, pre_iter, post_iter, bottom_iter, max_iterations):
    """Benchmark a single problem size"""
    print(f"\nBenchmarking {nx}x{ny}x{nz} grid with {num_levels} levels...")
    # solver = SimpleGMG(nx, ny, nz, num_levels, verbose, tolerance, pre_iter, post_iter, bottom_iter)
    solver = SimpleGMGPYTHON(nx, ny, nz, num_levels, verbose, abs_tolerance, pre_iter, post_iter, bottom_iter)
    start_time = time.time()
    # residual, iterations = solver.solve(max_iterations)
    rho_max, iterations = solver.solve_iterative(max_iterations)
    reltol = solver.rel_tolerance
    total_time = time.time() - start_time
    return rho_max, iterations, total_time, reltol


def main():
    """Run benchmark tests"""
    print("=" * 70)
    print("Simple Geometric Multigrid Solver - Python Implementation Benchmark")
    print("=" * 70)
    parser = argparse.ArgumentParser(description='Geometric Multigrid Solver')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print detailed level information')
    args = parser.parse_args()
    
    # Test problems (nx, ny, nz, levels, max_iterations, abs_tolerance, pre_iter, post_iter, bottom_iter)
    problems = [
        (4, 4, 4, 2, 100, 1e-4, 6, 6, 10),   # Tiny problem (7 host, 8 device)
        (8, 8, 8, 3, 100, 1e-4, 6, 6, 10),   # Tiny problem (7 host, 8 device)
        (16, 16, 16, 4, 100, 1e-4, 6, 6, 10),   # Small problem
        (32, 32, 32, 5, 100, 1e-4, 6, 6, 10),   # Small problem
        (64, 64, 64, 6, 100, 1e-4, 6, 6, 10),   # Medium problem - increased smoothing and bottom solver
        (128, 128, 128, 7, 100, 1e-4, 6, 6, 10),   # Large problem - increased smoothing and bottom solver
        (256, 256, 256, 8, 100, 1e-4, 6, 6, 50),   # Very large: reduce pre/post (40 was over-smoothing), massively increase bottom solver
        (512, 512, 512, 9, 100, 1e-4, 6, 6, 90),   # Very large: reduce pre/post (40 was over-smoothing), massively increase bottom solver

        # experiments
        # (256, 256, 256, 6, 200, 1e-4, 6, 6, 1),
        # (256, 256, 256, 6, 200, 1e-4, 6, 6, 5),
        # (256, 256, 256, 6, 200, 1e-4, 6, 6, 11),
        # (256, 256, 256, 6, 200, 1e-4, 6, 6, 12),
        # (256, 256, 256, 6, 200, 1e-4, 6, 6, 13),
        # (256, 256, 256, 6, 200, 1e-4, 6, 6, 15),
        # (256, 256, 256, 6, 200, 1e-4, 6, 6, 25),
        # (256, 256, 256, 6, 200, 1e-4, 6, 6, 50),

        # # 32 x 32 x 32 problem
        # (32, 32, 32, 5, 5, 1e-5, 6, 6, 10),
        # (32, 32, 32, 5, 6, 1e-5, 6, 6, 5),
        # (32, 32, 32, 5, 10, 1e-5, 6, 6, 6),
        # (32, 32, 32, 5, 100, 1e-5, 6, 6, 10),

        # # 64
        # (64, 64, 64, 6, 100, 1e-5, 6, 6, 10),

        # 128x128x128 problem(works and finishes and correct on H and D)
        # (128, 128, 128, 5, 100, 1e-4, 6, 6, 10),
        # (128, 128, 128, 5, 100, 1e-4, 6, 6, 50),

        # 256x256x256 problem
        # (256, 256, 256, 6, 1, 1e-4, 1, 1, 1),
        # (256, 256, 256, 6, 1, 1e-4, 6, 6, 1),
        # (256, 256, 256, 6, 1, 1e-4, 6, 6, 10),
        # (256, 256, 256, 6, 20, 1e-4, 6, 6, 10),
        # (256, 256, 256, 6, 50, 1e-4, 6, 6, 10),

        # 512x512x512 problem
        # (512, 512, 512, 7, 1, 1e-6, 1, 1, 1),
        # (512, 512, 512, 7, 1, 1e-6, 6, 6, 30),
        # (512, 512, 512, 7, 50, 1e-4, 6, 6, 30),
    ]
    
    results = []
    
    for nx, ny, nz, levels, max_iter, abs_tolerance, pre_iter, post_iter, bottom_iter in problems:
        try:
            rho_max, iterations, total_time, reltol = benchmark_problem(nx, ny, nz, levels, args.verbose, abs_tolerance, pre_iter, post_iter, bottom_iter, max_iter)
            results.append((nx, ny, nz, rho_max, iterations, total_time, reltol))
        except Exception as e:
            print(f"Failed: {e}")
            results.append((nx, ny, nz, float('inf'), 0, 0, float('inf')))
    
    # Summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    header = (
        f"{'Grid Size':<15} | "
        f"{'|rho|_inf':>11} | "
        f"{'(Rel Tolerance)^2':>17} | "
        f"{'Absolute Tol.':>14} | "
        f"{'Iters':>7} | "
        f"{'Time (s)':>9} | "
        f"{'Converged':>9}"
    )
    print(header)
    print("-" * len(header))
    
    for problem_idx, (nx, ny, nz, rho_max, iterations, total_time, reltol) in enumerate(results):
        grid_size = f"{nx}x{ny}x{nz}"
        tolerance = reltol
        converged = "Yes" if rho_max < tolerance else "No"
        print(
            f"{grid_size:<15} | "
            f"{rho_max:>11.2e} | "
            f"{tolerance:>17.2e} | "
            f"{abs_tolerance:>14.2e} | "
            f"{iterations:>7} | "
            f"{total_time:>9.4f} | "
            f"{converged:>9}"
        )

if __name__ == "__main__":
    main()