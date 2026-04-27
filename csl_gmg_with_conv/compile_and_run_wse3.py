"""
Compile and run GMG V-cycle for WSE3
Simple port of commands_vcycle_wse3.sh with flag support
Supports multiple problem sizes with artifact caching
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import glob
import shutil
import logging

# Cerebras SDK imports - only needed for compile/run, not for --check-memory
try:
    from cerebras.sdk.client import SdkCompiler
    from cerebras.sdk.client import SdkLauncher
    from cerebras.appliance import logger
    HAS_CEREBRAS = True
except ImportError:
    HAS_CEREBRAS = False
    SdkCompiler = SdkLauncher = None
    logger = None
# logging.basicConfig(level=logging.INFO)

# All output artifacts (out_dir_*/, shallow_*/, artifact_cache.json) live under BUILD_DIR
# to keep them out of app_path="./src" (avoids polluting the csl-file upload walk and
# ensures os.walk on src/ does not traverse extracted tarballs / nested cs_* dirs).
BUILD_DIR = "build"

# Cache file for storing artifact paths (lives under BUILD_DIR)
ARTIFACT_CACHE_FILE = os.path.join(BUILD_DIR, "artifact_cache.json")

###############################################################################
# Build commands (similar to shell script)
###############################################################################

def load_artifact_cache():
    """
    Loads the artifact cache from file. Returns a dictionary mapping out_path to list of artifact_paths.
    """
    if os.path.exists(ARTIFACT_CACHE_FILE):
        try:
            with open(ARTIFACT_CACHE_FILE, "r") as f:
                cache = json.load(f)
                # Convert to list format if old format (single artifact_path)
                # Use list() to avoid RuntimeError if cache is modified during iteration
                for key in list(cache.keys()):
                    if isinstance(cache[key], str):
                        cache[key] = [cache[key]]
                return cache
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load artifact cache: {e}")
            return {}
    return {}

def save_artifact_cache(cache):
    """
    Saves the artifact cache to file.
    """
    os.makedirs(os.path.dirname(ARTIFACT_CACHE_FILE) or ".", exist_ok=True)
    with open(ARTIFACT_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def get_cached_artifact(out_path):
    """
    Checks if there's a cached artifact for the given out_path.
    Returns the artifact_path if found and file exists, None otherwise.
    """
    cache = load_artifact_cache()
    if out_path in cache:
        # Check all artifact paths for this out_path
        for artifact_path in cache[out_path]:
            if os.path.exists(artifact_path):
                print(f"Found cached artifact for {out_path}: {artifact_path}")
                return artifact_path
        # If none exist, remove from cache
        del cache[out_path]
        save_artifact_cache(cache)
    return None

def add_artifact_to_cache(out_path, artifact_path):
    """
    Adds an artifact_path to the cache for the given out_path.
    """
    cache = load_artifact_cache()
    if out_path not in cache:
        cache[out_path] = []
    if artifact_path not in cache[out_path]:
        cache[out_path].append(artifact_path)
    save_artifact_cache(cache)


def get_out_name(size, levels, max_ite, pre_iter, post_iter, bottom_iter, suffix=""):
    """Logical out-dir name (no BUILD_DIR prefix). Written to response.txt and matched
    by plot regexes like r'out_dir_S(\\d+)x_'. Do NOT change this format."""
    return f"out_dir_S{size}x_L{levels}_M{max_ite}_P{pre_iter}_P{post_iter}_B{bottom_iter}{suffix}"


def get_out_path(size, levels, max_ite, pre_iter, post_iter, bottom_iter, suffix=""):
    """Filesystem path where this run's out_dir lives: build/out_dir_S*_L*_..."""
    return os.path.join(
        BUILD_DIR,
        get_out_name(size, levels, max_ite, pre_iter, post_iter, bottom_iter, suffix),
    )


def run_check_memory_for_outputs(script_dir=None, out_paths=None):
    """
    For each output folder in out_paths (or all out_dir_* if out_paths is None), find .tar.gz
    files, extract them, run ./check_memory_usage.sh on the extracted dir, and append output
    to response.txt. When running from this script after device jobs, pass the exact list of
    out_paths that were used so only those folders are processed.
    """
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    orig_cwd = os.getcwd()
    check_script = os.path.join(script_dir, "check_memory_usage.sh")
    if not os.path.exists(check_script):
        print(f"Warning: {check_script} not found, skipping memory checks")
        return

    try:
        os.chdir(script_dir)
    except OSError as e:
        print(f"Warning: could not chdir to {script_dir}: {e}")
        return

    try:
        if out_paths is not None:
            out_dirs = [p for p in out_paths if os.path.isdir(p)]
            if not out_dirs:
                print("None of the run output folders exist yet for memory check (jobs may still be running)")
                return
        else:
            out_dirs = sorted(glob.glob(os.path.join(BUILD_DIR, "out_dir_*")))
            if not out_dirs:
                print(f"No {BUILD_DIR}/out_dir_* folders found for memory check")
                return

        for out_path in out_dirs:
            if not os.path.isdir(out_path):
                continue
            # Parse size from out_path: out_dir_S4x_L2_... -> 4
            size_match = re.search(r"out_dir_S(\d+)x_", out_path)
            size = int(size_match.group(1)) if size_match else None
            artifact_name = f"out_vcycle{size}" if size is not None else "out_vcycle"

            tar_files = glob.glob(os.path.join(out_path, "*.tar.gz"))
            if not tar_files:
                print(f"  {out_path}: no .tar.gz found, skip")
                continue

            response_file = os.path.join(out_path, "response.txt")
            # IMPORTANT: extract to a scratch /tmp dir, NOT into out_path. Extracting
                # in-place created out_dir_*/cs_*/csl/out_dir_*/... recursion that
                # later polluted os.walk in SdkCompiler and caused HTTP 413 failures.
                # Scratch dir is cleaned up by the context manager after the check runs.
            import tempfile
            for tar_path in tar_files:
                tar_basename = os.path.basename(tar_path)
                extract_base = os.path.splitext(os.path.splitext(tar_basename)[0])[0]
                try:
                    # Extract under BUILD_DIR (not /tmp) so cs_readelf inside
                    # the Singularity container can see the ELFs.  /tmp is NOT
                    # bind-mounted into the container.
                    scratch = tempfile.mkdtemp(prefix="cslgmg_check_", dir=os.path.abspath(BUILD_DIR))
                    with tarfile.open(tar_path, "r:gz") as tf:
                        tf.extractall(scratch)
                        names = tf.getnames()
                    top_dirs = {n.split("/")[0] for n in names if "/" in n} | {n for n in names if not os.path.dirname(n)}
                    if len(top_dirs) == 1 and "/" not in list(top_dirs)[0]:
                        extract_dir = os.path.join(scratch, list(top_dirs)[0])
                    else:
                        extract_dir = scratch
                except Exception as e:
                    print(f"  {out_path}: failed to extract {tar_path}: {e}")
                    continue

                # Find the dir that has bin/ with .elf files (elf_dir for check_memory_usage.sh)
                elf_dir_candidates = [
                    extract_dir,
                    os.path.join(extract_dir, artifact_name),
                ]
                elf_dir = None
                for cand in elf_dir_candidates:
                    bin_dir = os.path.join(cand, "bin")
                    if os.path.isdir(bin_dir) and glob.glob(os.path.join(bin_dir, "*.elf")):
                        elf_dir = cand
                        break
                if elf_dir is None:
                    # Search recursively for bin/ with .elf
                    for root, dirs, _ in os.walk(extract_dir):
                        if "bin" in dirs:
                            b = os.path.join(root, "bin")
                            if glob.glob(os.path.join(b, "*.elf")):
                                elf_dir = root
                                break
                        if elf_dir:
                            break
                if elf_dir is None:
                    print(f"  {out_path}: no bin/*.elf found in extracted contents, skip")
                    continue

                print(f"  {out_path}: running check_memory_usage.sh {elf_dir}")
                try:
                    result = subprocess.run(
                        [check_script, elf_dir],
                        capture_output=True,
                        text=True,
                        cwd=script_dir,
                        timeout=60,
                    )
                    output = result.stdout or ""
                    if result.stderr:
                        output += f"\n{result.stderr}"
                    if result.returncode != 0:
                        output += f"\n(script exited with code {result.returncode})"
                except subprocess.TimeoutExpired:
                    output = "\n(check_memory_usage.sh timed out)\n"
                except Exception as e:
                    output = f"\n(check_memory_usage.sh failed: {e})\n"

                with open(response_file, "a") as f:
                    f.write(f"\n{'='*70}\n")
                    f.write("Memory usage check:\n")
                    f.write(f"{'='*70}\n")
                    f.write(output)
                    f.write("\n")
                print(f"  {out_path}: appended memory check to response.txt")
                # Remove scratch extraction so it never leaks back into the source tree.
                shutil.rmtree(scratch, ignore_errors=True)
    finally:
        try:
            os.chdir(orig_cwd)
        except OSError:
            pass


def write_run_info(out_path, size, levels, channels, max_ite, pre_iter, post_iter, bottom_iter, Compile_command, Run_command):
    """
    Writes compile and run information into a response.txt file in the specified output directory.
    """
    # write the folder name as well
    print(f"Compile command: {Compile_command}")
    print(f"Run command: {Run_command}")
    with open(f"./{out_path}/response.txt", "w") as f:
        # Write the logical out_dir name (not the build/ prefix) so plot regexes
        # like r'out_dir_S(\d+)x_' continue to match.
        f.write(f"Output directory: {os.path.basename(out_path)}\n")
        f.write(f"########################################################\n")
        f.write(f"Parameters: size={size}, levels={levels}, channels={channels}\n")
        f.write(f"Run parameters: max_ite={max_ite}, pre_iter={pre_iter}, post_iter={post_iter}, bottom_iter={bottom_iter}\n")
        f.write(f"Fabric dims: 762,1172\n")
        f.write(f"Compile command: {Compile_command}\n")
        f.write(f"Run command: {Run_command}\n")
        f.write(f"########################################################\n")

# COMPILE FUNCTION
def compile_app(layout_file, Compile_command, out_path):
    """
    Compiles the app using the SdkCompiler and returns artifact_path and compile_duration.
    First checks cache, and only compiles if no valid cached artifact exists.
    """
    # Check cache first
    cached_artifact = get_cached_artifact(out_path)
    if cached_artifact is not None:
        print(f"Using cached artifact, skipping compilation for {out_path}")
        return cached_artifact, 0.0  # No compile time for cached artifacts
    
    print("Compiling...")
    compile_start = time.time()
    # IMPORTANT: app_path points only at the CSL source tree (./src) so that
    # SdkCompiler's os.walk uploads only legitimate kernel sources (~40 .csl
    # files). Using app_path="." used to walk the whole working tree including
    # out_dir_*/ extractions (hundreds of thousands of .csl files), which blew
    # past the HTTP proxy 413 limit and caused every compile to be cancelled.
    with SdkCompiler(disable_version_check=True) as compiler:
        artifact_path = compiler.compile(
            app_path="./src",
            csl_main=layout_file,   # filename relative to app_path, e.g. "layout_gmg_vcycle.csl"
            options=Compile_command,
            out_path=out_path,
        )
    compile_end = time.time()
    compile_duration = compile_end - compile_start
    
    # Save to cache
    add_artifact_to_cache(out_path, artifact_path)
    
    return artifact_path, compile_duration

# RUN FUNCTION - fire and forget
def run_on_appliance(artifact_path, out_path, Run_command):
    """Submit job to queue - fires in background thread and returns immediately"""
    import threading
    
    def _run():
        with SdkLauncher(artifact_path, simulator=False, disable_version_check=True) as launcher:
            files_to_stage = [
                "cmd_parser.py",
                "run_gmg_vcycle.py",
                "util.py", 
                "python_gmg/gmgoscar.py",
            ]
            for file_to_stage in files_to_stage:
                launcher.stage(file_to_stage)
            run_start = time.time()
            response = launcher.run(Run_command)
            run_end = time.time()
            run_duration = run_end - run_start
            with open(f"./{out_path}/response.txt", "a") as f:
                f.write(f"\n{'='*70}\n")
                f.write("Run output:\n")
                f.write(f"{'='*70}\n")
                f.write(response)
                f.write(f"\nRun time (s): {run_duration:.6f}\n")
    
    # Fire job in background - don't wait
    threading.Thread(target=_run, daemon=False).start()

def process_on_device(size, levels, channels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter):
    """
    Fire compile + run job for a single problem.
    Submits job and returns immediately - queue handles execution.
    """
    import threading
    
    
    # csl_main is relative to app_path (which is "./src"), so just the filename here.
    layout_file = "layout_gmg_vcycle.csl"
    bsizemap = {
        4: 4,
        8: 8,
        16: 16,
        32: 32,
        64: 64,
        128: 128, # Let's keep a sweet spot of totalsize/4, so 1/4 th size is block size.
        256: 256,
        512: 256, # seems heuristically better when tested with different block sizes
    }
    BSIZE  = bsizemap[size]
    out_path = get_out_path(size, levels, max_ite, pre_iter, post_iter, bottom_iter)
    os.makedirs(out_path, exist_ok=True)

    INLINE_THRESHOLD = 256    # inline always good for speed, # code segment
    artifact_name = f"out_vcycle{size}"

    # prepare commands
    Compile_command = (
        f"--arch=wse3 --fabric-dims=762,1172 "
        f"--fabric-offsets=4,1 "
        f"--params=width:{size},height:{size},MAX_ZDIM:{size},LEVELS:{levels} "
        f"--params=BLOCK_SIZE:{BSIZE} --memcpy --channels={channels} "
        f"--width-west-buf=0 --width-east-buf=0 -o {artifact_name} "
        f"--llvm-option=--inline-threshold={INLINE_THRESHOLD} "
        f"--llvm-option=--unroll-threshold=256"
    )

    Run_command = (
        f"cs_python run_gmg_vcycle.py -m={size} -n={size} -k={size} "
        f"--latestlink {artifact_name} --channels={channels} "
        f"--width-west-buf=0 --width-east-buf=0 --zDim={size} --run-only "
        f"--levels={levels} --max-ite={max_ite} --blockSize={BSIZE} "
        f"--pre-iter={pre_iter} --post-iter={post_iter} --bottom-iter={bottom_iter} "
        f"--tolerance={abs_tolerance} "
        f"--cmaddr %CMADDR%"
    )

    write_run_info(out_path, size, levels, channels, max_ite, pre_iter, post_iter, bottom_iter, Compile_command, Run_command)
    
    # Fire compile + run job in background - returns immediately
    def _compile_and_run():
        artifact_path, compile_duration = compile_app(layout_file, Compile_command, out_path)
        with open(f"./{out_path}/response.txt", "a") as response_file:
            response_file.write(f"\nCompile time (s): {compile_duration:.6f}\n")
        run_on_appliance(artifact_path, out_path, Run_command)
    
    print(f"Firing compile+run job for {size}x{size}x{size}...")
    threading.Thread(target=_compile_and_run, daemon=False).start()
    print(f"Job submitted for {size}x{size}x{size} - moving to next problem")


import argparse
import time


def process_on_host(size, num_levels, verbose, max_iterations, abs_tolerance, pre_iter, post_iter, bottom_iter):
    """Benchmark a single problem size"""
    from python_gmg.gmgoscar import SimpleGMG as SimpleGMGOSCAR
    print(f"\nProcessing on host {size}x{size}x{size} grid with {num_levels} levels...")
    solver = SimpleGMGOSCAR(size, size, size, num_levels, verbose, abs_tolerance, pre_iter, post_iter, bottom_iter)
    start_time = time.time()
    rho_max, iterations = solver.solve_iterative(max_iterations)
    reltol = solver.rel_tolerance
    total_time = time.time() - start_time
    return rho_max, iterations, total_time, reltol

def main():
    # add a cmd line option called --only-host, --only-device, --host-and-device, --check-memory
    parser = argparse.ArgumentParser(description='Geometric Multigrid Solver')
    parser.add_argument('--only-host', action='store_true', default=False, help='Process on host')
    parser.add_argument('--only-device', action='store_true', default=False, help='Process on device')
    parser.add_argument('--host-and-device', action='store_true', default=False, help='Process on host and device')
    parser.add_argument('--check-memory', action='store_true', default=False,
                        help='Post-process: for each out_dir_*, extract .tar.gz, run check_memory_usage.sh, append to response.txt')
    args = parser.parse_args()

    # Require at least one of the options, otherwise print help and exit
    if not (args.only_host or args.only_device or args.host_and_device):
        parser.print_help()
        exit(1)
    
    # Test problems (size, levels, max_ite, pre_iter, post_iter, bottom_iter)
    # Format: (size, levels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter)
    # NOTE: MANUALLY DELETE FOLDER IF THERE IS SOME CHANGES IN THE SOURCE CODE AS IT WILL SKIP COMPILATION DUE TO CACHING.

#  2022  ls -d out_dir_S*x*_P4_P4_B100*
#  2023  ls -d out_dir_S*x*_P4_P4_B6*
#  2024  ls -d out_dir_S*x*_P6_P6_B6*
    problems = [
        # 6/6/100
        # ls -d out_dir_S*x*_P6_P6_B100 | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_6_6_100.txt
        (4, 2, 100, 1e-2, 6, 6, 100),   # DONE
        (8, 3, 100, 1e-2, 6, 6, 100),   # DONE
        (16, 4, 100, 1e-2, 6, 6, 100),   # DONE
        (32, 5, 100, 1e-2, 6, 6, 100),   # DONE
        (64, 6, 100, 1e-2, 6, 6, 100),   # DONE
        (128, 7, 100, 1e-2, 6, 6, 100),   # DONE
        (256, 8, 100, 1e-2, 6, 6, 100),   # DONE
        (512, 9, 100, 1e-2, 6, 6, 100),   # DONE

        # 4/4/100
        # ls -d out_dir_S*x*_P4_P4_B100 | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_4_4_100.txt
        (4, 2, 100, 1e-2, 4, 4, 100),   # DONE
        (8, 3, 100, 1e-2, 4, 4, 100),   # DONE
        (16, 4, 100, 1e-2, 4, 4, 100),   # DONE
        (32, 5, 100, 1e-2, 4, 4, 100),   # DONE
        (64, 6, 100, 1e-2, 4, 4, 100),   # DONE
        (128, 7, 100, 1e-2, 4, 4, 100),   # DONE
        (256, 8, 100, 1e-2, 4, 4, 100),   # DONE
        (512, 9, 100, 1e-2, 4, 4, 100),   # DONE

        # OSCAR matching problems
        # 4/4/6
        # ls -d out_dir_S*x*_P4_P4_B6 | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_4_4_6.txt
        (4, 2, 100, 1e-2, 4, 4, 6),   # DONE
        (8, 3, 100, 1e-2, 4, 4, 6),   # DONE
        (16, 4, 100, 1e-2, 4, 4, 6),   # DONE
        (32, 5, 100, 1e-2, 4, 4, 6),   # DONE
        (64, 6, 100, 1e-2, 4, 4, 6),   # DONE
        (128, 7, 100, 1e-2, 4, 4, 6),   # DONE
        (256, 8, 100, 1e-2, 4, 4, 6),   # DONE
        (512, 9, 100, 1e-2, 4, 4, 6),   # DONE

        # 6/6/6
        # ls -d out_dir_S*x*_P6_P6_B6 | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_6_6_6.txt
        (4, 2, 100, 1e-2, 6, 6, 6),   # DONE
        (8, 3, 100, 1e-2, 6, 6, 6),   # DONE
        (16, 4, 100, 1e-2, 6, 6, 6),   # DONE
        (32, 5, 100, 1e-2, 6, 6, 6),   # DONE
        (64, 6, 100, 1e-2, 6, 6, 6),   # DONE
        (128, 7, 100, 1e-2, 6, 6, 6),   # DONE
        (256, 8, 100, 1e-2, 6, 6, 6),   # DONE
        (512, 9, 100, 1e-2, 6, 6, 6),   # TEST: u=0, f=sin(2pi*x*hx) matching HPGMG

        # Deep vs shallow V cycle depth experiments
        # ls -d shallow_* | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_6_6_6_shallow.txt
        # Removed the smaller problems as we can't reduce the levels on these problems.
        
        (16, 2, 100, 1e-2, 6, 6, 6),   # DONE
        (32, 3, 100, 1e-2, 6, 6, 6),   # DONE
        (64, 4, 100, 1e-2, 6, 6, 6),   # DONE
        (128, 5, 100, 1e-2, 6, 6, 6),   # DONE
        (256, 6, 100, 1e-2, 6, 6, 6),   # DONE
        (512, 7, 100, 1e-2, 6, 6, 6),   # DONE

        # Unoptimized 6/6/6
        # Add "_unoptimized" suffix
        # ls -d out_dir_S*unoptimized* | sed 's/.*S\([0-9]*\)x.*/\1 &/' | sort -n | cut -d' ' -f2- | xargs -I{} cat {}/response.txt > all_responses_6_6_6_unoptimized.txt
        # (4, 2, 100, 1e-2, 6, 6, 6),   # Tiny problem
        # (8, 3, 100, 1e-2, 6, 6, 6),   # Tiny problem
        # (16, 4, 100, 1e-2, 6, 6, 6),   # Small problem
        # (32, 5, 100, 1e-2, 6, 6, 6),   # Small problem
        # (64, 6, 100, 1e-2, 6, 6, 6),   # Medium problem
        # (128, 7, 100, 1e-2, 6, 6, 6),   # Large problem
        # (256, 8, 100, 1e-2, 6, 6, 6),   # Very large
        # (512, 9, 100, 1e-2, 6, 6, 6),   # Very large       
    ]
    verbose = False    
    results = []
    
    # Process host runs sequentially (if needed)
    if args.only_host or args.host_and_device:
        for size, levels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter in problems:
            try:
                rho_max, iterations, total_time, reltol = process_on_host(size, levels, verbose, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter)
                results.append((size, size, size, rho_max, abs_tolerance, iterations, total_time, reltol))
            except Exception as e:
                print(f"Failed for size={size}, levels={levels}: {e}")
                results.append((size, size, size, float('inf'), float('inf'), 0, 0, float('inf')))
        
                # Summary
        print("\n" + "=" * 70)
        print("Summary of all runs on HOST")
        print("=" * 70)
        header = (
            f"{'Grid Size':<15} | "
            f"{'|rho|_inf':>11} | "
            f"{'(Rel Tolerance)':>17} | "
            f"{'Absolute Tol.':>14} | "
            f"{'Iters':>7} | "
            f"{'Time (s)':>9} | "
        )
        print(header)
        print("-" * len(header))
        
        for problem_idx, (nx, ny, nz, rho_max, abs_tolerance, iterations, total_time, reltol) in enumerate(results):
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
    
    # Process device runs
    if args.only_device or args.host_and_device:
        run_out_paths = [
            get_out_path(size, levels, max_ite, pre_iter, post_iter, bottom_iter)
            for (size, levels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter) in problems
        ]
        for size, levels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter in problems:
            if size > 16:
                channels = 16
            else:
                channels = size
            try:
                process_on_device(size, levels, channels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter)
            except Exception as e:
                print(f"Failed for size={size}, levels={levels}: {e}")
        run_check_memory_for_outputs(out_paths=run_out_paths)

if __name__ == "__main__":
    main()
