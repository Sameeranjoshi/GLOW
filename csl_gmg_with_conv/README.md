# GLOW — Geometric Multigrid (GMG) V-Cycle Solver on Cerebras WSE-3

GLOW is a device-resident Geometric Multigrid (GMG) V-cycle solver for the 3D
Poisson equation, running entirely on the Cerebras WSE-3 wafer-scale engine.
This repo is the artifact for the GLOW paper of a multigrid algorithm on WSE-3.

## This README is a self-contained reproduction guide: prerequisites, build,

device runs, and figure regeneration, reproduces all the tables and figures from the paper.

## 1. What is in this repository

### Algorithm

GLOW solves `A·x = b` for the discrete 3D Poisson operator on an `N×N×N`
uniform grid with Dirichlet boundaries using a V-cycle:

```
finest ─► smooth (Jacobi) ─► residual ─► restrict ─┐
                                                    ▼
                            ... recurse on coarser levels ...
                                                    ▲
   ◄─ smooth ◄─ correct ◄─ interpolate ◄────────────┘
```

Per level, the kernel performs a 7-point stencil SpMV (Laplacian),
weighted-Jacobi smoothing, residual computation, restriction (full-weighting),
prolongation, and a windowed allreduce for the residual norm.
The coarsest level is solved with additional Jacobi sweeps (`bottom_iter`).

The whole V-cycle, including the coarse solve, executes on-device as a
single CSL state machine (28 states) — there is no host-controlled outer
loop. A strided *distributed mapping* keeps every level on the same PE
rectangle and a *forwarding* communication optimization reduces the
per-level halo cost.

### Repository layout

```
GLOW/
├── README.md                       # this file
├── requirements.txt                # Python deps for host + plots
└── csl_gmg_with_conv/              # V-cycle solver (this artifact)
    ├── compile_and_run_wse3.py     # entry point: compile + run + cache
    ├── run_gmg_vcycle.py           # device execution orchestration
    ├── cmd_parser.py               # CLI argument parsing
    ├── util.py                     # host data-layout helpers
    ├── check_memory_usage.sh       # ELF code/data analysis (Table 5)
    ├── clean.sh                    # remove build artifacts
    ├── commands_vcycle_wse3.sh     # quick smoke-test commands
    │
    ├── src/                        # CSL device code
    │   ├── kernel_gmg_vcycle.csl   # 28-state V-cycle kernel
    │   ├── layout_gmg_vcycle.csl   # PE grid layout + colors
    │   ├── blas.csl                # linear algebra utilities
    │   ├── timer_modified.csl      # hardware timestamp helpers
    │   └── modified_csl_lib_hops/  # forked stencil + allreduce w/ hops
    │
    ├── python_gmg/                 # CPU reference solver (validation)
    │   └── gmgoscar.py
    │
    ├── plots/                      # figure / table generators
    │   ├── GENERATEFIGURES.sh      # one-shot driver for all figures
    │   ├── plot_gmg_performance.py # per-operation timing analysis
    │   ├── h200_vs_cs3.py          # Fig. 5 — GH200 vs CS-3 bar chart
    │   ├── print_512_table.py      # Table 4 — 512³ comparison
    │   ├── memory_utilization_table.py  # Table 5 — memory per level
    │   ├── v_vs_w_cycle.py         # V vs W cycle comparison
    │   ├── roofline_analysis.py    # Fig. 9 — roofline plot
    │   ├── gpu_numbers.txt         # GH200 baseline numbers (input)
    │   └── wse_numbers.txt         # WSE-3 baseline numbers (input)
    │
    └── build/                      # run outputs (created by runs, gitignored)
        ├── out_dir_S*x_*/          # per-config response.txt + ELF tarballs
        └── all_responses_*.txt     # aggregated per-config logs
```

---

## 2. Prerequisites

### Hardware

- Cerebras CS-3 appliance (required for device runs)
- x86_64 Linux host for compilation and job dispatch
- Tested on Rocky Linux 8.10 (`4.18.0-553.53.1.el8_10.x86_64`)

### Software


| Component           | Version                                                                        |
| ------------------- | ------------------------------------------------------------------------------ |
| Cerebras SDK        | **1.4.0** (release 2.5.0 — `cerebras-sdk==2.5.0`, `cerebras-appliance==2.5.0`) |
| SDK container image | `sdk-cbcore-202505010205-2-ef181f81.sif`                                       |
| `cslc`, `cs_python` | bundled with the SDK (no separate install)                                     |
| Python              | **3.8.17** (3.8+ required)                                                     |


The Cerebras SDK is gated — request access at
[https://sdk.cerebras.net/installation-guide](https://sdk.cerebras.net/installation-guide) and follow the installer
shipped with the SDK tarball (`Cerebras-SDK-1.4.0.tar.gz`).

### Time budget


| Phase                          | Estimate                                   |
| ------------------------------ | ------------------------------------------ |
| Setup (SDK + venv + deps)      | 30–45 min                                  |
| Compile + CS-3 sweep (4³–512³) | 45–90 min (dominated by appliance staging) |
| Figure regeneration            | 2–5 min                                    |
| **Total**                      | ~80–140 min                                |


---

## 3. Setup

```bash
# 1. Activate the SDK's pinned Python environment
source /path/to/sdk_venv/bin/activate
pip install -r /path/to/sdk/req.txt        # cerebras-sdk + cerebras-appliance

# 2. Install this repo's host-side and plotting dependencies
git clone https://github.com/Sameeranjoshi/GLOW.git
cd GLOW/
pip install -r requirements.txt
```

`requirements.txt` pins `numpy==1.24.4`, `matplotlib==3.7.5`,
`scipy==1.13.1`, `pandas==2.0.3`, and (optional) `numba==0.58.1`. All are
compatible with the SDK 1.4.0 venv.


| Package            | Used by                              |
| ------------------ | ------------------------------------ |
| `numpy`            | host solver, plots                   |
| `matplotlib`       | plots                                |
| `scipy`            | host reference solver                |
| `pandas`           | plot scripts                         |
| `numba` (optional) | accelerates `python_gmg/gmgoscar.py` |


---

## 4. Reproducing the figures end-to-end

All paths below are relative to `csl_gmg_with_conv/`.

### Step 1 — choose problem sizes

Edit the `configs` list in `compile_and_run_wse3.py`. The full sweep used in
the paper covers `4³ … 512³`:

```python
# (size, levels, max_ite, abs_tolerance, pre_iter, post_iter, bottom_iter)
configs = [
    (4,   2, 100, 1e-2, 6, 6,   6),
    (8,   3, 100, 1e-2, 6, 6,   6),
    (16,  4, 100, 1e-2, 6, 6,   6),
    (32,  5, 100, 1e-2, 6, 6,   6),
    (64,  6, 100, 1e-2, 6, 6,   6),
    (128, 7, 100, 1e-2, 6, 6,   6),
    (256, 8, 100, 1e-2, 6, 6,   6),
    (512, 9, 100, 1e-2, 6, 6,   6),
]
```

**Parameter guide:**


| Field                    | Meaning                                                           |
| ------------------------ | ----------------------------------------------------------------- |
| `size`                   | grid is `size³`; PE rectangle is `size × size`                    |
| `levels`                 | multigrid depth (deep = `log₂(size)+1`, shallow = `log₂(size)−1`) |
| `max_ite`                | maximum V-cycle iterations                                        |
| `abs_tolerance`          | convergence tolerance                                             |
| `pre_iter` / `post_iter` | Jacobi sweeps before / after recursion                            |
| `bottom_iter`            | Jacobi sweeps on the coarsest level                               |


To reproduce **all** paper figures and tables, also run the auxiliary sweeps
referenced by `GENERATEFIGURES.sh` (different smoother counts and a shallow
variant). The simplest way is to run the script once per `(pre, post, bottom)`
combination it expects:


| Sweep                        | `(pre, post, bottom)`                | Used by               |
| ---------------------------- | ------------------------------------ | --------------------- |
| Primary                      | `(6, 6, 6)`                          | Figs. 6–9, Tables 4–5 |
| Heavy bottom                 | `(6, 6, 100)`                        | comparison            |
| Light pre/post               | `(4, 4, 6)`                          | comparison            |
| Heavy bottom, light pre/post | `(4, 4, 100)`                        | comparison            |
| Shallow V                    | `(6, 6, 6)`, levels = `log₂(size)−1` | shallow vs deep study |


### Step 2 — compile + run on CS-3

```bash
python compile_and_run_wse3.py --only-device
```

Each entry in `configs` produces:

```
build/out_dir_S{size}x_L{levels}_M{max_ite}_P{pre}_P{post}_B{bottom}/
├── response.txt    # compile + run log: timings, convergence, memory
├── *.tar.gz        # compiled ELF artifacts (used for memory analysis)
└── cs_<hash>/      # extracted compile artifacts
```

Shallow runs land in `build/shallow_*/`.

**Artifact cache:** `artifact_cache.json` maps each output directory to its
compiled artifact. Subsequent invocations skip compilation for cached
configurations. Delete the JSON to force a clean rebuild.

Note: This step takes the longest, mainly because the data is staged onto the appliance and also depends on the availability of allocation of a node on a shared cluster.

### Step 3 — regenerate all figures

```bash
cd plots/
bash GENERATEFIGURES.sh
```

The script runs the full pipeline:

1. **Memory analysis** — appends ELF code/data sizes to each `response.txt`
  via `check_memory_usage.sh`.
2. **Aggregation** — concatenates `build/out_dir_*/response.txt` per
  configuration into `build/all_responses_*.txt`.
3. **Performance parsing** — runs `plot_gmg_performance.py` on each
  aggregate, producing `out_*.txt` summaries plus the per-operation
   timing PNGs (`spmv_internal.png`, `interpolation_internal.png`,
   `per_operation_timing_*.png`).
4. **Comparison plots/tables**
  - `h200_vs_cs3.py` → `hpgmg_speedup_barplot.pdf` (Fig. 5)
  - `memory_utilization_table.py` → Table 5
  - `print_512_table.py` → Table 4
5. **Standalone plots** — `v_vs_w_cycle.py` (V vs W comparison).
6. **Roofline** — `roofline_analysis.py` on the 512³ 6/6/6 sample →
  `roofline_plot.png` (Fig. 9).

All figures and tables land in `plots/`. A combined log is written to
`plots/GENERATEFIGURES.log`.

---

## 5. Expected results

After 3 warmup iterations followed by timed runs, the pipeline should
reproduce, within the tolerances reported in the paper:


| Output                                     | Expected behavior                                                                      |
| ------------------------------------------ | -------------------------------------------------------------------------------------- |
| Fig. 5 (`hpgmg_speedup_barplot.pdf`)       | GLOW above the GH200 baseline at every size; up to ~25× single-V-cycle speedup at 512³ |
| Fig. 6 (per-level operator timings)        | Smooth/residual/restriction follow the U-shape predicted by the communication model    |
| Figs. 7–8 (SpMV / interpolation internals) | Per-operator breakdowns matching the paper within <1%                                  |
| Fig. 9 (`roofline_plot.png`)               | Finest level near 43% of per-PE peak; coarsest near 1%                                 |
| Table 4                                    | GH200 vs WSE-3 entries within ~1%                                                      |
| Table 5                                    | Reproduces exactly — derived from compile-time ELF inspection                          |


---

## 7. Troubleshooting

- `**cs_python` / `cslc` not found** — the SDK venv is not active. Re-run
`source /path/to/sdk_venv/bin/activate`.
- **Stale ELFs after editing `src/`** — delete `artifact_cache.json` (or the
affected `build/out_dir_*/`) to force a recompile.
- **Empty `all_responses_*.txt`** — `GENERATEFIGURES.sh` skipped a sweep:
check that the corresponding `(pre, post, bottom)` configs from §4 Step 1
actually ran and produced `response.txt` files.
- **Plot script reports missing `gpu_numbers.txt` / `wse_numbers.txt`** —
both are checked into `plots/`; ensure you `cd plots/` before running
comparison scripts.

