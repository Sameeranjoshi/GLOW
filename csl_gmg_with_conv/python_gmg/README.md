# Geometric Multigrid Solver - Python Implementation

This directory contains a simple, clean Python implementation of a Geometric Multigrid (GMG) solver for the 3D Poisson equation.

## Simple GMG Implementation

The `gmg.py` file contains the cleanest, most straightforward GMG implementation without any complex considerations for distributed computing, ghost cells, or hardware-specific optimizations.

### Key Features
- **Easy to understand** - clean, straightforward code
- **Easy to modify** - simple parameter changes
- **Pure CPU implementation** - no GPU considerations
- **Simple grid hierarchy** - just divide by 2 each level
- **Standard multigrid components** - restriction, interpolation, smoothing
- **Clean separation** - each component does one thing
- **Easy to understand** - minimal complexity

### Algorithm Details

#### Key Parameters
- **7-point Poisson stencil**: α = -6, β = 1
- **Jacobi relaxation coefficient**: 0.5 (configurable)
- **Pre-smooth iterations**: 6 (configurable)
- **Post-smooth iterations**: 6 (configurable)
- **Bottom solver iterations**: 100 (configurable)
- **Convergence tolerance**: 1e-6 (configurable)

#### Multigrid Components
- **Operator**: 7-point finite difference stencil for 3D Poisson equation
- **Smoother**: Jacobi relaxation with configurable coefficient
- **Restriction**: Full weighting (averages 8 fine points to 1 coarse point)
- **Interpolation**: Linear interpolation (copies coarse value to 8 fine points)
- **Cycle**: V-cycle multigrid

#### Grid Hierarchy
- Automatically creates multigrid hierarchy with 2:1 coarsening
- Default: 4 levels (configurable)
- Minimum grid size: 1 point per level

### Usage
#### Dependencies

- Python 3.7+
- NumPy


#### Basic Usage

```bash
python3 gmg.py
```

#### Example: Simple Test
```bash
python3 gmg.py -s 16,16,16 -l 3 -n 10 --tolerance 1e-3 -v
```

Options:
- `-s, --size`: Grid size as nx,ny,nz (default: 32,32,32)
- `-l, --levels`: Number of multigrid levels (default: 4)
- `-n, --max_iter`: Maximum number of iterations (default: 20)
- `-v, --verbose`: Print detailed level information
- `--tolerance`: Convergence tolerance (default: 1e-6)
- `--pre-iter`: Number of pre-smoothing iterations (default: 6)
- `--post-iter`: Number of post-smoothing iterations (default: 6)
- `--bottom-iter`: Number of bottom solver iterations (default: 100)



### Implementation Details

#### File Structure
- `gmg.py`: Main simple GMG solver implementation
- `README.md`: This documentation
- `requirements.txt`: Python dependencies

#### Core Classes
- `SimpleGMG`: Main solver class containing all multigrid operations

#### Key Methods
- `apply_operator()`: Applies 7-point Poisson stencil
- `jacobi_smooth()`: Jacobi relaxation smoother
- `restrict()`: Full weighting restriction
- `interpolate()`: Linear interpolation
- `v_cycle()`: Complete V-cycle multigrid
- `solve()`: Main solver loop with convergence checking

#### Right-Hand Side
The solver initializes the right-hand side with a simple test function:
```
f = sin(πx) * sin(πy) * sin(πz)
```

### Example Output

```
$ python3 gmg.py -s 16,16,16 -l 3 -n 10 --tolerance 1e-3 -v
============================================================
Simple Geometric Multigrid Solver
============================================================

Multigrid Level Hierarchy:
==================================================
Level  0:  16 x  16 x  16 (h = 0.066667, points = 4,096)
Level  1:   8 x   8 x   8 (h = 0.142857, points = 512)
Level  2:   4 x   4 x   4 (h = 0.333333, points = 64)
==================================================

Data Structures at Each Level:
========================================================================================================================
Level  Grid Size    u (solution)    f (RHS)         r (residual)    Au (operator)   Stencil             
------------------------------------------------------------------------------------------------------------------------
0      16×16×16     (16,16,16)      (16,16,16)      (16,16,16)      (16,16,16)      7-point Poisson     
1      8×8×8        (8,8,8)         (8,8,8)         (8,8,8)         (8,8,8)         7-point Poisson     
2      4×4×4        (4,4,4)         (4,4,4)         (4,4,4)         (4,4,4)         7-point Poisson     
------------------------------------------------------------------------------------------------------------------------
Legend:
  u     = Solution vector (unknown)
  f     = Right-hand side vector (source term)
  r     = Residual vector (f - Au)
  Au    = Operator applied to solution (A*u)
  Stencil = Finite difference stencil used
========================================================================================================================

Stencil Coefficients:
----------------------------------------
  α (center) = -6.0
  β (neighbors) = 1.0
  7-point stencil: α*u[i,j,k] + β*(u[i±1,j,k] + u[i,j±1,k] + u[i,j,k±1])
----------------------------------------

Memory Usage Summary:
--------------------------------------------------
  Level 0: 4,096 points × 4 arrays = 0.12 MB
  Level 1: 512 points × 4 arrays = 0.02 MB
  Level 2: 64 points × 4 arrays = 0.00 MB
  Total: 4,672 points × 4 arrays = 0.14 MB
--------------------------------------------------

Initialization Parameters:
==================================================
Grid dimensions: 16 × 16 × 16
Number of levels: 3
Grid spacing (h): 0.066667

Stencil Coefficients:
  α (center coefficient): -6.0
  β (neighbor coefficient): 1.0
  7-point stencil: α*u[i,j,k] + β*(u[i±1,j,k] + u[i,j±1,k] + u[i,j,k±1])
  Grid spacing scaling: h² = 0.004444 (used in residual computation)

Relaxation Parameters:
  Jacobi coefficient: 0.500000
  Pre-smooth iterations: 6
  Post-smooth iterations: 6
  Bottom solver iterations: 100

Convergence Parameters:
  Tolerance: 1.00e-03

Right-hand Side Initialization:
  Function: sin(πx) * sin(πy) * sin(πz)
  Domain: [0,1] × [0,1] × [0,1]
  Grid points: 16 × 16 × 16

Boundary Conditions:
  Dirichlet boundary conditions (u = 0 on boundary)
  Interior points only for stencil application
Initial values:
  x (solution) - 0.0:
  f (RHS) - sin function values:
  r (residual) - 0.0:
  Au (operator) - 0.0:

==================================================

Starting GMG solve with 10 max iterations
Grid size: 16x16x16, Levels: 3
Tolerance: 0.001
--------------------------------------------------
Iteration  1: Residual = 6.459505e-01, Time = 0.2974s
Iteration  2: Residual = 2.088473e-01, Time = 0.2923s
Iteration  3: Residual = 7.883476e-02, Time = 0.3311s
Iteration  4: Residual = 2.939147e-02, Time = 0.3409s
Iteration  5: Residual = 1.115661e-02, Time = 0.2926s
Iteration  6: Residual = 4.255705e-03, Time = 0.2926s
Iteration  7: Residual = 1.630359e-03, Time = 0.2897s
Iteration  8: Residual = 6.257447e-04, Time = 0.2926s
--------------------------------------------------
After 8 iterations
Final residual: 6.257447e-04
Tolerance: 1.000000e-03
Converged: Yes
Total time: 2.4884s
Final residual: 6.257447e-04
Iterations: 8

```
