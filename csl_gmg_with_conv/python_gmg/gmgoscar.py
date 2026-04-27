#!/usr/bin/env python3
"""
Simplest Geometric Multigrid Solver for 3D Poisson equation
Optimized CPU implementation with optional Numba JIT compilation

Performance: 
- With Numba: 10-100x faster (JIT-compiled loops)
- Without Numba: 2-5x faster (vectorized NumPy operations)
- Original: Pure Python loops (baseline)

To install Numba for maximum speed:
    pip install numba

Run: simplest example
python3 gmg.py -s 16,16,16 -l 3 -n 10 --tolerance 1e-3 -v
"""

import numpy as np
import time
from typing import Tuple, List

# Try to import Numba for JIT compilation (optional dependency)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define dummy decorator if Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(x):
        return range(x)

DTYPE = np.float32

# ============================================================================
# Optimized helper functions with Numba JIT compilation
# ============================================================================
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True)
    def _apply_operator_jit(u, Au, nx, ny, nz, hx, hy, hz, beta):
        """JIT-compiled 7-point Laplacian operator"""
        hx2_inv = 1.0 / (hx * hx)
        hy2_inv = 1.0 / (hy * hy)
        hz2_inv = 1.0 / (hz * hz)
        
        for i in prange(nx):
            for j in prange(ny):
                for k in prange(nz):
                    val_x = -2.0 * u[i, j, k]
                    val_y = -2.0 * u[i, j, k]
                    val_z = -2.0 * u[i, j, k]
                    
                    # west/east
                    if i > 0:
                        val_x += beta * u[i-1, j, k]
                    if i < nx-1:
                        val_x += beta * u[i+1, j, k]
                    if i == 0:     
                        val_x += beta * -1.0 * u[0, j, k]
                    if i == nx-1:     
                        val_x += beta * -1.0 * u[nx-1, j, k]
                    val_x *= hx2_inv
                    
                    # south/north
                    if j > 0:
                        val_y += beta * u[i, j-1, k]
                    if j < ny-1:
                        val_y += beta * u[i, j+1, k]
                    if j == 0:     
                        val_y += beta * -1.0 * u[i, 0, k]
                    if j == ny-1:     
                        val_y += beta * -1.0 * u[i, ny-1, k]
                    val_y *= hy2_inv
                    
                    # bottom/top
                    if k > 0:
                        val_z += beta * u[i, j, k-1]
                    if k < nz-1:
                        val_z += beta * u[i, j, k+1]
                    if k == 0:     
                        val_z += beta * -1.0 * u[i, j, 0]
                    if k == nz-1:     
                        val_z += beta * -1.0 * u[i, j, nz-1]
                    val_z *= hz2_inv
                    
                    Au[i, j, k] = val_x + val_y + val_z
    
    @jit(nopython=True, parallel=True, fastmath=True)
    def _restrict_jit(fine_r, coarse_f, Xdim, Ydim, Zdim):
        """JIT-compiled restriction operator"""
        inv_8 = 1.0 / 8.0
        for i in prange(Xdim):
            for j in prange(Ydim):
                for k in prange(Zdim):
                    f_i = 2 * i
                    f_j = 2 * j
                    f_k = 2 * k
                    total_val = (
                        fine_r[f_i, f_j, f_k] +
                        fine_r[f_i+1, f_j, f_k] +
                        fine_r[f_i, f_j+1, f_k] +
                        fine_r[f_i+1, f_j+1, f_k] +
                        fine_r[f_i, f_j, f_k+1] +
                        fine_r[f_i+1, f_j, f_k+1] +
                        fine_r[f_i, f_j+1, f_k+1] +
                        fine_r[f_i+1, f_j+1, f_k+1]
                    )
                    coarse_f[i, j, k] = total_val * inv_8
    
    @jit(nopython=True, parallel=True, fastmath=True)
    def _interpolate_jit(coarse_u, fine_u, coarse_nx, coarse_ny, coarse_nz):
        """JIT-compiled interpolation operator"""
        for i in prange(coarse_nx):
            for j in prange(coarse_ny):
                for k in prange(coarse_nz):
                    fi, fj, fk = 2*i, 2*j, 2*k
                    coarse_val = coarse_u[i, j, k]
                    
                    fine_u[fi,   fj,   fk] += coarse_val
                    fine_u[fi+1, fj,   fk] += coarse_val
                    fine_u[fi,   fj+1, fk] += coarse_val
                    fine_u[fi+1, fj+1, fk] += coarse_val
                    fine_u[fi,   fj,   fk+1] += coarse_val
                    fine_u[fi+1, fj,   fk+1] += coarse_val
                    fine_u[fi,   fj+1, fk+1] += coarse_val
                    fine_u[fi+1, fj+1, fk+1] += coarse_val

class SimpleGMG:
    """Simplest possible GMG solver for 3D Poisson equation"""
    
    def __init__(self, nx: int, ny: int, nz: int, num_levels: int = 4, 
                verbose: bool = False, abs_tolerance: float = 1e-5, 
                pre_iter: int = 6, post_iter: int = 6, bottom_iter: int = 100):
        """
        Initialize the GMG solver
        
        Args:
            nx, ny, nz: Grid dimensions
            num_levels: Number of multigrid levels
            verbose: Whether to print detailed level information
            tolerance: Convergence tolerance
            pre_iter: Number of pre-smoothing iterations
            post_iter: Number of post-smoothing iterations
            bottom_iter: Number of bottom solver iterations
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.num_levels = num_levels
        self.verbose = verbose
        
        # Create grid hierarchy
        self.grids = self._create_grids()
        
        # Initialize solution x
        self._init_x(0.0)
        # Initialize RHS
        self._init_rhs()
        
        # Simple parameters
        self.omega = 2.0/3.0  # Jacobi relaxation parameter

        self.nrm_b = np.linalg.norm(self.grids[0]['f'].ravel(), np.inf)
        #print norm
        print("Norm of B =", self.nrm_b)
        self.abs_tolerance = abs_tolerance
        self.rel_tolerance = self.abs_tolerance * self.nrm_b
        self.ALPHA = -6.0
        self.BETA = 1.0

        # Iteration counts
        self.PRE_SMOOTH_ITER = pre_iter
        self.POST_SMOOTH_ITER = post_iter
        self.BOTTOM_SOLVER_ITER = bottom_iter
        
        # Print level information if verbose
        if self.verbose:
            self._print_level_info()
            self._print_initialization_info()
        
    def _create_grids(self):
        """Create simple grid hierarchy with full coarsening (all dimensions reduced)"""
        grids = []
        
        for level in range(self.num_levels):
            # Calculate dimensions for this level
            # Full coarsening: reduce x, y, and z dimensions by factor of 2 at each level
            nx = max(1, self.nx // (2 ** level))
            ny = max(1, self.ny // (2 ** level))
            nz = max(1, self.nz // (2 ** level))  # Also reduce z dimension for full coarsening
            
            # Grid spacing (full coarsening)
            hx = 1.0 / nx
            hy = 1.0 / ny
            hz = 1.0 / nz
            
            grid = {
                'nx': nx, 'ny': ny, 'nz': nz, 
                'hx': hx,
                'hy': hy,
                'hz': hz,
                'u': np.zeros((nx, ny, nz), dtype=DTYPE),      # Solution
                'f': np.zeros((nx, ny, nz), dtype=DTYPE),      # Right-hand side
                'r': np.zeros((nx, ny, nz), dtype=DTYPE),      # Residual
                'Au': np.zeros((nx, ny, nz), dtype=DTYPE),      # A*u
                'rho': 0.0,                                    # L2 norm squared of the residual
                'rho_up': 0.0                                   # L2 norm squared of the residual after up cycle
            }
            grids.append(grid)
            
        return grids
    
    def _print_level_info(self):
        """Print detailed information about the multigrid hierarchy and data structures"""
        print("\nMultigrid Level Hierarchy:")
        print("=" * 50)
        for level, grid in enumerate(self.grids):
            print(f"Level {level:2d}: {grid['nx']:3d} x {grid['ny']:3d} x {grid['nz']:3d} "
                f"(h = {grid['hx']:.6f}, {grid['hy']:.6f}, {grid['hz']:.6f}, points = {grid['nx']*grid['ny']*grid['nz']:,})")
        print("=" * 50)

        print("\nData Structures at Each Level:")
        print("=" * 120)
        print(f"{'Level':<6} {'Grid Size (nx×ny×nz)':<22} {'u shape (nx,ny,nz)':<22} "
            f"{'f shape':<15} {'r shape':<15} {'Au shape':<15} {'Stencil':<20}")
        print("-" * 120)

        for level, grid in enumerate(self.grids):
            nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
            grid_size = f"{nx}×{ny}×{nz}"
            u_shape  = str(grid['u'].shape)
            f_shape  = str(grid['f'].shape)
            r_shape  = str(grid['r'].shape)
            au_shape = str(grid['Au'].shape)
            stencil_info = "7-point Poisson"
            print(f"{level:<6} {grid_size:<22} {u_shape:<22} {f_shape:<15} "
                f"{r_shape:<15} {au_shape:<15} {stencil_info:<20}")

        print("-" * 120)
        print("Legend:")
        print("  Arrays are stored as (nx, ny, nz) = (x, y, z)")
        print("-" * 120)
        print("Legend:")
        print("  u     = Solution vector (unknown)")
        print("  f     = Right-hand side vector (source term)")
        print("  r     = Residual vector (f - Au)")
        print("  Au    = Operator applied to solution (A*u)")
        print("  Stencil = Finite difference stencil used")
        print("=" * 120)
        
        print("\nStencil Coefficients:")
        print("-" * 40)
        print(f"  α (center) = {self.ALPHA}")
        print(f"  β (neighbors) = {self.BETA}")
        print("  7-point stencil: α*u[i,j,k] + β*(u[i±1,j,k] + u[i,j±1,k] + u[i,j,k±1])")
        print("-" * 40)
        
        print("\nMemory Usage Summary:")
        print("-" * 50)
        total_points = 0
        for level, grid in enumerate(self.grids):
            points = grid['nx'] * grid['ny'] * grid['nz']
            total_points += points
            bytes_per_elem = self.grids[0]['u'].dtype.itemsize  # f32
            memory_mb = (points * 4 * bytes_per_elem)/1024
            print(f"  Level {level}: {points:,} points × 4 arrays = {memory_mb:.2f} KB")
        total_bytes_per_elem = self.grids[0]['u'].dtype.itemsize
        total_memory = (total_points * 4 * total_bytes_per_elem)/1024
        print(f"  Total: {total_points:,} points × 4 arrays = {total_memory:.2f} KB")
        print("-" * 50)
        print()
    
    def _print_initialization_info(self):
        """Print all initialization parameters and coefficients"""
        print("Initialization Parameters:")
        print("=" * 50)
        print(f"Grid dimensions: {self.nx} × {self.ny} × {self.nz}")
        print(f"Number of levels: {self.num_levels}")
        print(f"Grid spacing (h): {self.grids[0]['hx']:.6f}, {self.grids[0]['hy']:.6f}, {self.grids[0]['hz']:.6f}")
        print()
        
        print("Stencil Coefficients:")
        print(f"  α (center coefficient): {self.ALPHA}")
        print(f"  β (neighbor coefficient): {self.BETA}")
        print(f"  7-point stencil: α*u[i,j,k] + β*(u[i±1,j,k] + u[i,j±1,k] + u[i,j,k±1])")
        print(f"  Grid spacing scaling: h² = {self.grids[0]['hx']**2:.6f}, {self.grids[0]['hy']**2:.6f}, {self.grids[0]['hz']**2:.6f} (used in residual computation)")
        print()
        
        print("Relaxation Parameters:")
        print(f"  Jacobi coefficient: {self.omega:.6f}")
        print(f"  Pre-smooth iterations: {self.PRE_SMOOTH_ITER}")
        print(f"  Post-smooth iterations: {self.POST_SMOOTH_ITER}")
        print(f"  Bottom solver iterations: {self.BOTTOM_SOLVER_ITER}")
        print()
        
        print("Convergence Parameters:")
        print(f"  Relative Tolerance: {self.rel_tolerance:.6e}")
        print(f"  Absolute Tolerance: {self.abs_tolerance:.6e}")
        print()
        
        print("Right-hand Side Initialization:")
        print(f"  Function: sin(πx) * sin(πy) * sin(πz)")
        print(f"  Domain: [0,1] × [0,1] × [0,1]")
        print(f"  Grid points: {self.nx} × {self.ny} × {self.nz}")
        print()
        
        print("Boundary Conditions:")
        print("  Dirichlet boundary conditions (u = 0 on boundary)")
        print("  Interior points only for stencil application")

        print("Initial values:")
        print("  x (solution) - 0.0:")
        print("  f (RHS) - sin function values:")
        print("  r (residual) - 0.0:")
        print("  Au (operator) - 0.0:")
        print()

        print("=" * 50)
        print()
    
    def _init_x(self, value: float = 0.0):
        """Initialize solution x with zeros"""
        for level, grid in enumerate(self.grids):
           if level == 0:
               grid['u'].fill(value)
            # else 0.0
    
    def _init_rhs(self):
        """Initialize right-hand side with simple test function"""
        for level, grid in enumerate(self.grids):
            nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
            
            # Cell-centered coordinates: (i+0.5)*h, matching HPGMG FV
            hx = grid['hx']
            hy = grid['hy']
            hz = grid['hz']
            x = (np.arange(nx, dtype=DTYPE) + 0.5) * hx
            y = (np.arange(ny, dtype=DTYPE) + 0.5) * hy
            z = (np.arange(nz, dtype=DTYPE) + 0.5) * hz
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

            # HPGMG-matching: f = sin(2πx)·sin(2πy)·sin(2πz)
            pi = DTYPE(np.pi)
            grid['f'] = np.sin(2*pi * X) * np.sin(2*pi * Y) * np.sin(2*pi * Z)
            #print(f"f: {grid['f']}")

    def calculate_rho(self, residual_3d):
        """Calculate the L2 norm squared of the residual vector
        
        This function computes ||r||² where r is the residual vector.
        This is equivalent to the xi calculation in the CSL code.
        
        Args:
            residual_3d: 3D numpy array of shape (height, width, zDim) containing residual values
            
        Returns:
            float: The L2 norm squared of the residual vector
            
        Example:
            rho = self.calculate_rho(residual_3d_first)
        """
        # rho = np.dot(residual_3d.flatten(), residual_3d.flatten())
        rho = np.max(np.abs(residual_3d.flatten()))
        # print(f"rho: {rho}")
        # norm_inf = np.linalg.norm(residual_3d.flatten(), np.inf)
        # print(f"norm_inf: {norm_inf}")
        # Flatten the 3D array to 1D vector
        # residual_1d = residual_3d.flatten()
        
        # Compute L2 norm squared: ||r||² = Σ(r[i]²)
        # rho = np.sum(residual_1d * residual_1d)
        
        return rho

    def apply_operator(self, level: int):
        """Apply 7-point Laplacian operator"""
        """
        General forumula: 
        1/hx2 ( u(i-1, j, k) - 2*u(i,j,k) + u(i+1, j, k) ) + 
        1/hy2 ( u(i, j-1, k) - 2*u(i,j,k) + u(i, j+1, k) ) +
        1/hz2 ( u(i, j, k-1) - 2*u(i,j,k) + u(i, j, k+1) )
        """
        grid = self.grids[level]
        u = grid['u']
        Au = grid['Au']
        hx = grid['hx']
        hy = grid['hy']
        hz = grid['hz']
        nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
        beta = DTYPE(self.BETA)
        
        # Clear Au (handled inside vectorized version, but needed for JIT version)
        Au.fill(0.0)
        _apply_operator_jit(u, Au, nx, ny, nz, hx, hy, hz, beta)

    def compute_residual(self, level: int):
        """Compute residual r = f - Au"""
        grid = self.grids[level]
        f = grid['f']
        r = grid['r']
        
        # Apply operator
        self.apply_operator(level)
        Au = grid['Au']
        
        # Residual: r = f - Au
        r[:] = f - Au
    
    def jacobi_smooth(self, level: int, num_iter: int = 1):
        """Jacobi smoothing"""
        grid = self.grids[level]
        u = grid['u']
        f = grid['f']
        hx = grid['hx']
        hy = grid['hy']
        hz = grid['hz']
        nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
        omega = DTYPE(self.omega)
        
        for _ in range(num_iter):
            # Apply operator
            self.apply_operator(level)
            Au = grid['Au']
            
            # Jacobi update: u_new = u + ω * (f - Au) / diagonal
            # For 7-point stencil, diagonal is -6/h²
            diagonal = DTYPE(-2.0) / (hx*hx) + DTYPE(-2.0) / (hy*hy) + DTYPE(-2.0) / (hz*hz)
            diag_inv = 1.0 / diagonal
            update = self.omega * diag_inv * (f - Au)  # Correct sign: f - Au, if you make it Au - f, it will give nan
            
            u[:] += update[:]  # update all cells, boundary handled by apply_operator
    
    def restrict(self, fine_level: int):
        """
        Full coarsening restriction - average 8 points in 2x2x2 block (all 3 dimensions)
        """
        fine = self.grids[fine_level]
        coarse = self.grids[fine_level + 1]
        
        fine_r = fine['r'] # (nx, ny, nz)
        coarse_f = coarse['f']
        
        Xdim, Ydim, Zdim = coarse_f.shape
        
        # Use optimized JIT-compiled function if Numba is available
        _restrict_jit(fine_r, coarse_f, Xdim, Ydim, Zdim)
        

    def interpolate(self, fine_level: int):
        """Full coarsening interpolation - broadcast to 8 fine points in 2x2x2 block
        
        Reverse of restriction: for each coarse point (i,j,k), broadcast its value to
        8 fine points in the 2x2x2 block. This is the inverse operation of full coarsening.
        
        For each coarse position (i,j,k), adds coarse_u[i,j,k] to all 8 fine positions:
        - Fine positions: (2i, 2j, 2k), (2i+1, 2j, 2k), (2i, 2j+1, 2k), (2i+1, 2j+1, 2k),
                           (2i, 2j, 2k+1), (2i+1, 2j, 2k+1), (2i, 2j+1, 2k+1), (2i+1, 2j+1, 2k+1)
        """
        fine = self.grids[fine_level]
        coarse = self.grids[fine_level + 1]
        np.set_printoptions(linewidth=1000)
        coarse_u = coarse['u']
        fine_u = fine['u']
        
        coarse_nx, coarse_ny, coarse_nz = coarse['nx'], coarse['ny'], coarse['nz']
        
        # Use optimized JIT-compiled function if Numba is available
        _interpolate_jit(coarse_u, fine_u, coarse_nx, coarse_ny, coarse_nz)
            
    def v_cycle(self, level: int = 0):
        """V-cycle multigrid"""
        if level == self.num_levels - 1:
            # Coarsest level: solve directly
            self.solve_coarse()
            return
        
        # Pre-smooth
        self.jacobi_smooth(level, self.PRE_SMOOTH_ITER)
        
        # Compute residual
        self.compute_residual(level)
        
        # Restrict residual to coarser level
        self.restrict(level)
        
        # Initialize coarse solution
        self.grids[level + 1]['u'].fill(0.0)
        
        # Recursive call to coarser level
        self.v_cycle(level + 1)
                
        # Interpolate correction
        self.interpolate(level)
        
        # Post-smooth
        self.jacobi_smooth(level, self.POST_SMOOTH_ITER)

    def only_down_cycle(self, level: int = 0):
        """Only down cycle multigrid"""
        if level == self.num_levels - 1:
            return
        
        # Pre-smooth
        self.jacobi_smooth(level, self.PRE_SMOOTH_ITER)
        # print u after smooth
        u = self.grids[level]['u']
        f = self.grids[level]['f']

        # Compute residual
        self.compute_residual(level)

        self.grids[level]['rho'] = self.calculate_rho(self.grids[level]['r'])
        
        # Restrict residual to coarser level
        self.restrict(level)
        
        # Initialize coarse solution
        self.grids[level + 1]['u'].fill(0.0)
        
        # Recursive call to coarser level
        self.only_down_cycle(level + 1)

    # Note level must be (coarse_level - 2)
    def only_up_cycle(self, level: int):
            """Only up cycle multigrid"""
            if level == -1:
                return
            
            # Interpolate correction
            self.interpolate(level)
            
            # Post-smooth
            self.jacobi_smooth(level, self.POST_SMOOTH_ITER)

            # residual
            self.compute_residual(level)

            self.grids[level]['rho_up'] = self.calculate_rho(self.grids[level]['r'])
            # Recursive call to finer level
            self.only_up_cycle(level - 1)

    def solve_coarse(self):
        """Solve on coarsest level using Jacobi"""

        # if level != self.num_levels - 1:
        #     raise ValueError("solve_coarse can only be called on the coarsest level")
        #     return
        coarse_level = self.num_levels - 1
        grid = self.grids[coarse_level]

        self.jacobi_smooth(coarse_level, self.BOTTOM_SOLVER_ITER)

        # Compute residual
        self.compute_residual(coarse_level)

        self.grids[coarse_level]['rho'] = self.calculate_rho(self.grids[coarse_level]['r'])

    #########################################################
    ## User facing functions
    #########################################################

    # recursive version
    def solve(self, max_iter: int = 20):
        """Solve using V-cycles"""
        # Initialize solution
        # for grid in self.grids:
        #     grid['u'].fill(0.0)
        
        print(f"Starting GMG solve with {max_iter} max iterations")
        print(f"Grid size: {self.nx}x{self.ny}x{self.nz}, Levels: {self.num_levels}")
        print(f"Tolerance: {self.tolerance:.6e}")
        print("-" * 50)
        
        # calculate residual and print before any cycle starts at level 0.
        self.compute_residual(0)
        residual = self.calculate_rho(self.grids[0]['r'])
        print(f"Initial residual at level 0: {residual:.6e}")
        start_time = time.time()
        iterations = 0
        residual = 0.0
        
        while residual > self.tolerance and iterations < max_iter:
        # while iterations < max_iter:
            # Perform V-cycle
            cycle_start = time.time()
            self.v_cycle()
            cycle_time = time.time() - cycle_start
            
            # Check convergence
            self.compute_residual(0)
            residual = self.calculate_rho(self.grids[0]['r'])
            iterations += 1
            
            print(f"Iteration {iterations:2d}: Residual = {residual:.6e}, Time = {cycle_time:.4f}s")
        
        total_time = time.time() - start_time
        
        print(f"After {iterations} iterations")
        print(f"Final residual: {residual:.6e}")
        print(f"Tolerance: {self.tolerance:.6e}")
        converged = "Yes" if residual < self.tolerance else "No"
        print(f"Converged: {converged}")
        print(f"Total time: {total_time:.4f}s")
        
        return residual, iterations
  
    def solve_iterative(self, max_iter: int = 20):
        """Solve using iterative version"""
        # Initialize solution
        # for grid in self.grids:
        #     grid['u'].fill(0.0)
        print("=" * 50)
        print(f"Starting GMG solve with {max_iter} max iterations")
        print(f"Grid size: {self.nx}x{self.ny}x{self.nz}, Levels: {self.num_levels}")
        print(f"Tolerance(abs): {self.abs_tolerance:.6e}")
        print(f"|b|_2 = {self.nrm_b:.6e}")
        print(f"Tolerance(rel): {self.rel_tolerance:.6e}")
        print(f"max_ite = {max_iter}")
        print("-" * 50)
        
        # calculate residual and print before any cycle starts at level 0.
        self.compute_residual(0)
        xi_max = self.calculate_rho(self.grids[0]['r'])
        print(f"Initial |rho|_max at level 0: {xi_max:.6e}")
        
        start_time = time.time()
        iterations = 0        
        while (xi_max > self.rel_tolerance) and (iterations < max_iter):
            # Perform V-cycle
            cycle_start = time.time()
            self.only_down_cycle()
            self.solve_coarse()

            # temporary fix.
            self.grids[self.num_levels - 1]['rho_up'] = self.grids[self.num_levels - 1]['rho']
            self.only_up_cycle(self.num_levels - 2)
            cycle_time = time.time() - cycle_start
            
            # Check convergence
            self.compute_residual(0)
            xi_max = self.calculate_rho(self.grids[0]['r'])
            print(f"Iteration {iterations:2d}: |rho|_max = {xi_max:.6e}, tolerance = {self.rel_tolerance:.6e}, Time = {cycle_time:.4f}s")
            iterations += 1


        total_time = time.time() - start_time
        
        print("-" * 50)
        print(f"After {iterations} iterations")
        print(f"Final |rho|_inf: {xi_max:.6e}")
        converged = "Yes" if xi_max < self.rel_tolerance else "No"
        print(f"Converged: {converged}")
        print(f"Total time: {total_time:.4f}s")
        print("=" * 50)
        
        return xi_max, iterations

    #########################################################

