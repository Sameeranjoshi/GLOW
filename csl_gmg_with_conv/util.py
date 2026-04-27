# Copyright 2024 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import numpy as np

from scipy.sparse import coo_matrix


def COL_MAJOR(h, w, l, height, width, pe_length):
    assert 0 <= h and h < height
    assert 0 <= w and w < width
    assert 0 <= l and l < pe_length

    return (h + w*height + l*height*width)


def hwl_2_oned_colmajor(
    height: int,
    width: int,
    pe_length: int,
    A_hwl: np.ndarray,
    dtype
):
  """
    Given a 3-D tensor A[height][width][pe_length], transform it to
    1D array by column-major
  """
  A_1d = np.zeros(height*width*pe_length, dtype)
  idx = 0
  for l in range(pe_length):
    for w in range(width):
      for h in range(height):
        A_1d[idx] = A_hwl[(h, w, l)]
        idx = idx + 1
  return A_1d


def oned_to_hwl_colmajor(
    height: int,
    width: int,
    pe_length: int,
    A_1d: np.ndarray,
    dtype
):
    """
    Given a 1-D tensor A_1d[height*width*pe_length], transform it to
    3-D tensor A[height][width][pe_length] by column-major
    """
    if dtype == np.float32:
        # only support f32 to f32
        assert A_1d.dtype == np.float32, "only support f32 to f32"
        A_hwl = np.reshape(A_1d, (height, width, pe_length), order='F')

    elif dtype == np.uint16:
        # only support u32 to u16 by dropping upper 16-bit
        assert A_1d.dtype == np.uint32, "only support u32 to u16"
        A_hwl = np.zeros((height, width, pe_length), dtype)
        idx = 0
        for l in range(pe_length):
            for w in range(width):
                for h in range(height):
                    x = A_1d[idx]
                    x = x & 0x0000FFFF # drop upper 16-bit
                    A_hwl[(h, w, l)] = np.uint16(x)
                    idx = idx + 1
    else:
        raise RuntimeError(f"{dtype} is not supported")

    return A_hwl



#  y = Laplacian(x) for z=0,1,..,zDim-1
#
# The capacity of x and y can be bigger than zDim, but the physical domain is [0,zDim)
#
# The coordinates of physical domain are x,y,z.
# The physical layout of WSE is width, height.
# To avoid confusion, the kernel is written based on the layout of
# WSE, not physical domain of the application.
# For example, the user can match x-coordinate to x direction of
# WSE and y-coordinate to y-direction of WSE.
#              x-coord
#            +--------+
#    y-coord |        |
#            +--------+
#
# The stencil coefficients "stencil_coeff" can vary along x-y direction,
# but universal along z-direction. Each PE can have seven coefficents,
# west, east, south, north, bottom, top and center.
#
# Input:
#   stencil_coeff: size is (h,w,7)
#   x: size is (h,w,l)
# Output:
#   y: size is (h,w,l)
#
def laplacian(stencil_coeff, zDim, x, y):
  (height, width, pe_length) = x.shape
  assert zDim <= pe_length
  # y and x must have the same dimensions
  (m, n, k) = y.shape
  assert m == height
  assert n == width
  assert pe_length == k
  # stencil_coeff must be (h,w,7)
  (m, n, k) = stencil_coeff.shape
  assert m == height
  assert n == width
  assert 7 == k

#          North
#           j
#        +------+
# West i |      | East
#        +------+
#          south
  for i in range(height):
    for j in range(width):
      for k in range(zDim):
        c_west = stencil_coeff[(i,j,0)]
        c_east = stencil_coeff[(i,j,1)]
        c_south = stencil_coeff[(i,j,2)]
        c_north = stencil_coeff[(i,j,3)]
        c_bottom = stencil_coeff[(i,j,4)]
        c_top = stencil_coeff[(i,j,5)]
        c_center = stencil_coeff[(i,j,6)]

        west_buf = 0 # x[(i,-1,k)]
        if 0 < j:
          west_buf = x[(i,j-1,k)]
        east_buf = 0  # x[(i,w,k)]
        if j < width-1:
          east_buf = x[(i,j+1,k)]
        north_buf = 0; # x[(-1,j,k)]
        if 0 < i:
          north_buf = x[(i-1,j,k)]
        south_buf = 0  # x[(h,j,k)]
        if i < height-1:
          south_buf = x[(i+1,j,k)]
        bottom_buf = 0 # x[(i,j,-1)]
        if 0 < k:
          bottom_buf = x[(i,j,k-1)]
        top_buf = 0    # x[(i,j,l)]
        if k < zDim-1:
          top_buf = x[(i,j,k+1)]
        center_buf = x[(i,j,k)]
        y[(i,j,k)] = c_west*west_buf + c_east*east_buf + \
                     c_south*south_buf + c_north*north_buf + \
                     c_bottom*bottom_buf + c_top*top_buf + \
                     c_center*center_buf

def laplacian_modified(stencil_coeff, zDim, x, y, hops=1, factor=1):
  (height, width, pe_length) = x.shape
  assert zDim <= pe_length
  # y and x must have the same dimensions
  (m, n, k) = y.shape
  assert m == height
  assert n == width
  assert pe_length == k
  # stencil_coeff must be (h,w,7)
  (m, n, k) = stencil_coeff.shape
  assert m == height
  assert n == width
  assert 7 == k

#          North
#           j
#        +------+
# West i |      | East
#        +------+
#          south
  for i in range(height):
    for j in range(width):
      # Check if this PE is active (matches WSE logic)
      is_active_pe = (i % factor == 0) and (j % factor == 0)
      
      for k in range(zDim):
        c_west = stencil_coeff[(i,j,0)]
        c_east = stencil_coeff[(i,j,1)]
        c_south = stencil_coeff[(i,j,2)]
        c_north = stencil_coeff[(i,j,3)]
        c_bottom = stencil_coeff[(i,j,4)]
        c_top = stencil_coeff[(i,j,5)]
        c_center = stencil_coeff[(i,j,6)]

        # Use hop-based neighbors instead of immediate neighbors
        west_buf = 0
        if j >= hops:  # Check if west neighbor is within bounds
          west_neighbor_j = j - hops
          # Check if west neighbor is also active
          if (i % factor == 0) and (west_neighbor_j % factor == 0):
            west_buf = x[(i, west_neighbor_j, k)]
            
        east_buf = 0
        if j < width - hops:  # Check if east neighbor is within bounds
          east_neighbor_j = j + hops
          # Check if east neighbor is also active
          if (i % factor == 0) and (east_neighbor_j % factor == 0):
            east_buf = x[(i, east_neighbor_j, k)]
            
        north_buf = 0
        if i >= hops:  # Check if north neighbor is within bounds
          north_neighbor_i = i - hops
          # Check if north neighbor is also active
          if (north_neighbor_i % factor == 0) and (j % factor == 0):
            north_buf = x[(north_neighbor_i, j, k)]
            
        south_buf = 0
        if i < height - hops:  # Check if south neighbor is within bounds
          south_neighbor_i = i + hops
          # Check if south neighbor is also active
          if (south_neighbor_i % factor == 0) and (j % factor == 0):
            south_buf = x[(south_neighbor_i, j, k)]
            
        # Z-direction neighbors remain immediate (no hop-based logic for Z)
        bottom_buf = 0
        if 0 < k:
          bottom_buf = x[(i,j,k-1)]
        top_buf = 0
        if k < zDim-1:
          top_buf = x[(i,j,k+1)]
        center_buf = x[(i,j,k)]
        
        # Only compute stencil for active PEs
        if is_active_pe:
          y[(i,j,k)] = c_west*west_buf + c_east*east_buf + \
                        c_south*south_buf + c_north*north_buf + \
                        c_bottom*bottom_buf + c_top*top_buf + \
                        c_center*center_buf
        else:
          y[(i,j,k)] = 0  # Non-active PEs produce zero


# Given a 7-point stencil, generate sparse matrix A.
# A is represented by CSR.
# The order of grids is column-major
def csr_7_pt_stencil(stencil_coeff, height, width, pe_length):
  # stencil_coeff must be (h,w,7)
  (m, n, k) = stencil_coeff.shape
  assert m == height
  assert n == width
  assert 7 == k

  N = height * width * pe_length

  # each point has 7 coefficents at most
  cooRows = np.zeros(7*N, np.int32)
  cooCols = np.zeros(7*N, np.int32)
  cooVals = np.zeros(7*N, np.float32)

#          North
#           j
#        +------+
# West i |      | East
#        +------+
#          south
  nnz = 0
  for i in range(height):
    for j in range(width):
      for k in range(pe_length):
        c_west = stencil_coeff[(i,j,0)]
        c_east = stencil_coeff[(i,j,1)]
        c_south = stencil_coeff[(i,j,2)]
        c_north = stencil_coeff[(i,j,3)]
        c_bottom = stencil_coeff[(i,j,4)]
        c_top = stencil_coeff[(i,j,5)]
        c_center = stencil_coeff[(i,j,6)]

        center_idx = COL_MAJOR(i, j, k, height, width, pe_length)
        cooRows[nnz] = center_idx 
        cooCols[nnz] = center_idx
        cooVals[nnz] = c_center
        nnz += 1
        #west_buf = 0 # x[(i,-1,k)]
        if 0 < j:
          west_idx = COL_MAJOR(i, j-1, k, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = west_idx
          cooVals[nnz] = c_west;
          nnz += 1
        #east_buf = 0  # x[(i,w,k)]
        if j < width-1:
          east_idx = COL_MAJOR(i,j+1,k, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = east_idx
          cooVals[nnz] = c_east
          nnz += 1 
        #north_buf = 0; # x[(-1,j,k)]
        if 0 < i:
          north_idx = COL_MAJOR(i-1,j,k, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = north_idx
          cooVals[nnz] = c_north
          nnz += 1
        #south_buf = 0  # x[(h,j,k)]
        if i < height-1:
          south_idx = COL_MAJOR(i+1,j,k, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = south_idx
          cooVals[nnz] = c_south
          nnz += 1
        #bottom_buf = 0 # x[(i,j,-1)]
        if 0 < k:
          bottom_idx = COL_MAJOR(i,j,k-1, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = bottom_idx
          cooVals[nnz] = c_bottom 
          nnz += 1
        #top_buf = 0    # x[(i,j,l)]
        if k < pe_length-1:
          top_idx = COL_MAJOR(i,j,k+1, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = top_idx
          cooVals[nnz] = c_top 
          nnz += 1

  A_coo = coo_matrix((cooVals, (cooRows, cooCols)), shape=(N, N))

  A_csr = A_coo.tocsr(copy=True)
  # sort column indices
  A_csr = A_csr.sorted_indices().astype(np.float32)
  assert 1 == A_csr.has_sorted_indices, "Error: A is not sorted"

  return A_csr


import numpy as np
import matplotlib.pyplot as plt


def plot_3d_shapes(arrays, titles=None, colors=None, alphas=None, filename=None):
    """Visualize multiple 3D arrays using voxels as subplots."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Handle single input by converting to a list
    if not isinstance(arrays, list):
        arrays = [arrays]

    # Default values
    num_arrays = len(arrays)
    titles = titles or [f"Shape {i+1}" for i in range(num_arrays)]
    colors = colors or ['blue'] * num_arrays
    alphas = alphas or [0.7] * num_arrays

    # Single plot if only one array is provided
    if num_arrays == 1:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(titles[0])
        ax.voxels(np.ones(arrays[0].shape, dtype=bool), 
                  facecolors=colors[0], edgecolor='k', alpha=alphas[0])
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('Depth')
        plt.show()

    # Subplots if more than one array
    else:
        cols = min(num_arrays, 3)   # Limit to 3 columns
        rows = (num_arrays + cols - 1) // cols  # Calculate number of rows
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), 
                                 subplot_kw={'projection': '3d'})
        axes = np.array(axes).reshape(-1)  # Flatten in case of multiple rows

        for i, (array, title, color, alpha) in enumerate(zip(arrays, titles, colors, alphas)):
            ax = axes[i]
            ax.set_title(title)
            ax.voxels(np.ones(array.shape, dtype=bool), 
                      facecolors=color, edgecolor='k', alpha=alpha)
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            ax.set_zlabel('Depth')

        # Hide extra axes if fewer arrays
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])


    plt.savefig(filename)
    plt.close()
