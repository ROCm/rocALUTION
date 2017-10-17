#ifndef PARALUTION_GPU_CUDA_KERNELS_DIA_HPP_
#define PARALUTION_GPU_CUDA_KERNELS_DIA_HPP_

#include "../matrix_formats_ind.hpp"
#include "gpu_complex.hpp"

namespace paralution {

// Nathan Bell and Michael Garland
// Efficient Sparse Matrix-Vector Multiplication on {CUDA}
// NVR-2008-004 / NVIDIA Technical Report
template <typename ValueType, typename IndexType>
__global__ void kernel_dia_spmv(const IndexType num_rows,
                                const IndexType num_cols,
                                const IndexType num_diags,
                                const IndexType *Aoffsets,
                                const ValueType *Aval,
                                const ValueType *x,
                                ValueType *y) {

    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < num_rows) {

      ValueType sum;
      make_ValueType(sum, 0.0);

      for (IndexType n=0; n<num_diags; ++n) {

        const IndexType ind = DIA_IND(row, n, num_rows, num_diags);
        const IndexType col = row + Aoffsets[n];

        if ((col >= 0) && (col < num_cols))
          sum = sum + Aval[ind] * x[col];

      }

      y[row] = sum;

    }

}

// Nathan Bell and Michael Garland
// Efficient Sparse Matrix-Vector Multiplication on {CUDA}
// NVR-2008-004 / NVIDIA Technical Report
template <typename ValueType, typename IndexType>
__global__ void kernel_dia_add_spmv(const IndexType num_rows,
                                    const IndexType num_cols,
                                    const IndexType num_diags,
                                    const IndexType *Aoffsets,
                                    const ValueType *Aval,
                                    const ValueType scalar,
                                    const ValueType *x,
                                    ValueType *y) {

    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < num_rows) {

      ValueType sum;
      make_ValueType(sum, 0.0);

      for (IndexType n=0; n<num_diags; ++n) {

        const IndexType ind = DIA_IND(row, n, num_rows, num_diags);
        const IndexType col = row + Aoffsets[n];

        if ((col >= 0) && (col < num_cols)) {

          sum = sum + Aval[ind] * x[col];

        }

      }

      y[row] = y[row] + scalar*sum;

    }

}

template <typename IndexType>
__global__ void kernel_dia_diag_map(const IndexType nrow, const IndexType *row_offset,
                                    const IndexType *col, IndexType *diag_map) {

  int row = blockDim.x * blockIdx.x + threadIdx.x;
  IndexType map_index;

  if (row < nrow) {

    for (IndexType j=row_offset[row]; j<row_offset[row+1]; ++j) {

      map_index = col[j] - row + nrow;
      diag_map[map_index] = 1;

    }

  }

}

template <typename IndexType>
__global__ void kernel_dia_fill_offset(const IndexType nrow, const IndexType ncol,
                                       IndexType *diag_map, const IndexType *offset_map,
                                       IndexType *offset) {

  // 1D and 2D indexing
  int i =  threadIdx.x + ( ((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x );
    
  if (i < nrow+ncol)
    if (diag_map[i] == 1) {
      offset[offset_map[i]] = i - nrow;
      diag_map[i] = offset_map[i];
    }
  
}

template <typename ValueType, typename IndexType>
__global__ void kernel_dia_convert(const IndexType nrow, const IndexType ndiag,
                                   const IndexType *row_offset, const IndexType *col,
                                   const ValueType *val, const IndexType *diag_map,
                                   ValueType *dia_val) {

  int row = blockDim.x * blockIdx.x + threadIdx.x;
  IndexType map_index;

  if (row < nrow) {

    for (int j=row_offset[row]; j<row_offset[row+1]; ++j) {

      map_index = col[j] - row + nrow;
      dia_val[DIA_IND(row, diag_map[map_index], nrow, ndiag)] = val[j];

    }

  }

}


}

#endif
