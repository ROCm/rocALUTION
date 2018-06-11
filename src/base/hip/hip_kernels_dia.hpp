#ifndef ROCALUTION_HIP_HIP_KERNELS_DIA_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_DIA_HPP_

#include "../matrix_formats_ind.hpp"
#include "hip_complex.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

// Nathan Bell and Michael Garland
// Efficient Sparse Matrix-Vector Multiplication
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
// Efficient Sparse Matrix-Vector Multiplication
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
__global__ void kernel_dia_diag_idx(const IndexType nrow, const IndexType *row_offset,
                                    const IndexType *col, IndexType *diag_idx) {

  int row = blockDim.x * blockIdx.x + threadIdx.x;
  if (row < nrow) {

    for (IndexType j=row_offset[row]; j<row_offset[row+1]; ++j) {

      int idx = col[j] - row + nrow - 1;
      diag_idx[idx] = true;

    }

  }

}

template <typename IndexType>
__global__ void kernel_dia_fill_offset(const IndexType nrow, const IndexType ncol,
                                       IndexType *diag_idx, const IndexType *offset_map,
                                       IndexType *offset) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;
    
  if (i < nrow+ncol)
    if (diag_idx[i]) {
      offset[offset_map[i]] = i - nrow + 1;
      diag_idx[i] = offset_map[i];
    }
  
}

template <typename ValueType, typename IndexType>
__global__ void kernel_dia_convert(const IndexType nrow, const IndexType ndiag,
                                   const IndexType *row_offset, const IndexType *col,
                                   const ValueType *val, const IndexType *diag_idx,
                                   ValueType *dia_val) {

  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < nrow) {

    for (int j=row_offset[row]; j<row_offset[row+1]; ++j) {

      IndexType idx = col[j] - row + nrow - 1;
      dia_val[DIA_IND(row, diag_idx[idx], nrow, ndiag)] = val[j];

    }

  }

}


}

#endif // ROCALUTION_HIP_HIP_KERNELS_DIA_HPP_
