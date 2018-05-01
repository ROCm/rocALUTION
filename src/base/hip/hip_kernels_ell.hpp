#ifndef ROCALUTION_HIP_HIP_KERNELS_ELL_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_ELL_HPP_

#include "../matrix_formats_ind.hpp"
#include "hip_complex.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

// Nathan Bell and Michael Garland
// Efficient Sparse Matrix-Vector Multiplication
// NVR-2008-004 / NVIDIA Technical Report
template <typename ValueType, typename IndexType>
__global__ void kernel_ell_spmv(const IndexType num_rows,
                                const IndexType num_cols,
                                const IndexType num_cols_per_row,
                                const IndexType *Acol,
                                const ValueType *Aval,
                                const ValueType *x,
                                ValueType *y) {

    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < num_rows) {

      ValueType sum;
      make_ValueType(sum, 0.0);

      for (IndexType n=0; n<num_cols_per_row; ++n) {

        const IndexType ind = ELL_IND(row, n, num_rows, num_cols_per_row);
        const IndexType col = Acol[ind];

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
__global__ void kernel_ell_add_spmv(const IndexType num_rows,
                                    const IndexType num_cols,
                                    const IndexType num_cols_per_row,
                                    const IndexType *Acol,
                                    const ValueType *Aval,
                                    const ValueType scalar,
                                    const ValueType *x,
                                    ValueType *y) {

    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < num_rows) {

      ValueType sum;
      make_ValueType(sum, 0.0);

      for (IndexType n=0; n<num_cols_per_row; ++n) {

        const IndexType ind = ELL_IND(row, n, num_rows, num_cols_per_row);
        const IndexType col = Acol[ind];

        if ((col >= 0) && (col < num_cols))
          sum = sum + Aval[ind] * x[col];

      }

      y[row] = y[row] + scalar*sum;

    }

}

template <typename ValueType, typename IndexType, unsigned int BLOCK_SIZE>
__global__ void kernel_ell_max_row(const IndexType  nrow,
                                   const ValueType *data,
                                         ValueType *out,
                                   const IndexType  GROUP_SIZE,
                                   const IndexType  LOCAL_SIZE) {

    int tid = threadIdx.x;

    __shared__ IndexType sdata[BLOCK_SIZE];
    sdata[tid] = 0;

    IndexType max;

    // get global id
    int gid = GROUP_SIZE * blockIdx.x + tid;

    for (int i = 0; i < LOCAL_SIZE; ++i, gid += BLOCK_SIZE) {

      if ( gid < nrow ) {
        // accessing global memory quite often - fix that
        max = data[gid+1] - data[gid];
        if (max > sdata[tid]) sdata[tid] = max;
      }

    }

    __syncthreads();

    for (int i = BLOCK_SIZE/2; i > 0; i /= 2) {

      if ( tid < i )
        if (sdata[tid+i] > sdata[tid]) sdata[tid] = sdata[tid+i];

      __syncthreads();

    }

    if (tid == 0)
      out[blockIdx.x] = sdata[tid];

}

template <typename ValueType, typename IndexType>
__global__ void kernel_ell_csr_to_ell(const IndexType nrow, const IndexType max_row,
                                  const IndexType *src_row_offset, const IndexType *src_col,
                                  const ValueType *src_val, IndexType *ell_col, ValueType *ell_val) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;
  IndexType n;
  IndexType ell_ind;

  if (ai < nrow) {

    n = 0;

    // warp divergence!
    for (aj=src_row_offset[ai]; aj<src_row_offset[ai+1]; ++aj) {

      ell_ind = n * nrow + ai;

      ell_col[ell_ind] = src_col[aj];
      ell_val[ell_ind] = src_val[aj];

      ++n;

    }

    // warp divergence!
    for (aj=src_row_offset[ai+1]-src_row_offset[ai]; aj<max_row; ++aj) {

      ell_ind = n * nrow + ai;

      ell_col[ell_ind] = IndexType(-1);
      make_ValueType(ell_val[ell_ind], 0.0);

      ++n;

    }

  }

}


}

#endif // ROCALUTION_HIP_HIP_KERNELS_ELL_HPP_
