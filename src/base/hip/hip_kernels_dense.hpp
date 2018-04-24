#ifndef PARALUTION_HIP_HIP_KERNELS_DENSE_HPP_
#define PARALUTION_HIP_HIP_KERNELS_DENSE_HPP_

#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace paralution {

// Replace column vector
template <typename ValueType, typename IndexType>
__global__ void kernel_dense_replace_column_vector(const ValueType *vec, const IndexType idx, const IndexType nrow,
                                                   const IndexType ncol, ValueType *mat) {

  IndexType ai = blockIdx.x * blockDim.x + threadIdx.x;

  if(ai < nrow)
    mat[DENSE_IND(ai, idx, nrow, ncol)] = vec[ai];

}

// Replace row vector
template <typename ValueType, typename IndexType>
__global__ void kernel_dense_replace_row_vector(const ValueType *vec, const IndexType idx, const IndexType nrow,
                                                const IndexType ncol, ValueType *mat) {

  IndexType aj = blockIdx.x * blockDim.x + threadIdx.x;

  if (aj < ncol)
    mat[DENSE_IND(idx, aj, nrow, ncol)] = vec[aj];

}

// Extract column vector
template <typename ValueType, typename IndexType>
__global__ void kernel_dense_extract_column_vector(ValueType *vec, const IndexType idx, const IndexType nrow,
                                                   const IndexType ncol, const ValueType *mat) {

  IndexType ai = blockIdx.x * blockDim.x + threadIdx.x;

  if (ai < nrow)
    vec[ai] = mat[DENSE_IND(ai, idx, nrow, ncol)];

}

// Extract row vector
template <typename ValueType, typename IndexType>
__global__ void kernel_dense_extract_row_vector(ValueType *vec, const IndexType idx, const IndexType nrow,
                                                const IndexType ncol, const ValueType *mat) {

  IndexType aj = blockIdx.x * blockDim.x + threadIdx.x;

  if (aj < ncol)
    vec[aj] = mat[DENSE_IND(idx, aj, nrow, ncol)];

}


}

#endif // PARALUTION_HIP_HIP_KERNELS_DENSE_HPP_
