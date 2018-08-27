#ifndef ROCALUTION_HIP_HIP_KERNELS_DENSE_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_DENSE_HPP_

#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

// Replace column vector
template <typename ValueType, typename IndexType>
__global__ void kernel_dense_replace_column_vector(const ValueType* __restrict__ vec,
                                                   IndexType idx,
                                                   IndexType nrow,
                                                   IndexType ncol,
                                                   ValueType* __restrict__ mat)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    mat[DENSE_IND(ai, idx, nrow, ncol)] = vec[ai];
}

// Replace row vector
template <typename ValueType, typename IndexType>
__global__ void kernel_dense_replace_row_vector(const ValueType* __restrict__ vec,
                                                IndexType idx,
                                                IndexType nrow,
                                                IndexType ncol,
                                                ValueType* __restrict__ mat)
{
    IndexType aj = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(aj >= ncol)
    {
        return;
    }

    mat[DENSE_IND(idx, aj, nrow, ncol)] = vec[aj];
}

// Extract column vector
template <typename ValueType, typename IndexType>
__global__ void kernel_dense_extract_column_vector(ValueType* __restrict__ vec,
                                                   IndexType idx,
                                                   IndexType nrow,
                                                   IndexType ncol,
                                                   const ValueType* __restrict__ mat)
{
    IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= nrow)
    {
        return;
    }

    vec[ai] = mat[DENSE_IND(ai, idx, nrow, ncol)];
}

// Extract row vector
template <typename ValueType, typename IndexType>
__global__ void kernel_dense_extract_row_vector(ValueType* __restrict__ vec,
                                                IndexType idx,
                                                IndexType nrow,
                                                IndexType ncol,
                                                const ValueType* __restrict__ mat)
{
    IndexType aj = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(aj >= ncol)
    {
        return;
    }

    vec[aj] = mat[DENSE_IND(idx, aj, nrow, ncol)];
}

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_DENSE_HPP_
