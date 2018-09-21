/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HIP_HIP_KERNELS_COO_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_COO_HPP_

#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType, typename IndexType>
__global__ void kernel_coo_permute(IndexType nnz,
                                   const IndexType* __restrict__ in_row,
                                   const IndexType* __restrict__ in_col,
                                   const IndexType* __restrict__ perm,
                                   IndexType* __restrict__ out_row,
                                   IndexType* __restrict__ out_col)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    for(int i = ind; i < nnz; i += hipGridDim_x)
    {
        out_row[i] = perm[in_row[i]];
        out_col[i] = perm[in_col[i]];
    }
}

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_COO_HPP_
