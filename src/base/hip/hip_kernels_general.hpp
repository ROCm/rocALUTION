#ifndef ROCALUTION_HIP_HIP_KERNELS_GENERAL_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_GENERAL_HPP_

#include "hip_complex.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType, typename IndexType>
__global__ void kernel_set_to_ones(IndexType n, ValueType* __restrict__ data)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    make_ValueType(data[ind], 1);
}

template <typename IndexType>
__global__ void
kernel_reverse_index(IndexType n, const IndexType* __restrict__ perm, IndexType* __restrict__ out)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    out[perm[ind]] = ind;
}

template <typename ValueType, typename IndexType>
__global__ void kernel_buffer_addscalar(IndexType n, ValueType scalar, ValueType* __restrict__ buff)
{
    IndexType ind = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ind >= n)
    {
        return;
    }

    buff[ind] = buff[ind] + scalar;
}

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_GENERAL_HPP_
