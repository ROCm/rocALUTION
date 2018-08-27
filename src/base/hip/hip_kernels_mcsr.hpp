#ifndef ROCALUTION_HIP_HIP_KERNELS_MCSR_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_MCSR_HPP_

#include "hip_complex.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

template <unsigned int BLOCK_SIZE, unsigned int WARP_SIZE, typename ValueType, typename IndexType>
__global__ void kernel_mcsr_spmv(IndexType nrow,
                                 const IndexType* __restrict__ row_offset,
                                 const IndexType* __restrict__ col,
                                 const ValueType* __restrict__ val,
                                 const ValueType* __restrict__ in,
                                 ValueType* __restrict__ out)
{
    IndexType gid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    IndexType tid    = hipThreadIdx_x;
    IndexType laneid = tid % WARP_SIZE;
    IndexType warpid = gid / WARP_SIZE;
    IndexType nwarps = hipGridDim_x * hipBlockDim_x / WARP_SIZE;

    __shared__ volatile ValueType sdata[BLOCK_SIZE + WARP_SIZE / 2];

    for(IndexType ai = warpid; ai < nrow; ai += nwarps)
    {
        ValueType sum;
        make_ValueType(sum, 0);

        for(IndexType aj = row_offset[ai] + laneid; aj < row_offset[ai + 1]; aj += WARP_SIZE)
        {
            sum += val[aj] * in[col[aj]];
        }

        assign_volatile_ValueType(&sum, &sdata[tid]);

        __syncthreads();
        if(WARP_SIZE > 32)
            sum = add_volatile_ValueType(&sdata[tid + 32], &sum);
        __syncthreads();
        assign_volatile_ValueType(&sum, &sdata[tid]);
        if(WARP_SIZE > 16)
            sum = add_volatile_ValueType(&sdata[tid + 16], &sum);
        __syncthreads();
        assign_volatile_ValueType(&sum, &sdata[tid]);
        if(WARP_SIZE > 8)
            sum = add_volatile_ValueType(&sdata[tid + 8], &sum);
        __syncthreads();
        assign_volatile_ValueType(&sum, &sdata[tid]);
        if(WARP_SIZE > 4)
            sum = add_volatile_ValueType(&sdata[tid + 4], &sum);
        __syncthreads();
        assign_volatile_ValueType(&sum, &sdata[tid]);
        if(WARP_SIZE > 2)
            sum = add_volatile_ValueType(&sdata[tid + 2], &sum);
        __syncthreads();
        assign_volatile_ValueType(&sum, &sdata[tid]);
        if(WARP_SIZE > 1)
            sum = add_volatile_ValueType(&sdata[tid + 1], &sum);

        if(laneid == 0)
        {
            out[ai] = sum + val[ai] * in[ai];
        }
    }
}

template <unsigned int BLOCK_SIZE, unsigned int WARP_SIZE, typename ValueType, typename IndexType>
__global__ void kernel_mcsr_add_spmv(IndexType nrow,
                                     const IndexType* __restrict__ row_offset,
                                     const IndexType* __restrict__ col,
                                     const ValueType* __restrict__ val,
                                     ValueType scalar,
                                     const ValueType* __restrict__ in,
                                     ValueType* __restrict__ out)
{
    IndexType gid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    IndexType tid    = hipThreadIdx_x;
    IndexType laneid = tid % WARP_SIZE;
    IndexType warpid = gid / WARP_SIZE;
    IndexType nwarps = hipGridDim_x * hipBlockDim_x / WARP_SIZE;

    __shared__ volatile ValueType sdata[BLOCK_SIZE + WARP_SIZE / 2];

    for(IndexType ai = warpid; ai < nrow; ai += nwarps)
    {
        ValueType sum;
        make_ValueType(sum, 0.0);

        for(IndexType aj = row_offset[ai] + laneid; aj < row_offset[ai + 1]; aj += WARP_SIZE)
        {
            sum += scalar * val[aj] * in[col[aj]];
        }

        assign_volatile_ValueType(&sum, &sdata[tid]);

        __syncthreads();
        if(WARP_SIZE > 32)
            sum = add_volatile_ValueType(&sdata[tid + 32], &sum);
        __syncthreads();
        assign_volatile_ValueType(&sum, &sdata[tid]);
        if(WARP_SIZE > 16)
            sum = add_volatile_ValueType(&sdata[tid + 16], &sum);
        __syncthreads();
        assign_volatile_ValueType(&sum, &sdata[tid]);
        if(WARP_SIZE > 8)
            sum = add_volatile_ValueType(&sdata[tid + 8], &sum);
        __syncthreads();
        assign_volatile_ValueType(&sum, &sdata[tid]);
        if(WARP_SIZE > 4)
            sum = add_volatile_ValueType(&sdata[tid + 4], &sum);
        __syncthreads();
        assign_volatile_ValueType(&sum, &sdata[tid]);
        if(WARP_SIZE > 2)
            sum = add_volatile_ValueType(&sdata[tid + 2], &sum);
        __syncthreads();
        assign_volatile_ValueType(&sum, &sdata[tid]);
        if(WARP_SIZE > 1)
            sum = add_volatile_ValueType(&sdata[tid + 1], &sum);

        if(laneid == 0)
        {
            out[ai] = out[ai] + sum + scalar * val[ai] * in[ai];
        }
    }
}

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_MCSR_HPP_
