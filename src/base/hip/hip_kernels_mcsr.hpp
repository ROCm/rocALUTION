#ifndef ROCALUTION_HIP_HIP_KERNELS_MCSR_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_MCSR_HPP_

#include "../matrix_formats_ind.hpp"
#include "hip_complex.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

template <unsigned int BLOCK_SIZE, unsigned int WARP_SIZE, typename ValueType, typename IndexType>
__global__ void kernel_mcsr_spmv(const IndexType nrow,
                                 const IndexType *row_offset,
                                 const IndexType *col,
                                 const ValueType *val,
                                 const ValueType *in,
                                       ValueType *out) {

  IndexType gid = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType tid = threadIdx.x;
  IndexType laneid = tid % WARP_SIZE;
  IndexType warpid = gid / WARP_SIZE;
  IndexType nwarps = gridDim.x * blockDim.x / WARP_SIZE;

  __shared__ volatile ValueType sdata[BLOCK_SIZE + WARP_SIZE / 2];

  for (IndexType ai = warpid; ai<nrow; ai+=nwarps) {

    ValueType sum;
    make_ValueType(sum, 0.0);

    for (IndexType aj = row_offset[ai] + laneid; aj < row_offset[ai+1]; aj+=WARP_SIZE) {
      sum += val[aj] * in[col[aj]];
    }

    assign_volatile_ValueType(&sum, &sdata[tid]);

    __syncthreads();
    if(WARP_SIZE > 32) sum = add_volatile_ValueType(&sdata[tid+32], &sum); __syncthreads(); assign_volatile_ValueType(&sum, &sdata[tid]);
    if(WARP_SIZE > 16) sum = add_volatile_ValueType(&sdata[tid+16], &sum); __syncthreads(); assign_volatile_ValueType(&sum, &sdata[tid]);
    if(WARP_SIZE >  8) sum = add_volatile_ValueType(&sdata[tid+ 8], &sum); __syncthreads(); assign_volatile_ValueType(&sum, &sdata[tid]);
    if(WARP_SIZE >  4) sum = add_volatile_ValueType(&sdata[tid+ 4], &sum); __syncthreads(); assign_volatile_ValueType(&sum, &sdata[tid]);
    if(WARP_SIZE >  2) sum = add_volatile_ValueType(&sdata[tid+ 2], &sum); __syncthreads(); assign_volatile_ValueType(&sum, &sdata[tid]);
    if(WARP_SIZE >  1) sum = add_volatile_ValueType(&sdata[tid+ 1], &sum);

    if (laneid == 0) {
      out[ai] = sum + val[ai] * in[ai];
    }

  }

}

template <unsigned int BLOCK_SIZE, unsigned int WARP_SIZE, typename ValueType, typename IndexType>
__global__ void kernel_mcsr_add_spmv(const IndexType nrow,
                                     const IndexType *row_offset,
                                     const IndexType *col,
                                     const ValueType *val,
                                     const ValueType scalar,
                                     const ValueType *in,
                                           ValueType *out) {

  IndexType gid = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType tid = threadIdx.x;
  IndexType laneid = tid % WARP_SIZE;
  IndexType warpid = gid / WARP_SIZE;
  IndexType nwarps = gridDim.x * blockDim.x / WARP_SIZE;

  __shared__ volatile ValueType sdata[BLOCK_SIZE + WARP_SIZE / 2];

  for (IndexType ai = warpid; ai<nrow; ai+=nwarps) {

    ValueType sum;
    make_ValueType(sum, 0.0);

    for (IndexType aj = row_offset[ai] + laneid; aj < row_offset[ai+1]; aj+=WARP_SIZE) {
      sum += scalar * val[aj] * in[col[aj]];
    }

    assign_volatile_ValueType(&sum, &sdata[tid]);

    __syncthreads();
    if(WARP_SIZE > 32) sum = add_volatile_ValueType(&sdata[tid+32], &sum); __syncthreads(); assign_volatile_ValueType(&sum, &sdata[tid]);
    if(WARP_SIZE > 16) sum = add_volatile_ValueType(&sdata[tid+16], &sum); __syncthreads(); assign_volatile_ValueType(&sum, &sdata[tid]);
    if(WARP_SIZE >  8) sum = add_volatile_ValueType(&sdata[tid+ 8], &sum); __syncthreads(); assign_volatile_ValueType(&sum, &sdata[tid]);
    if(WARP_SIZE >  4) sum = add_volatile_ValueType(&sdata[tid+ 4], &sum); __syncthreads(); assign_volatile_ValueType(&sum, &sdata[tid]);
    if(WARP_SIZE >  2) sum = add_volatile_ValueType(&sdata[tid+ 2], &sum); __syncthreads(); assign_volatile_ValueType(&sum, &sdata[tid]);
    if(WARP_SIZE >  1) sum = add_volatile_ValueType(&sdata[tid+ 1], &sum);

    if (laneid == 0) {
      out[ai] = out[ai] + sum + scalar * val[ai] * in[ai];
    }

  }

}

}

#endif // ROCALUTION_HIP_HIP_KERNELS_MCSR_HPP_
