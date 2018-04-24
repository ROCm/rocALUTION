#ifndef PARALUTION_HIP_HIP_KERNELS_MCSR_HPP_
#define PARALUTION_HIP_HIP_KERNELS_MCSR_HPP_

#include "../matrix_formats_ind.hpp"
#include "hip_complex.hpp"

#include <hip/hip_runtime.h>

namespace paralution {

// ----------------------------------------------------------
// function spmv_csr_vector_kernel(...)
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.5.1, 
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - adapted interface
// - other modifications
// - modified for MCSR
// ----------------------------------------------------------
template <unsigned int BLOCK_SIZE, unsigned int WARP_SIZE, typename ValueType, typename IndexType>
__global__ void kernel_mcsr_spmv(const IndexType nrow,
                                 const IndexType nthreads,
                                 const IndexType *row_offset,
                                 const IndexType *col,
                                 const ValueType *val,
                                 const ValueType *in,
                                       ValueType *out) {

  IndexType gid = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType tid = threadIdx.x;
  IndexType lid = tid & (nthreads - 1);
  IndexType vid = gid / nthreads;
  IndexType nvec = gridDim.x * blockDim.x / nthreads;

  __shared__ volatile ValueType sdata[BLOCK_SIZE + WARP_SIZE / 2];

  for (IndexType ai=vid; ai<nrow; ai+=nvec) {

    IndexType row_begin = row_offset[ai];
    IndexType row_end = row_offset[ai+1];

    ValueType sum;
    make_ValueType(sum, 0.0);

    for(IndexType j=row_begin+lid; j<row_end; j+=nthreads) {
      sum = sum + val[j] * in[col[j]];
    }

    assign_volatile_ValueType(&sum, &sdata[tid]);

    __syncthreads();

    if (nthreads > 32) sum = add_volatile_ValueType(&sdata[tid+32], &sum); assign_volatile_ValueType(&sum, &sdata[tid]); __syncthreads();
    if (nthreads > 16) sum = add_volatile_ValueType(&sdata[tid+16], &sum); assign_volatile_ValueType(&sum, &sdata[tid]); __syncthreads();
    if (nthreads >  8) sum = add_volatile_ValueType(&sdata[tid+ 8], &sum); assign_volatile_ValueType(&sum, &sdata[tid]); __syncthreads();
    if (nthreads >  4) sum = add_volatile_ValueType(&sdata[tid+ 4], &sum); assign_volatile_ValueType(&sum, &sdata[tid]); __syncthreads();
    if (nthreads >  2) sum = add_volatile_ValueType(&sdata[tid+ 2], &sum); assign_volatile_ValueType(&sum, &sdata[tid]); __syncthreads();
    if (nthreads >  1) sum = add_volatile_ValueType(&sdata[tid+ 1], &sum);

    if (lid == 0) {
      out[ai] = sum + val[ai] * in[ai];
    }

  }

}

// ----------------------------------------------------------
// function spmv_csr_vector_kernel(...)
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.5.1, 
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - adapted interface
// - other modifications
// - modified for MCSR
// ----------------------------------------------------------
template <unsigned int BLOCK_SIZE, unsigned int WARP_SIZE, typename ValueType, typename IndexType>
__global__ void kernel_mcsr_add_spmv(const IndexType nrow,
                                     const IndexType nthreads,
                                     const IndexType *row_offset,
                                     const IndexType *col,
                                     const ValueType *val,
                                     const ValueType scalar,
                                     const ValueType *in,
                                           ValueType *out) {

  IndexType gid = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType tid = threadIdx.x;
  IndexType lid = tid & (nthreads - 1);
  IndexType vid = gid / nthreads;
  IndexType nvec = gridDim.x * blockDim.x / nthreads;

  __shared__ volatile ValueType sdata[BLOCK_SIZE + WARP_SIZE / 2];

  for (IndexType ai=vid; ai<nrow; ai+=nvec) {

    IndexType row_begin = row_offset[ai];
    IndexType row_end = row_offset[ai+1];

    ValueType sum;
    make_ValueType(sum, 0.0);

    for(IndexType j=row_begin+lid; j<row_end; j+=nthreads) {
      sum = sum + scalar * val[j] * in[col[j]];
    }

    assign_volatile_ValueType(&sum, &sdata[tid]);

    __syncthreads();

    if (nthreads > 32) sum = add_volatile_ValueType(&sdata[tid+32], &sum); assign_volatile_ValueType(&sum, &sdata[tid]); __syncthreads();
    if (nthreads > 16) sum = add_volatile_ValueType(&sdata[tid+16], &sum); assign_volatile_ValueType(&sum, &sdata[tid]); __syncthreads();
    if (nthreads >  8) sum = add_volatile_ValueType(&sdata[tid+ 8], &sum); assign_volatile_ValueType(&sum, &sdata[tid]); __syncthreads();
    if (nthreads >  4) sum = add_volatile_ValueType(&sdata[tid+ 4], &sum); assign_volatile_ValueType(&sum, &sdata[tid]); __syncthreads();
    if (nthreads >  2) sum = add_volatile_ValueType(&sdata[tid+ 2], &sum); assign_volatile_ValueType(&sum, &sdata[tid]); __syncthreads();
    if (nthreads >  1) sum = add_volatile_ValueType(&sdata[tid+ 1], &sum);

    if (lid == 0) {
      out[ai] = out[ai] + sum + scalar * val[ai] * in[ai];
    }

  }

}


}

#endif // PARALUTION_HIP_HIP_KERNELS_MCSR_HPP_
