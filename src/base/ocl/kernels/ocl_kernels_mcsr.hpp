#ifndef PARALUTION_OCL_KERNELS_MCSR_HPP_
#define PARALUTION_OCL_KERNELS_MCSR_HPP_

namespace paralution {

const char *ocl_kernels_mcsr = CL_KERNEL(

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
__kernel void kernel_mcsr_spmv(         const IndexType nrow,
                                        const IndexType nthreads,
                               __global const IndexType *row_offset,
                               __global const IndexType *col,
                               __global const ValueType *val,
                               __global const ValueType *in,
                               __global       ValueType *out) {

  IndexType gid = get_global_id(0);
  IndexType tid = get_local_id(0);
  IndexType lid = tid & (nthreads - 1);
  IndexType vid = gid / nthreads;
  IndexType nvec = get_global_size(0) / nthreads;

  __local volatile ValueType sdata[BLOCK_SIZE + WARP_SIZE / 2];

  for (IndexType ai=vid; ai<nrow; ai+=nvec) {

    IndexType row_begin = row_offset[ai];
    IndexType row_end = row_offset[ai+1];

    ValueType sum = ocl_set((ValueType) 0);

    for(IndexType j=row_begin+lid; j<row_end; j+=nthreads) {
      sum = sum + ocl_mult(val[j], in[col[j]]);
    }

    sdata[tid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (nthreads > 32) sdata[tid] = sum = sum + sdata[tid + 32]; barrier(CLK_LOCAL_MEM_FENCE);
    if (nthreads > 16) sdata[tid] = sum = sum + sdata[tid + 16]; barrier(CLK_LOCAL_MEM_FENCE);
    if (nthreads >  8) sdata[tid] = sum = sum + sdata[tid +  8]; barrier(CLK_LOCAL_MEM_FENCE);
    if (nthreads >  4) sdata[tid] = sum = sum + sdata[tid +  4]; barrier(CLK_LOCAL_MEM_FENCE);
    if (nthreads >  2) sdata[tid] = sum = sum + sdata[tid +  2]; barrier(CLK_LOCAL_MEM_FENCE);
    if (nthreads >  1)              sum = sum + sdata[tid +  1];

    if (lid == 0) {
      out[ai] = sum + ocl_mult(val[ai], in[ai]);
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
__kernel void kernel_mcsr_add_spmv(         const IndexType nrow,
                                            const IndexType nthreads,
                                   __global const IndexType *row_offset,
                                   __global const IndexType *col,
                                   __global const ValueType *val,
                                            const ValueType scalar,
                                   __global const ValueType *in,
                                   __global       ValueType *out) {

  IndexType gid = get_global_id(0);
  IndexType tid = get_local_id(0);
  IndexType lid = tid & (nthreads - 1);
  IndexType vid = gid / nthreads;
  IndexType nvec = get_global_size(0) / nthreads;

  __local volatile ValueType sdata[BLOCK_SIZE + WARP_SIZE / 2];

  for (IndexType ai=vid; ai<nrow; ai+=nvec) {

    IndexType row_begin = row_offset[ai];
    IndexType row_end = row_offset[ai+1];

    ValueType sum = ocl_set((ValueType) 0);

    for(IndexType j=row_begin+lid; j<row_end; j+=nthreads) {
      sum = sum + ocl_mult(scalar, ocl_mult(val[j], in[col[j]]));
    }

    sdata[tid] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (nthreads > 32) sdata[tid] = sum = sum + sdata[tid + 32]; barrier(CLK_LOCAL_MEM_FENCE);
    if (nthreads > 16) sdata[tid] = sum = sum + sdata[tid + 16]; barrier(CLK_LOCAL_MEM_FENCE);
    if (nthreads >  8) sdata[tid] = sum = sum + sdata[tid +  8]; barrier(CLK_LOCAL_MEM_FENCE);
    if (nthreads >  4) sdata[tid] = sum = sum + sdata[tid +  4]; barrier(CLK_LOCAL_MEM_FENCE);
    if (nthreads >  2) sdata[tid] = sum = sum + sdata[tid +  2]; barrier(CLK_LOCAL_MEM_FENCE);
    if (nthreads >  1)              sum = sum + sdata[tid +  1];

    if (lid == 0) {
      out[ai] += sum + ocl_mult(scalar, ocl_mult(val[ai], in[ai]));
    }

  }

}

);

}

#endif // PARALUTION_OCL_KERNELS_MCSR_HPP_
