#ifndef PARALUTION_OCL_KERNELS_DENSE_HPP_
#define PARALUTION_OCL_KERNELS_DENSE_HPP_

namespace paralution {

const char *ocl_kernels_dense = CL_KERNEL(

__kernel void kernel_dense_spmv(         const IndexType nrow,
                                         const IndexType ncol,
                                __global const ValueType *val,
                                __global const ValueType *in,
                                __global       ValueType *out) {

  IndexType ai = get_global_id(0);

  if (ai < nrow) {

    ValueType sum = ocl_set((ValueType) 0);

    for (IndexType aj=0; aj<ncol; ++aj)
      sum += ocl_mult(val[ai+aj*nrow], in[aj]);

    out[ai] = sum;

  }

}

// Replace column vector
__kernel void kernel_dense_replace_column_vector(__global const ValueType *vec,
                                                          const IndexType idx,
                                                          const IndexType nrow,
                                                 __global       ValueType *mat) {

  IndexType ai = get_global_id(0);

  if(ai < nrow)
    mat[ai+idx*nrow] = vec[ai];

}

// Replace row vector
__kernel void kernel_dense_replace_row_vector(__global const ValueType *vec,
                                                       const IndexType idx,
                                                       const IndexType nrow,
                                                       const IndexType ncol,
                                              __global       ValueType *mat) {

  IndexType aj = get_global_id(0);

  if (aj < ncol)
    mat[idx+aj*nrow] = vec[aj];

}

// Extract column vector
__kernel void kernel_dense_extract_column_vector(__global       ValueType *vec,
                                                          const IndexType idx,
                                                          const IndexType nrow,
                                                 __global const ValueType *mat) {

  IndexType ai = get_global_id(0);

  if (ai < nrow)
    vec[ai] = mat[ai+idx*nrow];

}

// Extract row vector
__kernel void kernel_dense_extract_row_vector(__global       ValueType *vec,
                                                       const IndexType idx,
                                                       const IndexType nrow,
                                                       const IndexType ncol,
                                              __global const ValueType *mat) {

  IndexType aj = get_global_id(0);

  if (aj < ncol)
    vec[aj] = mat[idx+aj*nrow];

}

);

}

#endif // PARALUTION_OCL_KERNELS_DENSE_HPP_
