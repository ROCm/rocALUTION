#ifndef PARALUTION_OCL_KERNELS_CSR_HPP_
#define PARALUTION_OCL_KERNELS_CSR_HPP_

namespace paralution {

const char *ocl_kernels_csr = CL_KERNEL(

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
// ----------------------------------------------------------
__kernel void kernel_csr_spmv(         const IndexType nrow,
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
      out[ai] = sum;
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
// ----------------------------------------------------------
__kernel void kernel_csr_add_spmv(         const IndexType nrow,
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
      out[ai] += sum;
    }

  }

}

__kernel void kernel_csr_scale_diagonal(         const IndexType nrow,
                                        __global const IndexType *row_offset,
                                        __global const IndexType *col,
                                                 const ValueType alpha,
                                        __global       ValueType *val) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow)
    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj)
      if (ai == col[aj])
        val[aj] = ocl_mult(alpha, val[aj]);

}

__kernel void kernel_csr_scale_offdiagonal(         const IndexType nrow,
                                           __global const IndexType *row_offset,
                                           __global const IndexType *col,
                                                    const ValueType alpha,
                                           __global       ValueType *val) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow)
    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj)
      if (ai != col[aj])
        val[aj] = ocl_mult(alpha, val[aj]);

}

__kernel void kernel_csr_add_diagonal(         const IndexType nrow,
                                      __global const IndexType *row_offset,
                                      __global const IndexType *col,
                                               const ValueType alpha,
                                      __global       ValueType *val) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow)
    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj)
      if (ai == col[aj])
        val[aj] += alpha;

}

__kernel void kernel_csr_add_offdiagonal(         const IndexType nrow,
                                         __global const IndexType *row_offset,
                                         __global const IndexType *col,
                                                  const ValueType alpha,
                                         __global       ValueType *val) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow)
    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj)
      if (ai != col[aj])
        val[aj] += alpha;

}

__kernel void kernel_csr_extract_diag(         const IndexType nrow,
                                      __global const IndexType *row_offset,
                                      __global const IndexType *col,
                                      __global const ValueType *val,
                                      __global       ValueType *vec) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow)
    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj)
      if (ai == col[aj])
        vec[ai] = val[aj];

}

__kernel void kernel_csr_extract_inv_diag(         const IndexType nrow,
                                          __global const IndexType *row_offset,
                                          __global const IndexType *col,
                                          __global const ValueType *val,
                                          __global       ValueType *vec) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow)
    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj)
      if (ai == col[aj])
        vec[ai] = ocl_div(ocl_set((ValueType) 1), val[aj]);

}

__kernel void kernel_csr_extract_submatrix_row_nnz(__global const IndexType *row_offset,
                                                   __global const IndexType *col,
                                                   __global const ValueType *val,
                                                            const IndexType smrow_offset,
                                                            const IndexType smcol_offset,
                                                            const IndexType smrow_size,
                                                            const IndexType smcol_size,
                                                   __global       IndexType *row_nnz) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < smrow_size) {

    IndexType nnz = 0;
    IndexType ind = ai + smrow_offset;

    for (aj=row_offset[ind]; aj<row_offset[ind+1]; ++aj)

      if ( (col[aj] >= smcol_offset) &&
           (col[aj] <  smcol_offset + smcol_size) )
        ++nnz;

    row_nnz[ai] = nnz;

  }

}

__kernel void kernel_csr_extract_submatrix_copy(__global const IndexType *row_offset,
                                                __global const IndexType *col,
                                                __global const ValueType *val,
                                                         const IndexType smrow_offset,
                                                         const IndexType smcol_offset,
                                                         const IndexType smrow_size,
                                                         const IndexType smcol_size,
                                                __global const IndexType *sm_row_offset,
                                                __global       IndexType *sm_col,
                                                __global       ValueType *sm_val) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < smrow_size) {

    IndexType row_nnz = sm_row_offset[ai];
    IndexType ind = ai + smrow_offset;

    for (aj=row_offset[ind]; aj<row_offset[ind+1]; ++aj) {

      if ( (col[aj] >= smcol_offset) &&
           (col[aj] <  smcol_offset + smcol_size) ) {

        sm_col[row_nnz] = col[aj] - smcol_offset;
        sm_val[row_nnz] = val[aj];
        ++row_nnz;

      }

    }

  }

}

__kernel void kernel_csr_diagmatmult_r(         const IndexType nrow,
                                       __global const IndexType *row_offset,
                                       __global const IndexType *col,
                                       __global const ValueType *diag,
                                       __global       ValueType *val) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow)
    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj)
      val[aj] = ocl_mult(val[aj], diag[ col[aj] ]);

}

__kernel void kernel_csr_diagmatmult_l(         const IndexType nrow,
                                       __global const IndexType *row_offset,
                                       __global const ValueType *diag,
                                       __global       ValueType *val) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow)
    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj)
      val[aj] = val[aj] * diag[ai];

}

__kernel void kernel_csr_add_csr_same_struct(         const IndexType nrow,
                                             __global const IndexType *out_row_offset,
                                             __global const IndexType *out_col,
                                             __global const IndexType *in_row_offset,
                                             __global const IndexType *in_col,
                                             __global const ValueType *in_val,
                                                      const ValueType alpha,
                                                      const ValueType beta,
                                             __global       ValueType *out_val) {

  IndexType ai = get_global_id(0);
  IndexType aj, ajj;

  if (ai < nrow) {

    IndexType first_col = in_row_offset[ai];
      
    for (ajj=out_row_offset[ai]; ajj<out_row_offset[ai+1]; ++ajj)
      for (aj=first_col; aj<in_row_offset[ai+1]; ++aj)
        if (in_col[aj] == out_col[ajj]) {
          
          out_val[ajj] = ocl_mult(alpha, out_val[ajj]) + ocl_mult(beta, in_val[aj]);
          ++first_col;
          break;

        }

  }

}

__kernel void kernel_buffer_addscalar(const IndexType size, const ValueType scalar, __global ValueType *buff) {

  IndexType gid = get_global_id(0);

  if (gid < size)
    buff[gid] += scalar;

}

__kernel void kernel_reverse_index(const IndexType size, __global const IndexType *perm, __global IndexType *out) {

  IndexType gid = get_global_id(0);

  if (gid < size)
    out[perm[gid]] = gid;

}

__kernel void kernel_csr_calc_row_nnz(         const IndexType nrow,
                                      __global const IndexType *row_offset,
                                      __global       IndexType *row_nnz) {

  IndexType ai = get_global_id(0);

  if (ai < nrow)
    row_nnz[ai] = row_offset[ai+1]-row_offset[ai];

}

__kernel void kernel_csr_permute_row_nnz(         const IndexType nrow,
                                         __global const IndexType *row_nnz_src,
                                         __global const IndexType *perm_vec,
                                         __global       IndexType *row_nnz_dst) {

  IndexType ai = get_global_id(0);

  if (ai < nrow)
    row_nnz_dst[perm_vec[ai]] = row_nnz_src[ai];

}

__kernel void kernel_csr_permute_rows(       const IndexType nrow,
                                    __global const IndexType *row_offset,
                                    __global const IndexType *perm_row_offset,
                                    __global const IndexType *col,
                                    __global const ValueType *data,
                                    __global const IndexType *perm_vec,
                                    __global const IndexType *row_nnz,
                                    __global       IndexType *perm_col,
                                    __global       ValueType *perm_data) {

  IndexType ai = get_global_id(0);

  if (ai < nrow) {

    IndexType num_elems = row_nnz[ai];
    IndexType perm_index = perm_row_offset[perm_vec[ai]];
    IndexType prev_index = row_offset[ai];

    for (IndexType i = 0; i < num_elems; ++i) {
      perm_data[perm_index + i] = data[prev_index + i];
      perm_col[perm_index + i]  = col[prev_index + i];
    }

  }

}

__kernel void kernel_csr_permute_cols(         const IndexType nrow,
                                      __global const IndexType *row_offset,
                                      __global const IndexType *perm_vec,
                                      __global const IndexType *row_nnz,
                                      __global const IndexType *perm_col,
                                      __global const ValueType *perm_data,
                                      __global       IndexType *col,
                                      __global       ValueType *data) {

  IndexType ai = get_global_id(0);
  IndexType j;

  if (ai < nrow) {

    IndexType num_elems = row_nnz[ai];
    IndexType elem_index = row_offset[ai];

    for (IndexType i = 0; i < num_elems; ++i) {

      IndexType comp = perm_vec[perm_col[elem_index+i]];

      for (j = i-1; j >= 0 ; --j) {

        if (col[elem_index+j]>comp) {
          data[elem_index+j+1] = data[elem_index+j];
          col[elem_index+j+1]  = col[elem_index+j];
        } else
          break;
      }

      data[elem_index+j+1] = perm_data[elem_index+i];
      col[elem_index+j+1]  = comp;

    }

  }

}

__kernel void kernel_csr_extract_l_triangular(         const IndexType nrow,
                                              __global const IndexType *src_row_offset,
                                              __global const IndexType *src_col,
                                              __global const ValueType *src_val,
                                              __global       IndexType *nnz_per_row,
                                              __global       IndexType *dst_col,
                                              __global       ValueType *dst_val) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow) {

      IndexType dst_index = nnz_per_row[ai];
      IndexType src_index = src_row_offset[ai];

    for (aj=0; aj<nnz_per_row[ai+1]-nnz_per_row[ai]; ++aj) {
      
      dst_col[dst_index] = src_col[src_index];
      dst_val[dst_index] = src_val[src_index];
      
      ++dst_index;
      ++src_index;
      
    }
  }

}

__kernel void kernel_csr_extract_u_triangular(         const IndexType nrow,
                                              __global const IndexType *src_row_offset,
                                              __global const IndexType *src_col,
                                              __global const ValueType *src_val,
                                              __global IndexType *nnz_per_row,
                                              __global IndexType *dst_col,
                                              __global ValueType *dst_val) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow) {

      IndexType num_elements = nnz_per_row[ai+1]-nnz_per_row[ai];
      IndexType dst_index = nnz_per_row[ai];
      IndexType src_index = src_row_offset[ai+1]-num_elements;

    for (aj=0; aj<num_elements; ++aj) {
      
      dst_col[dst_index] = src_col[src_index];
      dst_val[dst_index] = src_val[src_index];
      
      ++dst_index;
      ++src_index;
      
    }

  }

}

__kernel void kernel_csr_slower_nnz_per_row(         const IndexType nrow,
                                            __global const IndexType *src_row_offset,
                                            __global const IndexType *src_col,
                                            __global       IndexType *nnz_per_row) {

  IndexType ai = get_global_id(0);
  IndexType aj;
  
  if (ai < nrow) {
    nnz_per_row[ai+1] = 0;
    for (aj=src_row_offset[ai]; aj<src_row_offset[ai+1]; ++aj)
      if (src_col[aj] < ai)
        ++nnz_per_row[ai+1];
  }

}

__kernel void kernel_csr_supper_nnz_per_row(         const IndexType nrow,
                                            __global const IndexType *src_row_offset,
                                            __global const IndexType *src_col,
                                            __global       IndexType *nnz_per_row) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow) {
    nnz_per_row[ai+1] = 0;
    for (aj=src_row_offset[ai]; aj<src_row_offset[ai+1]; ++aj)
      if (src_col[aj] > ai)
        ++nnz_per_row[ai+1];
  }

}

__kernel void kernel_csr_lower_nnz_per_row(         const IndexType nrow,
                                           __global const IndexType *src_row_offset,
                                           __global const IndexType *src_col,
                                           __global       IndexType *nnz_per_row) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow) {
    nnz_per_row[ai+1] = 0;
    for (aj=src_row_offset[ai]; aj<src_row_offset[ai+1]; ++aj)
      if (src_col[aj] <= ai)
        ++nnz_per_row[ai+1];
  }

}

__kernel void kernel_csr_upper_nnz_per_row(         const IndexType nrow,
                                           __global const IndexType *src_row_offset,
                                           __global const IndexType *src_col,
                                           __global       IndexType *nnz_per_row) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow) {
    nnz_per_row[ai+1] = 0;
    for (aj=src_row_offset[ai]; aj<src_row_offset[ai+1]; ++aj)
      if (src_col[aj] >= ai)
        ++nnz_per_row[ai+1];
  }

}

__kernel void kernel_csr_compress_count_nrow(__global const IndexType *row_offset,
                                             __global const IndexType *col,
                                             __global const ValueType *val,
                                                      const IndexType nrow,
                                                      const double drop_off,
                                             __global       IndexType *row_offset_new) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow) {
    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {

      if ((ocl_abs(val[aj]) > drop_off) || (col[aj] == ai))
        row_offset_new[ai]++;
    }
  }

}

__kernel void kernel_csr_compress_copy(__global const IndexType *row_offset,
                                       __global const IndexType *col,
                                       __global const ValueType *val,
                                                const IndexType nrow,
                                                const double drop_off,
                                       __global const IndexType *row_offset_new,
                                       __global       IndexType *col_new,
                                       __global       ValueType *val_new) {

  IndexType ai = get_global_id(0);
  IndexType aj;
  IndexType ajj = row_offset_new[ai];

  if (ai < nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {

      if ((ocl_abs(val[aj]) > drop_off) || (col[aj] == ai)) {
        col_new[ajj] = col[aj];
        val_new[ajj] = val[aj];
        ajj++;
      }
    }

  }

}

// Extract column vector
__kernel void kernel_csr_extract_column_vector(__global const IndexType *row_offset,
                                               __global const IndexType *col,
                                               __global const ValueType *val,
                                                        const IndexType nrow,
                                                        const IndexType idx,
                                               __global       ValueType *vec) {

  IndexType ai = get_global_id(0);
  IndexType aj;

  if (ai < nrow) {

    vec[ai] = ocl_set((ValueType) 0);

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj)
      if (idx == col[aj])
        vec[ai] = val[aj];

  }

}

// Replace column vector - compute new offset
__kernel void kernel_csr_replace_column_vector_offset(__global const IndexType *row_offset,
                                                      __global const IndexType *col,
                                                               const IndexType nrow,
                                                               const IndexType idx,
                                                      __global const ValueType *vec,
                                                      __global IndexType *offset) {

  IndexType ai = get_global_id(0);
  IndexType aj;
  IndexType add = 1;

  if (ai < nrow) {

    offset[ai+1] = row_offset[ai+1] - row_offset[ai];

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {
      if (col[aj] == idx) {
        add = 0;
        break;
      }
    }

    ValueType zero = ocl_set((ValueType) 0);

    if (add == 1 && ocl_nequal(vec[ai], zero))
      ++offset[ai+1];

    if (add == 0 && ocl_equal(vec[ai], zero))
      --offset[ai+1];

  }

}

// Replace column vector - compute new offset
__kernel void kernel_csr_replace_column_vector(__global const IndexType *row_offset,
                                               __global const IndexType *col,
                                               __global const ValueType *val,
                                                        const IndexType nrow,
                                                        const IndexType idx,
                                               __global const ValueType *vec,
                                               __global const IndexType *offset,
                                               __global       IndexType *new_col,
                                               __global       ValueType *new_val) {

  IndexType ai = get_global_id(0);
  IndexType aj = row_offset[ai];
  IndexType k  = offset[ai];

  if (ai < nrow) {

    for (; aj<row_offset[ai+1]; ++aj) {
      if (col[aj] < idx) {
        new_col[k] = col[aj];
        new_val[k] = val[aj];
        ++k;
      } else
        break;
    }

    ValueType zero = ocl_set((ValueType) 0);

    if (ocl_nequal(vec[ai], zero)) {
      new_col[k] = idx;
      new_val[k] = vec[ai];
      ++k;
      ++aj;
    }

    for (; aj<row_offset[ai+1]; ++aj) {
      if (col[aj] > idx) {
        new_col[k] = col[aj];
        new_val[k] = val[aj];
        ++k;
      }
    }

  }

}

);

}

#endif // PARALUTION_OCL_KERNELS_CSR_HPP_
