#ifndef PARALUTION_HIP_HIP_KERNELS_CSR_HPP_
#define PARALUTION_HIP_HIP_KERNELS_CSR_HPP_

#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace paralution {

// Nathan Bell and Michael Garland
// Efficient Sparse Matrix-Vector Multiplication
// NVR-2008-004 / NVIDIA Technical Report
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_spmv_scalar(const IndexType nrow, const IndexType *row_offset, 
                                       const IndexType *col, const ValueType *val, 
                                       const ValueType *in, ValueType *out) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai <nrow) {

    out[ai] = ValueType(0.0);

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {
      out[ai] = out[ai] + val[aj]*in[col[aj]];
    }

  }
}

// Nathan Bell and Michael Garland
// Efficient Sparse Matrix-Vector Multiplication on
// NVR-2008-004 / NVIDIA Technical Report
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_add_spmv_scalar(const IndexType nrow, const IndexType *row_offset, 
                                           const IndexType *col, const ValueType *val, 
                                           const ValueType scalar,
                                           const ValueType *in, ValueType *out) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai <nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {
      out[ai] = out[ai] + scalar*val[aj]*in[col[aj]];
    }

  }
}


template <typename ValueType, typename IndexType>
__global__ void kernel_csr_scale_diagonal(const IndexType nrow, const IndexType *row_offset, 
                                          const IndexType *col, const ValueType alpha, ValueType *val) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai <nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {
      if (ai == col[aj])
        val[aj] = alpha*val[aj];
    }

  }
}


template <typename ValueType, typename IndexType>
__global__ void kernel_csr_scale_offdiagonal(const IndexType nrow, const IndexType *row_offset, 
                                             const IndexType *col, const ValueType alpha, ValueType *val) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai <nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {
      if (ai != col[aj])
        val[aj] = alpha*val[aj];
    }

  }
}


template <typename ValueType, typename IndexType>
__global__ void kernel_csr_add_diagonal(const IndexType nrow, const IndexType *row_offset, 
                                        const IndexType *col, const ValueType alpha, ValueType *val) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai <nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {
      if (ai == col[aj])
        val[aj] = val[aj] + alpha;
    }

  }
}


template <typename ValueType, typename IndexType>
__global__ void kernel_csr_add_offdiagonal(const IndexType nrow, const IndexType *row_offset, 
                                           const IndexType *col, const ValueType alpha, ValueType *val) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai <nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {
      if (ai != col[aj])
        val[aj] = val[aj] + alpha;
    }

  }
}


template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_diag(const IndexType nrow, const IndexType *row_offset, const IndexType *col, const ValueType *val,
                                        ValueType *vec) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai <nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {
      if (ai == col[aj])
        vec[ai] = val[aj];
    }

  }
}


template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_inv_diag(const IndexType nrow, const IndexType *row_offset,
                                            const IndexType *col, const ValueType *val, ValueType *vec) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai <nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj)
      if (ai == col[aj]) {
        make_ValueType(vec[ai], 1.0);
        vec[ai] = vec[ai] / val[aj];
      }

  }

}

template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_submatrix_row_nnz(const IndexType *row_offset, const IndexType *col, const ValueType *val,
                                                     const IndexType smrow_offset, const IndexType smcol_offset, 
                                                     const IndexType smrow_size, const IndexType smcol_size,
                                                     IndexType *row_nnz) {
  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai <smrow_size) {

    IndexType nnz = 0 ;

    IndexType ind = ai+smrow_offset;

    for (aj=row_offset[ind]; aj<row_offset[ind+1]; ++aj) {

      IndexType c = col[aj];

      if ((c >= smcol_offset) &&
          (c < smcol_offset + smcol_size) )
        ++nnz;
    
    }
    
    row_nnz[ai] = nnz;

  }

}


template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_submatrix_copy(const IndexType *row_offset, const IndexType *col, const ValueType *val,
                                                  const IndexType smrow_offset, const IndexType smcol_offset, 
                                                  const IndexType smrow_size, const IndexType smcol_size,
                                                  const IndexType *sm_row_offset, IndexType *sm_col, ValueType *sm_val) {
  
  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai <smrow_size) {

    IndexType row_nnz = sm_row_offset[ai];
    IndexType ind = ai+smrow_offset;

    for (aj=row_offset[ind]; aj<row_offset[ind+1]; ++aj) {

      IndexType c = col[aj];
      if ((c >= smcol_offset) &&
          (c < smcol_offset + smcol_size) ) {

        sm_col[row_nnz] = c - smcol_offset;
        sm_val[row_nnz] = val[aj];
        ++row_nnz;

      }

    }

  }

}

template <typename ValueType, typename IndexType>
__global__ void kernel_csr_diagmatmult_r(const IndexType nrow, const IndexType *row_offset, 
                                       const IndexType *col, 
                                       const ValueType *diag, 
                                       ValueType *val) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai <nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {
      val[aj] = val[aj] * diag[ col[aj] ] ; 
    }

  }
}

template <typename ValueType, typename IndexType>
__global__ void kernel_csr_diagmatmult_l(const IndexType nrow, const IndexType *row_offset, 
                                       const ValueType *diag, 
                                       ValueType *val) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai <nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {
      val[aj] = val[aj] * diag[ ai ] ; 
    }

  }
}

// Calculates the number of non-zero elements per row
template <typename IndexType>
__global__ void kernel_calc_row_nnz( const IndexType nrow,
                              const IndexType *row_offset,
                              IndexType *row_nnz){
  IndexType ai = blockIdx.x*blockDim.x + threadIdx.x;
  if(ai < nrow){
    row_nnz[ai] = row_offset[ai+1]-row_offset[ai];
  }
}

// Performs a permutation on the vector of non-zero elements per row
//
// Inputs:   nrow:         number of rows in matrix
//           row_nnz_src:  original number of non-zero elements per row 
//           perm_vec:     permutation vector
// Outputs:  row_nnz_dst   permuted number of non-zero elements per row
template <typename IndexType>
__global__ void kernel_permute_row_nnz(const IndexType nrow,
                                       const IndexType *row_nnz_src,
                                       const IndexType *perm_vec,
                                       IndexType *row_nnz_dst) {

  IndexType ai = blockIdx.x*blockDim.x + threadIdx.x;

  if (ai < nrow) {
    row_nnz_dst[perm_vec[ai]] = row_nnz_src[ai];
																}
}

// Permutes rows
// 
// Inputs:   nrow:             number of rows in matrix
//           row_offset:       original row pointer
//           perm_row_offset:  permuted row pointer
//           col:              original column indices of elements
//           data:             original data vector
//           perm_vec:         permutation vector
//           row_nnz:          number of non-zero elements per row
// Outputs:  perm_col:         permuted column indices of elements
//           perm_data:        permuted data vector
template <typename ValueType, typename IndexType>
__global__ void kernel_permute_rows(const IndexType nrow,
                                    const IndexType *row_offset,
                                    const IndexType *perm_row_offset,
                                    const IndexType *col,
                                    const ValueType *data,
                                    const IndexType *perm_vec,
                                    const IndexType *row_nnz,
                                    IndexType *perm_col,
                                    ValueType *perm_data) {

  IndexType ai = blockIdx.x*blockDim.x + threadIdx.x;

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

// Permutes columns
//
// Inputs:   nrow:             number of rows in matrix
//           row_offset:       row pointer
//           perm_vec:         permutation vector
//           row_nnz:          number of non-zero elements per row
//           perm_col:         row-permuted column indices of elements
//           perm_data:        row-permuted data
// Outputs:  col:              fully permuted column indices of elements
//           data:             fully permuted data
template <unsigned int size, typename ValueType, typename IndexType>
__global__ void kernel_permute_cols(const IndexType nrow,
                                    const IndexType *row_offset,
                                    const IndexType *perm_vec,
                                    const IndexType *row_nnz,
                                    const IndexType *perm_col,
                                    const ValueType *perm_data,
                                    IndexType *col,
                                    ValueType *data) {

  IndexType ai = blockIdx.x*blockDim.x + threadIdx.x;
  IndexType j;

  IndexType ccol[size];
  ValueType cval[size];

  if (ai < nrow) {

    IndexType num_elems = row_nnz[ai];
    IndexType elem_index = row_offset[ai];

    for (IndexType i=0; i<num_elems; ++i) {
      ccol[i] = col[elem_index+i];
      cval[i] = data[elem_index+i];
    }

    for (IndexType i = 0; i < num_elems; ++i) {

      IndexType comp = perm_vec[perm_col[elem_index+i]];

      for (j = i-1; j >= 0 ; --j) {
        IndexType c = ccol[j];
        if(c>comp){
          cval[j+1] = cval[j];
          ccol[j+1] = c;
        } else
          break;
      }

      cval[j+1] = perm_data[elem_index+i];
      ccol[j+1] = comp;

    }

    for (IndexType i=0; i<num_elems; ++i) {
      col[elem_index+i] = ccol[i];
      data[elem_index+i] = cval[i];
    }

  }

}

// Permutes columns
//
// Inputs:   nrow:             number of rows in matrix
//           row_offset:       row pointer
//           perm_vec:         permutation vector
//           row_nnz:          number of non-zero elements per row
//           perm_col:         row-permuted column indices of elements
//           perm_data:        row-permuted data
// Outputs:  col:              fully permuted column indices of elements
//           data:             fully permuted data
template <typename ValueType, typename IndexType>
__global__ void kernel_permute_cols_fallback(const IndexType nrow,
                                             const IndexType *row_offset,
                                             const IndexType *perm_vec,
                                             const IndexType *row_nnz,
                                             const IndexType *perm_col,
                                             const ValueType *perm_data,
                                             IndexType *col,
                                             ValueType *data) {

  IndexType ai = blockIdx.x*blockDim.x + threadIdx.x;
  IndexType j;

  if (ai < nrow) {

    IndexType num_elems = row_nnz[ai];
    IndexType elem_index = row_offset[ai];

    for (IndexType i = 0; i < num_elems; ++i) {

      IndexType comp = perm_vec[perm_col[elem_index+i]];

      for (j = i-1; j >= 0 ; --j) {
        IndexType c = col[elem_index+j];
        if(c>comp){
          data[elem_index+j+1] = data[elem_index+j];
          col[elem_index+j+1] = c;
        } else
          break;
      }

      data[elem_index+j+1] = perm_data[elem_index+i];
      col[elem_index+j+1] = comp;

    }

  }

}

// TODO
// kind of ugly and inefficient ... but works 
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_add_csr_same_struct(const IndexType nrow,
                                               const IndexType *out_row_offset, const IndexType *out_col,
                                               const IndexType *in_row_offset,  const IndexType *in_col, const ValueType *in_val,
                                               const ValueType alpha, const ValueType beta,
                                               ValueType *out_val) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj, ajj;

  if (ai <nrow) {

    IndexType first_col = in_row_offset[ai];
      
    for (ajj=out_row_offset[ai]; ajj<out_row_offset[ai+1]; ++ajj)
      for (aj=first_col; aj<in_row_offset[ai+1]; ++aj)
        if (in_col[aj] == out_col[ajj]) {
          
          out_val[ajj] = alpha*out_val[ajj] + beta*in_val[aj];
          ++first_col;
          break ; 
          
        }
  }

}


// Computes the lower triangular part nnz per row
template <typename IndexType>
__global__ void kernel_csr_lower_nnz_per_row(const IndexType nrow, const IndexType *src_row_offset,
                                             const IndexType *src_col, IndexType *nnz_per_row) {
  
  IndexType ai = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType aj;
  
  if (ai < nrow) {
    nnz_per_row[ai] = 0;
    for (aj=src_row_offset[ai]; aj<src_row_offset[ai+1]; ++aj)
      if (src_col[aj] <= ai)
        ++nnz_per_row[ai];
  }
}

// Computes the upper triangular part nnz per row
template <typename IndexType>
__global__ void kernel_csr_upper_nnz_per_row(const IndexType nrow, const IndexType *src_row_offset,
                                             const IndexType *src_col, IndexType *nnz_per_row) {
  
  IndexType ai = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType aj;
  
  if (ai < nrow) {
    nnz_per_row[ai] = 0;
    for (aj=src_row_offset[ai]; aj<src_row_offset[ai+1]; ++aj)
      if (src_col[aj] >= ai)
        ++nnz_per_row[ai];
  }
}
  
// Computes the stricktly lower triangular part nnz per row
template <typename IndexType>
__global__ void kernel_csr_slower_nnz_per_row(const IndexType nrow, const IndexType *src_row_offset,
                                              const IndexType *src_col, IndexType *nnz_per_row) {
  
  IndexType ai = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType aj;
  
  if (ai < nrow) {
    nnz_per_row[ai] = 0;
    for (aj=src_row_offset[ai]; aj<src_row_offset[ai+1]; ++aj)
      if (src_col[aj] < ai)
        ++nnz_per_row[ai];
  }
}


// Computes the stricktly upper triangular part nnz per row
template <typename IndexType>
__global__ void kernel_csr_supper_nnz_per_row(const IndexType nrow, const IndexType *src_row_offset,
                                              const IndexType *src_col, IndexType *nnz_per_row) {
  
  IndexType ai = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType aj;
  
  if (ai < nrow) {
    nnz_per_row[ai] = 0;
    for (aj=src_row_offset[ai]; aj<src_row_offset[ai+1]; ++aj)
      if (src_col[aj] > ai)
        ++nnz_per_row[ai];
  }
}


// Extracts lower triangular part for given nnz per row array (partial sums nnz)
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_l_triangular(const IndexType nrow,
                                                const IndexType *src_row_offset, const IndexType *src_col,
                                                const ValueType *src_val, IndexType *nnz_per_row,
                                                IndexType *dst_col, ValueType *dst_val) {
  
  IndexType ai = blockIdx.x * blockDim.x + threadIdx.x;
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


// Extracts upper triangular part for given nnz per row array (partial sums nnz)
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_u_triangular(const IndexType nrow,
                                                const IndexType *src_row_offset, const IndexType *src_col,
                                                const ValueType *src_val, IndexType *nnz_per_row,
                                                IndexType *dst_col, ValueType *dst_val) {
  
  IndexType ai = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType aj;
  
  if (ai < nrow) {

    IndexType num_elements = nnz_per_row[ai+1]-nnz_per_row[ai];
    IndexType src_index = src_row_offset[ai+1]-num_elements;
    IndexType dst_index = nnz_per_row[ai];

    for (aj=0; aj<num_elements; ++aj) {

      dst_col[dst_index] = src_col[src_index];
      dst_val[dst_index] = src_val[src_index];

      ++dst_index;
      ++src_index;
      
    }
  }
}


// Compress 
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_compress_count_nrow(const IndexType *row_offset, const IndexType *col, const ValueType *val,
                                               const IndexType nrow,
                                               const double drop_off, 
                                               IndexType *row_offset_new) {
  
  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai <nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {

      if ( (hip_abs(val[aj]) > drop_off) ||
           ( col[aj] == ai))
        row_offset_new[ai]++;
    }
  }


}

// Compress 
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_compress_copy(const IndexType *row_offset, const IndexType *col, const ValueType *val,
                                         const IndexType nrow,
                                         const double drop_off, 
                                         const IndexType *row_offset_new,
                                         IndexType *col_new,
                                         ValueType *val_new) {
  
  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;
  IndexType ajj = row_offset_new[ai];

  if (ai <nrow) {

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj) {

      if ( (hip_abs(val[aj]) > drop_off) ||
           ( col[aj] == ai)) {
        col_new[ajj] = col[aj];
        val_new[ajj] = val[aj];
        ajj++;
      }
    }
  }

}

// Extract column vector
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_column_vector(const IndexType *row_offset, const IndexType *col, const ValueType *val,
                                                 const IndexType nrow, const IndexType idx, ValueType *vec) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj;

  if (ai < nrow) {

    make_ValueType(vec[ai], 0.0);

    for (aj=row_offset[ai]; aj<row_offset[ai+1]; ++aj)
      if (idx == col[aj])
        vec[ai] = val[aj];

  }

}

// Replace column vector - compute new offset
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_replace_column_vector_offset(const IndexType *row_offset, const IndexType *col,
                                                        const IndexType nrow, const IndexType idx,
                                                        const ValueType *vec, IndexType *offset) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
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

    if (add == 1 && hip_abs(vec[ai]) != 0.0)
      ++offset[ai+1];

    if (add == 0 && hip_abs(vec[ai]) == 0.0)
      --offset[ai+1];

  }

}

// Replace column vector - compute new offset
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_replace_column_vector(const IndexType *row_offset, const IndexType *col, const ValueType *val,
                                                 const IndexType nrow, const IndexType idx,
                                                 const ValueType *vec, const IndexType *offset,
                                                 IndexType *new_col, ValueType *new_val) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
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

    if (hip_abs(vec[ai]) != 0.0) {
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

// Extract row vector
template <typename ValueType, typename IndexType>
__global__ void kernel_csr_extract_row_vector(const IndexType *row_offset, const IndexType *col, const ValueType *val,
                                              const IndexType row_nnz, const IndexType idx, ValueType *vec) {

  IndexType ai = blockIdx.x*blockDim.x+threadIdx.x;
  IndexType aj = row_offset[idx] + ai;

  if (ai < row_nnz)
    vec[col[aj]] = val[aj];

}


}

#endif // PARALUTION_HIP_HIP_KERNELS_CSR_HPP_
