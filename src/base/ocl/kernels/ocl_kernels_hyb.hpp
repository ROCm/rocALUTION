#ifndef PARALUTION_OCL_KERNELS_HYB_HPP_
#define PARALUTION_OCL_KERNELS_HYB_HPP_

namespace paralution {

const char *ocl_kernels_hyb = CL_KERNEL(

__kernel void kernel_hyb_ell_nnz_coo(         const IndexType nrow,
                                              const IndexType max_row,
                                     __global const IndexType *row_offset,
                                     __global       IndexType *nnz_coo) {

  IndexType gid = get_global_id(0);

  if (gid < nrow) {

    nnz_coo[gid] = 0;
    IndexType nnz_per_row = row_offset[gid+1] - row_offset[gid];

    if (nnz_per_row > max_row)
      nnz_coo[gid] = nnz_per_row - max_row;

  }

}

__kernel void kernel_hyb_ell_fill_ell(         const IndexType nrow,
                                               const IndexType max_row,
                                      __global const IndexType *row_offset,
                                      __global const IndexType *col,
                                      __global const ValueType *val,
                                      __global       IndexType *ELL_col,
                                      __global       ValueType *ELL_val,
                                      __global       IndexType *nnz_ell) {

  IndexType gid = get_global_id(0);

  if (gid < nrow) {

    IndexType n = 0;

    for (IndexType i=row_offset[gid]; i<row_offset[gid+1]; ++i) {

      if (n >= max_row) break;

      IndexType idx = n * nrow + gid;

      ELL_col[idx] = col[i];
      ELL_val[idx] = val[i];

      ++n;

    }

    nnz_ell[gid] = n;

  }

}

__kernel void kernel_hyb_ell_fill_coo(         const IndexType nrow,
                                      __global const IndexType *row_offset,
                                      __global const IndexType *col,
                                      __global const ValueType *val,
                                      __global const IndexType *nnz_coo,
                                      __global const IndexType *nnz_ell,
                                      __global       IndexType *COO_row,
                                      __global       IndexType *COO_col,
                                      __global       ValueType *COO_val) {

  IndexType gid = get_global_id(0);

  if (gid < nrow) {

    IndexType row_ptr = row_offset[gid+1];

    for (IndexType i=row_ptr - nnz_coo[gid]; i<row_ptr; ++i) {

      IndexType idx = i - nnz_ell[gid];

      COO_row[idx] = gid;
      COO_col[idx] = col[i];
      COO_val[idx] = val[i];

    }

  }

}

);

}

#endif // PARALUTION_OCL_KERNELS_HYB_HPP_
