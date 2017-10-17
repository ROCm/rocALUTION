#ifndef PARALUTION_OCL_KERNELS_ELL_HPP_
#define PARALUTION_OCL_KERNELS_ELL_HPP_

namespace paralution {

const char *ocl_kernels_ell = CL_KERNEL(

// Nathan Bell and Michael Garland
// Efficient Sparse Matrix-Vector Multiplication on {CUDA}
// NVR-2008-004 / NVIDIA Technical Report
__kernel void kernel_ell_spmv(         const IndexType num_rows,
                                       const IndexType num_cols,
                                       const IndexType num_cols_per_row,
                              __global const IndexType *Acol,
                              __global const ValueType *Aval,
                              __global const ValueType *x,
                              __global       ValueType *y) {

  IndexType row = get_global_id(0);

  if (row < num_rows) {

    ValueType sum = ocl_set((ValueType) 0);

    for (IndexType n=0; n<num_cols_per_row; ++n) {

      const IndexType ind = n * num_rows + row;
      const IndexType col = Acol[ind];

      if ((col >= 0) && (col < num_cols))
        sum += ocl_mult(Aval[ind], x[col]);

    }

    y[row] = sum;

  }

}

// Nathan Bell and Michael Garland
// Efficient Sparse Matrix-Vector Multiplication on {CUDA}
// NVR-2008-004 / NVIDIA Technical Report
__kernel void kernel_ell_add_spmv(         const IndexType num_rows,
                                           const IndexType num_cols,
                                           const IndexType num_cols_per_row,
                                  __global const IndexType *Acol,
                                  __global const ValueType *Aval,
                                           const ValueType scalar,
                                  __global const ValueType *x,
                                  __global       ValueType *y) {

  IndexType row = get_global_id(0);

  if (row < num_rows) {

    ValueType sum = ocl_set((ValueType) 0);

    for (IndexType n=0; n<num_cols_per_row; ++n) {

      const IndexType ind = n * num_rows + row;
      const IndexType col = Acol[ind];
      
      if ((col >= 0) && (col < num_cols))
        sum += ocl_mult(Aval[ind], x[col]);

    }
        
    y[row] += ocl_mult(scalar, sum);

  }

}

__kernel void kernel_ell_max_row(         const IndexType nrow,
                                 __global const IndexType *data,
                                 __global       IndexType *out,
                                          const IndexType  GROUP_SIZE,
                                          const IndexType  LOCAL_SIZE) {

    IndexType tid = get_local_id(0);

    __local IndexType sdata[BLOCK_SIZE];

    sdata[tid] = 0;

    IndexType max;

    IndexType gid = GROUP_SIZE * get_group_id(0) + tid;

    for (IndexType i = 0; i < LOCAL_SIZE; ++i, gid += BLOCK_SIZE) {

      if (gid < nrow) {
        max = data[gid+1] - data[gid];
        if (max > sdata[tid])
          sdata[tid] = max;
      }

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (IndexType i = BLOCK_SIZE/2; i > 0; i /= 2) {

      if (tid < i)
        if (sdata[tid+i] > sdata[tid]) sdata[tid] = sdata[tid+i];

      barrier(CLK_LOCAL_MEM_FENCE);

    }

    if (tid == 0)
      out[get_group_id(0)] = sdata[tid];

}

__kernel void kernel_ell_csr_to_ell(         const IndexType nrow,
                                             const IndexType max_row,
                                    __global const IndexType *src_row_offset,
                                    __global const IndexType *src_col,
                                    __global const ValueType *src_val,
                                    __global       IndexType *ell_col,
                                    __global       ValueType *ell_val) {

  IndexType ai = get_global_id(0);
  IndexType aj;
  IndexType n = 0;
  IndexType ell_ind;

  if (ai < nrow) {

    for (aj=src_row_offset[ai]; aj<src_row_offset[ai+1]; ++aj) {

      ell_ind = n * nrow + ai;

      ell_col[ell_ind] = src_col[aj];
      ell_val[ell_ind] = src_val[aj];

      ++n;

    }

    for (aj=src_row_offset[ai+1]-src_row_offset[ai]; aj<max_row; ++aj) {

      ell_ind = n * nrow + ai;

      ell_col[ell_ind] = (int) -1;
      ell_val[ell_ind] = ocl_set((ValueType) 0);

      ++n;

    }

  }

}

);

}

#endif // PARALUTION_OCL_KERNELS_ELL_HPP_
