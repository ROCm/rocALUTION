#ifndef PARALUTION_OCL_KERNELS_DIA_HPP_
#define PARALUTION_OCL_KERNELS_DIA_HPP_

namespace paralution {

const char *ocl_kernels_dia = CL_KERNEL(

// Nathan Bell and Michael Garland
// Efficient Sparse Matrix-Vector Multiplication on {CUDA}
// NVR-2008-004 / NVIDIA Technical Report
__kernel void kernel_dia_spmv(         const IndexType num_rows,
                                       const IndexType num_cols,
                                       const IndexType num_diags,
                              __global const IndexType *Aoffsets,
                              __global const ValueType *Aval,
                              __global const ValueType *x,
                              __global       ValueType *y) {

  IndexType row = get_global_id(0);

  if (row < num_rows) {

    ValueType sum = ocl_set((ValueType) 0);

    for (IndexType n=0; n<num_diags; ++n) {

      const IndexType ind = n * num_rows + row;
      const IndexType col = row + Aoffsets[n];
      
      if ((col >= 0) && (col < num_cols))
        sum += ocl_mult(Aval[ind], x[col]);

    }
        
    y[row] = sum;

  }

}

// Nathan Bell and Michael Garland
// Efficient Sparse Matrix-Vector Multiplication on {CUDA}
// NVR-2008-004 / NVIDIA Technical Report
__kernel void kernel_dia_add_spmv(         const IndexType num_rows,
                                           const IndexType num_cols,
                                           const IndexType num_diags,
                                  __global const IndexType *Aoffsets,
                                  __global const ValueType *Aval,
                                           const ValueType scalar,
                                  __global const ValueType *x,
                                  __global       ValueType *y) {

  IndexType row = get_global_id(0);

  if (row < num_rows) {

    ValueType sum = ocl_set((ValueType) 0);

    for (IndexType n=0; n<num_diags; ++n) {

      const IndexType ind = n * num_rows + row;
      const IndexType col = row + Aoffsets[n];
      
      if ((col >= 0) && (col < num_cols))
        sum += ocl_mult(Aval[ind], x[col]);

    }

    y[row] += ocl_mult(scalar, sum);

  }

}

);

}

#endif // PARALUTION_OCL_KERNELS_DIA_HPP_
