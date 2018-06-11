#ifndef ROCALUTION_HIP_HIP_KERNELS_COO_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_COO_HPP_

#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType, typename IndexType>
__global__ void kernel_coo_permute(const IndexType nnz, 
                                   const IndexType *in_row, const IndexType *in_col, 
                                   const IndexType *perm,
                                   IndexType *out_row,  IndexType *out_col) {


  IndexType ind = blockIdx.x*blockDim.x+threadIdx.x;

  for (int i=ind; i<nnz; i+=gridDim.x) {

    out_row[i] = perm[ in_row[i] ];
    out_col[i] = perm[ in_col[i] ];

  }

}

}

#endif // ROCALUTION_HIP_HIP_KERNELS_COO_HPP_
