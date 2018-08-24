#ifndef ROCALUTION_HIP_HIP_KERNELS_DIA_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_DIA_HPP_

#include "../matrix_formats_ind.hpp"
#include "hip_complex.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

// Nathan Bell and Michael Garland
// Efficient Sparse Matrix-Vector Multiplication
// NVR-2008-004 / NVIDIA Technical Report
template <typename ValueType, typename IndexType>
__global__ void kernel_dia_spmv(IndexType num_rows,
                                IndexType num_cols,
                                IndexType num_diags,
                                const IndexType* __restrict__ Aoffsets,
                                const ValueType* __restrict__ Aval,
                                const ValueType* __restrict__ x,
                                ValueType* __restrict__ y)
{
    int row = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(row >= num_rows)
    {
        return;
    }

    ValueType sum;
    make_ValueType(sum, 0);

    for(IndexType n = 0; n < num_diags; ++n)
    {
        IndexType ind = DIA_IND(row, n, num_rows, num_diags);
        IndexType col = row + Aoffsets[n];

        if((col >= 0) && (col < num_cols))
        {
            sum = sum + Aval[ind] * x[col];
        }
    }

    y[row] = sum;
}

// Nathan Bell and Michael Garland
// Efficient Sparse Matrix-Vector Multiplication
// NVR-2008-004 / NVIDIA Technical Report
template <typename ValueType, typename IndexType>
__global__ void kernel_dia_add_spmv(IndexType num_rows,
                                    IndexType num_cols,
                                    IndexType num_diags,
                                    const IndexType* __restrict__ Aoffsets,
                                    const ValueType* __restrict__ Aval,
                                    ValueType scalar,
                                    const ValueType* __restrict__ x,
                                    ValueType* __restrict__ y)
{
    int row = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(row >= num_rows)
    {
        return;
    }

    ValueType sum;
    make_ValueType(sum, 0.0);

    for(IndexType n = 0; n < num_diags; ++n)
    {
        IndexType ind = DIA_IND(row, n, num_rows, num_diags);
        IndexType col = row + Aoffsets[n];

        if((col >= 0) && (col < num_cols))
        {
            sum = sum + Aval[ind] * x[col];
        }
    }

    y[row] = y[row] + scalar * sum;
}

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_DIA_HPP_
