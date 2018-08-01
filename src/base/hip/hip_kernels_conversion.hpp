#ifndef ROCALUTION_HIP_HIP_KERNELS_CONVERSION_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_CONVERSION_HPP_

#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

// Compute non-zero entries per row
__global__ void kernel_hyb_coo_nnz(int m,
                                   int ell_width,
                                   const int* csr_row_ptr,
                                   int* coo_row_nnz)
{
    int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(gid >= m)
    {
        return;
    }

    int row_nnz = csr_row_ptr[gid + 1] - csr_row_ptr[gid] - ell_width;
    coo_row_nnz[gid] = row_nnz > 0 ? row_nnz : 0;

}

// CSR to HYB format conversion kernel
template <typename ValueType>
__global__ void kernel_hyb_csr2hyb(int m,
                                   const ValueType* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   int ell_width,
                                   int* ell_col_ind,
                                   ValueType* ell_val,
                                   int* coo_row_ind,
                                   int* coo_col_ind,
                                   ValueType* coo_val,
                                   int* workspace)
{
    int ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(ai >= m)
    {
        return;
    }

    int p = 0;

    int row_begin = csr_row_ptr[ai];
    int row_end   = csr_row_ptr[ai + 1];
    int coo_idx   = coo_row_ind ? workspace[ai] : 0;

    // Fill HYB matrix
    for(int aj = row_begin; aj < row_end; ++aj)
    {
        if(p < ell_width)
        {
            // Fill ELL part
            int idx = ELL_IND(ai, p++, m, ell_width);
            ell_col_ind[idx]  = csr_col_ind[aj];
            ell_val[idx]      = csr_val[aj];
        }
        else
        {
            // Fill COO part
            coo_row_ind[coo_idx] = ai;
            coo_col_ind[coo_idx] = csr_col_ind[aj];
            coo_val[coo_idx]     = csr_val[aj];
            ++coo_idx;
        }
    }

    // Pad remaining ELL structure
    for(int aj = row_end - row_begin; aj < ell_width; ++aj)
    {
        int idx = ELL_IND(ai, p++, m, ell_width);
        ell_col_ind[idx]  = -1;
        ell_val[idx]      = static_cast<ValueType>(0);
    }
}

template <typename IndexType>
__global__ void kernel_dia_diag_idx(IndexType nrow,
                                    IndexType* row_offset,
                                    IndexType* col,
                                    IndexType* diag_idx)
{
    IndexType row = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(row >= nrow)
    {
        return;
    }

    for(IndexType j = row_offset[row]; j < row_offset[row + 1]; ++j)
    {
        IndexType idx = col[j] - row + nrow;
        diag_idx[idx] = 1;
    }
}

template <typename IndexType>
__global__ void kernel_dia_fill_offset(IndexType nrow,
                                       IndexType ncol,
                                       IndexType* diag_idx,
                                       const IndexType* offset_map,
                                       IndexType* offset)
{
    IndexType i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(i >= nrow + ncol)
    {
        return;
    }
    
    if(diag_idx[i] == 1)
    {
        offset[offset_map[i]] = i - nrow;
        diag_idx[i]           = offset_map[i];
    }
}

template <typename ValueType, typename IndexType>
__global__ void kernel_dia_convert(IndexType nrow,
                                   IndexType ndiag,
                                   const IndexType* row_offset,
                                   const IndexType* col,
                                   const ValueType* val,
                                   const IndexType* diag_idx,
                                   ValueType* dia_val)
{
    IndexType row = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(row >= nrow)
    {
        return;
    }

    for(IndexType j = row_offset[row]; j < row_offset[row + 1]; ++j)
    {
        IndexType idx = col[j] - row + nrow;

        dia_val[DIA_IND(row, diag_idx[idx], nrow, ndiag)] = val[j];
    }
}

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_CONVERSION_HPP_
