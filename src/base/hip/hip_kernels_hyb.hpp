#ifndef ROCALUTION_HIP_HIP_KERNELS_HYB_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_HYB_HPP_

#include "hip_kernels_ell.hpp"
#include "hip_kernels_coo.hpp"
#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

// Block reduce kernel computing sum
template <int BLOCKSIZE>
__device__ void sum_reduce(int tid, int* data)
{
    __syncthreads();

    for(int i = BLOCKSIZE >> 1; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            data[tid] += data[tid + i];
        }

        __syncthreads();
    }
}

// Compute non-zero entries per CSR row and do a block reduction over the sum
// to obtain the number of COO part non-zero entries and COO nnz per row.
// Store the result in a workspace for final reduction on part2
template <int BLOCKSIZE>
__global__ void kernel_hyb_coo_nnz_part1(int m,
                                         int ell_width,
                                         const int* csr_row_ptr,
                                         int* workspace,
                                         int* coo_row_nnz)
{
    int tid = hipThreadIdx_x;
    int gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    __shared__ int sdata[BLOCKSIZE];

    if(gid < m)
    {
        int row_nnz = csr_row_ptr[gid + 1] - csr_row_ptr[gid];

        if(row_nnz > ell_width)
        {
            row_nnz          = row_nnz - ell_width;
            sdata[tid]       = row_nnz;
            coo_row_nnz[gid] = row_nnz;
        }
        else
        {
            sdata[tid]       = 0;
            coo_row_nnz[gid] = 0;
        }
    }
    else
    {
        sdata[tid] = 0;
    }

    sum_reduce<BLOCKSIZE>(tid, sdata);

    if(tid == 0)
    {
        workspace[hipBlockIdx_x] = sdata[0];
    }
}

// Part2 kernel for final reduction over the sum of COO non-zero entries
template <int BLOCKSIZE>
__global__ void kernel_hyb_coo_nnz_part2(int m, int* workspace)
{
    int tid = hipThreadIdx_x;

    __shared__ int sdata[BLOCKSIZE];
    sdata[tid] = 0;

    for(int i = tid; i < m; i += BLOCKSIZE)
    {
        sdata[tid] += workspace[i];
    }

    __syncthreads();

    if(m < 32)
    {
        if(tid == 0)
        {
            for(int i = 1; i < m; ++i)
            {
                sdata[0] += sdata[i];
            }
        }
    }
    else
    {
        sum_reduce<BLOCKSIZE>(tid, sdata);
    }

    if(tid == 0)
    {
        workspace[0] = sdata[0];
    }
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

}

#endif // ROCALUTION_HIP_HIP_KERNELS_HYB_HPP_
