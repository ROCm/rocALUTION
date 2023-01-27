/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HIP_HIP_KERNELS_CSR_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_CSR_HPP_

#include "../matrix_formats_ind.hpp"
#include "hip_atomics.hpp"
#include "hip_unordered_map.hpp"
#include "hip_unordered_set.hpp"
#include "hip_utils.hpp"

#include <hip/hip_runtime.h>

namespace rocalution
{
    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_scale_diagonal(IndexType nrow,
                                              const IndexType* __restrict__ row_offset,
                                              const IndexType* __restrict__ col,
                                              ValueType alpha,
                                              ValueType* __restrict__ val)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai == col[aj])
            {
                val[aj] = alpha * val[aj];
            }
        }
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_scale_offdiagonal(IndexType nrow,
                                                 const IndexType* __restrict__ row_offset,
                                                 const IndexType* __restrict__ col,
                                                 ValueType alpha,
                                                 ValueType* __restrict__ val)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai != col[aj])
            {
                val[aj] = alpha * val[aj];
            }
        }
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_add_diagonal(IndexType nrow,
                                            const IndexType* __restrict__ row_offset,
                                            const IndexType* __restrict__ col,
                                            ValueType alpha,
                                            ValueType* __restrict__ val)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai == col[aj])
            {
                val[aj] = val[aj] + alpha;
            }
        }
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_add_offdiagonal(IndexType nrow,
                                               const IndexType* __restrict__ row_offset,
                                               const IndexType* __restrict__ col,
                                               ValueType alpha,
                                               ValueType* __restrict__ val)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai != col[aj])
            {
                val[aj] = val[aj] + alpha;
            }
        }
    }

    template <unsigned int WFSIZE, typename ValueType, typename IndexType>
    __global__ void kernel_csr_extract_diag(IndexType nrow,
                                            const IndexType* __restrict__ row_offset,
                                            const IndexType* __restrict__ col,
                                            const ValueType* __restrict__ val,
                                            ValueType* __restrict__ vec)
    {
        IndexType tid = hipThreadIdx_x;
        IndexType gid = hipBlockIdx_x * hipBlockDim_x + tid;
        IndexType lid = tid & (WFSIZE - 1);
        IndexType row = gid / WFSIZE;

        if(row >= nrow)
        {
            return;
        }

        IndexType start = row_offset[row];
        IndexType end   = row_offset[row + 1];

        for(IndexType aj = start + lid; aj < end; aj += WFSIZE)
        {
            if(row == col[aj])
            {
                vec[row] = val[aj];
            }
        }
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_extract_inv_diag(IndexType nrow,
                                                const IndexType* __restrict__ row_offset,
                                                const IndexType* __restrict__ col,
                                                const ValueType* __restrict__ val,
                                                ValueType* __restrict__ vec,
                                                int* __restrict__ detect_zero)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai == col[aj])
            {
                if(val[aj] != static_cast<ValueType>(0))
                {
                    vec[ai] = static_cast<ValueType>(1) / val[aj];
                }
                else
                {
                    vec[ai] = static_cast<ValueType>(1);

                    *detect_zero = 1;
                }
            }
        }
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_extract_submatrix_row_nnz(const IndexType* __restrict__ row_offset,
                                                         const IndexType* __restrict__ col,
                                                         const ValueType* __restrict__ val,
                                                         IndexType smrow_offset,
                                                         IndexType smcol_offset,
                                                         IndexType smrow_size,
                                                         IndexType smcol_size,
                                                         IndexType* __restrict__ row_nnz)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= smrow_size)
        {
            return;
        }

        IndexType nnz = 0;
        IndexType ind = ai + smrow_offset;

        for(IndexType aj = row_offset[ind]; aj < row_offset[ind + 1]; ++aj)
        {
            IndexType c = col[aj];

            if((c >= smcol_offset) && (c < smcol_offset + smcol_size))
            {
                ++nnz;
            }
        }

        row_nnz[ai] = nnz;
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_extract_submatrix_copy(const IndexType* __restrict__ row_offset,
                                                      const IndexType* __restrict__ col,
                                                      const ValueType* __restrict__ val,
                                                      IndexType smrow_offset,
                                                      IndexType smcol_offset,
                                                      IndexType smrow_size,
                                                      IndexType smcol_size,
                                                      const IndexType* __restrict__ sm_row_offset,
                                                      IndexType* __restrict__ sm_col,
                                                      ValueType* __restrict__ sm_val)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= smrow_size)
        {
            return;
        }

        IndexType row_nnz = sm_row_offset[ai];
        IndexType ind     = ai + smrow_offset;

        for(IndexType aj = row_offset[ind]; aj < row_offset[ind + 1]; ++aj)
        {
            IndexType c = col[aj];

            if((c >= smcol_offset) && (c < smcol_offset + smcol_size))
            {
                sm_col[row_nnz] = c - smcol_offset;
                sm_val[row_nnz] = val[aj];
                ++row_nnz;
            }
        }
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_diagmatmult_r(IndexType nrow,
                                             const IndexType* __restrict__ row_offset,
                                             const IndexType* __restrict__ col,
                                             const ValueType* __restrict__ diag,
                                             ValueType* __restrict__ val)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            val[aj] = val[aj] * diag[col[aj]];
        }
    }

    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_diagmatmult_l(IndexType nrow,
                                             const IndexType* __restrict__ row_offset,
                                             const ValueType* __restrict__ diag,
                                             ValueType* __restrict__ val)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            val[aj] = val[aj] * diag[ai];
        }
    }

    // Calculates the number of non-zero elements per row
    template <typename IndexType>
    __global__ void kernel_calc_row_nnz(IndexType nrow,
                                        const IndexType* __restrict__ row_offset,
                                        IndexType* __restrict__ row_nnz)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        row_nnz[ai] = row_offset[ai + 1] - row_offset[ai];
    }

    // Performs a permutation on the vector of non-zero elements per row
    //
    // Inputs:   nrow:         number of rows in matrix
    //           row_nnz_src:  original number of non-zero elements per row
    //           perm_vec:     permutation vector
    // Outputs:  row_nnz_dst   permuted number of non-zero elements per row
    template <typename IndexType>
    __global__ void kernel_permute_row_nnz(IndexType nrow,
                                           const IndexType* __restrict__ row_offset,
                                           const IndexType* __restrict__ perm_vec,
                                           IndexType* __restrict__ row_nnz_dst)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        if(ai == 0)
        {
            row_nnz_dst[0] = 0;
        }

        row_nnz_dst[perm_vec[ai] + 1] = row_offset[ai + 1] - row_offset[ai];
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
    template <unsigned int WF_SIZE, typename ValueType, typename IndexType>
    __global__ void kernel_permute_rows(IndexType nrow,
                                        const IndexType* __restrict__ row_offset,
                                        const IndexType* __restrict__ perm_row_offset,
                                        const IndexType* __restrict__ col,
                                        const ValueType* __restrict__ data,
                                        const IndexType* __restrict__ perm_vec,
                                        IndexType* __restrict__ perm_col,
                                        ValueType* __restrict__ perm_data)
    {
        IndexType tid = hipThreadIdx_x;
        IndexType gid = hipBlockIdx_x * hipBlockDim_x + tid;
        IndexType lid = tid & (WF_SIZE - 1);
        IndexType row = gid / WF_SIZE;

        if(row >= nrow)
        {
            return;
        }

        IndexType perm_index = perm_row_offset[perm_vec[row]];
        IndexType prev_index = row_offset[row];
        IndexType num_elems  = row_offset[row + 1] - prev_index;

        for(IndexType i = lid; i < num_elems; i += WF_SIZE)
        {
            perm_data[perm_index + i] = data[prev_index + i];
            perm_col[perm_index + i]  = col[prev_index + i];
        }
    }

    // Permutes columns
    //
    // Inputs:   nrow:             number of rows in matrix
    //           row_offset:       row pointer
    //           perm_vec:         permutation vector
    //           perm_col:         row-permuted column indices of elements
    //           perm_data:        row-permuted data
    // Outputs:  col:              fully permuted column indices of elements
    //           data:             fully permuted data
    template <unsigned int size, typename ValueType, typename IndexType>
    __launch_bounds__(256) __global__
        void kernel_permute_cols(IndexType nrow,
                                 const IndexType* __restrict__ row_offset,
                                 const IndexType* __restrict__ perm_vec,
                                 const IndexType* __restrict__ perm_col,
                                 const ValueType* __restrict__ perm_data,
                                 IndexType* __restrict__ col,
                                 ValueType* __restrict__ data)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        IndexType elem_index = row_offset[ai];
        IndexType num_elems  = row_offset[ai + 1] - elem_index;

        IndexType ccol[size];
        ValueType cval[size];

        col += elem_index;
        data += elem_index;
        perm_col += elem_index;
        perm_data += elem_index;

        for(IndexType i = 0; i < num_elems; ++i)
        {
            ccol[i] = col[i];
            cval[i] = data[i];
        }

        for(IndexType i = 0; i < num_elems; ++i)
        {
            IndexType comp = perm_vec[perm_col[i]];

            IndexType j;
            for(j = i - 1; j >= 0; --j)
            {
                IndexType c = ccol[j];
                if(c > comp)
                {
                    cval[j + 1] = cval[j];
                    ccol[j + 1] = c;
                }
                else
                {
                    break;
                }
            }

            cval[j + 1] = perm_data[i];
            ccol[j + 1] = comp;
        }

        for(IndexType i = 0; i < num_elems; ++i)
        {
            col[i]  = ccol[i];
            data[i] = cval[i];
        }
    }

    // Permutes columns
    //
    // Inputs:   nrow:             number of rows in matrix
    //           row_offset:       row pointer
    //           perm_vec:         permutation vector
    //           perm_col:         row-permuted column indices of elements
    //           perm_data:        row-permuted data
    // Outputs:  col:              fully permuted column indices of elements
    //           data:             fully permuted data
    template <typename ValueType, typename IndexType>
    __global__ void kernel_permute_cols_fallback(IndexType nrow,
                                                 const IndexType* __restrict__ row_offset,
                                                 const IndexType* __restrict__ perm_vec,
                                                 const IndexType* __restrict__ perm_col,
                                                 const ValueType* __restrict__ perm_data,
                                                 IndexType* __restrict__ col,
                                                 ValueType* __restrict__ data)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        IndexType elem_index = row_offset[ai];
        IndexType num_elems  = row_offset[ai + 1] - elem_index;

        col += elem_index;
        data += elem_index;
        perm_col += elem_index;
        perm_data += elem_index;

        for(IndexType i = 0; i < num_elems; ++i)
        {
            IndexType comp = perm_vec[perm_col[i]];

            IndexType j;
            for(j = i - 1; j >= 0; --j)
            {
                IndexType c = col[j];
                if(c > comp)
                {
                    data[j + 1] = data[j];
                    col[j + 1]  = c;
                }
                else
                {
                    break;
                }
            }

            data[j + 1] = perm_data[i];
            col[j + 1]  = comp;
        }
    }

    // TODO
    // kind of ugly and inefficient ... but works
    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_add_csr_same_struct(IndexType nrow,
                                                   const IndexType* __restrict__ out_row_offset,
                                                   const IndexType* __restrict__ out_col,
                                                   const IndexType* __restrict__ in_row_offset,
                                                   const IndexType* __restrict__ in_col,
                                                   const ValueType* __restrict__ in_val,
                                                   ValueType alpha,
                                                   ValueType beta,
                                                   ValueType* __restrict__ out_val)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        IndexType first_col = in_row_offset[ai];

        for(IndexType ajj = out_row_offset[ai]; ajj < out_row_offset[ai + 1]; ++ajj)
        {
            for(IndexType aj = first_col; aj < in_row_offset[ai + 1]; ++aj)
            {
                if(in_col[aj] == out_col[ajj])
                {
                    out_val[ajj] = alpha * out_val[ajj] + beta * in_val[aj];
                    ++first_col;
                    break;
                }
            }
        }
    }

    // Computes the lower triangular part nnz per row
    template <typename IndexType>
    __global__ void kernel_csr_lower_nnz_per_row(IndexType nrow,
                                                 const IndexType* __restrict__ src_row_offset,
                                                 const IndexType* __restrict__ src_col,
                                                 IndexType* __restrict__ nnz_per_row)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        nnz_per_row[ai] = 0;

        for(IndexType aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
        {
            if(src_col[aj] <= ai)
            {
                ++nnz_per_row[ai];
            }
        }
    }

    // Computes the upper triangular part nnz per row
    template <typename IndexType>
    __global__ void kernel_csr_upper_nnz_per_row(IndexType nrow,
                                                 const IndexType* __restrict__ src_row_offset,
                                                 const IndexType* __restrict__ src_col,
                                                 IndexType* __restrict__ nnz_per_row)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        nnz_per_row[ai] = 0;

        for(IndexType aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
        {
            if(src_col[aj] >= ai)
            {
                ++nnz_per_row[ai];
            }
        }
    }

    // Computes the stricktly lower triangular part nnz per row
    template <typename IndexType>
    __global__ void kernel_csr_slower_nnz_per_row(IndexType nrow,
                                                  const IndexType* __restrict__ src_row_offset,
                                                  const IndexType* __restrict__ src_col,
                                                  IndexType* __restrict__ nnz_per_row)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        nnz_per_row[ai] = 0;

        for(IndexType aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
        {
            if(src_col[aj] < ai)
            {
                ++nnz_per_row[ai];
            }
        }
    }

    // Computes the stricktly upper triangular part nnz per row
    template <typename IndexType>
    __global__ void kernel_csr_supper_nnz_per_row(IndexType nrow,
                                                  const IndexType* __restrict__ src_row_offset,
                                                  const IndexType* __restrict__ src_col,
                                                  IndexType* __restrict__ nnz_per_row)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        nnz_per_row[ai] = 0;

        for(IndexType aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
        {
            if(src_col[aj] > ai)
            {
                ++nnz_per_row[ai];
            }
        }
    }

    // Extracts lower triangular part for given nnz per row array (partial sums nnz)
    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_extract_l_triangular(IndexType nrow,
                                                    const IndexType* __restrict__ src_row_offset,
                                                    const IndexType* __restrict__ src_col,
                                                    const ValueType* __restrict__ src_val,
                                                    IndexType* __restrict__ nnz_per_row,
                                                    IndexType* __restrict__ dst_col,
                                                    ValueType* __restrict__ dst_val)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        IndexType dst_index = nnz_per_row[ai];
        IndexType src_index = src_row_offset[ai];

        for(IndexType aj = 0; aj < nnz_per_row[ai + 1] - nnz_per_row[ai]; ++aj)
        {
            dst_col[dst_index] = src_col[src_index];
            dst_val[dst_index] = src_val[src_index];

            ++dst_index;
            ++src_index;
        }
    }

    // Extracts upper triangular part for given nnz per row array (partial sums nnz)
    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_extract_u_triangular(IndexType nrow,
                                                    const IndexType* __restrict__ src_row_offset,
                                                    const IndexType* __restrict__ src_col,
                                                    const ValueType* __restrict__ src_val,
                                                    IndexType* __restrict__ nnz_per_row,
                                                    IndexType* __restrict__ dst_col,
                                                    ValueType* __restrict__ dst_val)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        IndexType num_elements = nnz_per_row[ai + 1] - nnz_per_row[ai];
        IndexType src_index    = src_row_offset[ai + 1] - num_elements;
        IndexType dst_index    = nnz_per_row[ai];

        for(IndexType aj = 0; aj < num_elements; ++aj)
        {
            dst_col[dst_index] = src_col[src_index];
            dst_val[dst_index] = src_val[src_index];

            ++dst_index;
            ++src_index;
        }
    }

    // Compress
    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_compress_count_nrow(const IndexType* __restrict__ row_offset,
                                                   const IndexType* __restrict__ col,
                                                   const ValueType* __restrict__ val,
                                                   IndexType nrow,
                                                   double    drop_off,
                                                   IndexType* __restrict__ row_offset_new)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if((hip_abs(val[aj]) > drop_off) || (col[aj] == ai))
            {
                ++row_offset_new[ai];
            }
        }
    }

    // Compress
    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_compress_copy(const IndexType* __restrict__ row_offset,
                                             const IndexType* __restrict__ col,
                                             const ValueType* __restrict__ val,
                                             IndexType nrow,
                                             double    drop_off,
                                             const IndexType* __restrict__ row_offset_new,
                                             IndexType* __restrict__ col_new,
                                             ValueType* __restrict__ val_new)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        IndexType ajj = row_offset_new[ai];

        for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if((hip_abs(val[aj]) > drop_off) || (col[aj] == ai))
            {
                col_new[ajj] = col[aj];
                val_new[ajj] = val[aj];
                ++ajj;
            }
        }
    }

    // Extract column vector
    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_extract_column_vector(const IndexType* __restrict__ row_offset,
                                                     const IndexType* __restrict__ col,
                                                     const ValueType* __restrict__ val,
                                                     IndexType nrow,
                                                     IndexType idx,
                                                     ValueType* __restrict__ vec)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        vec[ai] = static_cast<ValueType>(0);

        for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(idx == col[aj])
            {
                vec[ai] = val[aj];
            }
        }
    }

    // Replace column vector - compute new offset
    template <typename ValueType, typename IndexType>
    __global__ void
        kernel_csr_replace_column_vector_offset(const IndexType* __restrict__ row_offset,
                                                const IndexType* __restrict__ col,
                                                IndexType nrow,
                                                IndexType idx,
                                                const ValueType* __restrict__ vec,
                                                IndexType* __restrict__ offset)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        bool      add = true;
        IndexType val = row_offset[ai + 1] - row_offset[ai];

        for(IndexType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(col[aj] == idx)
            {
                add = false;
                break;
            }
        }

        if(add == true && hip_abs(vec[ai]) != 0.0)
        {
            ++val;
        }

        if(add == false && hip_abs(vec[ai]) == 0.0)
        {
            --val;
        }

        offset[ai + 1] = val;
    }

    // Replace column vector - compute new offset
    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_replace_column_vector(const IndexType* __restrict__ row_offset,
                                                     const IndexType* __restrict__ col,
                                                     const ValueType* __restrict__ val,
                                                     IndexType nrow,
                                                     IndexType idx,
                                                     const ValueType* __restrict__ vec,
                                                     const IndexType* __restrict__ offset,
                                                     IndexType* __restrict__ new_col,
                                                     ValueType* __restrict__ new_val)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= nrow)
        {
            return;
        }

        IndexType aj = row_offset[ai];
        IndexType k  = offset[ai];

        for(; aj < row_offset[ai + 1]; ++aj)
        {
            if(col[aj] < idx)
            {
                new_col[k] = col[aj];
                new_val[k] = val[aj];
                ++k;
            }
            else
            {
                break;
            }
        }

        if(hip_abs(vec[ai]) != 0.0)
        {
            new_col[k] = idx;
            new_val[k] = vec[ai];
            ++k;
            ++aj;
        }

        for(; aj < row_offset[ai + 1]; ++aj)
        {
            if(col[aj] > idx)
            {
                new_col[k] = col[aj];
                new_val[k] = val[aj];
                ++k;
            }
        }
    }

    // Extract row vector
    template <typename ValueType, typename IndexType>
    __global__ void kernel_csr_extract_row_vector(const IndexType* __restrict__ row_offset,
                                                  const IndexType* __restrict__ col,
                                                  const ValueType* __restrict__ val,
                                                  IndexType row_nnz,
                                                  IndexType idx,
                                                  ValueType* __restrict__ vec)
    {
        IndexType ai = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(ai >= row_nnz)
        {
            return;
        }

        IndexType aj = row_offset[idx] + ai;
        vec[col[aj]] = val[aj];
    }

    // AMG Connect
    template <unsigned int WF_SIZE, typename ValueType, typename IndexType>
    __global__ void kernel_csr_amg_connect(IndexType nrow,
                                           ValueType eps2,
                                           const IndexType* __restrict__ row_offset,
                                           const IndexType* __restrict__ col,
                                           const ValueType* __restrict__ val,
                                           const ValueType* __restrict__ diag,
                                           IndexType* __restrict__ connections)
    {
        IndexType tid = hipThreadIdx_x;
        IndexType gid = hipBlockIdx_x * hipBlockDim_x + tid;
        IndexType lid = tid & (WF_SIZE - 1);
        IndexType row = gid / WF_SIZE;

        if(row >= nrow)
        {
            return;
        }

        ValueType eps_diag_r = eps2 * diag[row];

        IndexType start = row_offset[row];
        IndexType end   = row_offset[row + 1];

        for(IndexType i = start + lid; i < end; i += WF_SIZE)
        {
            IndexType c = col[i];
            ValueType v = val[i];

            connections[i] = (c != row) && (hip_real(v * v) > hip_real(eps_diag_r * diag[c]));
        }
    }

    // AMG Aggregate
    __device__ __forceinline__ unsigned int hash(unsigned int x)
    {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return x;
    }

    template <typename IndexType>
    __global__ void kernel_csr_amg_init_mis_tuples(IndexType nrow,
                                                   const IndexType* __restrict__ row_offset,
                                                   const IndexType* __restrict__ connections,
                                                   mis_tuple* __restrict__ tuples)
    {
        IndexType row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(row >= nrow)
        {
            return;
        }

        IndexType state = -2;

        IndexType row_start = row_offset[row];
        IndexType row_end   = row_offset[row + 1];

        for(IndexType j = row_start; j < row_end; j++)
        {
            if(connections[j] == 1)
            {
                state = 0;
                break;
            }
        }

        tuples[row].s = state;
        tuples[row].v = hash(row);
        tuples[row].i = row;
    }

    __device__ __forceinline__ mis_tuple lexographical_max(mis_tuple* ti, mis_tuple* tj)
    {
        // find lexographical maximum
        if(tj->s > ti->s)
        {
            return *tj;
        }
        else if(tj->s == ti->s && (tj->v > ti->v))
        {
            return *tj;
        }

        return *ti;
    }

    template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_amg_find_max_mis_tuples_shared(IndexType nrow,
                                                       const IndexType* __restrict__ row_offset,
                                                       const IndexType* __restrict__ cols,
                                                       const IndexType* __restrict__ connections,
                                                       const mis_tuple* __restrict__ tuples,
                                                       mis_tuple* __restrict__ max_tuples,
                                                       bool* done)
    {
        IndexType tid = hipThreadIdx_x;
        IndexType gid = hipBlockIdx_x * BLOCKSIZE + tid;
        IndexType lid = tid & (WFSIZE - 1);
        IndexType wid = tid / WFSIZE;
        IndexType row = gid / WFSIZE;

        __shared__ mis_tuple smax_tuple[BLOCKSIZE];

        if(row >= nrow)
        {
            return;
        }

        smax_tuple[WFSIZE * wid] = max_tuples[row];

        __syncthreads();

        for(IndexType k = 0; k < 2; k++)
        {
            mis_tuple t_max = smax_tuple[WFSIZE * wid];

            IndexType row_start = row_offset[t_max.i];
            IndexType row_end   = row_offset[t_max.i + 1];

            for(IndexType j = row_start + lid; j < row_end; j += WFSIZE)
            {
                if(connections[j] == 1)
                {
                    IndexType col = cols[j];
                    mis_tuple tj  = tuples[col];

                    t_max = lexographical_max(&tj, &t_max);
                }
            }

            smax_tuple[WFSIZE * wid + lid] = t_max;

            __syncthreads();

            // Finish reducing the intra block lexographical max
            if(WFSIZE >= 64)
            {
                if(lid < 32)
                    smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                        &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 32]);
                __threadfence_block();
            }
            if(WFSIZE >= 32)
            {
                if(lid < 16)
                    smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                        &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 16]);
                __threadfence_block();
            }
            if(WFSIZE >= 16)
            {
                if(lid < 8)
                    smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                        &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 8]);
                __threadfence_block();
            }
            if(WFSIZE >= 8)
            {
                if(lid < 4)
                    smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                        &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 4]);
                __threadfence_block();
            }
            if(WFSIZE >= 4)
            {
                if(lid < 2)
                    smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                        &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 2]);
                __threadfence_block();
            }
            if(WFSIZE >= 2)
            {
                if(lid < 1)
                    smax_tuple[WFSIZE * wid + lid] = lexographical_max(
                        &smax_tuple[WFSIZE * wid + lid], &smax_tuple[WFSIZE * wid + lid + 1]);
                __threadfence_block();
            }

            __syncthreads();
        }

        if(lid == 0)
        {
            max_tuples[row] = smax_tuple[WFSIZE * wid];
        }

        if(gid == 0)
        {
            *done = true;
        }
    }

    template <typename IndexType>
    __global__ void kernel_csr_amg_update_mis_tuples(IndexType nrow,
                                                     const mis_tuple* __restrict__ max_tuples,
                                                     mis_tuple* __restrict__ tuples,
                                                     IndexType* __restrict__ aggregates,
                                                     bool* done)
    {
        IndexType row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(row >= nrow)
        {
            return;
        }

        if(tuples[row].s == 0)
        {
            mis_tuple t_max = max_tuples[row];

            if(t_max.i == row)
            {
                tuples[row].s   = 1;
                aggregates[row] = 1;
            }
            else if(t_max.s == 1)
            {
                tuples[row].s   = -1;
                aggregates[row] = 0;
            }
            else
            {
                *done = false;
            }
        }
    }

    template <typename IndexType>
    __global__ void kernel_csr_amg_update_aggregates(IndexType nrow,
                                                     const IndexType* __restrict__ row_offset,
                                                     const IndexType* __restrict__ cols,
                                                     const IndexType* __restrict__ connections,
                                                     const mis_tuple* __restrict__ max_tuples,
                                                     mis_tuple* __restrict__ tuples,
                                                     IndexType* __restrict__ aggregates)
    {
        IndexType row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(row >= nrow)
        {
            return;
        }

        mis_tuple t = max_tuples[row];

        if(t.s == -1)
        {
            IndexType row_start = row_offset[row];
            IndexType row_end   = row_offset[row + 1];

            for(IndexType j = row_start; j < row_end; j++)
            {
                if(connections[j] == 1)
                {
                    IndexType col = cols[j];

                    if(max_tuples[col].s == 1)
                    {
                        aggregates[row] = aggregates[col];
                        tuples[row].s   = 1;
                        break;
                    }
                }
            }
        }
        else if(t.s == -2)
        {
            aggregates[row] = -2;
        }
    }

    // AMGSmoothedAggregation
    template <unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              unsigned int HASHSIZE,
              typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_sa_prolong_nnz(IndexType nrow,
                                       const IndexType* __restrict__ csr_row_ptr,
                                       const IndexType* __restrict__ csr_col_ind,
                                       const IndexType* __restrict__ connections,
                                       const IndexType* __restrict__ aggregates,
                                       IndexType* __restrict__ row_nnz)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        if(warpSize == WFSIZE)
        {
            wid = __builtin_amdgcn_readfirstlane(wid);
        }

        // The row this thread operates on
        IndexType row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Shared memory for the unordered set
        __shared__ IndexType stable[BLOCKSIZE / WFSIZE * HASHSIZE];

        // Each wavefront operates on its own set
        unordered_set<IndexType, HASHSIZE, WFSIZE> set(&stable[wid * HASHSIZE]);

        // Row nnz counter
        IndexType nnz = 0;

        // Row entry and exit points
        IndexType row_begin = csr_row_ptr[row];
        IndexType row_end   = csr_row_ptr[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(IndexType j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Get the column index
            IndexType col = csr_col_ind[j];

            // Get connection state
            IndexType con = connections[j];

            // Get aggregation state for this column
            IndexType agg = aggregates[col];

            // When aggregate is defined, diagonal entries and connected columns will
            // generate a fill in
            if((row == col || con) && agg >= 0)
            {
                // Insert into unordered set, to discard duplicates
                nnz += set.insert(agg);
            }
        }

        // Sum up the row nnz from all lanes
        wf_reduce_sum<WFSIZE>(&nnz);

        if(lid == WFSIZE - 1)
        {
            // Write row nnz back to global memory
            row_nnz[row] = nnz;
        }
    }

    template <unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              unsigned int HASHSIZE,
              typename ValueType,
              typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_sa_prolong_fill(IndexType nrow,
                                        ValueType relax,
                                        int       lumping_strat,
                                        const IndexType* __restrict__ csr_row_ptr,
                                        const IndexType* __restrict__ csr_col_ind,
                                        const ValueType* __restrict__ csr_val,
                                        const IndexType* __restrict__ connections,
                                        const IndexType* __restrict__ aggregates,
                                        const IndexType* __restrict__ csr_row_ptr_P,
                                        IndexType* __restrict__ csr_col_ind_P,
                                        ValueType* __restrict__ csr_val_P)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        if(warpSize == WFSIZE)
        {
            wid = __builtin_amdgcn_readfirstlane(wid);
        }

        // The row this thread operates on
        IndexType row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Shared memory for the unordered map
        __shared__ IndexType stable[(BLOCKSIZE / WFSIZE) * HASHSIZE];
        __shared__ char      smem[(BLOCKSIZE / WFSIZE) * HASHSIZE * sizeof(ValueType)];

        ValueType* sdata = reinterpret_cast<ValueType*>(smem);

        // Each wavefront operates on its own map
        unordered_map<IndexType, ValueType, HASHSIZE, WFSIZE> map(&stable[wid * HASHSIZE],
                                                                  &sdata[wid * HASHSIZE]);

        // Row entry and exit points
        IndexType row_begin = csr_row_ptr[row];
        IndexType row_end   = csr_row_ptr[row + 1];

        // Accumulator
        ValueType dia = static_cast<ValueType>(0);

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(IndexType j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Get the column index
            IndexType col_j = csr_col_ind[j];

            // Get the value
            ValueType val_j = csr_val[j];

            // Add the diagonal
            if(col_j == row)
            {
                dia += val_j;
            }
            else if(connections[j] == false)
            {
                dia = (lumping_strat == 0) ? dia + val_j : dia - val_j;
            }
        }

        // Sum up the accumulator from all lanes
        for(unsigned int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            dia += hip_shfl_xor(dia, i);
        }

        dia = static_cast<ValueType>(1) / dia;

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(IndexType j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Get the column index
            IndexType col_j = csr_col_ind[j];

            // Get the value
            ValueType val_j = csr_val[j];

            // Get connection state
            IndexType con = connections[j];

            // Get aggregation state for this column
            IndexType agg = aggregates[col_j];

            // When aggregate is defined, diagonal entries and connected columns will
            // generate a fill in
            if((row == col_j || con) && agg >= 0)
            {
                // Insert or add if already present into unordered map
                ValueType val
                    = (col_j == row) ? static_cast<ValueType>(1) - relax : -relax * dia * val_j;

                map.insert_or_add(agg, val);
            }
        }

        // Access into P
        IndexType aj = csr_row_ptr_P[row];

        // Store key val pairs from map into P
        map.store_sorted(&csr_col_ind_P[aj], &csr_val_P[aj]);
    }

    template <unsigned int BLOCKSIZE, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_unsmoothed_prolong_nnz_per_row(IndexType nrow,
                                                       const IndexType* __restrict__ aggregates,
                                                       IndexType* __restrict__ prolong_row_offset)
    {
        IndexType tid = hipThreadIdx_x;
        IndexType row = hipBlockIdx_x * BLOCKSIZE + tid;

        if(row >= nrow)
        {
            return;
        }

        if(aggregates[row] >= 0)
        {
            prolong_row_offset[row + 1] = 1;
        }
        else
        {
            prolong_row_offset[row + 1] = 0;
        }

        if(row == 0)
        {
            prolong_row_offset[row] = 0;
        }
    }

    template <unsigned int BLOCKSIZE, typename ValueType, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_unsmoothed_prolong_fill_simple(IndexType nrow,
                                                       const IndexType* __restrict__ aggregates,
                                                       IndexType* __restrict__ prolong_cols,
                                                       ValueType* __restrict__ prolong_vals)
    {

        IndexType tid = hipThreadIdx_x;
        IndexType row = hipBlockIdx_x * BLOCKSIZE + tid;

        if(row >= nrow)
        {
            return;
        }

        prolong_cols[row] = aggregates[row];
        prolong_vals[row] = static_cast<ValueType>(1);
    }

    template <unsigned int BLOCKSIZE, typename ValueType, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_unsmoothed_prolong_fill(IndexType nrow,
                                                const IndexType* __restrict__ aggregates,
                                                const IndexType* __restrict__ prolong_row_offset,
                                                IndexType* __restrict__ prolong_cols,
                                                ValueType* __restrict__ prolong_vals)
    {

        IndexType tid = hipThreadIdx_x;
        IndexType row = hipBlockIdx_x * BLOCKSIZE + tid;

        if(row >= nrow)
        {
            return;
        }

        IndexType agg = aggregates[row];

        if(agg >= 0)
        {
            IndexType j = prolong_row_offset[row];

            prolong_cols[j] = agg;
            prolong_vals[j] = static_cast<ValueType>(1);
        }
    }

    // Determine strong influences
    template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename ValueType, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_pmis_strong_influences(IndexType nrow,
                                                  const IndexType* __restrict__ csr_row_ptr,
                                                  const IndexType* __restrict__ csr_col_ind,
                                                  const ValueType* __restrict__ csr_val,
                                                  float eps,
                                                  float* __restrict__ omega,
                                                  bool* __restrict__ S)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        // The row this wavefront operates on
        IndexType row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Determine minimum and maximum off-diagonal of the current row
        ValueType min_a_ik = static_cast<ValueType>(0);
        ValueType max_a_ik = static_cast<ValueType>(0);

        // Shared boolean that holds diagonal sign for each wavefront
        // where true means, the diagonal element is negative
        __shared__ bool sign[BLOCKSIZE / WFSIZE];

        IndexType row_begin = csr_row_ptr[row];
        IndexType row_end   = csr_row_ptr[row + 1];

        // Determine diagonal sign and min/max
        for(IndexType j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            IndexType col = csr_col_ind[j];
            ValueType val = csr_val[j];

            if(col == row)
            {
                // Get diagonal entry sign
                sign[wid] = val < static_cast<ValueType>(0);
            }
            else
            {
                // Get min / max entries
                min_a_ik = (min_a_ik < val) ? min_a_ik : val;
                max_a_ik = (max_a_ik > val) ? max_a_ik : val;
            }
        }

        __threadfence_block();

        // Maximum or minimum, depending on the diagonal sign
        ValueType cond = sign[wid] ? max_a_ik : min_a_ik;

        // Obtain extrema on all threads of the wavefront
        if(sign[wid])
        {
            wf_reduce_max<WFSIZE>(&cond);
        }
        else
        {
            wf_reduce_min<WFSIZE>(&cond);
        }

        // Threshold to check for strength of connection
        cond *= eps;

        // Fill S
        for(IndexType j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            IndexType col = csr_col_ind[j];
            ValueType val = csr_val[j];

            if(col != row && val < cond)
            {
                // col is strongly connected to row
                S[j] = true;

                // Increment omega, as it holds all strongly connected edges
                // of vertex col.
                // Additionally, omega holds a random number between 0 and 1 to
                // distinguish neighbor points with equal number of strong
                // connections.
                atomicAdd(&omega[col], 1.0f);
            }
        }
    }

    // Mark all vertices that have not been assigned yet, as coarse
    template <typename IndexType>
    __global__ void kernel_csr_rs_pmis_unassigned_to_coarse(IndexType nrow,
                                                            const float* __restrict__ omega,
                                                            int* __restrict__ cf,
                                                            IndexType* __restrict__ workspace)
    {
        // Each thread processes a row
        IndexType row = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // workspace keeps track, whether a vertex has been marked coarse
        // during the current iteration, or not.
        bool flag = false;

        // Check only undecided vertices
        if(cf[row] == 0)
        {
            // If this vertex has an edge, it might be a coarse one
            if(omega[row] >= 1.0f)
            {
                cf[row] = 1;

                // Keep in mind, that this vertex has been marked coarse in the
                // current iteration
                flag = true;
            }
            else
            {
                // This point does not influence any other points and thus is a
                // fine point
                cf[row] = 2;
            }
        }

        workspace[row] = flag;
    }

    // Correct previously marked vertices with respect to omega
    template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_pmis_correct_coarse(IndexType nrow,
                                               const IndexType* __restrict__ csr_row_ptr,
                                               const IndexType* __restrict__ csr_col_ind,
                                               const float* __restrict__ omega,
                                               const bool* __restrict__ S,
                                               int* __restrict__ cf,
                                               IndexType* __restrict__ workspace)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        // The row this wavefront operates on
        IndexType row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // If this vertex has been marked coarse in the current iteration,
        // process it for further checks
        if(workspace[row])
        {
            IndexType row_begin = csr_row_ptr[row];
            IndexType row_end   = csr_row_ptr[row + 1];

            // Get the weight of the current row for comparison
            float omega_row = omega[row];

            // Loop over the full row to compare weights of other vertices that
            // have been marked coarse in the current iteration
            for(IndexType j = row_begin + lid; j < row_end; j += WFSIZE)
            {
                // Process only vertices that are strongly connected
                if(S[j])
                {
                    IndexType col = csr_col_ind[j];

                    // If this vertex has been marked coarse in the current iteration,
                    // we need to check whether it is accepted as a coarse vertex or not.
                    if(workspace[col])
                    {
                        // Get the weight of the current vertex for comparison
                        float omega_col = omega[col];

                        if(omega_row > omega_col)
                        {
                            // The diagonal entry has more edges and will remain
                            // a coarse point, whereas this vertex gets reverted
                            // back to undecided, for further processing.
                            cf[col] = 0;
                        }
                        else if(omega_row < omega_col)
                        {
                            // The diagonal entry has fewer edges and gets
                            // reverted back to undecided for further processing,
                            // whereas this vertex stays
                            // a coarse one.
                            cf[row] = 0;
                        }
                    }
                }
            }
        }
    }

    // Mark remaining edges of a coarse point to fine
    template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_pmis_coarse_edges_to_fine(IndexType nrow,
                                                     const IndexType* __restrict__ csr_row_ptr,
                                                     const IndexType* __restrict__ csr_col_ind,
                                                     const bool* __restrict__ S,
                                                     int* __restrict__ cf)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        // The row this wavefront operates on
        IndexType row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Process only undecided vertices
        if(cf[row] == 0)
        {
            IndexType row_begin = csr_row_ptr[row];
            IndexType row_end   = csr_row_ptr[row + 1];

            // Loop over all edges of this undecided vertex
            // and check, if there is a coarse point connected
            for(IndexType j = row_begin + lid; j < row_end; j += WFSIZE)
            {
                // Check, whether this edge is strongly connected to the vertex
                if(S[j])
                {
                    IndexType col = csr_col_ind[j];

                    // If this edge is coarse, our vertex must be fine
                    if(cf[col] == 1)
                    {
                        cf[row] = 2;
                        return;
                    }
                }
            }
        }
    }

    // Check for undecided vertices
    template <unsigned int BLOCKSIZE, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_pmis_check_undecided(IndexType nrow,
                                                const int* __restrict__ cf,
                                                IndexType* __restrict__ undecided)
    {
        IndexType row = blockIdx.x * blockDim.x + threadIdx.x;

        if(row >= nrow)
        {
            return;
        }

        // Check whether current vertex is undecided
        if(cf[row] == 0)
        {
            *undecided = true;
        }
    }

    template <unsigned int BLOCKSIZE, typename ValueType, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_direct_interp_nnz(IndexType nrow,
                                             const IndexType* __restrict__ csr_row_ptr,
                                             const IndexType* __restrict__ csr_col_ind,
                                             const ValueType* __restrict__ csr_val,
                                             const bool* __restrict__ S,
                                             const int* __restrict__ cf,
                                             ValueType* __restrict__ Amin,
                                             ValueType* __restrict__ Amax,
                                             IndexType* __restrict__ row_nnz,
                                             IndexType* __restrict__ coarse_idx)
    {
        // The row this thread operates on
        IndexType row = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Counter
        IndexType nnz = 0;

        // Coarse points generate a single entry
        if(cf[row] == 1)
        {
            // Set coarse flag
            coarse_idx[row] = nnz = 1;
        }
        else
        {
            // Set non-coarse flag
            coarse_idx[row] = 0;

            ValueType amin = static_cast<ValueType>(0);
            ValueType amax = static_cast<ValueType>(0);

            IndexType row_begin = csr_row_ptr[row];
            IndexType row_end   = csr_row_ptr[row + 1];

            // Loop over the full row and determine minimum and maximum
            for(IndexType j = row_begin; j < row_end; ++j)
            {
                // Process only vertices that are strongly connected
                if(S[j])
                {
                    IndexType col = csr_col_ind[j];

                    // Process only coarse points
                    if(cf[col] == 1)
                    {
                        ValueType val = csr_val[j];

                        amin = (amin < val) ? amin : val;
                        amax = (amax > val) ? amax : val;
                    }
                }
            }

            Amin[row] = amin = amin * static_cast<ValueType>(0.2);
            Amax[row] = amax = amax * static_cast<ValueType>(0.2);

            // Loop over the full row to count eligible entries
            for(IndexType j = row_begin; j < row_end; ++j)
            {
                // Process only vertices that are strongly connected
                if(S[j])
                {
                    IndexType col = csr_col_ind[j];

                    // Process only coarse points
                    if(cf[col] == 1)
                    {
                        ValueType val = csr_val[j];

                        // If conditions are fulfilled, count up row nnz
                        if(val <= amin || val >= amax)
                        {
                            ++nnz;
                        }
                    }
                }
            }
        }

        // Write row nnz back to global memory
        row_nnz[row] = nnz;
    }

    template <unsigned int BLOCKSIZE, typename ValueType, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_direct_interp_fill(IndexType nrow,
                                              const IndexType* __restrict__ csr_row_ptr,
                                              const IndexType* __restrict__ csr_col_ind,
                                              const ValueType* __restrict__ csr_val,
                                              const IndexType* __restrict__ prolong_csr_row_ptr,
                                              IndexType* __restrict__ prolong_csr_col_ind,
                                              ValueType* __restrict__ prolong_csr_val,
                                              const bool* __restrict__ S,
                                              const int* __restrict__ cf,
                                              const ValueType* __restrict__ Amin,
                                              const ValueType* __restrict__ Amax,
                                              const IndexType* __restrict__ coarse_idx)
    {
        // The row this thread operates on
        IndexType row = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // The row of P this thread operates on
        IndexType row_P = prolong_csr_row_ptr[row];

        // If this is a coarse point, we can fill P and return
        if(cf[row] == 1)
        {
            prolong_csr_col_ind[row_P] = coarse_idx[row];
            prolong_csr_val[row_P]     = static_cast<ValueType>(1);

            return;
        }

        ValueType diag  = static_cast<ValueType>(0);
        ValueType a_num = static_cast<ValueType>(0), a_den = static_cast<ValueType>(0);
        ValueType b_num = static_cast<ValueType>(0), b_den = static_cast<ValueType>(0);
        ValueType d_neg = static_cast<ValueType>(0), d_pos = static_cast<ValueType>(0);

        IndexType row_begin = csr_row_ptr[row];
        IndexType row_end   = csr_row_ptr[row + 1];

        // Loop over the full row
        for(IndexType j = row_begin; j < row_end; ++j)
        {
            IndexType col = csr_col_ind[j];
            ValueType val = csr_val[j];

            // Do not process the vertex itself
            if(col == row)
            {
                diag = val;
                continue;
            }

            if(val < static_cast<ValueType>(0))
            {
                a_num += val;

                // Only process vertices that are strongly connected and coarse
                if(S[j] && cf[col] == 1)
                {
                    a_den += val;

                    if(val > Amin[row])
                    {
                        d_neg += val;
                    }
                }
            }
            else
            {
                b_num += val;

                // Only process vertices that are strongly connected and coarse
                if(S[j] && cf[col] == 1)
                {
                    b_den += val;

                    if(val < Amax[row])
                    {
                        d_pos += val;
                    }
                }
            }
        }

        ValueType cf_neg = static_cast<ValueType>(1);
        ValueType cf_pos = static_cast<ValueType>(1);

        if(abs(a_den - d_neg) > 1e-32)
        {
            cf_neg = a_den / (a_den - d_neg);
        }

        if(abs(b_den - d_pos) > 1e-32)
        {
            cf_pos = b_den / (b_den - d_pos);
        }

        if(b_num > static_cast<ValueType>(0) && abs(b_den) < 1e-32)
        {
            diag += b_num;
        }

        ValueType alpha
            = abs(a_den) > 1e-32 ? -cf_neg * a_num / (diag * a_den) : static_cast<ValueType>(0);
        ValueType beta
            = abs(b_den) > 1e-32 ? -cf_pos * b_num / (diag * b_den) : static_cast<ValueType>(0);

        // Loop over the full row to fill eligible entries
        for(IndexType j = row_begin; j < row_end; ++j)
        {
            // Process only vertices that are strongly connected
            if(S[j])
            {
                IndexType col = csr_col_ind[j];
                ValueType val = csr_val[j];

                // Process only coarse points
                if(cf[col] == 1)
                {
                    if(val > Amin[row] && val < Amax[row])
                    {
                        continue;
                    }

                    // Fill P
                    prolong_csr_col_ind[row_P] = coarse_idx[col];
                    prolong_csr_val[row_P] = (val < static_cast<ValueType>(0) ? alpha : beta) * val;
                    ++row_P;
                }
            }
        }
    }

    template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_extpi_interp_max(IndexType nrow,
                                            bool      FF1,
                                            const IndexType* __restrict__ csr_row_ptr,
                                            const IndexType* __restrict__ csr_col_ind,
                                            const bool* __restrict__ S,
                                            const int* __restrict__ cf,
                                            IndexType* __restrict__ row_max)
    {
        int lid = threadIdx.x & (WFSIZE - 1);
        int wid = threadIdx.x / WFSIZE;

        // The row this thread operates on
        IndexType row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Some helpers for readability
        constexpr int COARSE = 1;

        // Coarse points generate a single entry
        if(cf[row] == COARSE)
        {
            if(lid == 0)
            {
                // Set row nnz to one
                row_max[row] = 1;
            }

            return;
        }

        // Counter
        IndexType nnz = 0;

        // Row entry and exit points
        IndexType row_begin = csr_row_ptr[row];
        IndexType row_end   = csr_row_ptr[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(IndexType j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Skip points that do not influence the current point
            if(S[j] == false)
            {
                continue;
            }

            // Get the column index
            IndexType col_j = csr_col_ind[j];

            // Skip diagonal entries (i does not influence itself)
            if(col_j == row)
            {
                continue;
            }

            // Switch between coarse and fine points that influence the i-th point
            if(cf[col_j] == COARSE)
            {
                // This is a coarse point and thus contributes, count it
                ++nnz;
            }
            else
            {
                // This is a fine point, check for strongly connected coarse points

                // Row entry and exit of this fine point
                IndexType row_begin_j = csr_row_ptr[col_j];
                IndexType row_end_j   = csr_row_ptr[col_j + 1];

                // Loop over all columns of the fine point
                for(IndexType k = row_begin_j; k < row_end_j; ++k)
                {
                    // Skip points that do not influence the fine point
                    if(S[k] == false)
                    {
                        continue;
                    }

                    // Get the column index
                    IndexType col_k = csr_col_ind[k];

                    // Skip diagonal entries (the fine point does not influence itself)
                    if(col_k == col_j)
                    {
                        continue;
                    }

                    // Check whether k is a coarse point
                    if(cf[col_k] == COARSE)
                    {
                        // This is a coarse point, it contributes, count it
                        ++nnz;

                        // Stop if FF interpolation is limited
                        if(FF1 == true)
                        {
                            break;
                        }
                    }
                }
            }
        }

        // Sum up the row nnz from all lanes
        wf_reduce_sum<WFSIZE>(&nnz);

        if(lid == WFSIZE - 1)
        {
            // Write row nnz back to global memory
            row_max[row] = nnz;
        }
    }

    template <unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              unsigned int HASHSIZE,
              unsigned int HASHVAL,
              typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_extpi_interp_nnz(IndexType nrow,
                                            bool      FF1,
                                            const IndexType* __restrict__ csr_row_ptr,
                                            const IndexType* __restrict__ csr_col_ind,
                                            const bool* __restrict__ S,
                                            const int* __restrict__ cf,
                                            IndexType* __restrict__ row_nnz,
                                            IndexType* __restrict__ state)
    {
        int lid = threadIdx.x & (WFSIZE - 1);
        int wid = threadIdx.x / WFSIZE;

        // The row this thread operates on
        IndexType row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Some helpers for readability
        constexpr int COARSE = 1;

        // Coarse points generate a single entry
        if(cf[row] == COARSE)
        {
            if(lid == 0)
            {
                // Set this points state to coarse
                state[row] = 1;

                // Set row nnz to one
                row_nnz[row] = 1;
            }

            return;
        }

        // Counter
        IndexType nnz = 0;

        // Shared memory for the unordered set
        __shared__ IndexType suset[BLOCKSIZE / WFSIZE * HASHSIZE];

        // Each wavefront operates on its own set
        IndexType* uset = suset + wid * HASHSIZE;

        // Initialize the set
        for(unsigned int i = lid; i < HASHSIZE; i += WFSIZE)
        {
            uset[i] = INT32_MAX;
        }

        // Wait for all threads to finish initialization
        __threadfence_block();

        // Row entry and exit points
        IndexType row_begin = csr_row_ptr[row];
        IndexType row_end   = csr_row_ptr[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(IndexType j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Skip points that do not influence the current point
            if(S[j] == false)
            {
                continue;
            }

            // Get the column index
            IndexType col_j = csr_col_ind[j];

            // Skip diagonal entries (i does not influence itself)
            if(col_j == row)
            {
                continue;
            }

            // Switch between coarse and fine points that influence the i-th point
            if(cf[col_j] == COARSE)
            {
                // This is a coarse point and thus contributes, count it for the row nnz
                // We need to use a set here, to discard duplicates.
                nnz += insert_key<HASHVAL, HASHSIZE>(col_j, uset, INT32_MAX);
            }
            else
            {
                // This is a fine point, check for strongly connected coarse points

                // Row entry and exit of this fine point
                IndexType row_begin_j = csr_row_ptr[col_j];
                IndexType row_end_j   = csr_row_ptr[col_j + 1];

                // Loop over all columns of the fine point
                for(IndexType k = row_begin_j; k < row_end_j; ++k)
                {
                    // Skip points that do not influence the fine point
                    if(S[k] == false)
                    {
                        continue;
                    }

                    // Get the column index
                    IndexType col_k = csr_col_ind[k];

                    // Skip diagonal entries (the fine point does not influence itself)
                    if(col_k == col_j)
                    {
                        continue;
                    }

                    // Check whether k is a coarse point
                    if(cf[col_k] == COARSE)
                    {
                        // This is a coarse point, it contributes, count it for the row nnz
                        // We need to use a set here, to discard duplicates.
                        nnz += insert_key<HASHVAL, HASHSIZE>(col_k, uset, INT32_MAX);

                        // Stop if FF interpolation is limited
                        if(FF1 == true)
                        {
                            break;
                        }
                    }
                }
            }
        }

        // Sum up the row nnz from all lanes
        wf_reduce_sum<WFSIZE>(&nnz);

        if(lid == WFSIZE - 1)
        {
            // Write row nnz back to global memory
            row_nnz[row] = nnz;

            // Set this points state to fine
            state[row] = 0;
        }
    }

    template <unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              unsigned int HASHSIZE,
              unsigned int HASHVAL,
              typename ValueType,
              typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_extpi_interp_fill(IndexType nrow,
                                             bool      FF1,
                                             const IndexType* __restrict__ csr_row_ptr,
                                             const IndexType* __restrict__ csr_col_ind,
                                             const ValueType* __restrict__ csr_val,
                                             const ValueType* __restrict__ diag,
                                             const IndexType* __restrict__ prolong_csr_row_ptr,
                                             IndexType* __restrict__ prolong_csr_col_ind,
                                             ValueType* __restrict__ prolong_csr_val,
                                             const bool* __restrict__ S,
                                             const int* __restrict__ cf,
                                             const IndexType* __restrict__ coarse_idx)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        // The row this thread operates on
        IndexType row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Some helpers for readability
        constexpr ValueType zero = static_cast<ValueType>(0);

        constexpr int COARSE = 1;
        constexpr int FINE   = 2;

        // Coarse points generate a single entry
        if(cf[row] == COARSE)
        {
            if(lid == 0)
            {
                // Get index into P
                IndexType idx = prolong_csr_row_ptr[row];

                // Single entry in this row (coarse point)
                prolong_csr_col_ind[idx] = coarse_idx[row];
                prolong_csr_val[idx]     = static_cast<ValueType>(1);
            }

            return;
        }

        // Shared memory for the hash table
        extern __shared__ char smem[];

        IndexType* stable = reinterpret_cast<IndexType*>(smem);
        ValueType* sdata  = reinterpret_cast<ValueType*>(stable + BLOCKSIZE / WFSIZE * HASHSIZE);

        // Each wavefront operates on its own hash table
        IndexType* table = stable + wid * HASHSIZE;
        ValueType* data  = sdata + wid * HASHSIZE;

        // Initialize the hash table
        for(unsigned int i = lid; i < HASHSIZE; i += WFSIZE)
        {
            table[i] = INT32_MAX;
            data[i]  = zero;
        }

        // Wait for all threads to finish initialization
        __threadfence_block();

        // Fill the hash table according to the nnz pattern of P
        // This is identical to the nnz per row kernel

        // Row entry and exit points
        IndexType row_begin = csr_row_ptr[row];
        IndexType row_end   = csr_row_ptr[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(IndexType k = row_begin + lid; k < row_end; k += WFSIZE)
        {
            // Skip points that do not influence the current point
            if(S[k] == false)
            {
                continue;
            }

            // Get the column index
            IndexType col_ik = csr_col_ind[k];

            // Skip diagonal entries (i does not influence itself)
            if(col_ik == row)
            {
                continue;
            }

            // Switch between coarse and fine points that influence the i-th point
            if(cf[col_ik] == COARSE)
            {
                // This is a coarse point and thus contributes
                insert_key<HASHVAL, HASHSIZE>(col_ik, table, INT32_MAX);
            }
            else
            {
                // This is a fine point, check for strongly connected coarse points

                // Row entry and exit of this fine point
                IndexType row_begin_k = csr_row_ptr[col_ik];
                IndexType row_end_k   = csr_row_ptr[col_ik + 1];

                // Loop over all columns of the fine point
                for(IndexType l = row_begin_k; l < row_end_k; ++l)
                {
                    // Skip points that do not influence the fine point
                    if(S[l] == false)
                    {
                        continue;
                    }

                    // Get the column index
                    IndexType col_kl = csr_col_ind[l];

                    // Skip diagonal entries (the fine point does not influence itself)
                    if(col_kl == col_ik)
                    {
                        continue;
                    }

                    // Check whether l is a coarse point
                    if(cf[col_kl] == COARSE)
                    {
                        // This is a coarse point, it contributes
                        insert_key<HASHVAL, HASHSIZE>(col_kl, table, INT32_MAX);

                        // Stop if FF interpolation is limited
                        if(FF1 == true)
                        {
                            break;
                        }
                    }
                }
            }
        }

        // Now, we need to do the numerical part

        // Diagonal entry of i-th row
        ValueType val_ii = diag[row];

        // Sign of diagonal entry of i-th row
        bool pos_ii = val_ii >= zero;

        // Accumulators
        ValueType sum_k = zero;
        ValueType sum_n = zero;

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(IndexType k = row_begin + lid; k < row_end; k += WFSIZE)
        {
            // Get the column index
            IndexType col_ik = csr_col_ind[k];

            // Skip diagonal entries (i does not influence itself)
            if(col_ik == row)
            {
                continue;
            }

            // Get the column value
            ValueType val_ik = csr_val[k];

            // Check, whether the k-th entry of the row is a fine point and strongly
            // connected to the i-th point (e.g. k \in F^S_i)
            if(S[k] == true && cf[col_ik] == FINE)
            {
                // Accumulator for the sum over l
                ValueType sum_l = zero;

                // Diagonal entry of k-th row
                ValueType val_kk = diag[col_ik];

                // Store a_ki, if present
                ValueType val_ki = zero;

                // Row entry and exit of this fine point
                IndexType row_begin_k = csr_row_ptr[col_ik];
                IndexType row_end_k   = csr_row_ptr[col_ik + 1];

                // Loop over all columns of the fine point
                for(IndexType l = row_begin_k; l < row_end_k; ++l)
                {
                    // Get the column index
                    IndexType col_kl = csr_col_ind[l];

                    // Get the column value
                    ValueType val_kl = csr_val[l];

                    // Sign of a_kl
                    bool pos_kl = val_kl >= zero;

                    // Differentiate between diagonal and off-diagonal
                    if(col_kl == row)
                    {
                        // Column that matches the i-th row
                        // Since we sum up all l in C^hat_i and i, the diagonal need to
                        // be added to the sum over l, e.g. a^bar_kl
                        // a^bar contributes only, if the sign is different to the
                        // i-th row diagonal sign.
                        if(pos_ii != pos_kl)
                        {
                            sum_l += val_kl;
                        }

                        // If a_ki exists, keep it for later
                        val_ki = val_kl;
                    }
                    else if(cf[col_kl] == COARSE)
                    {
                        // Check if sign is different from i-th row diagonal
                        if(pos_ii != pos_kl)
                        {
                            // Entry contributes only, if it is a coarse point
                            // and part of C^hat (e.g. we need to check the hash table)
                            if(key_exists<HASHVAL, HASHSIZE>(col_kl, table, INT32_MAX))
                            {
                                sum_l += val_kl;
                            }
                        }
                    }
                }

                // Update sum over l with a_ik
                sum_l = val_ik / sum_l;

                // Compute the sign of a_kk and a_ki, we need this for a_bar
                bool pos_kk = val_kk >= zero;
                bool pos_ki = val_ki >= zero;

                // Additionally, for eq19 we need to add all coarse points in row k,
                // if they have different sign than the diagonal a_kk
                for(IndexType l = row_begin_k; l < row_end_k; ++l)
                {
                    // Get the column index
                    IndexType col_kl = csr_col_ind[l];

                    // Only coarse points contribute
                    if(cf[col_kl] != COARSE)
                    {
                        continue;
                    }

                    // Get the column value
                    ValueType val_kl = csr_val[l];

                    // Compute the sign of a_kl
                    bool pos_kl = val_kl >= zero;

                    // Check for different sign
                    if(pos_kk != pos_kl)
                    {
                        append_pair<HASHVAL, HASHSIZE>(
                            col_kl, val_kl * sum_l, table, data, INT32_MAX);
                    }
                }

                // If sign of a_ki and a_kk are different, a_ki contributes to the
                // sum over k in F^S_i
                if(pos_kk != pos_ki)
                {
                    sum_k += val_ki * sum_l;
                }
            }

            // Boolean, to flag whether a_ik is in C hat or not
            // (we can query the hash table for it)
            bool in_C_hat = false;

            // a_ik can only be in C^hat if it is coarse
            if(cf[col_ik] == COARSE)
            {
                // Append a_ik to the sum of eq19
                in_C_hat = append_pair<HASHVAL, HASHSIZE>(col_ik, val_ik, table, data, INT32_MAX);
            }

            // If a_ik is not in C^hat and does not strongly influence i, it contributes
            // to sum_n
            if(in_C_hat == false && S[k] == false)
            {
                sum_n += val_ik;
            }
        }

        // Each lane accumulates the sums (over n and l)
        ValueType a_ii_tilde = sum_n + sum_k;

        // Now, each lane of the wavefront should hold the global row sum
        for(unsigned int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            a_ii_tilde += hip_shfl_xor(a_ii_tilde, i);
        }

        // Precompute -1 / (a_ii_tilde + a_ii)
        a_ii_tilde = static_cast<ValueType>(-1) / (a_ii_tilde + val_ii);

        // Finally, extract the numerical values from the hash table and fill P such
        // that the resulting matrix is sorted by columns
        for(unsigned int i = lid; i < HASHSIZE; i += WFSIZE)
        {
            // Get column from hash table to fill into C hat
            IndexType col = table[i];

            // Skip, if table is empty
            if(col == INT32_MAX)
            {
                continue;
            }

            // Get index into P
            IndexType idx = prolong_csr_row_ptr[row];

            // Hash table index counter
            unsigned int cnt = 0;

            // Go through the hash table, until we reach its end
            while(cnt < HASHSIZE)
            {
                // We are searching for the right place in P to
                // insert the i-th hash table entry.
                // If the i-th hash table column entry is greater then the current one,
                // we need to leave a slot to its left.
                if(col > table[cnt])
                {
                    ++idx;
                }

                // Process next hash table entry
                ++cnt;
            }

            // Add hash table entry into P
            prolong_csr_col_ind[idx] = coarse_idx[col];
            prolong_csr_val[idx]     = a_ii_tilde * data[i];
        }
    }

    // Compress prolongation matrix
    template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename ValueType, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_rs_extpi_interp_compress_nnz(IndexType nrow,
                                                     const IndexType* __restrict__ csr_row_ptr,
                                                     const IndexType* __restrict__ csr_col_ind,
                                                     const ValueType* __restrict__ csr_val,
                                                     float trunc,
                                                     IndexType* __restrict__ row_nnz)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        // Current row
        IndexType row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Row nnz counter
        IndexType nnz = 0;

        double row_max = 0.0;

        IndexType row_begin = csr_row_ptr[row];
        IndexType row_end   = csr_row_ptr[row + 1];

        // Obtain numbers for processing the row
        for(IndexType j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Compute absolute row maximum
            row_max = max(row_max, hip_abs(csr_val[j]));
        }

        // Gather from other lanes
        wf_reduce_max<WFSIZE>(&row_max);

        // Threshold
        double threshold = row_max * trunc;

        // Count the row nnz
        for(IndexType j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Check whether we keep this entry or not
            if(hip_abs(csr_val[j]) >= threshold)
            {
                // Count nnz
                ++nnz;
            }
        }

        // Gather from other lanes
        for(unsigned int i = WFSIZE >> 1; i > 0; i >>= 1)
        {
            nnz += __shfl_xor(nnz, i);
        }

        if(lid == 0)
        {
            // Write back to global memory
            row_nnz[row] = nnz;
        }
    }

    // Compress
    template <typename ValueType, typename IndexType>
    __global__ void
        kernel_csr_rs_extpi_interp_compress_fill(IndexType nrow,
                                                 const IndexType* __restrict__ csr_row_ptr,
                                                 const IndexType* __restrict__ csr_col_ind,
                                                 const ValueType* __restrict__ csr_val,
                                                 float trunc,
                                                 const IndexType* __restrict__ comp_csr_row_ptr,
                                                 IndexType* __restrict__ comp_csr_col_ind,
                                                 ValueType* __restrict__ comp_csr_val)
    {
        // Current row
        IndexType row = blockIdx.x * blockDim.x + threadIdx.x;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Row entry and exit point
        IndexType row_begin = csr_row_ptr[row];
        IndexType row_end   = csr_row_ptr[row + 1];

        double    row_max = 0.0;
        ValueType row_sum = static_cast<ValueType>(0);

        // Obtain numbers for processing the row
        for(IndexType j = row_begin; j < row_end; ++j)
        {
            // Get column value
            ValueType val = csr_val[j];

            // Compute absolute row maximum
            row_max = max(row_max, hip_abs(val));

            // Compute row sum
            row_sum += val;
        }

        // Threshold
        double threshold = row_max * trunc;

        // Row entry point for the compressed matrix
        IndexType comp_row_begin = comp_csr_row_ptr[row];
        IndexType comp_row_end   = comp_csr_row_ptr[row + 1];

        // Row nnz counter
        IndexType nnz = 0;

        // Row sum of not-dropped entries
        ValueType comp_row_sum = static_cast<ValueType>(0);

        for(IndexType j = row_begin; j < row_end; ++j)
        {
            // Get column value
            ValueType val = csr_val[j];

            // Check whether we keep this entry or not
            if(hip_abs(val) >= threshold)
            {
                // Compute compressed row sum
                comp_row_sum += val;

                // Fill compressed structures
                comp_csr_col_ind[comp_row_begin + nnz] = csr_col_ind[j];
                comp_csr_val[comp_row_begin + nnz]     = csr_val[j];

                ++nnz;
            }
        }

        // Row scaling factor
        ValueType scale = row_sum / comp_row_sum;

        // Scale row entries
        for(IndexType j = comp_row_begin; j < comp_row_end; ++j)
        {
            comp_csr_val[j] *= scale;
        }
    }
} // namespace rocalution
#endif // ROCALUTION_HIP_HIP_KERNELS_CSR_HPP_
