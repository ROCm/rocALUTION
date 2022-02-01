/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
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
#include "hip_utils.hpp"

#include "../../utils/types.hpp"

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
    template <typename ValueType, typename IndexType, unsigned int WF_SIZE>
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
    template <typename ValueType, typename IndexType, unsigned int WF_SIZE>
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
    template <unsigned int BLOCKSIZE, unsigned int WFSIZE, unsigned int HASH, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_prolong_nnz_per_row(IndexType nrow,
                                            const IndexType* __restrict__ row_offset,
                                            const IndexType* __restrict__ cols,
                                            const IndexType* __restrict__ connections,
                                            const IndexType* __restrict__ aggregates,
                                            IndexType* __restrict__ prolong_row_offset)
    {
        IndexType tid = hipThreadIdx_x;
        IndexType gid = hipBlockIdx_x * BLOCKSIZE + tid;
        IndexType lid = tid & (WFSIZE - 1);
        IndexType wid = tid / WFSIZE;
        IndexType row = gid / WFSIZE;

        __shared__ IndexType stable[(BLOCKSIZE / WFSIZE) * HASH];

        if(row >= nrow)
        {
            return;
        }

        // Pointer to each wavefronts shared data
        IndexType* table = &stable[wid * HASH];

        // Initialize hash table with -1
        for(IndexType j = lid; j < HASH; j += WFSIZE)
        {
            table[j] = -1;
        }

        __syncthreads();

        IndexType row_start = row_offset[row];
        IndexType row_end   = row_offset[row + 1];

        for(IndexType j = row_start + lid; j < row_end; j += WFSIZE)
        {
            IndexType col = cols[j];

            if(row == col || row != col && connections[j] == 1)
            {
                IndexType key = aggregates[col];

                if(key >= 0)
                {
                    // Compute hash
                    IndexType hash = (key * 103) & (HASH - 1);

                    // Hash operation
                    while(true)
                    {
                        if(table[hash] == key)
                        {
                            // key is already inserted, done
                            break;
                        }
                        else if(table[hash] == -1)
                        {
                            if(atomicCAS(&table[hash], -1, key) == -1)
                            {
                                atomicAdd(&prolong_row_offset[row + 1], 1);
                                break;
                            }
                        }
                        else
                        {
                            // collision, compute new hash
                            hash = (hash + 1) & (HASH - 1);
                        }
                    }
                }
            }
        }
    }

    template <unsigned int BLOCKSIZE, unsigned int HASH, typename ValueType, typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_prolong_fill(IndexType nrow,
                                     ValueType relax,
                                     const IndexType* __restrict__ row_offset,
                                     const IndexType* __restrict__ cols,
                                     const ValueType* __restrict__ vals,
                                     const IndexType* __restrict__ connections,
                                     const IndexType* __restrict__ aggregates,
                                     const IndexType* __restrict__ prolong_row_offset,
                                     IndexType* __restrict__ prolong_cols,
                                     ValueType* __restrict__ prolong_vals)
    {
        IndexType tid = hipThreadIdx_x;
        IndexType row = hipBlockIdx_x * BLOCKSIZE + tid;

        __shared__ IndexType stable[BLOCKSIZE * HASH];
        __shared__ IndexType sdata[BLOCKSIZE * HASH];

        IndexType* table = &stable[tid * HASH];
        IndexType* data  = &sdata[tid * HASH];

        // Initialize hash table with -1
        for(IndexType j = 0; j < HASH; j++)
        {
            table[j] = -1;
            data[j]  = -1;
        }

        __syncthreads();

        if(row >= nrow)
        {
            return;
        }

        IndexType row_start = row_offset[row];
        IndexType row_end   = row_offset[row + 1];

        ValueType dia = static_cast<ValueType>(0);
        for(IndexType j = row_start; j < row_end; j++)
        {
            if(cols[j] == row)
            {
                dia += vals[j];
            }
            else if(!connections[j])
            {
                dia -= vals[j];
            }
        }

        dia = static_cast<ValueType>(1) / dia;

        IndexType prolong_row_start = prolong_row_offset[row];

        IndexType counter = prolong_row_start;
        for(IndexType j = row_start; j < row_end; j++)
        {
            IndexType col = cols[j];

            if(row == col || row != col && connections[j] == 1)
            {
                IndexType key = aggregates[col];

                if(key >= 0)
                {
                    ValueType val
                        = (col == row) ? static_cast<ValueType>(1) - relax : -relax * dia * vals[j];

                    IndexType hash = (key * 103) & (HASH - 1);

                    // Hash operation
                    while(true)
                    {
                        if(table[hash] == key)
                        {
                            prolong_vals[data[hash]] += val;
                            break;
                        }
                        else if(table[hash] == -1)
                        {
                            table[hash]           = key;
                            data[hash]            = counter;
                            prolong_cols[counter] = key;
                            prolong_vals[counter] = val;
                            counter++;
                            break;
                        }
                        else
                        {
                            // collision, compute new hash
                            hash = (hash + 1) & (HASH - 1);
                        }
                    }
                }
            }
        }
    }

    template <unsigned int BLOCKSIZE,
              unsigned int WFSIZE,
              unsigned int HASH,
              typename ValueType,
              typename IndexType>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_prolong_fill2(IndexType nrow,
                                      ValueType relax,
                                      const IndexType* __restrict__ row_offset,
                                      const IndexType* __restrict__ cols,
                                      const ValueType* __restrict__ vals,
                                      const IndexType* __restrict__ connections,
                                      const IndexType* __restrict__ aggregates,
                                      const IndexType* __restrict__ prolong_row_offset,
                                      IndexType* __restrict__ prolong_cols,
                                      ValueType* __restrict__ prolong_vals)
    {
        IndexType tid = hipThreadIdx_x;
        IndexType gid = hipBlockIdx_x * BLOCKSIZE + tid;
        IndexType lid = tid & (WFSIZE - 1);
        IndexType wid = tid / WFSIZE;
        IndexType row = gid / WFSIZE;

        __shared__ IndexType stable[(BLOCKSIZE / WFSIZE) * HASH];
        __shared__ IndexType sdata[(BLOCKSIZE / WFSIZE) * HASH];
        __shared__ IndexType scounter[(BLOCKSIZE / WFSIZE)];

        if(row >= nrow)
        {
            return;
        }

        IndexType* table   = &stable[wid * HASH];
        IndexType* data    = &sdata[wid * HASH];
        IndexType* counter = &scounter[wid];

        for(IndexType j = lid; j < HASH; j += WFSIZE)
        {
            table[j] = -1;
            data[j]  = -1;
        }

        *counter = prolong_row_offset[row];

        __syncthreads();

        IndexType row_start = row_offset[row];
        IndexType row_end   = row_offset[row + 1];

        ValueType dia = static_cast<ValueType>(0);
        for(IndexType j = row_start + lid; j < row_end; j += WFSIZE)
        {
            if(cols[j] == row)
            {
                dia += vals[j];
            }
            else if(!connections[j])
            {
                dia -= vals[j];
            }
        }

        wf_reduce_sum<WFSIZE>(&dia);

        dia = shfl(dia, 0, WFSIZE);

        dia = static_cast<ValueType>(1) / dia;

        for(IndexType j = row_start + lid; j < row_end; j += WFSIZE)
        {
            IndexType col = cols[j];

            if(row == col || row != col && connections[j] == 1)
            {
                IndexType key = aggregates[col];

                if(key >= 0)
                {
                    ValueType val
                        = (col == row) ? static_cast<ValueType>(1) - relax : -relax * dia * vals[j];

                    IndexType hash = (key * 103) & (HASH - 1);

                    // Hash operation
                    while(true)
                    {
                        if(table[hash] == key)
                        {
                            atomicAdd(&prolong_vals[data[hash]], val);
                            break;
                        }
                        else if(table[hash] == -1)
                        {
                            if(atomicCAS(&table[hash], -1, key) == -1)
                            {
                                IndexType old_counter = atomicAdd(counter, 1);

                                data[hash]                = old_counter;
                                prolong_cols[old_counter] = key;
                                prolong_vals[old_counter] = val;
                                __threadfence();
                                break;
                            }
                        }
                        else
                        {
                            // collision, compute new hash
                            hash = (hash + 1) & (HASH - 1);
                        }
                    }
                }
            }
        }
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

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_CSR_HPP_
