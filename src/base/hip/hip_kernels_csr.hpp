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
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_scale_diagonal(I nrow,
                                              const J* __restrict__ row_offset,
                                              const I* __restrict__ col,
                                              T alpha,
                                              T* __restrict__ val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai == col[aj])
            {
                val[aj] = alpha * val[aj];
            }
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_scale_offdiagonal(I nrow,
                                                 const J* __restrict__ row_offset,
                                                 const I* __restrict__ col,
                                                 T alpha,
                                                 T* __restrict__ val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai != col[aj])
            {
                val[aj] = alpha * val[aj];
            }
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_add_diagonal(I nrow,
                                            const J* __restrict__ row_offset,
                                            const I* __restrict__ col,
                                            T alpha,
                                            T* __restrict__ val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai == col[aj])
            {
                val[aj] = val[aj] + alpha;
            }
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_add_offdiagonal(I nrow,
                                               const J* __restrict__ row_offset,
                                               const I* __restrict__ col,
                                               T alpha,
                                               T* __restrict__ val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai != col[aj])
            {
                val[aj] = val[aj] + alpha;
            }
        }
    }

    template <unsigned int WFSIZE, typename T, typename I, typename J>
    __global__ void kernel_csr_extract_diag(I nrow,
                                            const J* __restrict__ row_offset,
                                            const I* __restrict__ col,
                                            const T* __restrict__ val,
                                            T* __restrict__ vec)
    {
        I tid = threadIdx.x;
        I gid = blockIdx.x * blockDim.x + tid;
        I lid = tid & (WFSIZE - 1);
        I row = gid / WFSIZE;

        if(row >= nrow)
        {
            return;
        }

        J start = row_offset[row];
        J end   = row_offset[row + 1];

        for(J aj = start + lid; aj < end; aj += WFSIZE)
        {
            if(row == col[aj])
            {
                vec[row] = val[aj];
            }
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_inv_diag(I nrow,
                                                const J* __restrict__ row_offset,
                                                const I* __restrict__ col,
                                                const T* __restrict__ val,
                                                T* __restrict__ vec,
                                                int* __restrict__ detect_zero)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(ai == col[aj])
            {
                if(val[aj] != static_cast<T>(0))
                {
                    vec[ai] = static_cast<T>(1) / val[aj];
                }
                else
                {
                    vec[ai] = static_cast<T>(1);

                    *detect_zero = 1;
                }
            }
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_submatrix_row_nnz(const J* __restrict__ row_offset,
                                                         const I* __restrict__ col,
                                                         const T* __restrict__ val,
                                                         I smrow_offset,
                                                         I smcol_offset,
                                                         I smrow_size,
                                                         I smcol_size,
                                                         J* __restrict__ row_nnz)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= smrow_size)
        {
            return;
        }

        I nnz = 0;
        I ind = ai + smrow_offset;

        for(J aj = row_offset[ind]; aj < row_offset[ind + 1]; ++aj)
        {
            I c = col[aj];

            if((c >= smcol_offset) && (c < smcol_offset + smcol_size))
            {
                ++nnz;
            }
        }

        row_nnz[ai] = nnz;
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_submatrix_copy(const J* __restrict__ row_offset,
                                                      const I* __restrict__ col,
                                                      const T* __restrict__ val,
                                                      I smrow_offset,
                                                      I smcol_offset,
                                                      I smrow_size,
                                                      I smcol_size,
                                                      const J* __restrict__ sm_row_offset,
                                                      I* __restrict__ sm_col,
                                                      T* __restrict__ sm_val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= smrow_size)
        {
            return;
        }

        J row_nnz = sm_row_offset[ai];
        I ind     = ai + smrow_offset;

        for(J aj = row_offset[ind]; aj < row_offset[ind + 1]; ++aj)
        {
            I c = col[aj];

            if((c >= smcol_offset) && (c < smcol_offset + smcol_size))
            {
                sm_col[row_nnz] = c - smcol_offset;
                sm_val[row_nnz] = val[aj];
                ++row_nnz;
            }
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_diagmatmult_r(I nrow,
                                             const J* __restrict__ row_offset,
                                             const I* __restrict__ col,
                                             const T* __restrict__ diag,
                                             T* __restrict__ val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            val[aj] = val[aj] * diag[col[aj]];
        }
    }

    template <typename T, typename I, typename J>
    __global__ void kernel_csr_diagmatmult_l(I nrow,
                                             const J* __restrict__ row_offset,
                                             const T* __restrict__ diag,
                                             T* __restrict__ val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            val[aj] = val[aj] * diag[ai];
        }
    }

    // Calculates the number of non-zero elements per row
    template <typename I, typename J>
    __global__ void
        kernel_calc_row_nnz(I nrow, const J* __restrict__ row_offset, I* __restrict__ row_nnz)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

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
    template <typename I, typename J>
    __global__ void kernel_permute_row_nnz(I nrow,
                                           const J* __restrict__ row_offset,
                                           const I* __restrict__ perm_vec,
                                           J* __restrict__ row_nnz_dst)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

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
    template <unsigned int WF_SIZE, typename T, typename I, typename J>
    __global__ void kernel_permute_rows(I nrow,
                                        const J* __restrict__ row_offset,
                                        const J* __restrict__ perm_row_offset,
                                        const I* __restrict__ col,
                                        const T* __restrict__ data,
                                        const I* __restrict__ perm_vec,
                                        I* __restrict__ perm_col,
                                        T* __restrict__ perm_data)
    {
        I tid = threadIdx.x;
        I gid = blockIdx.x * blockDim.x + tid;
        I lid = tid & (WF_SIZE - 1);
        I row = gid / WF_SIZE;

        if(row >= nrow)
        {
            return;
        }

        J perm_index = perm_row_offset[perm_vec[row]];
        J prev_index = row_offset[row];
        I num_elems  = row_offset[row + 1] - prev_index;

        for(I i = lid; i < num_elems; i += WF_SIZE)
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
    template <unsigned int size, typename T, typename I, typename J>
    __launch_bounds__(256) __global__ void kernel_permute_cols(I nrow,
                                                               const J* __restrict__ row_offset,
                                                               const I* __restrict__ perm_vec,
                                                               const I* __restrict__ perm_col,
                                                               const T* __restrict__ perm_data,
                                                               I* __restrict__ col,
                                                               T* __restrict__ data)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        J elem_index = row_offset[ai];
        I num_elems  = row_offset[ai + 1] - elem_index;

        I ccol[size];
        T cval[size];

        col += elem_index;
        data += elem_index;
        perm_col += elem_index;
        perm_data += elem_index;

        for(I i = 0; i < num_elems; ++i)
        {
            ccol[i] = col[i];
            cval[i] = data[i];
        }

        for(I i = 0; i < num_elems; ++i)
        {
            I comp = perm_vec[perm_col[i]];

            I j;
            for(j = i - 1; j >= 0; --j)
            {
                I c = ccol[j];
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

        for(I i = 0; i < num_elems; ++i)
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
    template <typename T, typename I, typename J>
    __global__ void kernel_permute_cols_fallback(I nrow,
                                                 const J* __restrict__ row_offset,
                                                 const I* __restrict__ perm_vec,
                                                 const I* __restrict__ perm_col,
                                                 const T* __restrict__ perm_data,
                                                 I* __restrict__ col,
                                                 T* __restrict__ data)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        J elem_index = row_offset[ai];
        I num_elems  = row_offset[ai + 1] - elem_index;

        col += elem_index;
        data += elem_index;
        perm_col += elem_index;
        perm_data += elem_index;

        for(I i = 0; i < num_elems; ++i)
        {
            I comp = perm_vec[perm_col[i]];

            I j;
            for(j = i - 1; j >= 0; --j)
            {
                I c = col[j];
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
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_add_csr_same_struct(I nrow,
                                                   const J* __restrict__ out_row_offset,
                                                   const I* __restrict__ out_col,
                                                   const J* __restrict__ in_row_offset,
                                                   const I* __restrict__ in_col,
                                                   const T* __restrict__ in_val,
                                                   T alpha,
                                                   T beta,
                                                   T* __restrict__ out_val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        J first_col = in_row_offset[ai];

        for(J ajj = out_row_offset[ai]; ajj < out_row_offset[ai + 1]; ++ajj)
        {
            for(J aj = first_col; aj < in_row_offset[ai + 1]; ++aj)
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
    template <typename I, typename J>
    __global__ void kernel_csr_lower_nnz_per_row(I nrow,
                                                 const J* __restrict__ src_row_offset,
                                                 const I* __restrict__ src_col,
                                                 J* __restrict__ nnz_per_row)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        nnz_per_row[ai] = 0;

        for(J aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
        {
            if(src_col[aj] <= ai)
            {
                ++nnz_per_row[ai];
            }
        }
    }

    // Computes the upper triangular part nnz per row
    template <typename I, typename J>
    __global__ void kernel_csr_upper_nnz_per_row(I nrow,
                                                 const J* __restrict__ src_row_offset,
                                                 const I* __restrict__ src_col,
                                                 J* __restrict__ nnz_per_row)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        nnz_per_row[ai] = 0;

        for(J aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
        {
            if(src_col[aj] >= ai)
            {
                ++nnz_per_row[ai];
            }
        }
    }

    // Computes the stricktly lower triangular part nnz per row
    template <typename I, typename J>
    __global__ void kernel_csr_slower_nnz_per_row(I nrow,
                                                  const J* __restrict__ src_row_offset,
                                                  const I* __restrict__ src_col,
                                                  J* __restrict__ nnz_per_row)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        nnz_per_row[ai] = 0;

        for(J aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
        {
            if(src_col[aj] < ai)
            {
                ++nnz_per_row[ai];
            }
        }
    }

    // Computes the stricktly upper triangular part nnz per row
    template <typename I, typename J>
    __global__ void kernel_csr_supper_nnz_per_row(I nrow,
                                                  const J* __restrict__ src_row_offset,
                                                  const I* __restrict__ src_col,
                                                  J* __restrict__ nnz_per_row)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        nnz_per_row[ai] = 0;

        for(J aj = src_row_offset[ai]; aj < src_row_offset[ai + 1]; ++aj)
        {
            if(src_col[aj] > ai)
            {
                ++nnz_per_row[ai];
            }
        }
    }

    // Extracts lower triangular part for given nnz per row array (partial sums nnz)
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_l_triangular(I nrow,
                                                    const J* __restrict__ src_row_offset,
                                                    const I* __restrict__ src_col,
                                                    const T* __restrict__ src_val,
                                                    J* __restrict__ nnz_per_row,
                                                    I* __restrict__ dst_col,
                                                    T* __restrict__ dst_val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        J dst_index = nnz_per_row[ai];
        J src_index = src_row_offset[ai];

        for(J aj = 0; aj < nnz_per_row[ai + 1] - nnz_per_row[ai]; ++aj)
        {
            dst_col[dst_index] = src_col[src_index];
            dst_val[dst_index] = src_val[src_index];

            ++dst_index;
            ++src_index;
        }
    }

    // Extracts upper triangular part for given nnz per row array (partial sums nnz)
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_u_triangular(I nrow,
                                                    const J* __restrict__ src_row_offset,
                                                    const I* __restrict__ src_col,
                                                    const T* __restrict__ src_val,
                                                    J* __restrict__ nnz_per_row,
                                                    I* __restrict__ dst_col,
                                                    T* __restrict__ dst_val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        I num_elements = nnz_per_row[ai + 1] - nnz_per_row[ai];
        J src_index    = src_row_offset[ai + 1] - num_elements;
        J dst_index    = nnz_per_row[ai];

        for(I aj = 0; aj < num_elements; ++aj)
        {
            dst_col[dst_index] = src_col[src_index];
            dst_val[dst_index] = src_val[src_index];

            ++dst_index;
            ++src_index;
        }
    }

    // Compress
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_compress_count_nrow(const J* __restrict__ row_offset,
                                                   const I* __restrict__ col,
                                                   const T* __restrict__ val,
                                                   I      nrow,
                                                   double drop_off,
                                                   I* __restrict__ row_offset_new)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if((hip_abs(val[aj]) > drop_off) || (col[aj] == ai))
            {
                ++row_offset_new[ai];
            }
        }
    }

    // Compress
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_compress_copy(const J* __restrict__ row_offset,
                                             const I* __restrict__ col,
                                             const T* __restrict__ val,
                                             I      nrow,
                                             double drop_off,
                                             const J* __restrict__ row_offset_new,
                                             I* __restrict__ col_new,
                                             T* __restrict__ val_new)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        J ajj = row_offset_new[ai];

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
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
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_column_vector(const J* __restrict__ row_offset,
                                                     const I* __restrict__ col,
                                                     const T* __restrict__ val,
                                                     I nrow,
                                                     I idx,
                                                     T* __restrict__ vec)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        vec[ai] = static_cast<T>(0);

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
        {
            if(idx == col[aj])
            {
                vec[ai] = val[aj];
            }
        }
    }

    // Replace column vector - compute new offset
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_replace_column_vector_offset(const J* __restrict__ row_offset,
                                                            const I* __restrict__ col,
                                                            I nrow,
                                                            I idx,
                                                            const T* __restrict__ vec,
                                                            J* __restrict__ offset)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        bool add = true;
        I    val = row_offset[ai + 1] - row_offset[ai];

        for(J aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
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

        offset[ai] = val;
    }

    // Replace column vector - compute new offset
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_replace_column_vector(const J* __restrict__ row_offset,
                                                     const I* __restrict__ col,
                                                     const T* __restrict__ val,
                                                     I nrow,
                                                     I idx,
                                                     const T* __restrict__ vec,
                                                     const J* __restrict__ offset,
                                                     I* __restrict__ new_col,
                                                     T* __restrict__ new_val)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= nrow)
        {
            return;
        }

        J aj = row_offset[ai];
        J k  = offset[ai];

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
    template <typename T, typename I, typename J>
    __global__ void kernel_csr_extract_row_vector(const J* __restrict__ row_offset,
                                                  const I* __restrict__ col,
                                                  const T* __restrict__ val,
                                                  I row_nnz,
                                                  I idx,
                                                  T* __restrict__ vec)
    {
        I ai = blockIdx.x * blockDim.x + threadIdx.x;

        if(ai >= row_nnz)
        {
            return;
        }

        J aj         = row_offset[idx] + ai;
        vec[col[aj]] = val[aj];
    }

    // AMG Connect
    template <unsigned int WF_SIZE, typename T, typename I, typename J>
    __global__ void kernel_csr_amg_connect(I nrow,
                                           T eps2,
                                           const J* __restrict__ row_offset,
                                           const I* __restrict__ col,
                                           const T* __restrict__ val,
                                           const T* __restrict__ diag,
                                           I* __restrict__ connections)
    {
        I tid = threadIdx.x;
        I gid = blockIdx.x * blockDim.x + tid;
        I lid = tid & (WF_SIZE - 1);
        I row = gid / WF_SIZE;

        if(row >= nrow)
        {
            return;
        }

        T eps_diag_r = eps2 * diag[row];

        J start = row_offset[row];
        J end   = row_offset[row + 1];

        for(J i = start + lid; i < end; i += WF_SIZE)
        {
            I c = col[i];
            T v = val[i];

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

    template <typename I, typename J>
    __global__ void kernel_csr_amg_init_mis_tuples(I nrow,
                                                   const J* __restrict__ row_offset,
                                                   const I* __restrict__ connections,
                                                   mis_tuple* __restrict__ tuples)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        if(row >= nrow)
        {
            return;
        }

        I state = -2;

        J row_start = row_offset[row];
        J row_end   = row_offset[row + 1];

        for(J j = row_start; j < row_end; j++)
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

    template <unsigned int BLOCKSIZE, unsigned int WFSIZE, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_amg_find_max_mis_tuples_shared(I nrow,
                                                       const J* __restrict__ row_offset,
                                                       const I* __restrict__ cols,
                                                       const I* __restrict__ connections,
                                                       const mis_tuple* __restrict__ tuples,
                                                       mis_tuple* __restrict__ max_tuples,
                                                       bool* done)
    {
        I tid = threadIdx.x;
        I gid = blockIdx.x * BLOCKSIZE + tid;
        I lid = tid & (WFSIZE - 1);
        I wid = tid / WFSIZE;
        I row = gid / WFSIZE;

        __shared__ mis_tuple smax_tuple[BLOCKSIZE];

        if(row >= nrow)
        {
            return;
        }

        smax_tuple[WFSIZE * wid] = max_tuples[row];

        __syncthreads();

        for(I k = 0; k < 2; k++)
        {
            mis_tuple t_max = smax_tuple[WFSIZE * wid];

            J row_start = row_offset[t_max.i];
            J row_end   = row_offset[t_max.i + 1];

            for(J j = row_start + lid; j < row_end; j += WFSIZE)
            {
                if(connections[j] == 1)
                {
                    I         col = cols[j];
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

    template <typename I>
    __global__ void kernel_csr_amg_update_mis_tuples(I nrow,
                                                     const mis_tuple* __restrict__ max_tuples,
                                                     mis_tuple* __restrict__ tuples,
                                                     I* __restrict__ aggregates,
                                                     bool* done)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

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

    template <typename I, typename J>
    __global__ void kernel_csr_amg_update_aggregates(I nrow,
                                                     const J* __restrict__ row_offset,
                                                     const I* __restrict__ cols,
                                                     const I* __restrict__ connections,
                                                     const mis_tuple* __restrict__ max_tuples,
                                                     mis_tuple* __restrict__ tuples,
                                                     I* __restrict__ aggregates)
    {
        I row = blockIdx.x * blockDim.x + threadIdx.x;

        if(row >= nrow)
        {
            return;
        }

        mis_tuple t = max_tuples[row];

        if(t.s == -1)
        {
            J row_start = row_offset[row];
            J row_end   = row_offset[row + 1];

            for(J j = row_start; j < row_end; j++)
            {
                if(connections[j] == 1)
                {
                    I col = cols[j];

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
              typename I,
              typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_sa_prolong_nnz(I nrow,
                                       const J* __restrict__ csr_row_ptr,
                                       const I* __restrict__ csr_col_ind,
                                       const I* __restrict__ connections,
                                       const I* __restrict__ aggregates,
                                       J* __restrict__ row_nnz)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        if(warpSize == WFSIZE)
        {
            wid = __builtin_amdgcn_readfirstlane(wid);
        }

        // The row this thread operates on
        I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Shared memory for the unordered set
        __shared__ I stable[BLOCKSIZE / WFSIZE * HASHSIZE];

        // Each wavefront operates on its own set
        unordered_set<I, HASHSIZE, WFSIZE> set(&stable[wid * HASHSIZE]);

        // Row nnz counter
        I nnz = 0;

        // Row entry and exit points
        I row_begin = csr_row_ptr[row];
        I row_end   = csr_row_ptr[row + 1];

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(I j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Get the column index
            I col = csr_col_ind[j];

            // Get connection state
            I con = connections[j];

            // Get aggregation state for this column
            I agg = aggregates[col];

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
              typename T,
              typename I,
              typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_sa_prolong_fill(I   nrow,
                                        T   relax,
                                        int lumping_strat,
                                        const J* __restrict__ csr_row_ptr,
                                        const I* __restrict__ csr_col_ind,
                                        const T* __restrict__ csr_val,
                                        const I* __restrict__ connections,
                                        const I* __restrict__ aggregates,
                                        const J* __restrict__ csr_row_ptr_P,
                                        I* __restrict__ csr_col_ind_P,
                                        T* __restrict__ csr_val_P)
    {
        unsigned int lid = threadIdx.x & (WFSIZE - 1);
        unsigned int wid = threadIdx.x / WFSIZE;

        if(warpSize == WFSIZE)
        {
            wid = __builtin_amdgcn_readfirstlane(wid);
        }

        // The row this thread operates on
        I row = blockIdx.x * BLOCKSIZE / WFSIZE + wid;

        // Do not run out of bounds
        if(row >= nrow)
        {
            return;
        }

        // Shared memory for the unordered map
        __shared__ I    stable[(BLOCKSIZE / WFSIZE) * HASHSIZE];
        __shared__ char smem[(BLOCKSIZE / WFSIZE) * HASHSIZE * sizeof(T)];

        T* sdata = reinterpret_cast<T*>(smem);

        // Each wavefront operates on its own map
        unordered_map<I, T, HASHSIZE, WFSIZE> map(&stable[wid * HASHSIZE], &sdata[wid * HASHSIZE]);

        // Row entry and exit points
        J row_begin = csr_row_ptr[row];
        J row_end   = csr_row_ptr[row + 1];

        // Accumulator
        T dia = static_cast<T>(0);

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Get the column index
            I col_j = csr_col_ind[j];

            // Get the value
            T val_j = csr_val[j];

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

        dia = static_cast<T>(1) / dia;

        // Loop over all columns of the i-th row, whereas each lane processes a column
        for(J j = row_begin + lid; j < row_end; j += WFSIZE)
        {
            // Get the column index
            I col_j = csr_col_ind[j];

            // Get the value
            T val_j = csr_val[j];

            // Get connection state
            I con = connections[j];

            // Get aggregation state for this column
            I agg = aggregates[col_j];

            // When aggregate is defined, diagonal entries and connected columns will
            // generate a fill in
            if((row == col_j || con) && agg >= 0)
            {
                // Insert or add if already present into unordered map
                T val = (col_j == row) ? static_cast<T>(1) - relax : -relax * dia * val_j;

                map.insert_or_add(agg, val);
            }
        }

        // Access into P
        J aj = csr_row_ptr_P[row];

        // Store key val pairs from map into P
        map.store_sorted(&csr_col_ind_P[aj], &csr_val_P[aj]);
    }

    template <unsigned int BLOCKSIZE, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_unsmoothed_prolong_nnz_per_row(I nrow,
                                                       const I* __restrict__ aggregates,
                                                       J* __restrict__ prolong_row_offset)
    {
        I tid = threadIdx.x;
        I row = blockIdx.x * BLOCKSIZE + tid;

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

    template <unsigned int BLOCKSIZE, typename T, typename I>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_unsmoothed_prolong_fill_simple(I nrow,
                                                       const I* __restrict__ aggregates,
                                                       I* __restrict__ prolong_cols,
                                                       T* __restrict__ prolong_vals)
    {

        I tid = threadIdx.x;
        I row = blockIdx.x * BLOCKSIZE + tid;

        if(row >= nrow)
        {
            return;
        }

        prolong_cols[row] = aggregates[row];
        prolong_vals[row] = static_cast<T>(1);
    }

    template <unsigned int BLOCKSIZE, typename T, typename I, typename J>
    __launch_bounds__(BLOCKSIZE) __global__
        void kernel_csr_unsmoothed_prolong_fill(I nrow,
                                                const I* __restrict__ aggregates,
                                                const J* __restrict__ prolong_row_offset,
                                                I* __restrict__ prolong_cols,
                                                T* __restrict__ prolong_vals)
    {

        I tid = threadIdx.x;
        I row = blockIdx.x * BLOCKSIZE + tid;

        if(row >= nrow)
        {
            return;
        }

        I agg = aggregates[row];

        if(agg >= 0)
        {
            J j = prolong_row_offset[row];

            prolong_cols[j] = agg;
            prolong_vals[j] = static_cast<T>(1);
        }
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_CSR_HPP_
