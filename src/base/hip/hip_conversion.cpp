/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

#include "hip_conversion.hpp"
#include "../../utils/def.hpp"
#include "../matrix_formats.hpp"
#include "hip_allocate_free.hpp"
#include "hip_kernels_conversion.hpp"
#include "hip_sparse.hpp"
#include "hip_blas.hpp"
#include "hip_utils.hpp"

#include <hip/hip_runtime_api.h>
#include <rocprim/rocprim.hpp>

#include <complex>

namespace rocalution
{
    template <typename ValueType, typename IndexType>
    bool csr_to_coo_hip(const rocsparse_handle                 handle,
                        IndexType                              nnz,
                        IndexType                              nrow,
                        IndexType                              ncol,
                        const MatrixCSR<ValueType, IndexType>& src,
                        MatrixCOO<ValueType, IndexType>*       dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(handle != NULL);

        allocate_hip(nnz, &dst->row);
        allocate_hip(nnz, &dst->col);
        allocate_hip(nnz, &dst->val);

        hipMemcpyAsync(dst->col, src.col, sizeof(IndexType) * nnz, hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpyAsync(dst->val, src.val, sizeof(ValueType) * nnz, hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        rocsparse_status status = rocsparse_csr2coo(
            handle, src.row_offset, nnz, nrow, dst->row, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Sync memcopy
        hipDeviceSynchronize();

        return true;
    }

    template <typename ValueType, typename IndexType>
    bool coo_to_csr_hip(const rocsparse_handle                 handle,
                        IndexType                              nnz,
                        IndexType                              nrow,
                        IndexType                              ncol,
                        const MatrixCOO<ValueType, IndexType>& src,
                        MatrixCSR<ValueType, IndexType>*       dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(handle != NULL);

        allocate_hip(nrow + 1, &dst->row_offset);
        allocate_hip(nnz, &dst->col);
        allocate_hip(nnz, &dst->val);

        hipMemcpyAsync(dst->col, src.col, sizeof(IndexType) * nnz, hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpyAsync(dst->val, src.val, sizeof(ValueType) * nnz, hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        rocsparse_status status = rocsparse_coo2csr(
            handle, src.row, nnz, nrow, dst->row_offset, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Sync memcopy
        hipDeviceSynchronize();

        return true;
    }

    template <typename ValueType, typename IndexType>
    bool csr_to_bcsr_hip(const rocsparse_handle                 handle,
                         IndexType                              nnz,
                         IndexType                              nrow,
                         IndexType                              ncol,
                         const MatrixCSR<ValueType, IndexType>& src,
                         const rocsparse_mat_descr              src_descr,
                         MatrixBCSR<ValueType, IndexType>*      dst,
                         const rocsparse_mat_descr              dst_descr)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(handle != NULL);

        IndexType blockdim = dst->blockdim;

        assert(blockdim > 1);

        // Matrix dimensions must be a multiple of blockdim
        if((nrow % blockdim) != 0 || (ncol % blockdim) != 0)
        {
            return false;
        }

        // BCSR row blocks
        IndexType mb = (nrow + blockdim - 1) / blockdim;
        IndexType nb = (ncol + blockdim - 1) / blockdim;
        IndexType nnzb;

        rocsparse_direction dir
            = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

        allocate_hip(mb + 1, &dst->row_offset);

        rocsparse_status status = rocsparse_csr2bsr_nnz(handle,
                                                        dir,
                                                        nrow,
                                                        ncol,
                                                        src_descr,
                                                        src.row_offset,
                                                        src.col,
                                                        blockdim,
                                                        dst_descr,
                                                        dst->row_offset,
                                                        &nnzb);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(nnzb, &dst->col);
        allocate_hip(nnzb * blockdim * blockdim, &dst->val);

        status = rocsparseTcsr2bsr(handle,
                                   dir,
                                   nrow,
                                   ncol,
                                   src_descr,
                                   src.val,
                                   src.row_offset,
                                   src.col,
                                   blockdim,
                                   dst_descr,
                                   dst->val,
                                   dst->row_offset,
                                   dst->col);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        dst->nrowb = mb;
        dst->ncolb = nb;
        dst->nnzb  = nnzb;

        return true;
    }

    template <typename ValueType, typename IndexType>
    bool bcsr_to_csr_hip(const rocsparse_handle                  handle,
                         IndexType                               nnz,
                         IndexType                               nrow,
                         IndexType                               ncol,
                         const MatrixBCSR<ValueType, IndexType>& src,
                         MatrixCSR<ValueType, IndexType>*        dst)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(handle != NULL);

        allocate_hip(nrow + 1, &dst->row_offset);
        allocate_hip(nnz, &dst->col);
        allocate_hip(nnz, &dst->val);

        hipMemcpyAsync(dst->col, src.col, sizeof(IndexType) * nnz, hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpyAsync(dst->val, src.val, sizeof(ValueType) * nnz, hipMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        rocsparse_status status = rocsparse_coo2csr(
            handle, src.row, nnz, nrow, dst->row_offset, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Sync memcopy
        hipDeviceSynchronize();

        return true;
    }

    template <typename ValueType, typename IndexType>
    bool csr_to_ell_hip(const rocsparse_handle                 handle,
                        IndexType                              nnz,
                        IndexType                              nrow,
                        IndexType                              ncol,
                        const MatrixCSR<ValueType, IndexType>& src,
                        const rocsparse_mat_descr              src_descr,
                        MatrixELL<ValueType, IndexType>*       dst,
                        const rocsparse_mat_descr              dst_descr,
                        IndexType*                             nnz_ell)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(nnz_ell != NULL);
        assert(handle != NULL);
        assert(src_descr != NULL);
        assert(dst_descr != NULL);

        rocsparse_status status;

        // Determine ELL width
        status = rocsparse_csr2ell_width(
            handle, nrow, src_descr, src.row_offset, dst_descr, &dst->max_row);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Limit ELL size to 5 times CSR nnz
        if(dst->max_row > 5 * (nnz / nrow))
        {
            return false;
        }

        // Compute ELL non-zeros
        *nnz_ell = dst->max_row * nrow;

        // Allocate ELL matrix
        allocate_hip(*nnz_ell, &dst->col);
        allocate_hip(*nnz_ell, &dst->val);

        // Conversion
        status = rocsparseTcsr2ell(handle,
                                   nrow,
                                   src_descr,
                                   src.val,
                                   src.row_offset,
                                   src.col,
                                   dst_descr,
                                   dst->max_row,
                                   dst->val,
                                   dst->col);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return true;
    }

    template <typename ValueType, typename IndexType>
    bool ell_to_csr_hip(const rocsparse_handle                 handle,
                        IndexType                              nnz,
                        IndexType                              nrow,
                        IndexType                              ncol,
                        const MatrixELL<ValueType, IndexType>& src,
                        const rocsparse_mat_descr              src_descr,
                        MatrixCSR<ValueType, IndexType>*       dst,
                        const rocsparse_mat_descr              dst_descr,
                        IndexType*                             nnz_csr)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(nnz_csr != NULL);
        assert(handle != NULL);
        assert(src_descr != NULL);
        assert(dst_descr != NULL);

        rocsparse_status status;

        // Allocate CSR row offset structure
        allocate_hip(nrow + 1, &dst->row_offset);

        // Determine CSR nnz
        status = rocsparse_ell2csr_nnz(handle,
                                       nrow,
                                       ncol,
                                       src_descr,
                                       src.max_row,
                                       src.col,
                                       dst_descr,
                                       dst->row_offset,
                                       nnz_csr);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        if(*nnz_csr < 0)
        {
            free_hip(&dst->row_offset);
            return false;
        }

        // Allocate CSR column and value structures
        allocate_hip(*nnz_csr, &dst->col);
        allocate_hip(*nnz_csr, &dst->val);

        // Conversion
        status = rocsparseTell2csr(handle,
                                   nrow,
                                   ncol,
                                   src_descr,
                                   src.max_row,
                                   src.val,
                                   src.col,
                                   dst_descr,
                                   dst->val,
                                   dst->row_offset,
                                   dst->col);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        return true;
    }

    template <typename ValueType, typename IndexType>
    bool csr_to_dia_hip(int                                    blocksize,
                        IndexType                              nnz,
                        IndexType                              nrow,
                        IndexType                              ncol,
                        const MatrixCSR<ValueType, IndexType>& src,
                        MatrixDIA<ValueType, IndexType>*       dst,
                        IndexType*                             nnz_dia,
                        IndexType*                             num_diag)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);
        assert(blocksize > 0);

        assert(dst != NULL);
        assert(nnz_dia != NULL);
        assert(num_diag != NULL);

        // Get diagonal mapping vector
        IndexType* diag_idx = NULL;
        allocate_hip(nrow + ncol, &diag_idx);
        set_to_zero_hip(blocksize, nrow + ncol, diag_idx);

        dim3 diag_blocks((nrow - 1) / blocksize + 1);
        dim3 diag_threads(blocksize);

        hipLaunchKernelGGL((kernel_dia_diag_idx<IndexType>),
                           diag_blocks,
                           diag_threads,
                           0,
                           0,
                           nrow,
                           src.row_offset,
                           src.col,
                           diag_idx);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Reduction to obtain number of occupied diagonals
        IndexType* d_num_diag = NULL;
        allocate_hip(1, &d_num_diag);

        size_t rocprim_size   = 0;
        void*  rocprim_buffer = NULL;

        // Get reduction buffer size
        rocprim::reduce(rocprim_buffer,
                        rocprim_size,
                        diag_idx,
                        d_num_diag,
                        0,
                        nrow + ncol,
                        rocprim::plus<IndexType>());

        // Allocate rocprim buffer
        hipMalloc(&rocprim_buffer, rocprim_size);

        // Do reduction
        rocprim::reduce(rocprim_buffer,
                        rocprim_size,
                        diag_idx,
                        d_num_diag,
                        0,
                        nrow + ncol,
                        rocprim::plus<IndexType>());

        // Clear rocprim buffer
        hipFree(rocprim_buffer);

        // Copy result to host
        hipMemcpy(num_diag, d_num_diag, sizeof(IndexType), hipMemcpyDeviceToHost);

        // Free device memory
        free_hip(&d_num_diag);

        // Conversion fails if DIA nnz exceeds 5 times CSR nnz
        IndexType size = (nrow > ncol) ? nrow : ncol;
        if(*num_diag > 5 * (nnz / size))
        {
            free_hip(&diag_idx);
            return false;
        }

        *nnz_dia = *num_diag * size;

        // Allocate DIA matrix
        allocate_hip(*num_diag, &dst->offset);
        allocate_hip(*nnz_dia, &dst->val);

        // Initialize values with zero
        set_to_zero_hip(blocksize, *num_diag, dst->offset);
        set_to_zero_hip(blocksize, *nnz_dia, dst->val);

        // Inclusive sum to obtain diagonal offsets
        IndexType* work = NULL;
        allocate_hip(nrow + ncol, &work);
        rocprim_buffer = NULL;
        rocprim_size   = 0;

        // Obtain rocprim buffer size
        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                diag_idx,
                                work,
                                0,
                                nrow + ncol,
                                rocprim::plus<IndexType>());

        // Allocate rocprim buffer
        hipMalloc(&rocprim_buffer, rocprim_size);

        // Do inclusive sum
        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                diag_idx,
                                work,
                                0,
                                nrow + ncol,
                                rocprim::plus<IndexType>());

        // Clear rocprim buffer
        hipFree(rocprim_buffer);

        // Fill DIA structures
        dim3 fill_blocks((nrow + ncol) / blocksize + 1);
        dim3 fill_threads(blocksize);

        hipLaunchKernelGGL((kernel_dia_fill_offset<IndexType>),
                           fill_blocks,
                           fill_threads,
                           0,
                           0,
                           nrow,
                           ncol,
                           diag_idx,
                           work,
                           dst->offset);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&work);

        dim3 conv_blocks((nrow - 1) / blocksize + 1);
        dim3 conv_threads(blocksize);

        hipLaunchKernelGGL((kernel_dia_convert<ValueType, IndexType>),
                           conv_blocks,
                           conv_threads,
                           0,
                           0,
                           nrow,
                           *num_diag,
                           src.row_offset,
                           src.col,
                           src.val,
                           diag_idx,
                           dst->val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Clear
        free_hip(&diag_idx);

        return true;
    }

    template <typename ValueType, typename IndexType>
    bool csr_to_hyb_hip(int                                    blocksize,
                        IndexType                              nnz,
                        IndexType                              nrow,
                        IndexType                              ncol,
                        const MatrixCSR<ValueType, IndexType>& src,
                        MatrixHYB<ValueType, IndexType>*       dst,
                        IndexType*                             nnz_hyb,
                        IndexType*                             nnz_ell,
                        IndexType*                             nnz_coo)
    {
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);
        assert(blocksize > 0);

        assert(dst != NULL);
        assert(nnz_hyb != NULL);
        assert(nnz_ell != NULL);
        assert(nnz_coo != NULL);

        // Determine ELL width by average nnz per row
        if(dst->ELL.max_row == 0)
        {
            dst->ELL.max_row = (nnz - 1) / nrow + 1;
        }

        // ELL nnz is ELL width times nrow
        *nnz_ell = dst->ELL.max_row * nrow;
        *nnz_coo = 0;

        // Allocate ELL part
        allocate_hip(*nnz_ell, &dst->ELL.col);
        allocate_hip(*nnz_ell, &dst->ELL.val);

        // Array to gold COO part nnz per row
        IndexType* coo_row_nnz = NULL;
        allocate_hip(nrow + 1, &coo_row_nnz);

        // If there is no ELL part, its easy
        if(*nnz_ell == 0)
        {
            *nnz_coo = nnz;
            hipMemcpy(coo_row_nnz,
                      src.row_offset,
                      sizeof(IndexType) * (nrow + 1),
                      hipMemcpyDeviceToDevice);
        }
        else
        {
            dim3 blocks((nrow - 1) / blocksize + 1);
            dim3 threads(blocksize);

            hipLaunchKernelGGL((kernel_hyb_coo_nnz),
                               blocks,
                               threads,
                               0,
                               0,
                               nrow,
                               dst->ELL.max_row,
                               src.row_offset,
                               coo_row_nnz);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            // Inclusive sum on coo_row_nnz
            size_t rocprim_size   = 0;
            void*  rocprim_buffer = NULL;

            // Obtain rocprim buffer size
            rocprim::exclusive_scan(rocprim_buffer,
                                    rocprim_size,
                                    coo_row_nnz,
                                    coo_row_nnz,
                                    0,
                                    nrow + 1,
                                    rocprim::plus<IndexType>());

            // Allocate rocprim buffer
            hipMalloc(&rocprim_buffer, rocprim_size);

            // Do exclusive sum
            rocprim::exclusive_scan(rocprim_buffer,
                                    rocprim_size,
                                    coo_row_nnz,
                                    coo_row_nnz,
                                    0,
                                    nrow + 1,
                                    rocprim::plus<IndexType>());

            // Clear rocprim buffer
            hipFree(rocprim_buffer);

            // Copy result to host
            hipMemcpy(nnz_coo, coo_row_nnz + nrow, sizeof(IndexType), hipMemcpyDeviceToHost);
        }

        *nnz_hyb = *nnz_coo + *nnz_ell;

        if(*nnz_hyb <= 0)
        {
            return false;
        }

        // Allocate COO part
        allocate_hip(*nnz_coo, &dst->COO.row);
        allocate_hip(*nnz_coo, &dst->COO.col);
        allocate_hip(*nnz_coo, &dst->COO.val);

        // Fill HYB structures
        dim3 blocks((nrow - 1) / blocksize + 1);
        dim3 threads(blocksize);

        hipLaunchKernelGGL((kernel_hyb_csr2hyb<ValueType>),
                           blocks,
                           threads,
                           0,
                           0,
                           nrow,
                           src.val,
                           src.row_offset,
                           src.col,
                           dst->ELL.max_row,
                           dst->ELL.col,
                           dst->ELL.val,
                           dst->COO.row,
                           dst->COO.col,
                           dst->COO.val,
                           coo_row_nnz);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&coo_row_nnz);

        return true;
    }

    template <typename ValueType, typename IndexType>
    bool csr_to_dense_hip(const rocsparse_handle                 sparse_handle,
                          const rocblas_handle                   blas_handle,
                          IndexType                              nrow,
                          IndexType                              ncol,
                          const MatrixCSR<ValueType, IndexType>& src,
                          const rocsparse_mat_descr              src_descr,
                          MatrixDENSE<ValueType>*                dst)
    {
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(sparse_handle != NULL);
        assert(blas_handle != NULL);
        assert(src_descr != NULL);

        allocate_hip(nrow * ncol, &dst->val);

        if(DENSE_IND_BASE == 0)
        {
            rocsparse_status status = rocsparseTcsr2dense(
            sparse_handle, nrow, ncol, src_descr, src.val, src.row_offset, src.col, dst->val, nrow);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
        else
        {
            ValueType* temp = NULL;
            allocate_hip(nrow * ncol, &temp);

            rocsparse_status sparse_status = rocsparseTcsr2dense(
            sparse_handle, nrow, ncol, src_descr, src.val, src.row_offset, src.col, temp, nrow);
            CHECK_ROCSPARSE_ERROR(sparse_status, __FILE__, __LINE__);

            ValueType alpha = static_cast<ValueType>(1);
            ValueType beta = static_cast<ValueType>(0);

            // Not actually used in following geam call as beta is zero
            ValueType* B; 

            // transpose matrix so that dst values are row major
            rocblas_status blas_status = rocblasTgeam(blas_handle,
                                                      rocblas_operation_transpose,
                                                      rocblas_operation_none,
                                                      nrow,
                                                      ncol,
                                                      &alpha,
                                                      temp,
                                                      nrow,
                                                      &beta,
                                                      B,
                                                      nrow,
                                                      dst->val,
                                                      nrow);
            CHECK_ROCBLAS_ERROR(blas_status, __FILE__, __LINE__);

            free_hip(&temp);
        }

        // Sync memcopy
        hipDeviceSynchronize();

        return true;
    }

    template <typename ValueType, typename IndexType>
    bool dense_to_csr_hip(const rocsparse_handle           sparse_handle,
                          const rocblas_handle             blas_handle,
                          IndexType                        nrow,
                          IndexType                        ncol,
                          const MatrixDENSE<ValueType>&    src,
                          MatrixCSR<ValueType, IndexType>* dst,
                          const rocsparse_mat_descr        dst_descr)
    {
        assert(nrow > 0);
        assert(ncol > 0);

        assert(dst != NULL);
        assert(sparse_handle != NULL);
        assert(blas_handle != NULL);
        assert(dst_descr != NULL);

        IndexType nnz_total;
        IndexType* nnz_per_row = NULL;

        if(DENSE_IND_BASE == 0)
        {
            rocsparse_status status;

            allocate_hip(nrow, &nnz_per_row);

            status = rocsparseTnnz(
                sparse_handle, rocsparse_direction_row, nrow, ncol, dst_descr, src.val, nrow, nnz_per_row, &nnz_total);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);  

            allocate_hip(nrow + 1, &dst->row_offset);
            allocate_hip(nnz_total, &dst->col);
            allocate_hip(nnz_total, &dst->val);

            status = rocsparseTdense2csr(
                sparse_handle, nrow, ncol, dst_descr, src.val, nrow, nnz_per_row, dst->val, dst->row_offset, dst->col);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            free_hip(&nnz_per_row);
        }
        else
        {
            ValueType* temp = NULL;
            allocate_hip(nrow * ncol, &temp);

            ValueType alpha = static_cast<ValueType>(1);
            ValueType beta = static_cast<ValueType>(0);

            // Not actually used in following geam call as beta is zero
            ValueType* B; 

            // transpose matrix so that src values are column major
            rocblas_status blas_status = rocblasTgeam(blas_handle,
                                                      rocblas_operation_transpose,
                                                      rocblas_operation_none,
                                                      nrow,
                                                      ncol,
                                                      &alpha,
                                                      src.val,
                                                      nrow,
                                                      &beta,
                                                      B,
                                                      nrow,
                                                      temp,
                                                      nrow);
            CHECK_ROCBLAS_ERROR(blas_status, __FILE__, __LINE__);

            allocate_hip(nrow, &nnz_per_row);

            rocsparse_status sparse_status;

            sparse_status = rocsparseTnnz(
                sparse_handle, rocsparse_direction_row, nrow, ncol, dst_descr, temp, nrow, nnz_per_row, &nnz_total);
            CHECK_ROCSPARSE_ERROR(sparse_status, __FILE__, __LINE__);  

            allocate_hip(nrow + 1, &dst->row_offset);
            allocate_hip(nnz_total, &dst->col);
            allocate_hip(nnz_total, &dst->val);

            sparse_status = rocsparseTdense2csr(
                sparse_handle, nrow, ncol, dst_descr, temp, nrow, nnz_per_row, dst->val, dst->row_offset, dst->col);
            CHECK_ROCSPARSE_ERROR(sparse_status, __FILE__, __LINE__);

            free_hip(&temp);
            free_hip(&nnz_per_row);
        }

        // Sync memcopy
        hipDeviceSynchronize();

        return true;
    }

    // csr_to_coo
    template bool csr_to_coo_hip(const rocsparse_handle       handle,
                                 int                          nnz,
                                 int                          nrow,
                                 int                          ncol,
                                 const MatrixCSR<float, int>& src,
                                 MatrixCOO<float, int>*       dst);

    template bool csr_to_coo_hip(const rocsparse_handle        handle,
                                 int                           nnz,
                                 int                           nrow,
                                 int                           ncol,
                                 const MatrixCSR<double, int>& src,
                                 MatrixCOO<double, int>*       dst);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_coo_hip(const rocsparse_handle                     handle,
                                 int                                        nnz,
                                 int                                        nrow,
                                 int                                        ncol,
                                 const MatrixCSR<std::complex<float>, int>& src,
                                 MatrixCOO<std::complex<float>, int>*       dst);

    template bool csr_to_coo_hip(const rocsparse_handle                      handle,
                                 int                                         nnz,
                                 int                                         nrow,
                                 int                                         ncol,
                                 const MatrixCSR<std::complex<double>, int>& src,
                                 MatrixCOO<std::complex<double>, int>*       dst);
#endif

    // coo_to_csr
    template bool coo_to_csr_hip(const rocsparse_handle       handle,
                                 int                          nnz,
                                 int                          nrow,
                                 int                          ncol,
                                 const MatrixCOO<float, int>& src,
                                 MatrixCSR<float, int>*       dst);

    template bool coo_to_csr_hip(const rocsparse_handle        handle,
                                 int                           nnz,
                                 int                           nrow,
                                 int                           ncol,
                                 const MatrixCOO<double, int>& src,
                                 MatrixCSR<double, int>*       dst);

#ifdef SUPPORT_COMPLEX
    template bool coo_to_csr_hip(const rocsparse_handle                     handle,
                                 int                                        nnz,
                                 int                                        nrow,
                                 int                                        ncol,
                                 const MatrixCOO<std::complex<float>, int>& src,
                                 MatrixCSR<std::complex<float>, int>*       dst);

    template bool coo_to_csr_hip(const rocsparse_handle                      handle,
                                 int                                         nnz,
                                 int                                         nrow,
                                 int                                         ncol,
                                 const MatrixCOO<std::complex<double>, int>& src,
                                 MatrixCSR<std::complex<double>, int>*       dst);
#endif

    // csr_to_bcsr
    template bool csr_to_bcsr_hip(const rocsparse_handle       handle,
                                  int                          nnz,
                                  int                          nrow,
                                  int                          ncol,
                                  const MatrixCSR<float, int>& src,
                                  const rocsparse_mat_descr    src_descr,
                                  MatrixBCSR<float, int>*      dst,
                                  const rocsparse_mat_descr    dst_descr);

    template bool csr_to_bcsr_hip(const rocsparse_handle        handle,
                                  int                           nnz,
                                  int                           nrow,
                                  int                           ncol,
                                  const MatrixCSR<double, int>& src,
                                  const rocsparse_mat_descr     src_descr,
                                  MatrixBCSR<double, int>*      dst,
                                  const rocsparse_mat_descr     dst_descr);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_bcsr_hip(const rocsparse_handle                     handle,
                                  int                                        nnz,
                                  int                                        nrow,
                                  int                                        ncol,
                                  const MatrixCSR<std::complex<float>, int>& src,
                                  const rocsparse_mat_descr                  src_descr,
                                  MatrixBCSR<std::complex<float>, int>*      dst,
                                  const rocsparse_mat_descr                  dst_descr);

    template bool csr_to_bcsr_hip(const rocsparse_handle                      handle,
                                  int                                         nnz,
                                  int                                         nrow,
                                  int                                         ncol,
                                  const MatrixCSR<std::complex<double>, int>& src,
                                  const rocsparse_mat_descr                   src_descr,
                                  MatrixBCSR<std::complex<double>, int>*      dst,
                                  const rocsparse_mat_descr                   dst_descr);
#endif

    // csr_to_ell
    template bool csr_to_ell_hip(const rocsparse_handle       handle,
                                 int                          nnz,
                                 int                          nrow,
                                 int                          ncol,
                                 const MatrixCSR<float, int>& src,
                                 const rocsparse_mat_descr    src_descr,
                                 MatrixELL<float, int>*       dst,
                                 const rocsparse_mat_descr    dst_descr,
                                 int*                         nnz_ell);

    template bool csr_to_ell_hip(const rocsparse_handle        handle,
                                 int                           nnz,
                                 int                           nrow,
                                 int                           ncol,
                                 const MatrixCSR<double, int>& src,
                                 const rocsparse_mat_descr     src_descr,
                                 MatrixELL<double, int>*       dst,
                                 const rocsparse_mat_descr     dst_descr,
                                 int*                          nnz_ell);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_ell_hip(const rocsparse_handle                     handle,
                                 int                                        nnz,
                                 int                                        nrow,
                                 int                                        ncol,
                                 const MatrixCSR<std::complex<float>, int>& src,
                                 const rocsparse_mat_descr                  src_descr,
                                 MatrixELL<std::complex<float>, int>*       dst,
                                 const rocsparse_mat_descr                  dst_descr,
                                 int*                                       nnz_ell);

    template bool csr_to_ell_hip(const rocsparse_handle                      handle,
                                 int                                         nnz,
                                 int                                         nrow,
                                 int                                         ncol,
                                 const MatrixCSR<std::complex<double>, int>& src,
                                 const rocsparse_mat_descr                   src_descr,
                                 MatrixELL<std::complex<double>, int>*       dst,
                                 const rocsparse_mat_descr                   dst_descr,
                                 int*                                        nnz_ell);
#endif

    // ell_to_csr
    template bool ell_to_csr_hip(const rocsparse_handle       handle,
                                 int                          nnz,
                                 int                          nrow,
                                 int                          ncol,
                                 const MatrixELL<float, int>& src,
                                 const rocsparse_mat_descr    src_descr,
                                 MatrixCSR<float, int>*       dst,
                                 const rocsparse_mat_descr    dst_descr,
                                 int*                         nnz_csr);

    template bool ell_to_csr_hip(const rocsparse_handle        handle,
                                 int                           nnz,
                                 int                           nrow,
                                 int                           ncol,
                                 const MatrixELL<double, int>& src,
                                 const rocsparse_mat_descr     src_descr,
                                 MatrixCSR<double, int>*       dst,
                                 const rocsparse_mat_descr     dst_descr,
                                 int*                          nnz_csr);

#ifdef SUPPORT_COMPLEX
    template bool ell_to_csr_hip(const rocsparse_handle                     handle,
                                 int                                        nnz,
                                 int                                        nrow,
                                 int                                        ncol,
                                 const MatrixELL<std::complex<float>, int>& src,
                                 const rocsparse_mat_descr                  src_descr,
                                 MatrixCSR<std::complex<float>, int>*       dst,
                                 const rocsparse_mat_descr                  dst_descr,
                                 int*                                       nnz_csr);

    template bool ell_to_csr_hip(const rocsparse_handle                      handle,
                                 int                                         nnz,
                                 int                                         nrow,
                                 int                                         ncol,
                                 const MatrixELL<std::complex<double>, int>& src,
                                 const rocsparse_mat_descr                   src_descr,
                                 MatrixCSR<std::complex<double>, int>*       dst,
                                 const rocsparse_mat_descr                   dst_descr,
                                 int*                                        nnz_csr);
#endif

    // csr_to_dia
    template bool csr_to_dia_hip(int                          blocksize,
                                 int                          nnz,
                                 int                          nrow,
                                 int                          ncol,
                                 const MatrixCSR<float, int>& src,
                                 MatrixDIA<float, int>*       dst,
                                 int*                         nnz_dia,
                                 int*                         num_diag);

    template bool csr_to_dia_hip(int                           blocksize,
                                 int                           nnz,
                                 int                           nrow,
                                 int                           ncol,
                                 const MatrixCSR<double, int>& src,
                                 MatrixDIA<double, int>*       dst,
                                 int*                          nnz_dia,
                                 int*                          num_diag);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_dia_hip(int                                        blocksize,
                                 int                                        nnz,
                                 int                                        nrow,
                                 int                                        ncol,
                                 const MatrixCSR<std::complex<float>, int>& src,
                                 MatrixDIA<std::complex<float>, int>*       dst,
                                 int*                                       nnz_dia,
                                 int*                                       num_diag);

    template bool csr_to_dia_hip(int                                         blocksize,
                                 int                                         nnz,
                                 int                                         nrow,
                                 int                                         ncol,
                                 const MatrixCSR<std::complex<double>, int>& src,
                                 MatrixDIA<std::complex<double>, int>*       dst,
                                 int*                                        nnz_dia,
                                 int*                                        num_diag);
#endif

    // csr_to_hyb
    template bool csr_to_hyb_hip(int                          blocksize,
                                 int                          nnz,
                                 int                          nrow,
                                 int                          ncol,
                                 const MatrixCSR<float, int>& src,
                                 MatrixHYB<float, int>*       dst,
                                 int*                         nnz_hyb,
                                 int*                         nnz_ell,
                                 int*                         nnz_coo);

    template bool csr_to_hyb_hip(int                           blocksize,
                                 int                           nnz,
                                 int                           nrow,
                                 int                           ncol,
                                 const MatrixCSR<double, int>& src,
                                 MatrixHYB<double, int>*       dst,
                                 int*                          nnz_hyb,
                                 int*                          nnz_ell,
                                 int*                          nnz_coo);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_hyb_hip(int                                        blocksize,
                                 int                                        nnz,
                                 int                                        nrow,
                                 int                                        ncol,
                                 const MatrixCSR<std::complex<float>, int>& src,
                                 MatrixHYB<std::complex<float>, int>*       dst,
                                 int*                                       nnz_hyb,
                                 int*                                       nnz_ell,
                                 int*                                       nnz_coo);

    template bool csr_to_hyb_hip(int                                         blocksize,
                                 int                                         nnz,
                                 int                                         nrow,
                                 int                                         ncol,
                                 const MatrixCSR<std::complex<double>, int>& src,
                                 MatrixHYB<std::complex<double>, int>*       dst,
                                 int*                                        nnz_hyb,
                                 int*                                        nnz_ell,
                                 int*                                        nnz_coo);
#endif

    // csr_to_dense
    template bool csr_to_dense_hip(const rocsparse_handle       sparse_handle,
                                   const rocblas_handle         blas_handle,
                                   int                          nrow,
                                   int                          ncol,
                                   const MatrixCSR<float, int>& src,
                                   const rocsparse_mat_descr    src_descr,
                                   MatrixDENSE<float>*          dst);

    template bool csr_to_dense_hip(const rocsparse_handle        sparse_handle,
                                   const rocblas_handle          blas_handle,
                                   int                           nrow,
                                   int                           ncol,
                                   const MatrixCSR<double, int>& src,
                                   const rocsparse_mat_descr     src_descr,
                                   MatrixDENSE<double>*          dst);

#ifdef SUPPORT_COMPLEX
    template bool csr_to_dense_hip(const rocsparse_handle                     sparse_handle,
                                   const rocblas_handle                       blas_handle,
                                   int                                        nrow,
                                   int                                        ncol,
                                   const MatrixCSR<std::complex<float>, int>& src,
                                   const rocsparse_mat_descr                  src_descr,
                                   MatrixDENSE<std::complex<float>>*          dst);


    template bool csr_to_dense_hip(const rocsparse_handle                      sparse_handle,
                                   const rocblas_handle                        blas_handle,
                                   int                                         nrow,
                                   int                                         ncol,
                                   const MatrixCSR<std::complex<double>, int>& src,
                                   const rocsparse_mat_descr                   src_descr,
                                   MatrixDENSE<std::complex<double>>*          dst);
#endif

    // dense_to_csr
    template bool dense_to_csr_hip(const rocsparse_handle    sparse_handle,
                                   const rocblas_handle      blas_handle,
                                   int                       nrow,
                                   int                       ncol,
                                   const MatrixDENSE<float>& src,
                                   MatrixCSR<float, int>*    dst,
                                   const rocsparse_mat_descr dst_descr);

    template bool dense_to_csr_hip(const rocsparse_handle     sparse_handle,
                                   const rocblas_handle       blas_handle,
                                   int                        nrow,
                                   int                        ncol,
                                   const MatrixDENSE<double>& src,
                                   MatrixCSR<double, int>*    dst,
                                   const rocsparse_mat_descr  dst_descr);

#ifdef SUPPORT_COMPLEX
    template bool dense_to_csr_hip(const rocsparse_handle                  sparse_handle,
                                   const rocblas_handle                    blas_handle,
                                   int                                     nrow,
                                   int                                     ncol,
                                   const MatrixDENSE<std::complex<float>>& src,
                                   MatrixCSR<std::complex<float>, int>*    dst,
                                   const rocsparse_mat_descr               dst_descr);

    template bool dense_to_csr_hip(const rocsparse_handle                   sparse_handle,
                                   const rocblas_handle                     blas_handle,
                                   int                                      nrow,
                                   int                                      ncol,
                                   const MatrixDENSE<std::complex<double>>& src,
                                   MatrixCSR<std::complex<double>, int>*    dst,
                                   const rocsparse_mat_descr                dst_descr);
#endif

} // namespace rocalution
