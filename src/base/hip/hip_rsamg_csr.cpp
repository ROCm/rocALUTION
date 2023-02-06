/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../../utils/def.hpp"
#include "hip_matrix_csr.hpp"

#include "hip_allocate_free.hpp"
#include "hip_kernels_rsamg_csr.hpp"
#include "hip_utils.hpp"
#include "hip_vector.hpp"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>

namespace rocalution
{
    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::RSPMISCoarsening(float             eps,
                                                              BaseVector<int>*  CFmap,
                                                              BaseVector<bool>* S) const
    {
        assert(CFmap != NULL);
        assert(S != NULL);

        HIPAcceleratorVector<int>*  cast_cf = dynamic_cast<HIPAcceleratorVector<int>*>(CFmap);
        HIPAcceleratorVector<bool>* cast_S  = dynamic_cast<HIPAcceleratorVector<bool>*>(S);

        assert(cast_cf != NULL);
        assert(cast_S != NULL);

        // Allocate CF mapping
        cast_cf->Clear();
        cast_cf->Allocate(this->nrow_);

        // Allocate S
        // S is the auxiliary strength matrix such that
        //
        // S_ij = { 1   if i != j and -a_ij > eps * max(-a_ik) for k != i
        //        { 0   otherwise
        //
        // This means, S_ij is 1, only if i strongly depends on j.
        // S has been extended to also work with matrices that do not have
        // fully non-positive off-diagonal entries.

        cast_S->Clear();
        cast_S->Allocate(this->nnz_);

        // Initialize S to false (no dependencies)
        cast_S->Zeros();

        // Sample rng
        HIPAcceleratorVector<float> omega(this->local_backend_);
        omega.Allocate(this->nrow_);
        omega.SetRandomUniform(1234ULL, 0, 1);

        dim3 BlockSize(256);
        dim3 GridSize((this->nrow_ * 8 - 1) / 256 + 1);

        // Determine strong influences in the matrix
        kernel_csr_rs_pmis_strong_influences<256, 8>
            <<<GridSize, BlockSize, 0, HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->nrow_,
                this->mat_.row_offset,
                this->mat_.col,
                this->mat_.val,
                eps,
                omega.vec_,
                cast_S->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Mark all vertices as undecided
        cast_cf->Zeros();

        bool* workspace = NULL;
        allocate_hip(this->nrow_ + 1, &workspace);

        // Iteratively find coarse and fine vertices until all undecided vertices have
        // been marked (JPL approach)
        int iter = 0;

        while(true)
        {
            // First, mark all vertices that have not been assigned yet, as coarse
            kernel_csr_rs_pmis_unassigned_to_coarse<<<
                (this->nrow_ - 1) / 256 + 1,
                256,
                0,
                HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->nrow_, omega.vec_, cast_cf->vec_, workspace);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            // Now, correct previously marked vertices with respect to omega
            kernel_csr_rs_pmis_correct_coarse<256, 8>
                <<<GridSize, BlockSize, 0, HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    this->mat_.row_offset,
                    this->mat_.col,
                    omega.vec_,
                    cast_S->vec_,
                    cast_cf->vec_,
                    workspace);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            // Mark remaining edges of a coarse point to fine
            kernel_csr_rs_pmis_coarse_edges_to_fine<256, 8>
                <<<GridSize, BlockSize, 0, HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    this->mat_.row_offset,
                    this->mat_.col,
                    cast_S->vec_,
                    cast_cf->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            // Now, we need to check whether we have vertices left that are marked
            // undecided, in order to restart the loop
            hipMemsetAsync(
                workspace, 0, sizeof(bool), HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            kernel_csr_rs_pmis_check_undecided<1024>
                <<<(this->nrow_ - 1) / 1024 + 1,
                   1024,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_, cast_cf->vec_, workspace);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            bool undecided;

            copy_d2h(1, workspace, &undecided);

            // If no more undecided vertices are left, we are done
            if(undecided == false)
            {
                break;
            }

            ++iter;

            // Print some warning if number of iteration is getting huge
            if(iter > 20)
            {
                LOG_VERBOSE_INFO(2,
                                 "*** warning: HIPAcceleratorMatrixCSR::RSPMISCoarsening() Current "
                                 "number of iterations: "
                                     << iter);
            }
        }

        free_hip(&workspace);

        omega.Clear();

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::RSDirectInterpolation(
        const BaseVector<int>&  CFmap,
        const BaseVector<bool>& S,
        BaseMatrix<ValueType>*  prolong,
        BaseMatrix<ValueType>* restrict) const
    {
        assert(prolong != NULL);

        HIPAcceleratorMatrixCSR<ValueType>* cast_prolong
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(prolong);
        HIPAcceleratorMatrixCSR<ValueType>* cast_restrict
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(restrict);
        const HIPAcceleratorVector<int>* cast_cf
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&CFmap);
        const HIPAcceleratorVector<bool>* cast_S
            = dynamic_cast<const HIPAcceleratorVector<bool>*>(&S);

        assert(cast_prolong != NULL);
        assert(cast_cf != NULL);
        assert(cast_S != NULL);

        // Allocate
        cast_prolong->Clear();

        ValueType* Amin = NULL;
        ValueType* Amax = NULL;

        int* workspace = NULL;

        allocate_hip(this->nrow_, &Amin);
        allocate_hip(this->nrow_, &Amax);
        allocate_hip(this->nrow_ + 1, &workspace);

        // Allocate P row pointer array
        allocate_hip(this->nrow_ + 1, &cast_prolong->mat_.row_offset);

        cast_prolong->nrow_ = this->nrow_;

        dim3 BlockSize(256);
        dim3 GridSize((this->nrow_ - 1) / 256 + 1);

        // Determine nnz per row of P
        kernel_csr_rs_direct_interp_nnz<256>
            <<<GridSize, BlockSize, 0, HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->nrow_,
                this->mat_.row_offset,
                this->mat_.col,
                this->mat_.val,
                cast_S->vec_,
                cast_cf->vec_,
                Amin,
                Amax,
                cast_prolong->mat_.row_offset,
                workspace);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        size_t rocprim_size;
        void*  rocprim_buffer;

        // Exclusive sum to obtain row offset pointers of P
        rocprim::exclusive_scan(NULL,
                                rocprim_size,
                                cast_prolong->mat_.row_offset,
                                cast_prolong->mat_.row_offset,
                                0,
                                this->nrow_ + 1,
                                rocprim::plus<int>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);
        hipMalloc(&rocprim_buffer, rocprim_size);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                cast_prolong->mat_.row_offset,
                                cast_prolong->mat_.row_offset,
                                0,
                                this->nrow_ + 1,
                                rocprim::plus<int>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);
        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                workspace,
                                workspace,
                                0,
                                this->nrow_ + 1,
                                rocprim::plus<int>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);
        hipFree(rocprim_buffer);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Copy ncol and nnz back to host
        copy_d2h(1, cast_prolong->mat_.row_offset + this->nrow_, &cast_prolong->nnz_);
        copy_d2h(1, workspace + this->nrow_, &cast_prolong->ncol_);

        // Allocate column and value arrays
        allocate_hip(cast_prolong->nnz_, &cast_prolong->mat_.col);
        allocate_hip(cast_prolong->nnz_, &cast_prolong->mat_.val);

        // Fill column indices and values of P
        kernel_csr_rs_direct_interp_fill<256>
            <<<GridSize, BlockSize, 0, HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->nrow_,
                this->mat_.row_offset,
                this->mat_.col,
                this->mat_.val,
                cast_prolong->mat_.row_offset,
                cast_prolong->mat_.col,
                cast_prolong->mat_.val,
                cast_S->vec_,
                cast_cf->vec_,
                Amin,
                Amax,
                workspace);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Free temporary buffers
        free_hip(&Amin);
        free_hip(&Amax);
        free_hip(&workspace);

        // Transpose P to obtain R
        if(cast_restrict != NULL)
        {
            cast_prolong->Transpose(cast_restrict);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::RSExtPIInterpolation(
        const BaseVector<int>&  CFmap,
        const BaseVector<bool>& S,
        bool                    FF1,
        float                   trunc,
        BaseMatrix<ValueType>*  prolong,
        BaseMatrix<ValueType>* restrict) const
    {
        assert(trunc >= 0.0f);
        assert(prolong != NULL);

        HIPAcceleratorMatrixCSR<ValueType>* cast_prolong
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(prolong);
        HIPAcceleratorMatrixCSR<ValueType>* cast_restrict
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(restrict);

        const HIPAcceleratorVector<int>* cast_cf
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&CFmap);
        const HIPAcceleratorVector<bool>* cast_S
            = dynamic_cast<const HIPAcceleratorVector<bool>*>(&S);

        assert(cast_prolong != NULL);
        assert(cast_cf != NULL);
        assert(cast_S != NULL);

        // Start with fresh operators
        cast_prolong->Clear();

        // Allocate P row pointer array
        allocate_hip(this->nrow_ + 1, &cast_prolong->mat_.row_offset);

        // We already know the number of rows of P
        cast_prolong->nrow_ = this->nrow_;

        // Temporary buffer
        int* workspace = NULL;
        allocate_hip(this->nrow_ + 1, &workspace);

#define BLOCKSIZE 256
        // Obtain maximum row nnz
        kernel_csr_rs_extpi_interp_max<BLOCKSIZE, 16>
            <<<(this->nrow_ * 16 - 1) / BLOCKSIZE + 1,
               BLOCKSIZE,
               0,
               HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                     FF1,
                                                                     this->mat_.row_offset,
                                                                     this->mat_.col,
                                                                     cast_S->vec_,
                                                                     cast_cf->vec_,
                                                                     cast_prolong->mat_.row_offset);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        size_t rocprim_size;
        void*  rocprim_buffer;

        // Determine maximum row nnz
        rocprim::reduce(NULL,
                        rocprim_size,
                        cast_prolong->mat_.row_offset,
                        cast_prolong->mat_.row_offset + this->nrow_,
                        0,
                        this->nrow_,
                        rocprim::maximum<int>(),
                        HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);
        hipMalloc(&rocprim_buffer, rocprim_size);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
        rocprim::reduce(rocprim_buffer,
                        rocprim_size,
                        cast_prolong->mat_.row_offset,
                        cast_prolong->mat_.row_offset + this->nrow_,
                        0,
                        this->nrow_,
                        rocprim::maximum<int>(),
                        HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        int max_nnz;
        copy_d2h(1, cast_prolong->mat_.row_offset + this->nrow_, &max_nnz);

        // Determine nnz per row of P
        if(max_nnz < 16)
        {
            kernel_csr_rs_extpi_interp_nnz<BLOCKSIZE, 8, 16>
                <<<(this->nrow_ * 8 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    cast_S->vec_,
                    cast_cf->vec_,
                    cast_prolong->mat_.row_offset,
                    workspace);
        }
        else if(max_nnz < 32)
        {
            kernel_csr_rs_extpi_interp_nnz<BLOCKSIZE, 16, 32>
                <<<(this->nrow_ * 16 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    cast_S->vec_,
                    cast_cf->vec_,
                    cast_prolong->mat_.row_offset,
                    workspace);
        }
        else if(max_nnz < 64)
        {
            kernel_csr_rs_extpi_interp_nnz<BLOCKSIZE, 32, 64>
                <<<(this->nrow_ * 32 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    cast_S->vec_,
                    cast_cf->vec_,
                    cast_prolong->mat_.row_offset,
                    workspace);
        }
        else if(max_nnz < 128)
        {
            kernel_csr_rs_extpi_interp_nnz<BLOCKSIZE, 64, 128>
                <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    cast_S->vec_,
                    cast_cf->vec_,
                    cast_prolong->mat_.row_offset,
                    workspace);
        }
        else if(max_nnz < 256)
        {
            kernel_csr_rs_extpi_interp_nnz<BLOCKSIZE, 64, 256>
                <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    cast_S->vec_,
                    cast_cf->vec_,
                    cast_prolong->mat_.row_offset,
                    workspace);
        }
        else if(max_nnz < 512)
        {
            kernel_csr_rs_extpi_interp_nnz<BLOCKSIZE, 64, 512>
                <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    cast_S->vec_,
                    cast_cf->vec_,
                    cast_prolong->mat_.row_offset,
                    workspace);
        }
        else if(max_nnz < 1024)
        {
            kernel_csr_rs_extpi_interp_nnz<BLOCKSIZE, 64, 1024>
                <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    cast_S->vec_,
                    cast_cf->vec_,
                    cast_prolong->mat_.row_offset,
                    workspace);
        }
        else if(max_nnz < 2048)
        {
            kernel_csr_rs_extpi_interp_nnz<BLOCKSIZE, 64, 2048>
                <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    cast_S->vec_,
                    cast_cf->vec_,
                    cast_prolong->mat_.row_offset,
                    workspace);
        }
        else if(max_nnz < 4096)
        {
            kernel_csr_rs_extpi_interp_nnz<BLOCKSIZE, 64, 4096>
                <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    cast_S->vec_,
                    cast_cf->vec_,
                    cast_prolong->mat_.row_offset,
                    workspace);
        }
        else
        {
            // More nnz per row will not fit into LDS
            // Fall back to host

            // Free temporary buffer
            free_hip(&workspace);
            hipFree(rocprim_buffer);

            return false;
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Determine maximum hash table fill
        rocprim::reduce(rocprim_buffer,
                        rocprim_size,
                        cast_prolong->mat_.row_offset,
                        cast_prolong->mat_.row_offset + this->nrow_,
                        0,
                        this->nrow_,
                        rocprim::maximum<int>(),
                        HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);
        hipFree(rocprim_buffer);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        int max_hash_fill;
        copy_d2h(1, cast_prolong->mat_.row_offset + this->nrow_, &max_hash_fill);

        // Exclusive sum to obtain row offset pointers of P
        rocprim::exclusive_scan(NULL,
                                rocprim_size,
                                cast_prolong->mat_.row_offset,
                                cast_prolong->mat_.row_offset,
                                0,
                                this->nrow_ + 1,
                                rocprim::plus<int>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);
        hipMalloc(&rocprim_buffer, rocprim_size);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                cast_prolong->mat_.row_offset,
                                cast_prolong->mat_.row_offset,
                                0,
                                this->nrow_ + 1,
                                rocprim::plus<int>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);
        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                workspace,
                                workspace,
                                0,
                                this->nrow_ + 1,
                                rocprim::plus<int>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);
        hipFree(rocprim_buffer);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Copy number of columns of P and nnz of P back to host
        copy_d2h(1, cast_prolong->mat_.row_offset + this->nrow_, &cast_prolong->nnz_);
        copy_d2h(1, workspace + this->nrow_, &cast_prolong->ncol_);

        // Allocate column and value arrays
        allocate_hip(cast_prolong->nnz_, &cast_prolong->mat_.col);
        allocate_hip(cast_prolong->nnz_, &cast_prolong->mat_.val);

        // Extract diagonal matrix entries
        HIPAcceleratorVector<ValueType> diag(this->local_backend_);
        diag.Allocate(this->nrow_);

        this->ExtractDiagonal(&diag);

        // Fill column indices and values of P

        if(max_hash_fill < 16)
        {
            size_t ssize = BLOCKSIZE / 8 * 16 * (sizeof(int) + sizeof(ValueType));
            kernel_csr_rs_extpi_interp_fill<BLOCKSIZE, 8, 16>
                <<<(this->nrow_ * 8 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   ssize,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    diag.vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val,
                    cast_S->vec_,
                    cast_cf->vec_,
                    workspace);
        }
        else if(max_hash_fill < 32)
        {
            size_t ssize = BLOCKSIZE / 16 * 32 * (sizeof(int) + sizeof(ValueType));
            kernel_csr_rs_extpi_interp_fill<BLOCKSIZE, 16, 32>
                <<<(this->nrow_ * 16 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   ssize,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    diag.vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val,
                    cast_S->vec_,
                    cast_cf->vec_,
                    workspace);
        }
        else if(max_hash_fill < 64)
        {
            size_t ssize = BLOCKSIZE / 32 * 64 * (sizeof(int) + sizeof(ValueType));
            kernel_csr_rs_extpi_interp_fill<BLOCKSIZE, 32, 64>
                <<<(this->nrow_ * 32 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   ssize,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    diag.vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val,
                    cast_S->vec_,
                    cast_cf->vec_,
                    workspace);
        }
        else if(max_hash_fill < 128)
        {
            size_t ssize = BLOCKSIZE / 64 * 128 * (sizeof(int) + sizeof(ValueType));
            kernel_csr_rs_extpi_interp_fill<BLOCKSIZE, 64, 128>
                <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   ssize,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    diag.vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val,
                    cast_S->vec_,
                    cast_cf->vec_,
                    workspace);
        }
        else if(max_hash_fill < 256)
        {
            size_t ssize = BLOCKSIZE / 64 * 256 * (sizeof(int) + sizeof(ValueType));
            kernel_csr_rs_extpi_interp_fill<BLOCKSIZE, 64, 256>
                <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   ssize,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    diag.vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val,
                    cast_S->vec_,
                    cast_cf->vec_,
                    workspace);
        }
        else if(max_hash_fill < 512)
        {
            size_t ssize = BLOCKSIZE / 64 * 512 * (sizeof(int) + sizeof(ValueType));
            kernel_csr_rs_extpi_interp_fill<BLOCKSIZE, 64, 512>
                <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   ssize,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    FF1,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    diag.vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val,
                    cast_S->vec_,
                    cast_cf->vec_,
                    workspace);
        }
        else
        {
            // More nnz per row will not fit into LDS
            // Fall back to host

            // Free temporary buffer
            free_hip(&workspace);

            return false;
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);
#undef BLOCKSIZE

        // Free temporary buffer
        free_hip(&workspace);

        // Apply dropping strategy, if enabled
        if(trunc > 0.0f)
        {
            // Allocate structures for "compressed" P
            int compressed_nrow = cast_prolong->nrow_;
            int compressed_ncol = cast_prolong->ncol_;

            int*       compressed_csr_row_ptr = NULL;
            int*       compressed_csr_col_ind = NULL;
            ValueType* compressed_csr_val     = NULL;

            allocate_hip(compressed_nrow + 1, &compressed_csr_row_ptr);

            kernel_csr_rs_extpi_interp_compress_nnz<256, 8>
                <<<(cast_prolong->nrow_ * 8 - 1) / 256 + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    cast_prolong->nrow_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val,
                    trunc,
                    compressed_csr_row_ptr);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            rocprim::exclusive_scan(NULL,
                                    rocprim_size,
                                    compressed_csr_row_ptr,
                                    compressed_csr_row_ptr,
                                    0,
                                    compressed_nrow + 1,
                                    rocprim::plus<int>(),
                                    HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);
            hipMalloc(&rocprim_buffer, rocprim_size);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
            rocprim::exclusive_scan(rocprim_buffer,
                                    rocprim_size,
                                    compressed_csr_row_ptr,
                                    compressed_csr_row_ptr,
                                    0,
                                    compressed_nrow + 1,
                                    rocprim::plus<int>(),
                                    HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);
            hipFree(rocprim_buffer);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            // Get the new compressed nnz
            int compressed_nnz;
            copy_d2h(1, compressed_csr_row_ptr + compressed_nrow, &compressed_nnz);

            // Allocate structures for "compressed" P
            allocate_hip(compressed_nnz, &compressed_csr_col_ind);
            allocate_hip(compressed_nnz, &compressed_csr_val);

            // Copy column and value entries
            kernel_csr_rs_extpi_interp_compress_fill<<<
                (cast_prolong->nrow_ - 1) / 256 + 1,
                256,
                0,
                HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(cast_prolong->nrow_,
                                                                      cast_prolong->mat_.row_offset,
                                                                      cast_prolong->mat_.col,
                                                                      cast_prolong->mat_.val,
                                                                      trunc,
                                                                      compressed_csr_row_ptr,
                                                                      compressed_csr_col_ind,
                                                                      compressed_csr_val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            // Update P
            cast_prolong->Clear();
            cast_prolong->SetDataPtrCSR(&compressed_csr_row_ptr,
                                        &compressed_csr_col_ind,
                                        &compressed_csr_val,
                                        compressed_nnz,
                                        compressed_nrow,
                                        compressed_ncol);
        }

        // Transpose P to obtain R
        if(cast_restrict != NULL)
        {
            cast_prolong->Transpose(cast_restrict);
        }

        return true;
    }

    template class HIPAcceleratorMatrixCSR<double>;
    template class HIPAcceleratorMatrixCSR<float>;
#ifdef SUPPORT_COMPLEX
    template class HIPAcceleratorMatrixCSR<std::complex<double>>;
    template class HIPAcceleratorMatrixCSR<std::complex<float>>;
#endif

} // namespace rocalution
