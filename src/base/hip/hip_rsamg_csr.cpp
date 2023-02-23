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
    bool HIPAcceleratorMatrixCSR<ValueType>::RSPMISStrongInfluences(
        float                        eps,
        BaseVector<bool>*            S,
        BaseVector<float>*           omega,
        unsigned long long           seed,
        const BaseMatrix<ValueType>& ghost) const
    {
        assert(S != NULL);
        assert(omega != NULL);

        HIPAcceleratorVector<bool>*  cast_S = dynamic_cast<HIPAcceleratorVector<bool>*>(S);
        HIPAcceleratorVector<float>* cast_w = dynamic_cast<HIPAcceleratorVector<float>*>(omega);
        const HIPAcceleratorMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&ghost);

        assert(cast_S != NULL);
        assert(cast_w != NULL);
        assert(cast_gst != NULL);

        // Do we need communication?
        bool global = cast_gst->nrow_ > 0;

        // Initialize S to false (no dependencies)
        cast_S->Zeros();

        // Sample rng
        omega->SetRandomUniform(seed, 0, 1); // TODO only fill up to this->nrow_!!!

        // Determine strong influences in the matrix
        if(global == false)
        {
            kernel_csr_rs_pmis_strong_influences<false, 256, 8>
                <<<(this->nrow_ * 8 - 1) / 256 + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->nnz_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         this->mat_.val,
                                                                         eps,
                                                                         cast_w->vec_,
                                                                         cast_S->vec_);
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::RSPMISUnassignedToCoarse(
        BaseVector<int>* CFmap, BaseVector<bool>* marked, const BaseVector<float>& omega) const
    {
        assert(CFmap != NULL);
        assert(marked != NULL);

        HIPAcceleratorVector<int>*  cast_cf = dynamic_cast<HIPAcceleratorVector<int>*>(CFmap);
        HIPAcceleratorVector<bool>* cast_m  = dynamic_cast<HIPAcceleratorVector<bool>*>(marked);
        const HIPAcceleratorVector<float>* cast_w
            = dynamic_cast<const HIPAcceleratorVector<float>*>(&omega);

        assert(cast_cf != NULL);
        assert(cast_m != NULL);
        assert(cast_w != NULL);

        // First, mark all vertices that have not been assigned yet, as coarse
        kernel_csr_rs_pmis_unassigned_to_coarse<<<(cast_cf->size_ - 1) / 256 + 1,
                                                  256,
                                                  0,
                                                  HIPSTREAM(
                                                      this->local_backend_.HIP_stream_current)>>>(
            cast_cf->size_, cast_w->vec_, cast_cf->vec_, cast_m->vec_);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::RSPMISCorrectCoarse(
        BaseVector<int>*             CFmap,
        const BaseVector<bool>&      S,
        const BaseVector<bool>&      marked,
        const BaseVector<float>&     omega,
        const BaseMatrix<ValueType>& ghost) const
    {
        assert(CFmap != NULL);

        HIPAcceleratorVector<int>*        cast_cf = dynamic_cast<HIPAcceleratorVector<int>*>(CFmap);
        const HIPAcceleratorVector<bool>* cast_S
            = dynamic_cast<const HIPAcceleratorVector<bool>*>(&S);
        const HIPAcceleratorVector<bool>* cast_m
            = dynamic_cast<const HIPAcceleratorVector<bool>*>(&marked);
        const HIPAcceleratorVector<float>* cast_w
            = dynamic_cast<const HIPAcceleratorVector<float>*>(&omega);
        const HIPAcceleratorMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&ghost);

        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_m != NULL);
        assert(cast_w != NULL);
        assert(cast_gst != NULL);

        // Do we need communication?
        bool global = cast_gst->nrow_ > 0;

        // Now, correct previously marked vertices with respect to omega
        if(global == false)
        {
            kernel_csr_rs_pmis_correct_coarse<false, 256, 8>
                <<<(this->nrow_ * 8 - 1) / 256 + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->nnz_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_w->vec_,
                                                                         cast_S->vec_,
                                                                         cast_cf->vec_,
                                                                         cast_m->vec_);
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::RSPMISCoarseEdgesToFine(
        BaseVector<int>* CFmap, const BaseVector<bool>& S, const BaseMatrix<ValueType>& ghost) const
    {
        assert(CFmap != NULL);

        HIPAcceleratorVector<int>*        cast_cf = dynamic_cast<HIPAcceleratorVector<int>*>(CFmap);
        const HIPAcceleratorVector<bool>* cast_S
            = dynamic_cast<const HIPAcceleratorVector<bool>*>(&S);
        const HIPAcceleratorMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&ghost);

        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_gst != NULL);

        // Do we need communication?
        bool global = cast_gst->nrow_ > 0;

        // Mark remaining edges of a coarse point to fine
        if(global == false)
        {
            kernel_csr_rs_pmis_coarse_edges_to_fine<false, 256, 8>
                <<<(this->nrow_ * 8 - 1) / 256 + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->nnz_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_S->vec_,
                                                                         cast_cf->vec_);
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        return true;
    }

    template <typename ValueType>
    bool
        HIPAcceleratorMatrixCSR<ValueType>::RSPMISCheckUndecided(bool&                  undecided,
                                                                 const BaseVector<int>& CFmap) const
    {
        const HIPAcceleratorVector<int>* cast_cf
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&CFmap);

        assert(cast_cf != NULL);

        bool* d_undecided = NULL;
        allocate_hip(1, &d_undecided);
        set_to_zero_hip(this->local_backend_.HIP_block_size,
                        1,
                        d_undecided,
                        true,
                        HIPSTREAM(this->local_backend_.HIP_stream_current));

        kernel_csr_rs_pmis_check_undecided<256>
            <<<(this->nrow_ - 1) / 256 + 1,
               256,
               0,
               HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->nrow_, cast_cf->vec_, d_undecided);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        copy_d2h(1, d_undecided, &undecided);
        free_hip(&d_undecided);

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::RSDirectProlongNnz(
        const BaseVector<int>&       CFmap,
        const BaseVector<bool>&      S,
        const BaseMatrix<ValueType>& ghost,
        BaseVector<ValueType>*       Amin,
        BaseVector<ValueType>*       Amax,
        BaseVector<int>*             f2c,
        BaseMatrix<ValueType>*       prolong_int,
        BaseMatrix<ValueType>*       prolong_gst) const
    {
        const HIPAcceleratorVector<int>* cast_cf
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&CFmap);
        const HIPAcceleratorVector<bool>* cast_S
            = dynamic_cast<const HIPAcceleratorVector<bool>*>(&S);
        const HIPAcceleratorMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&ghost);
        HIPAcceleratorVector<ValueType>* cast_Amin
            = dynamic_cast<HIPAcceleratorVector<ValueType>*>(Amin);
        HIPAcceleratorVector<ValueType>* cast_Amax
            = dynamic_cast<HIPAcceleratorVector<ValueType>*>(Amax);
        HIPAcceleratorVector<int>* cast_f2c = dynamic_cast<HIPAcceleratorVector<int>*>(f2c);
        HIPAcceleratorMatrixCSR<ValueType>* cast_pi
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(prolong_int);
        HIPAcceleratorMatrixCSR<ValueType>* cast_pg
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(prolong_gst);

        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_f2c != NULL);
        assert(cast_pi != NULL);
        assert(cast_Amin != NULL);
        assert(cast_Amax != NULL);
        assert(cast_Amin->size_ == this->nrow_);
        assert(cast_Amax->size_ == this->nrow_);

        // Do we need communication?
        bool global = prolong_gst != NULL;

        // Start with fresh P
        cast_pi->Clear();

        // Allocate P row pointer array
        allocate_hip(this->nrow_ + 1, &cast_pi->mat_.row_offset);

        // We already know the number of rows of P
        cast_pi->nrow_ = this->nrow_;

        // Determine row nnz of P
        if(global == false)
        {
            kernel_csr_rs_direct_interp_nnz<false, 256>
                <<<(this->nrow_ - 1) / 256 + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->nnz_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         this->mat_.val,
                                                                         cast_S->vec_,
                                                                         cast_cf->vec_,
                                                                         cast_Amin->vec_,
                                                                         cast_Amax->vec_,
                                                                         cast_pi->mat_.row_offset,
                                                                         cast_f2c->vec_);
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        cast_f2c->ExclusiveSum(*cast_f2c);

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::RSDirectProlongFill(
        const BaseVector<int64_t>&   l2g,
        const BaseVector<int>&       f2c,
        const BaseVector<int>&       CFmap,
        const BaseVector<bool>&      S,
        const BaseMatrix<ValueType>& ghost,
        const BaseVector<ValueType>& Amin,
        const BaseVector<ValueType>& Amax,
        BaseMatrix<ValueType>*       prolong_int,
        BaseMatrix<ValueType>*       prolong_gst,
        BaseVector<int64_t>*         global_ghost_col) const
    {
        const HIPAcceleratorVector<int64_t>* cast_l2g
            = dynamic_cast<const HIPAcceleratorVector<int64_t>*>(&l2g);
        const HIPAcceleratorVector<int>* cast_f2c
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&f2c);
        const HIPAcceleratorVector<int>* cast_cf
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&CFmap);
        const HIPAcceleratorVector<bool>* cast_S
            = dynamic_cast<const HIPAcceleratorVector<bool>*>(&S);
        const HIPAcceleratorMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&ghost);
        const HIPAcceleratorVector<ValueType>* cast_Amin
            = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&Amin);
        const HIPAcceleratorVector<ValueType>* cast_Amax
            = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&Amax);
        HIPAcceleratorMatrixCSR<ValueType>* cast_pi
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(prolong_int);
        HIPAcceleratorMatrixCSR<ValueType>* cast_pg
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(prolong_gst);
        HIPAcceleratorVector<int64_t>* cast_glo
            = dynamic_cast<HIPAcceleratorVector<int64_t>*>(global_ghost_col);

        assert(cast_f2c != NULL);
        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_pi != NULL);
        assert(cast_Amin != NULL);
        assert(cast_Amax != NULL);
        assert(cast_Amin->size_ == this->nrow_);
        assert(cast_Amax->size_ == this->nrow_);

        // Do we need communication?
        bool global = prolong_gst != NULL;

        // rocprim buffer
        size_t rocprim_size;
        char*  rocprim_buffer = NULL;

        // Exclusive sum to obtain row offset pointers of P
        // P contains only nnz per row, so far
        rocprim::exclusive_scan(NULL,
                                rocprim_size,
                                cast_pi->mat_.row_offset,
                                cast_pi->mat_.row_offset,
                                0,
                                this->nrow_ + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                cast_pi->mat_.row_offset,
                                cast_pi->mat_.row_offset,
                                0,
                                this->nrow_ + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Initialize nnz of P
        PtrType tmp;
        copy_d2h(1, cast_pi->mat_.row_offset + this->nrow_, &tmp);
        cast_pi->nnz_ = tmp;

        // Initialize ncol of P
        int ncol32;
        copy_d2h(1, cast_f2c->vec_ + this->nrow_, &ncol32);
        cast_pi->ncol_ = ncol32;

        // Allocate column and value arrays
        allocate_hip(cast_pi->nnz_, &cast_pi->mat_.col);
        allocate_hip(cast_pi->nnz_, &cast_pi->mat_.val);

        free_hip(&rocprim_buffer);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Fill column indices and values of P
        if(global == false)
        {
            kernel_csr_rs_direct_interp_fill<false, 256>
                <<<(this->nrow_ - 1) / 256 + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->nnz_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         this->mat_.val,
                                                                         cast_pi->mat_.row_offset,
                                                                         cast_pi->mat_.col,
                                                                         cast_pi->mat_.val,
                                                                         cast_S->vec_,
                                                                         cast_cf->vec_,
                                                                         cast_Amin->vec_,
                                                                         cast_Amax->vec_,
                                                                         cast_f2c->vec_);
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::RSExtPIProlongNnz(
        int64_t                      global_column_begin,
        int64_t                      global_column_end,
        bool                         FF1,
        const BaseVector<int64_t>&   l2g,
        const BaseVector<int>&       CFmap,
        const BaseVector<bool>&      S,
        const BaseMatrix<ValueType>& ghost,
        const BaseVector<PtrType>&   bnd_csr_row_ptr,
        const BaseVector<int64_t>&   bnd_csr_col_ind,
        BaseVector<int>*             f2c,
        BaseMatrix<ValueType>*       prolong_int,
        BaseMatrix<ValueType>*       prolong_gst) const
    {
        const HIPAcceleratorVector<int64_t>* cast_l2g
            = dynamic_cast<const HIPAcceleratorVector<int64_t>*>(&l2g);
        const HIPAcceleratorVector<int>* cast_cf
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&CFmap);
        const HIPAcceleratorVector<bool>* cast_S
            = dynamic_cast<const HIPAcceleratorVector<bool>*>(&S);
        const HIPAcceleratorMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&ghost);
        const HIPAcceleratorVector<int>* cast_ptr
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&bnd_csr_row_ptr);
        const HIPAcceleratorVector<int64_t>* cast_col
            = dynamic_cast<const HIPAcceleratorVector<int64_t>*>(&bnd_csr_col_ind);
        HIPAcceleratorVector<int>* cast_f2c = dynamic_cast<HIPAcceleratorVector<int>*>(f2c);
        HIPAcceleratorMatrixCSR<ValueType>* cast_pi
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(prolong_int);
        HIPAcceleratorMatrixCSR<ValueType>* cast_pg
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(prolong_gst);

        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_f2c != NULL);
        assert(cast_pi != NULL);

        // Do we need communication?
        bool global = prolong_gst != NULL;

        // Start with fresh P
        cast_pi->Clear();

        // Allocate P row pointer array
        allocate_hip(this->nrow_ + 1, &cast_pi->mat_.row_offset);

        // We already know the number of rows of P
        cast_pi->nrow_ = this->nrow_;

        // rocprim buffer
        size_t rocprim_size;
        char*  rocprim_buffer = NULL;

#define BLOCKSIZE 256
        // Determine max row nnz

        if(global == false)
        {
            kernel_csr_rs_extpi_interp_max<false, BLOCKSIZE, 16>
                <<<(this->nrow_ * 16 - 1) / BLOCKSIZE + 1,
                   BLOCKSIZE,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->nnz_,
                                                                         FF1,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_S->vec_,
                                                                         cast_cf->vec_,
                                                                         cast_pi->mat_.row_offset);
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Obtain maximum row nnz
        rocprim::reduce(NULL,
                        rocprim_size,
                        cast_pi->mat_.row_offset,
                        cast_pi->mat_.row_offset + this->nrow_,
                        0,
                        this->nrow_,
                        rocprim::maximum<PtrType>(),
                        HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        rocprim::reduce(rocprim_buffer,
                        rocprim_size,
                        cast_pi->mat_.row_offset,
                        cast_pi->mat_.row_offset + this->nrow_,
                        0,
                        this->nrow_,
                        rocprim::maximum<PtrType>(),
                        HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&rocprim_buffer);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        PtrType max_nnz;
        copy_d2h(1, cast_pi->mat_.row_offset + this->nrow_, &max_nnz);

        // Determine nnz per row of P

        if(global == false)
        {
            if(max_nnz < 16)
            {
                kernel_csr_rs_extpi_interp_nnz<false, BLOCKSIZE, 8, 16>
                    <<<(this->nrow_ * 8 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_pi->mat_.row_offset,
                        cast_f2c->vec_);
            }
            else if(max_nnz < 32)
            {
                kernel_csr_rs_extpi_interp_nnz<false, BLOCKSIZE, 16, 32>
                    <<<(this->nrow_ * 16 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_pi->mat_.row_offset,
                        cast_f2c->vec_);
            }
            else if(max_nnz < 64)
            {
                kernel_csr_rs_extpi_interp_nnz<false, BLOCKSIZE, 32, 64>
                    <<<(this->nrow_ * 32 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_pi->mat_.row_offset,
                        cast_f2c->vec_);
            }
            else if(max_nnz < 128)
            {
                kernel_csr_rs_extpi_interp_nnz<false, BLOCKSIZE, 64, 128>
                    <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_pi->mat_.row_offset,
                        cast_f2c->vec_);
            }
            else if(max_nnz < 256)
            {
                kernel_csr_rs_extpi_interp_nnz<false, BLOCKSIZE, 64, 256>
                    <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_pi->mat_.row_offset,
                        cast_f2c->vec_);
            }
            else if(max_nnz < 512)
            {
                kernel_csr_rs_extpi_interp_nnz<false, BLOCKSIZE, 64, 512>
                    <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_pi->mat_.row_offset,
                        cast_f2c->vec_);
            }
            else if(max_nnz < 1024)
            {
                kernel_csr_rs_extpi_interp_nnz<false, BLOCKSIZE, 64, 1024>
                    <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_pi->mat_.row_offset,
                        cast_f2c->vec_);
            }
            else if(max_nnz < 2048)
            {
                kernel_csr_rs_extpi_interp_nnz<false, BLOCKSIZE, 64, 2048>
                    <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_pi->mat_.row_offset,
                        cast_f2c->vec_);
            }
            else if(max_nnz < 4096)
            {
                kernel_csr_rs_extpi_interp_nnz<false, 128, 64, 4096>
                    <<<(this->nrow_ * 64 - 1) / 128 + 1,
                       128,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_pi->mat_.row_offset,
                        cast_f2c->vec_);
            }
            else if(max_nnz < 8192)
            {
                kernel_csr_rs_extpi_interp_nnz<false, 64, 64, 8192>
                    <<<(this->nrow_ * 64 - 1) / 64 + 1,
                       64,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_pi->mat_.row_offset,
                        cast_f2c->vec_);
            }
            else
            {
                // More nnz per row will not fit into LDS
                // Fall back to host

                free_hip(&cast_pi->mat_.row_offset);
                cast_pi->nrow_ = 0;

                return false;
            }
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);
#undef BLOCKSIZE

        cast_f2c->ExclusiveSum(*cast_f2c);

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::RSExtPIProlongFill(
        int64_t                      global_column_begin,
        int64_t                      global_column_end,
        bool                         FF1,
        const BaseVector<int64_t>&   l2g,
        const BaseVector<int>&       f2c,
        const BaseVector<int>&       CFmap,
        const BaseVector<bool>&      S,
        const BaseMatrix<ValueType>& ghost,
        const BaseVector<PtrType>&   bnd_csr_row_ptr,
        const BaseVector<int64_t>&   bnd_csr_col_ind,
        const BaseVector<PtrType>&   ext_csr_row_ptr,
        const BaseVector<int64_t>&   ext_csr_col_ind,
        const BaseVector<ValueType>& ext_csr_val,
        BaseMatrix<ValueType>*       prolong_int,
        BaseMatrix<ValueType>*       prolong_gst,
        BaseVector<int64_t>*         global_ghost_col) const
    {
        const HIPAcceleratorVector<int64_t>* cast_l2g
            = dynamic_cast<const HIPAcceleratorVector<int64_t>*>(&l2g);
        const HIPAcceleratorVector<int>* cast_f2c
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&f2c);
        const HIPAcceleratorVector<int>* cast_cf
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&CFmap);
        const HIPAcceleratorVector<bool>* cast_S
            = dynamic_cast<const HIPAcceleratorVector<bool>*>(&S);
        const HIPAcceleratorMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&ghost);
        const HIPAcceleratorVector<int>* cast_ptr
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&bnd_csr_row_ptr);
        const HIPAcceleratorVector<int64_t>* cast_col
            = dynamic_cast<const HIPAcceleratorVector<int64_t>*>(&bnd_csr_col_ind);
        const HIPAcceleratorVector<int>* cast_ext_ptr
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&ext_csr_row_ptr);
        const HIPAcceleratorVector<int64_t>* cast_ext_col
            = dynamic_cast<const HIPAcceleratorVector<int64_t>*>(&ext_csr_col_ind);
        const HIPAcceleratorVector<ValueType>* cast_ext_val
            = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&ext_csr_val);
        HIPAcceleratorMatrixCSR<ValueType>* cast_pi
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(prolong_int);
        HIPAcceleratorMatrixCSR<ValueType>* cast_pg
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(prolong_gst);
        HIPAcceleratorVector<int64_t>* cast_glo
            = dynamic_cast<HIPAcceleratorVector<int64_t>*>(global_ghost_col);

        assert(cast_f2c != NULL);
        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_pi != NULL);

        // Do we need communication?
        bool global = prolong_gst != NULL;

        // rocprim buffer
        size_t rocprim_size;
        char*  rocprim_buffer = NULL;

        // Determine maximum hash table fill
        int* d_max_hash = NULL;
        allocate_hip(2, &d_max_hash);

        rocprim::reduce(NULL,
                        rocprim_size,
                        cast_pi->mat_.row_offset,
                        d_max_hash,
                        0,
                        this->nrow_,
                        rocprim::maximum<int>(),
                        HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);

        rocprim::reduce(rocprim_buffer,
                        rocprim_size,
                        cast_pi->mat_.row_offset,
                        d_max_hash,
                        0,
                        this->nrow_,
                        rocprim::maximum<int>(),
                        HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Maximum fill
        int max_hash_fill;
        copy_d2h(1, d_max_hash, &max_hash_fill);

        free_hip(&d_max_hash);
        free_hip(&rocprim_buffer);

        // Exclusive sum to obtain row offset pointers of P
        // P contains only nnz per row, so far
        rocprim::exclusive_scan(NULL,
                                rocprim_size,
                                cast_pi->mat_.row_offset,
                                cast_pi->mat_.row_offset,
                                0,
                                this->nrow_ + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        rocprim_buffer = NULL;
        allocate_hip(rocprim_size, &rocprim_buffer);

        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                cast_pi->mat_.row_offset,
                                cast_pi->mat_.row_offset,
                                0,
                                this->nrow_ + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Initialize nnz of P
        PtrType tmp;
        copy_d2h(1, cast_pi->mat_.row_offset + this->nrow_, &tmp);
        cast_pi->nnz_ = tmp;

        // Initialize ncol of P
        int ncol32;
        copy_d2h(1, cast_f2c->vec_ + this->nrow_, &ncol32);
        cast_pi->ncol_ = ncol32;

        // Allocate column and value arrays
        allocate_hip(cast_pi->nnz_, &cast_pi->mat_.col);
        allocate_hip(cast_pi->nnz_, &cast_pi->mat_.val);

        free_hip(&rocprim_buffer);

        // Extract diagonal matrix entries
        HIPAcceleratorVector<ValueType> diag(this->local_backend_);
        diag.Allocate(this->nrow_);

        this->ExtractDiagonal(&diag);

#define BLOCKSIZE 256
        // Fill column indices and values of P

        if(global == false)
        {
            if(max_hash_fill < 16)
            {
                size_t ssize = BLOCKSIZE / 8 * 16 * (sizeof(int) + sizeof(ValueType));
                kernel_csr_rs_extpi_interp_fill<false, BLOCKSIZE, 8, 16>
                    <<<(this->nrow_ * 8 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       ssize,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->ncol_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        diag.vec_,
                        cast_pi->mat_.row_offset,
                        cast_pi->mat_.col,
                        cast_pi->mat_.val,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_f2c->vec_);
            }
            else if(max_hash_fill < 32)
            {
                size_t ssize = BLOCKSIZE / 16 * 32 * (sizeof(int) + sizeof(ValueType));
                kernel_csr_rs_extpi_interp_fill<false, BLOCKSIZE, 16, 32>
                    <<<(this->nrow_ * 16 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       ssize,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->ncol_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        diag.vec_,
                        cast_pi->mat_.row_offset,
                        cast_pi->mat_.col,
                        cast_pi->mat_.val,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_f2c->vec_);
            }
            else if(max_hash_fill < 64)
            {
                size_t ssize = BLOCKSIZE / 32 * 64 * (sizeof(int) + sizeof(ValueType));
                kernel_csr_rs_extpi_interp_fill<false, BLOCKSIZE, 32, 64>
                    <<<(this->nrow_ * 32 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       ssize,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->ncol_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        diag.vec_,
                        cast_pi->mat_.row_offset,
                        cast_pi->mat_.col,
                        cast_pi->mat_.val,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_f2c->vec_);
            }
            else if(max_hash_fill < 128)
            {
                size_t ssize = BLOCKSIZE / 64 * 128 * (sizeof(int) + sizeof(ValueType));
                kernel_csr_rs_extpi_interp_fill<false, BLOCKSIZE, 64, 128>
                    <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       ssize,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->ncol_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        diag.vec_,
                        cast_pi->mat_.row_offset,
                        cast_pi->mat_.col,
                        cast_pi->mat_.val,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_f2c->vec_);
            }
            else if(max_hash_fill < 256)
            {
                size_t ssize = BLOCKSIZE / 64 * 256 * (sizeof(int) + sizeof(ValueType));
                kernel_csr_rs_extpi_interp_fill<false, BLOCKSIZE, 64, 256>
                    <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       ssize,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->ncol_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        diag.vec_,
                        cast_pi->mat_.row_offset,
                        cast_pi->mat_.col,
                        cast_pi->mat_.val,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_f2c->vec_);
            }
            else if(max_hash_fill < 512)
            {
                size_t ssize = BLOCKSIZE / 64 * 512 * (sizeof(int) + sizeof(ValueType));
                kernel_csr_rs_extpi_interp_fill<false, BLOCKSIZE, 64, 512>
                    <<<(this->nrow_ * 64 - 1) / BLOCKSIZE + 1,
                       BLOCKSIZE,
                       ssize,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->ncol_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        diag.vec_,
                        cast_pi->mat_.row_offset,
                        cast_pi->mat_.col,
                        cast_pi->mat_.val,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_f2c->vec_);
            }
            else if(max_hash_fill < 1024)
            {
                size_t ssize = 128 / 64 * 1024 * (sizeof(int) + sizeof(ValueType));
                kernel_csr_rs_extpi_interp_fill<false, 128, 64, 1024>
                    <<<(this->nrow_ * 64 - 1) / 128 + 1,
                       128,
                       ssize,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->ncol_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        diag.vec_,
                        cast_pi->mat_.row_offset,
                        cast_pi->mat_.col,
                        cast_pi->mat_.val,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_f2c->vec_);
            }
            else if(max_hash_fill < 2048)
            {
                size_t ssize = 64 / 64 * 2048 * (sizeof(int) + sizeof(ValueType));
                kernel_csr_rs_extpi_interp_fill<false, 64, 64, 2048>
                    <<<(this->nrow_ * 64 - 1) / 64 + 1,
                       64,
                       ssize,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->ncol_,
                        this->nnz_,
                        FF1,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        diag.vec_,
                        cast_pi->mat_.row_offset,
                        cast_pi->mat_.col,
                        cast_pi->mat_.val,
                        cast_S->vec_,
                        cast_cf->vec_,
                        cast_f2c->vec_);
            }
            else
            {
                // More nnz per row will not fit into LDS
                // Fall back to host

                free_hip(&cast_pi->mat_.col);
                free_hip(&cast_pi->mat_.val);

                cast_pi->nnz_  = 0;
                cast_pi->ncol_ = 0;

                return false;
            }
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);
#undef BLOCKSIZE

        return true;
    }

    template class HIPAcceleratorMatrixCSR<double>;
    template class HIPAcceleratorMatrixCSR<float>;
#ifdef SUPPORT_COMPLEX
    template class HIPAcceleratorMatrixCSR<std::complex<double>>;
    template class HIPAcceleratorMatrixCSR<std::complex<float>>;
#endif

} // namespace rocalution
