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

#include "hip_matrix_csr.hpp"
#include "../../utils/def.hpp"

#include "hip_matrix_coo.hpp"
#include "hip_matrix_dia.hpp"
#include "hip_matrix_ell.hpp"
#include "hip_matrix_hyb.hpp"
#include "hip_matrix_mcsr.hpp"

#include "hip_matrix_bcsr.hpp"
#include "hip_matrix_dense.hpp"
#include "hip_vector.hpp"

#include "hip_conversion.hpp"

#include "../host/host_matrix_csr.hpp"

#include "../base_matrix.hpp"
#include "../base_vector.hpp"

#include "../backend_manager.hpp"

#include "../../utils/log.hpp"

#include "../../utils/allocate_free.hpp"
#include "hip_utils.hpp"

#include "hip_kernels_general.hpp"

#include "hip_kernels_csr.hpp"
#include "hip_kernels_vector.hpp"

#include "hip_allocate_free.hpp"
#include "hip_blas.hpp"

#include "../matrix_formats_ind.hpp"

#include "hip_sparse.hpp"

#include <limits>
#include <vector>

#include <rocprim/rocprim.hpp>

#include <hip/hip_runtime.h>

namespace rocalution
{
    template <typename ValueType>
    HIPAcceleratorMatrixCSR<ValueType>::HIPAcceleratorMatrixCSR()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixCSR<ValueType>::HIPAcceleratorMatrixCSR(
        const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this,
                  "HIPAcceleratorMatrixCSR::HIPAcceleratorMatrixCSR()",
                  "constructor with local_backend");

        this->mat_.row_offset = NULL;
        this->mat_.col        = NULL;
        this->mat_.val        = NULL;
        this->set_backend(local_backend);

        this->L_mat_descr_ = 0;
        this->U_mat_descr_ = 0;

        this->mat_descr_ = 0;
        this->mat_info_  = 0;

        this->mat_buffer_size_ = 0;
        this->mat_buffer_      = NULL;

        this->tmp_vec_ = NULL;

        CHECK_HIP_ERROR(__FILE__, __LINE__);

        rocsparse_status status;

        status = rocsparse_create_mat_descr(&this->mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_create_mat_info(&this->mat_info_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixCSR<ValueType>::~HIPAcceleratorMatrixCSR()
    {
        log_debug(this, "HIPAcceleratorMatrixCSR::~HIPAcceleratorMatrixCSR()", "destructor");

        this->Clear();

        rocsparse_status status;

        status = rocsparse_destroy_mat_descr(this->mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_mat_info(this->mat_info_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::Info(void) const
    {
        LOG_INFO("HIPAcceleratorMatrixCSR<ValueType>");
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::AllocateCSR(int64_t nnz, int nrow, int ncol)
    {
        assert(nnz >= 0);
        assert(ncol >= 0);
        assert(nrow >= 0);

        this->Clear();

        allocate_hip(nrow + 1, &this->mat_.row_offset);
        set_to_zero_hip(this->local_backend_.HIP_block_size, nrow + 1, mat_.row_offset);

        allocate_hip(nnz, &this->mat_.col);
        allocate_hip(nnz, &this->mat_.val);

        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, mat_.col);
        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, mat_.val);

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::SetDataPtrCSR(
        PtrType** row_offset, int** col, ValueType** val, int64_t nnz, int nrow, int ncol)
    {
        assert(nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);
        assert(*row_offset != NULL);

        if(nnz > 0)
        {
            assert(*col != NULL);
            assert(*val != NULL);
        }

        this->Clear();

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;

        hipDeviceSynchronize();

        this->mat_.row_offset = *row_offset;
        this->mat_.col        = *col;
        this->mat_.val        = *val;

        this->ApplyAnalysis();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::LeaveDataPtrCSR(PtrType**   row_offset,
                                                             int**       col,
                                                             ValueType** val)
    {
        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);
        assert(this->nnz_ >= 0);

        hipDeviceSynchronize();

        // see free_host function for details
        *row_offset = this->mat_.row_offset;
        *col        = this->mat_.col;
        *val        = this->mat_.val;

        this->mat_.row_offset = NULL;
        this->mat_.col        = NULL;
        this->mat_.val        = NULL;

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::Clear(void)
    {
        free_hip(&this->mat_.row_offset);
        free_hip(&this->mat_.col);
        free_hip(&this->mat_.val);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;

        this->LAnalyseClear();
        this->UAnalyseClear();
        this->LUAnalyseClear();
        this->LLAnalyseClear();
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::Zeros()
    {
        if(this->nnz_ > 0)
        {
            set_to_zero_hip(this->local_backend_.HIP_block_size, this->nnz_, mat_.val);
        }

        return true;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::CopyFromHost(const HostMatrix<ValueType>& src)
    {
        const HostMatrixCSR<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());
        assert(this->GetMatBlockDimension() == src.GetMatBlockDimension());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixCSR<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateCSR(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            if(cast_mat->mat_.row_offset != NULL)
            {
                copy_h2d(this->nrow_ + 1, cast_mat->mat_.row_offset, this->mat_.row_offset);
            }

            copy_h2d(this->nnz_, cast_mat->mat_.col, this->mat_.col);
            copy_h2d(this->nnz_, cast_mat->mat_.val, this->mat_.val);
        }
        else
        {
            LOG_INFO("Error unsupported HIP matrix type");
            this->Info();
            src.Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        this->ApplyAnalysis();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType>& src)
    {
        const HostMatrixCSR<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());
        assert(this->GetMatBlockDimension() == src.GetMatBlockDimension());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixCSR<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateCSR(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            if(cast_mat->mat_.row_offset != NULL)
            {
                copy_h2d(this->nrow_ + 1,
                         cast_mat->mat_.row_offset,
                         this->mat_.row_offset,
                         true,
                         HIPSTREAM(this->local_backend_.HIP_stream_current));
            }

            copy_h2d(this->nnz_,
                     cast_mat->mat_.col,
                     this->mat_.col,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_h2d(this->nnz_,
                     cast_mat->mat_.val,
                     this->mat_.val,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
        }
        else
        {
            LOG_INFO("Error unsupported HIP matrix type");
            this->Info();
            src.Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        this->ApplyAnalysis();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::CopyToHost(HostMatrix<ValueType>* dst) const
    {
        HostMatrixCSR<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixCSR<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateCSR(this->nnz_, this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->ncol_ == cast_mat->ncol_);

            if(this->mat_.row_offset != NULL)
            {
                copy_d2h(this->nrow_ + 1, this->mat_.row_offset, cast_mat->mat_.row_offset);
            }

            copy_d2h(this->nnz_, this->mat_.col, cast_mat->mat_.col);
            copy_d2h(this->nnz_, this->mat_.val, cast_mat->mat_.val);
        }
        else
        {
            LOG_INFO("Error unsupported HIP matrix type");
            this->Info();
            dst->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::CopyToHostAsync(HostMatrix<ValueType>* dst) const
    {
        HostMatrixCSR<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixCSR<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateCSR(this->nnz_, this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->ncol_ == cast_mat->ncol_);

            if(this->mat_.row_offset != NULL)
            {
                copy_d2h(this->nrow_ + 1,
                         this->mat_.row_offset,
                         cast_mat->mat_.row_offset,
                         true,
                         HIPSTREAM(this->local_backend_.HIP_stream_current));
            }

            copy_d2h(this->nnz_,
                     this->mat_.col,
                     cast_mat->mat_.col,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_d2h(this->nnz_,
                     this->mat_.val,
                     cast_mat->mat_.val,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
        }
        else
        {
            LOG_INFO("Error unsupported HIP matrix type");
            this->Info();
            dst->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixCSR<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());
        assert(this->GetMatBlockDimension() == src.GetMatBlockDimension());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateCSR(hip_cast_mat->nnz_, hip_cast_mat->nrow_, hip_cast_mat->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            if(hip_cast_mat->mat_.row_offset)
            {
                copy_d2d(this->nrow_ + 1, hip_cast_mat->mat_.row_offset, this->mat_.row_offset);
            }

            copy_d2d(this->nnz_, hip_cast_mat->mat_.col, this->mat_.col);
            copy_d2d(this->nnz_, hip_cast_mat->mat_.val, this->mat_.val);
        }
        else
        {
            // CPU to HIP
            if((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*>(&src)) != NULL)
            {
                this->CopyFromHost(*host_cast_mat);
            }
            else
            {
                LOG_INFO("Error unsupported HIP matrix type");
                this->Info();
                src.Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }

        this->ApplyAnalysis();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::CopyFromAsync(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixCSR<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());
        assert(this->GetMatBlockDimension() == src.GetMatBlockDimension());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateCSR(hip_cast_mat->nnz_, hip_cast_mat->nrow_, hip_cast_mat->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            if(hip_cast_mat->mat_.row_offset != NULL)
            {
                copy_d2d(this->nrow_ + 1,
                         hip_cast_mat->mat_.row_offset,
                         this->mat_.row_offset,
                         true,
                         HIPSTREAM(this->local_backend_.HIP_stream_current));
            }

            copy_d2d(this->nnz_,
                     hip_cast_mat->mat_.col,
                     this->mat_.col,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_d2d(this->nnz_,
                     hip_cast_mat->mat_.val,
                     this->mat_.val,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
        }
        else
        {
            // CPU to HIP
            if((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*>(&src)) != NULL)
            {
                this->CopyFromHostAsync(*host_cast_mat);
            }
            else
            {
                LOG_INFO("Error unsupported HIP matrix type");
                this->Info();
                src.Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }

        this->ApplyAnalysis();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::CopyTo(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixCSR<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateCSR(this->nnz_, this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            if(this->mat_.row_offset != NULL)
            {
                copy_d2d(this->nrow_ + 1, this->mat_.row_offset, hip_cast_mat->mat_.row_offset);
            }

            copy_d2d(this->nnz_, this->mat_.col, hip_cast_mat->mat_.col);
            copy_d2d(this->nnz_, this->mat_.val, hip_cast_mat->mat_.val);
        }
        else
        {
            // HIP to CPU
            if((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*>(dst)) != NULL)
            {
                this->CopyToHost(host_cast_mat);
            }
            else
            {
                LOG_INFO("Error unsupported HIP matrix type");
                this->Info();
                dst->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::CopyToAsync(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixCSR<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateCSR(this->nnz_, this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            if(this->mat_.row_offset != NULL)
            {
                copy_d2h(this->nrow_ + 1,
                         this->mat_.row_offset,
                         hip_cast_mat->mat_.row_offset,
                         true,
                         HIPSTREAM(this->local_backend_.HIP_stream_current));
            }

            copy_d2h(this->nnz_,
                     this->mat_.col,
                     hip_cast_mat->mat_.col,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_d2h(this->nnz_,
                     this->mat_.val,
                     hip_cast_mat->mat_.val,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
        }
        else
        {
            // HIP to CPU
            if((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*>(dst)) != NULL)
            {
                this->CopyToHostAsync(host_cast_mat);
            }
            else
            {
                LOG_INFO("Error unsupported HIP matrix type");
                this->Info();
                dst->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::CopyFromCSR(const PtrType*   row_offsets,
                                                         const int*       col,
                                                         const ValueType* val)
    {
        copy_d2d(this->nrow_ + 1, row_offsets, this->mat_.row_offset);

        if(this->nnz_ > 0)
        {
            assert(this->nrow_ > 0);
            assert(this->ncol_ > 0);
        }

        copy_d2d(this->nnz_, col, this->mat_.col);
        copy_d2d(this->nnz_, val, this->mat_.val);

        this->ApplyAnalysis();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::CopyToCSR(PtrType*   row_offsets,
                                                       int*       col,
                                                       ValueType* val) const
    {
        copy_d2d(this->nrow_ + 1, this->mat_.row_offset, row_offsets);

        if(this->nnz_ > 0)
        {
            assert(this->nrow_ > 0);
            assert(this->ncol_ > 0);
        }

        copy_d2d(this->nnz_, this->mat_.col, col);
        copy_d2d(this->nnz_, this->mat_.val, val);
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // empty matrix is empty matrix
        if(mat.GetNnz() == 0)
        {
            return true;
        }

        const HIPAcceleratorMatrixCSR<ValueType>* cast_mat_csr;
        if((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&mat)) != NULL)
        {
            this->CopyFrom(*cast_mat_csr);
            return true;
        }

        // Convert from COO to CSR
        const HIPAcceleratorMatrixCOO<ValueType>* cast_mat_coo;
        if((cast_mat_coo = dynamic_cast<const HIPAcceleratorMatrixCOO<ValueType>*>(&mat)) != NULL)
        {
            this->Clear();

            if(coo_to_csr_hip(&this->local_backend_,
                              cast_mat_coo->nnz_,
                              cast_mat_coo->nrow_,
                              cast_mat_coo->ncol_,
                              cast_mat_coo->mat_,
                              &this->mat_)
               == true)
            {
                this->nrow_ = cast_mat_coo->nrow_;
                this->ncol_ = cast_mat_coo->ncol_;
                this->nnz_  = cast_mat_coo->nnz_;

                this->ApplyAnalysis();

                return true;
            }
        }

        const HIPAcceleratorMatrixELL<ValueType>* cast_mat_ell;
        if((cast_mat_ell = dynamic_cast<const HIPAcceleratorMatrixELL<ValueType>*>(&mat)) != NULL)
        {
            this->Clear();
            int64_t nnz;

            if(ell_to_csr_hip(&this->local_backend_,
                              cast_mat_ell->nnz_,
                              cast_mat_ell->nrow_,
                              cast_mat_ell->ncol_,
                              cast_mat_ell->mat_,
                              cast_mat_ell->mat_descr_,
                              &this->mat_,
                              this->mat_descr_,
                              &nnz)
               == true)
            {
                this->nrow_ = cast_mat_ell->nrow_;
                this->ncol_ = cast_mat_ell->ncol_;
                this->nnz_  = nnz;

                this->ApplyAnalysis();

                return true;
            }
        }

        const HIPAcceleratorMatrixDENSE<ValueType>* cast_mat_dense;
        if((cast_mat_dense = dynamic_cast<const HIPAcceleratorMatrixDENSE<ValueType>*>(&mat))
           != NULL)
        {
            this->Clear();
            int64_t nnz = 0;

            if(dense_to_csr_hip(&this->local_backend_,
                                cast_mat_dense->nrow_,
                                cast_mat_dense->ncol_,
                                cast_mat_dense->mat_,
                                &this->mat_,
                                this->mat_descr_,
                                &nnz)
               == true)
            {
                this->nrow_ = cast_mat_dense->nrow_;
                this->ncol_ = cast_mat_dense->ncol_;
                this->nnz_  = nnz;

                return true;
            }
        }

        const HIPAcceleratorMatrixBCSR<ValueType>* cast_mat_bcsr;
        if((cast_mat_bcsr = dynamic_cast<const HIPAcceleratorMatrixBCSR<ValueType>*>(&mat)) != NULL)
        {
            this->Clear();

            int     nrow = cast_mat_bcsr->mat_.nrowb * cast_mat_bcsr->mat_.blockdim;
            int     ncol = cast_mat_bcsr->mat_.ncolb * cast_mat_bcsr->mat_.blockdim;
            int64_t nnz  = cast_mat_bcsr->mat_.nnzb * cast_mat_bcsr->mat_.blockdim
                          * cast_mat_bcsr->mat_.blockdim;

            if(bcsr_to_csr_hip(&this->local_backend_,
                               nnz,
                               nrow,
                               ncol,
                               cast_mat_bcsr->mat_,
                               cast_mat_bcsr->mat_descr_,
                               &this->mat_,
                               this->mat_descr_)
               == true)
            {
                this->nrow_ = nrow;
                this->ncol_ = ncol;
                this->nnz_  = nnz;

                return true;
            }
        }

        /*
    const HIPAcceleratorMatrixDIA<ValueType>   *cast_mat_dia;
    if ((cast_mat_dia = dynamic_cast<const HIPAcceleratorMatrixDIA<ValueType>*> (&mat)) != NULL) {
      this->Clear();
      int64_t nnz = 0;

      FATAL_ERROR(__FILE__, __LINE__);

      this->nrow_ = cast_mat_dia->nrow_;
      this->ncol_ = cast_mat_dia->ncol_;
      this->nnz_  = nnz ;

      return true;

    }
    */

        /*
    const HIPAcceleratorMatrixMCSR<ValueType>  *cast_mat_mcsr;
    if ((cast_mat_mcsr = dynamic_cast<const HIPAcceleratorMatrixMCSR<ValueType>*> (&mat)) != NULL) {
      this->Clear();

      FATAL_ERROR(__FILE__, __LINE__);

      this->nrow_ = cast_mat_mcsr->nrow_;
      this->ncol_ = cast_mat_mcsr->ncol_;
      this->nnz_  = cast_mat_mcsr->nnz_;

      return true;

    }
    */

        /*
    const HIPAcceleratorMatrixHYB<ValueType>   *cast_mat_hyb;
    if ((cast_mat_hyb = dynamic_cast<const HIPAcceleratorMatrixHYB<ValueType>*> (&mat)) != NULL) {
      this->Clear();

      FATAL_ERROR(__FILE__, __LINE__);
      int64_t nnz = 0;

      this->nrow_ = cast_mat_hyb->nrow_;
      this->ncol_ = cast_mat_hyb->ncol_;
      this->nnz_  = nnz;

      return true;

    }
    */

        return false;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::CopyFromHostCSR(const PtrType*   row_offset,
                                                             const int*       col,
                                                             const ValueType* val,
                                                             int64_t          nnz,
                                                             int              nrow,
                                                             int              ncol)
    {
        assert(nnz >= 0);
        assert(ncol >= 0);
        assert(nrow >= 0);
        assert(row_offset != NULL);

        if(nnz > 0)
        {
            assert(col != NULL);
            assert(val != NULL);
        }

        // Allocate matrix
        this->Clear();
        this->AllocateCSR(nnz, nrow, ncol);

        copy_h2d(this->nrow_ + 1, row_offset, this->mat_.row_offset);
        copy_h2d(this->nnz_, col, this->mat_.col);
        copy_h2d(this->nnz_, val, this->mat_.val);

        this->ApplyAnalysis();
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::Sort(void)
    {
        if(this->nnz_ > 0)
        {
            rocsparse_status status;

            assert(this->nnz_ <= std::numeric_limits<int>::max());

            size_t buffer_size = 0;
            status             = rocsparse_csrsort_buffer_size(
                ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                this->nrow_,
                this->ncol_,
                this->nnz_,
                this->mat_.row_offset,
                this->mat_.col,
                &buffer_size);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            char* buffer = nullptr;
            allocate_hip(buffer_size, &buffer);

            int* perm = nullptr;
            allocate_hip(this->nnz_, &perm);
            status = rocsparse_create_identity_permutation(
                ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle), this->nnz_, perm);

            status = rocsparse_csrsort(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                       this->nrow_,
                                       this->ncol_,
                                       this->nnz_,
                                       this->mat_descr_,
                                       this->mat_.row_offset,
                                       this->mat_.col,
                                       perm,
                                       buffer);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            // Gather sorted csr_val array
            ValueType* csr_val_sorted = nullptr;
            allocate_hip(this->nnz_, &csr_val_sorted);
            status = rocsparseTgthr(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                    this->nnz_,
                                    this->mat_.val,
                                    csr_val_sorted,
                                    perm,
                                    rocsparse_index_base_zero);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            free_hip(&perm);
            free_hip(&this->mat_.val);

            this->mat_.val = csr_val_sorted;

            free_hip(&buffer);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::Permute(const BaseVector<int>& permutation)
    {
        if(this->nnz_ > 0)
        {
            const HIPAcceleratorVector<int>* cast_perm
                = dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);

            assert(cast_perm != NULL);
            assert(cast_perm->size_ == this->nrow_);
            assert(cast_perm->size_ == this->ncol_);

            PtrType*   d_nnzPerm = NULL;
            int*       d_offset  = NULL;
            ValueType* d_data    = NULL;

            allocate_hip((this->nrow_ + 1), &d_nnzPerm);
            allocate_hip(this->nnz_, &d_data);
            allocate_hip(this->nnz_, &d_offset);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->nrow_ / this->local_backend_.HIP_block_size + 1);

            kernel_permute_row_nnz<<<GridSize,
                                     BlockSize,
                                     0,
                                     HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->nrow_, this->mat_.row_offset, cast_perm->vec_, d_nnzPerm);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            // rocprim buffer
            size_t size   = 0;
            char*  buffer = NULL;

            // Determine maximum
            PtrType* d_max = NULL;
            allocate_hip(1, &d_max);
            rocprim::reduce(buffer,
                            size,
                            d_nnzPerm,
                            d_max,
                            0,
                            this->nrow_,
                            rocprim::maximum<PtrType>(),
                            HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            allocate_hip(size, &buffer);

            rocprim::reduce(buffer,
                            size,
                            d_nnzPerm,
                            d_max,
                            0,
                            this->nrow_,
                            rocprim::maximum<PtrType>(),
                            HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            free_hip(&buffer);
            buffer = NULL;

            PtrType maxnnzrow;
            copy_d2h(1, d_max, &maxnnzrow);
            free_hip(&d_max);

            // Inclusive sum
            rocprim::inclusive_scan(buffer,
                                    size,
                                    d_nnzPerm + 1,
                                    d_nnzPerm + 1,
                                    this->nrow_,
                                    rocprim::plus<PtrType>(),
                                    HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            allocate_hip(size, &buffer);

            rocprim::inclusive_scan(buffer,
                                    size,
                                    d_nnzPerm + 1,
                                    d_nnzPerm + 1,
                                    this->nrow_,
                                    rocprim::plus<PtrType>(),
                                    HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            free_hip(&buffer);
            buffer = NULL;

            BlockSize = dim3(this->local_backend_.HIP_block_size);
            GridSize  = dim3((this->local_backend_.HIP_warp * this->nrow_ - 1)
                                / this->local_backend_.HIP_block_size
                            + 1);

            if(this->local_backend_.HIP_warp == 32)
            {
                kernel_permute_rows<32>
                    <<<GridSize,
                       BlockSize,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             d_nnzPerm,
                                                                             this->mat_.col,
                                                                             this->mat_.val,
                                                                             cast_perm->vec_,
                                                                             d_offset,
                                                                             d_data);
            }
            else if(this->local_backend_.HIP_warp == 64)
            {
                kernel_permute_rows<64>
                    <<<GridSize,
                       BlockSize,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             d_nnzPerm,
                                                                             this->mat_.col,
                                                                             this->mat_.val,
                                                                             cast_perm->vec_,
                                                                             d_offset,
                                                                             d_data);
            }
            else
            {
                LOG_INFO("Unsupported HIP warp size of " << this->local_backend_.HIP_warp);
                FATAL_ERROR(__FILE__, __LINE__);
            }

            CHECK_HIP_ERROR(__FILE__, __LINE__);

            free_hip(&this->mat_.row_offset);

            this->mat_.row_offset = d_nnzPerm;

            if(maxnnzrow > 64)
            {
                kernel_permute_cols_fallback<<<
                    GridSize,
                    BlockSize,
                    0,
                    HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                          this->mat_.row_offset,
                                                                          cast_perm->vec_,
                                                                          d_offset,
                                                                          d_data,
                                                                          this->mat_.col,
                                                                          this->mat_.val);
            }
            else if(maxnnzrow > 32)
            {
                kernel_permute_cols<64>
                    <<<GridSize,
                       BlockSize,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             cast_perm->vec_,
                                                                             d_offset,
                                                                             d_data,
                                                                             this->mat_.col,
                                                                             this->mat_.val);
            }
            else if(maxnnzrow > 16)
            {
                kernel_permute_cols<32>
                    <<<GridSize,
                       BlockSize,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             cast_perm->vec_,
                                                                             d_offset,
                                                                             d_data,
                                                                             this->mat_.col,
                                                                             this->mat_.val);
            }
            else if(maxnnzrow > 8)
            {
                kernel_permute_cols<16>
                    <<<GridSize,
                       BlockSize,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             cast_perm->vec_,
                                                                             d_offset,
                                                                             d_data,
                                                                             this->mat_.col,
                                                                             this->mat_.val);
            }
            else if(maxnnzrow > 4)
            {
                kernel_permute_cols<8>
                    <<<GridSize,
                       BlockSize,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             cast_perm->vec_,
                                                                             d_offset,
                                                                             d_data,
                                                                             this->mat_.col,
                                                                             this->mat_.val);
            }
            else
            {
                kernel_permute_cols<4>
                    <<<GridSize,
                       BlockSize,
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             cast_perm->vec_,
                                                                             d_offset,
                                                                             d_data,
                                                                             this->mat_.col,
                                                                             this->mat_.val);
            }
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            free_hip(&d_offset);
            free_hip(&d_data);
        }

        this->ApplyAnalysis();

        return true;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::ApplyAnalysis(void) const
    {
        if(this->nnz_ > 0)
        {
            rocsparse_status status;
            status
                = rocsparseTcsrmv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                           rocsparse_operation_none,
                                           this->nrow_,
                                           this->ncol_,
                                           this->nnz_,
                                           this->mat_descr_,
                                           this->mat_.val,
                                           this->mat_.row_offset,
                                           this->mat_.col,
                                           this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::Apply(const BaseVector<ValueType>& in,
                                                   BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(out != NULL);

            const HIPAcceleratorVector<ValueType>* cast_in
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
            HIPAcceleratorVector<ValueType>* cast_out
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);
            assert(cast_in->size_ == this->ncol_);
            assert(cast_out->size_ == this->nrow_);

            ValueType alpha = static_cast<ValueType>(1);
            ValueType beta  = static_cast<ValueType>(0);

            rocsparse_status status;
            status = rocsparseTcsrmv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     rocsparse_operation_none,
                                     this->nrow_,
                                     this->ncol_,
                                     this->nnz_,
                                     &alpha,
                                     this->mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_info_,
                                     cast_in->vec_,
                                     &beta,
                                     cast_out->vec_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
                                                      ValueType                    scalar,
                                                      BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(out != NULL);

            const HIPAcceleratorVector<ValueType>* cast_in
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
            HIPAcceleratorVector<ValueType>* cast_out
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);
            assert(cast_in->size_ == this->ncol_);
            assert(cast_out->size_ == this->nrow_);

            ValueType beta = static_cast<ValueType>(1);

            rocsparse_status status;
            status = rocsparseTcsrmv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     rocsparse_operation_none,
                                     this->nrow_,
                                     this->ncol_,
                                     this->nnz_,
                                     &scalar,
                                     this->mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_info_,
                                     cast_in->vec_,
                                     &beta,
                                     cast_out->vec_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ILU0Factorize(void)
    {
        if(this->nnz_ > 0)
        {
            rocsparse_status status;

            assert(this->nnz_ <= std::numeric_limits<int>::max());

            // Create buffer, if not already available
            size_t buffer_size = 0;
            status             = rocsparseTcsrilu0_buffer_size(
                ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                this->nrow_,
                this->nnz_,
                this->mat_descr_,
                this->mat_.val,
                this->mat_.row_offset,
                this->mat_.col,
                this->mat_info_,
                &buffer_size);

            // Buffer is shared with ILU0 and other solve functions
            if(this->mat_buffer_ == NULL)
            {
                this->mat_buffer_size_ = buffer_size;
                allocate_hip(buffer_size, &this->mat_buffer_);
            }

            assert(this->mat_buffer_size_ >= buffer_size);
            assert(this->mat_buffer_ != NULL);

            status = rocsparseTcsrilu0_analysis(
                ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                this->nrow_,
                this->nnz_,
                this->mat_descr_,
                this->mat_.val,
                this->mat_.row_offset,
                this->mat_.col,
                this->mat_info_,
                rocsparse_analysis_policy_reuse,
                rocsparse_solve_policy_auto,
                this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            status = rocsparseTcsrilu0(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                       this->nrow_,
                                       this->nnz_,
                                       this->mat_descr_,
                                       this->mat_.val,
                                       this->mat_.row_offset,
                                       this->mat_.col,
                                       this->mat_info_,
                                       rocsparse_solve_policy_auto,
                                       this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            status = rocsparse_csrilu0_clear(
                ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle), this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ICFactorize(BaseVector<ValueType>* inv_diag)
    {
        if(this->nnz_ > 0)
        {
            rocsparse_status status;

            assert(this->nnz_ <= std::numeric_limits<int>::max());

            // Create buffer, if not already available
            size_t buffer_size = 0;
            status             = rocsparseTcsric0_buffer_size(
                ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                this->nrow_,
                this->nnz_,
                this->mat_descr_,
                this->mat_.val,
                this->mat_.row_offset,
                this->mat_.col,
                this->mat_info_,
                &buffer_size);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            // Buffer is shared with IC0 and other solve functions
            if(this->mat_buffer_ == NULL)
            {
                this->mat_buffer_size_ = buffer_size;
                allocate_hip(buffer_size, &this->mat_buffer_);
            }
            else if(this->mat_buffer_size_ < buffer_size)
            {
                this->mat_buffer_size_ = buffer_size;
                free_hip(&this->mat_buffer_);
                allocate_hip(buffer_size, &this->mat_buffer_);
            }

            assert(this->mat_buffer_size_ >= buffer_size);
            assert(this->mat_buffer_ != NULL);

            status = rocsparseTcsric0_analysis(
                ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                this->nrow_,
                this->nnz_,
                this->mat_descr_,
                this->mat_.val,
                this->mat_.row_offset,
                this->mat_.col,
                this->mat_info_,
                rocsparse_analysis_policy_reuse,
                rocsparse_solve_policy_auto,
                this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            status = rocsparseTcsric0(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                      this->nrow_,
                                      this->nnz_,
                                      this->mat_descr_,
                                      this->mat_.val,
                                      this->mat_.row_offset,
                                      this->mat_.col,
                                      this->mat_info_,
                                      rocsparse_solve_policy_auto,
                                      this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            status = rocsparse_csric0_clear(
                ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle), this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::LUAnalyse(void)
    {
        assert(this->ncol_ == this->nrow_);
        assert(this->tmp_vec_ == NULL);

        this->tmp_vec_ = new HIPAcceleratorVector<ValueType>(this->local_backend_);

        assert(this->tmp_vec_ != NULL);

        // Status
        rocsparse_status status;

        // L part
        status = rocsparse_create_mat_descr(&this->L_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->L_mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->L_mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_fill_mode(this->L_mat_descr_, rocsparse_fill_mode_lower);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_diag_type(this->L_mat_descr_, rocsparse_diag_type_unit);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // U part
        status = rocsparse_create_mat_descr(&this->U_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->U_mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->U_mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_fill_mode(this->U_mat_descr_, rocsparse_fill_mode_upper);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_diag_type(this->U_mat_descr_, rocsparse_diag_type_non_unit);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        assert(this->nnz_ <= std::numeric_limits<int>::max());

        // Create buffer, if not already available
        size_t buffer_size = 0;

        status
            = rocsparseTcsrsv_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          rocsparse_operation_none,
                                          this->nrow_,
                                          this->nnz_,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_info_,
                                          &buffer_size);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Buffer is shared with ILU0 and other solve functions
        if(this->mat_buffer_ == NULL)
        {
            this->mat_buffer_size_ = buffer_size;
            allocate_hip(buffer_size, &this->mat_buffer_);
        }

        assert(this->mat_buffer_size_ >= buffer_size);
        assert(this->mat_buffer_ != NULL);

        // L part analysis
        status = rocsparseTcsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          rocsparse_operation_none,
                                          this->nrow_,
                                          this->nnz_,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_info_,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // U part analysis
        status = rocsparseTcsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          rocsparse_operation_none,
                                          this->nrow_,
                                          this->nnz_,
                                          this->U_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_info_,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Allocate temporary vector
        this->tmp_vec_->Allocate(this->nrow_);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::LUAnalyseClear(void)
    {
        rocsparse_status status;

        // Clear analysis info
        if(this->L_mat_descr_ != NULL)
        {
            status = rocsparse_csrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                           this->L_mat_descr_,
                                           this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        if(this->U_mat_descr_ != NULL)
        {
            status = rocsparse_csrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                           this->U_mat_descr_,
                                           this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        // Clear matrix descriptor
        if(this->L_mat_descr_ != NULL)
        {
            status = rocsparse_destroy_mat_descr(this->L_mat_descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        if(this->U_mat_descr_ != NULL)
        {
            status = rocsparse_destroy_mat_descr(this->U_mat_descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        this->L_mat_descr_ = 0;
        this->U_mat_descr_ = 0;

        // Clear buffer
        if(this->mat_buffer_ != NULL)
        {
            free_hip(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        this->mat_buffer_size_ = 0;

        // Clear temporary vector
        if(this->tmp_vec_ != NULL)
        {
            delete this->tmp_vec_;
            this->tmp_vec_ = NULL;
        }
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::LUSolve(const BaseVector<ValueType>& in,
                                                     BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(out != NULL);
            assert(this->L_mat_descr_ != 0);
            assert(this->U_mat_descr_ != 0);
            assert(this->mat_info_ != 0);
            assert(this->ncol_ == this->nrow_);
            assert(this->tmp_vec_ != NULL);

            const HIPAcceleratorVector<ValueType>* cast_in
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
            HIPAcceleratorVector<ValueType>* cast_out
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);
            assert(cast_in->size_ == this->ncol_);
            assert(cast_out->size_ == this->nrow_);

            rocsparse_status status;

            ValueType alpha = static_cast<ValueType>(1);

            assert(this->nnz_ <= std::numeric_limits<int>::max());

            // Solve L
            status = rocsparseTcsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     rocsparse_operation_none,
                                     this->nrow_,
                                     this->nnz_,
                                     &alpha,
                                     this->L_mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_info_,
                                     cast_in->vec_,
                                     this->tmp_vec_->vec_,
                                     rocsparse_solve_policy_auto,
                                     this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            // Solve U
            status = rocsparseTcsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     rocsparse_operation_none,
                                     this->nrow_,
                                     this->nnz_,
                                     &alpha,
                                     this->U_mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_info_,
                                     this->tmp_vec_->vec_,
                                     cast_out->vec_,
                                     rocsparse_solve_policy_auto,
                                     this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::LLAnalyse(void)
    {
        assert(this->ncol_ == this->nrow_);
        assert(this->tmp_vec_ == NULL);

        this->tmp_vec_ = new HIPAcceleratorVector<ValueType>(this->local_backend_);

        assert(this->tmp_vec_ != NULL);

        rocsparse_status status;

        status = rocsparse_create_mat_descr(&this->L_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->L_mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->L_mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_fill_mode(this->L_mat_descr_, rocsparse_fill_mode_lower);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_diag_type(this->L_mat_descr_, rocsparse_diag_type_non_unit);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        assert(this->nnz_ <= std::numeric_limits<int>::max());

        // Create buffer, if not already available
        size_t buffer_size_L  = 0;
        size_t buffer_size_Lt = 0;

        status
            = rocsparseTcsrsv_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          rocsparse_operation_none,
                                          this->nrow_,
                                          this->nnz_,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_info_,
                                          &buffer_size_L);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status
            = rocsparseTcsrsv_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          rocsparse_operation_transpose,
                                          this->nrow_,
                                          this->nnz_,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_info_,
                                          &buffer_size_Lt);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        size_t buffer_size = std::max(buffer_size_L, buffer_size_Lt);

        // Buffer is shared with ILU0, IC0 and other solve functions
        if(this->mat_buffer_ == NULL)
        {
            this->mat_buffer_size_ = buffer_size;
            allocate_hip(buffer_size, &this->mat_buffer_);
        }
        else if(this->mat_buffer_size_ < buffer_size)
        {
            this->mat_buffer_size_ = buffer_size;
            free_hip(&this->mat_buffer_);
            allocate_hip(buffer_size, &this->mat_buffer_);
        }

        assert(this->mat_buffer_size_ >= buffer_size);
        assert(this->mat_buffer_ != NULL);

        // L part analysis
        status = rocsparseTcsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          rocsparse_operation_none,
                                          this->nrow_,
                                          this->nnz_,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_info_,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // L^T part analysis
        status = rocsparseTcsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          rocsparse_operation_transpose,
                                          this->nrow_,
                                          this->nnz_,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_info_,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Allocate temporary vector
        this->tmp_vec_->Allocate(this->nrow_);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::LLAnalyseClear(void)
    {
        rocsparse_status status;

        // Clear analysis info
        if(this->L_mat_descr_ != 0)
        {
            status = rocsparse_csrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                           this->L_mat_descr_,
                                           this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        // Clear matrix descriptor
        if(this->L_mat_descr_ != NULL)
        {
            status = rocsparse_destroy_mat_descr(this->L_mat_descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        this->L_mat_descr_ = 0;

        // Clear buffer
        if(this->mat_buffer_ != NULL)
        {
            free_hip(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        this->mat_buffer_size_ = 0;

        // Clear temporary vector
        if(this->tmp_vec_ != NULL)
        {
            delete this->tmp_vec_;
            this->tmp_vec_ = NULL;
        }
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::LLSolve(const BaseVector<ValueType>& in,
                                                     BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(out != NULL);
            assert(this->L_mat_descr_ != 0);
            assert(this->mat_info_ != 0);
            assert(this->ncol_ == this->nrow_);
            assert(this->tmp_vec_ != NULL);
            assert(this->mat_buffer_ != NULL);

            const HIPAcceleratorVector<ValueType>* cast_in
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
            HIPAcceleratorVector<ValueType>* cast_out
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);
            assert(cast_in->size_ == this->ncol_);
            assert(cast_out->size_ == this->nrow_);

            rocsparse_status status;

            ValueType alpha = static_cast<ValueType>(1);

            assert(this->nnz_ <= std::numeric_limits<int>::max());

            // Solve L
            status = rocsparseTcsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     rocsparse_operation_none,
                                     this->nrow_,
                                     this->nnz_,
                                     &alpha,
                                     this->L_mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_info_,
                                     cast_in->vec_,
                                     this->tmp_vec_->vec_,
                                     rocsparse_solve_policy_auto,
                                     this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            // Solve L^T
            status = rocsparseTcsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     rocsparse_operation_transpose,
                                     this->nrow_,
                                     this->nnz_,
                                     &alpha,
                                     this->L_mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_info_,
                                     this->tmp_vec_->vec_,
                                     cast_out->vec_,
                                     rocsparse_solve_policy_auto,
                                     this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::LLSolve(const BaseVector<ValueType>& in,
                                                     const BaseVector<ValueType>& inv_diag,
                                                     BaseVector<ValueType>*       out) const
    {
        return LLSolve(in, out);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::LAnalyse(bool diag_unit)
    {
        rocsparse_status status;

        // L part
        status = rocsparse_create_mat_descr(&this->L_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->L_mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->L_mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_fill_mode(this->L_mat_descr_, rocsparse_fill_mode_lower);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        if(diag_unit == true)
        {
            status = rocsparse_set_mat_diag_type(this->L_mat_descr_, rocsparse_diag_type_unit);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
        else
        {
            status = rocsparse_set_mat_diag_type(this->L_mat_descr_, rocsparse_diag_type_non_unit);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        assert(this->nnz_ <= std::numeric_limits<int>::max());

        // Create buffer, if not already available
        size_t buffer_size = 0;
        status
            = rocsparseTcsrsv_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          rocsparse_operation_none,
                                          this->nrow_,
                                          this->nnz_,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_info_,
                                          &buffer_size);

        // Buffer is shared with ILU0 and other solve functions
        if(this->mat_buffer_ == NULL)
        {
            this->mat_buffer_size_ = buffer_size;
            allocate_hip(buffer_size, &this->mat_buffer_);
        }

        assert(this->mat_buffer_size_ >= buffer_size);
        assert(this->mat_buffer_ != NULL);

        status = rocsparseTcsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          rocsparse_operation_none,
                                          this->nrow_,
                                          this->nnz_,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_info_,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::UAnalyse(bool diag_unit)
    {
        rocsparse_status status;

        // U part
        status = rocsparse_create_mat_descr(&this->U_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->U_mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->U_mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_fill_mode(this->U_mat_descr_, rocsparse_fill_mode_upper);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        if(diag_unit == true)
        {
            status = rocsparse_set_mat_diag_type(this->U_mat_descr_, rocsparse_diag_type_unit);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
        else
        {
            status = rocsparse_set_mat_diag_type(this->U_mat_descr_, rocsparse_diag_type_non_unit);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        assert(this->nnz_ <= std::numeric_limits<int>::max());

        // Create buffer, if not already available
        size_t buffer_size = 0;
        status
            = rocsparseTcsrsv_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          rocsparse_operation_none,
                                          this->nrow_,
                                          this->nnz_,
                                          this->U_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_info_,
                                          &buffer_size);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Buffer is shared with ILU0 and other solve functions
        if(this->mat_buffer_ == NULL)
        {
            this->mat_buffer_size_ = buffer_size;
            allocate_hip(buffer_size, &this->mat_buffer_);
        }

        assert(this->mat_buffer_size_ >= buffer_size);
        assert(this->mat_buffer_ != NULL);

        status = rocsparseTcsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          rocsparse_operation_none,
                                          this->nrow_,
                                          this->nnz_,
                                          this->U_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_info_,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::LAnalyseClear(void)
    {
        rocsparse_status status;

        // Clear analysis info
        if(this->L_mat_descr_ != NULL)
        {
            status = rocsparse_csrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                           this->L_mat_descr_,
                                           this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        // Clear buffer
        if(this->mat_buffer_ != NULL)
        {
            free_hip(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        this->mat_buffer_size_ = 0;

        // Clear matrix descriptor
        if(this->L_mat_descr_ != NULL)
        {
            status = rocsparse_destroy_mat_descr(this->L_mat_descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        this->L_mat_descr_ = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCSR<ValueType>::UAnalyseClear(void)
    {
        rocsparse_status status;

        // Clear analysis info
        if(this->U_mat_descr_ != NULL)
        {
            status = rocsparse_csrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                           this->U_mat_descr_,
                                           this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        // Clear buffer
        if(this->mat_buffer_ != NULL)
        {
            free_hip(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        this->mat_buffer_size_ = 0;

        // Clear matrix descriptor
        if(this->U_mat_descr_ != NULL)
        {
            status = rocsparse_destroy_mat_descr(this->U_mat_descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        this->U_mat_descr_ = 0;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::LSolve(const BaseVector<ValueType>& in,
                                                    BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(out != NULL);
            assert(this->L_mat_descr_ != 0);
            assert(this->mat_info_ != 0);
            assert(this->ncol_ == this->nrow_);
            assert(this->mat_buffer_size_ > 0);
            assert(this->mat_buffer_ != NULL);

            const HIPAcceleratorVector<ValueType>* cast_in
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
            HIPAcceleratorVector<ValueType>* cast_out
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);
            assert(cast_in->size_ == this->ncol_);
            assert(cast_out->size_ == this->nrow_);

            rocsparse_status status;

            ValueType alpha = static_cast<ValueType>(1);

            assert(this->nnz_ <= std::numeric_limits<int>::max());

            // Solve L
            status = rocsparseTcsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     rocsparse_operation_none,
                                     this->nrow_,
                                     this->nnz_,
                                     &alpha,
                                     this->L_mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_info_,
                                     cast_in->vec_,
                                     cast_out->vec_,
                                     rocsparse_solve_policy_auto,
                                     this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::USolve(const BaseVector<ValueType>& in,
                                                    BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(out != NULL);
            assert(this->U_mat_descr_ != 0);
            assert(this->mat_info_ != 0);
            assert(this->ncol_ == this->nrow_);
            assert(this->mat_buffer_size_ > 0);
            assert(this->mat_buffer_ != NULL);

            const HIPAcceleratorVector<ValueType>* cast_in
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
            HIPAcceleratorVector<ValueType>* cast_out
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);
            assert(cast_in->size_ == this->ncol_);
            assert(cast_out->size_ == this->nrow_);

            rocsparse_status status;

            ValueType alpha = static_cast<ValueType>(1);

            assert(this->nnz_ <= std::numeric_limits<int>::max());

            // Solve U
            status = rocsparseTcsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     rocsparse_operation_none,
                                     this->nrow_,
                                     this->nnz_,
                                     &alpha,
                                     this->U_mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_info_,
                                     cast_in->vec_,
                                     cast_out->vec_,
                                     rocsparse_solve_policy_auto,
                                     this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ExtractDiagonal(BaseVector<ValueType>* vec_diag) const
    {
        if(this->nnz_ > 0)
        {
            assert(vec_diag != NULL);

            HIPAcceleratorVector<ValueType>* cast_vec_diag
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(vec_diag);

            assert(cast_vec_diag != NULL);
            assert(cast_vec_diag->size_ == this->nrow_);

            int     nrow            = this->nrow_;
            int64_t avg_nnz_per_row = this->nnz_ / nrow;

            if(avg_nnz_per_row <= 8)
            {
                kernel_csr_extract_diag<1>
                    <<<dim3((this->nrow_ * 1 - 1) / this->local_backend_.HIP_block_size + 1),
                       dim3(this->local_backend_.HIP_block_size),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(nrow,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             this->mat_.val,
                                                                             cast_vec_diag->vec_);
            }
            else if(avg_nnz_per_row <= 16)
            {
                kernel_csr_extract_diag<2>
                    <<<dim3((this->nrow_ * 2 - 1) / this->local_backend_.HIP_block_size + 1),
                       dim3(this->local_backend_.HIP_block_size),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(nrow,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             this->mat_.val,
                                                                             cast_vec_diag->vec_);
            }
            else if(avg_nnz_per_row <= 32)
            {
                kernel_csr_extract_diag<4>
                    <<<dim3((this->nrow_ * 4 - 1) / this->local_backend_.HIP_block_size + 1),
                       dim3(this->local_backend_.HIP_block_size),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(nrow,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             this->mat_.val,
                                                                             cast_vec_diag->vec_);
            }
            else if(avg_nnz_per_row <= 64)
            {
                kernel_csr_extract_diag<8>
                    <<<dim3((this->nrow_ * 8 - 1) / this->local_backend_.HIP_block_size + 1),
                       dim3(this->local_backend_.HIP_block_size),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(nrow,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             this->mat_.val,
                                                                             cast_vec_diag->vec_);
            }
            else if(avg_nnz_per_row <= 128)
            {
                kernel_csr_extract_diag<16>
                    <<<dim3((this->nrow_ * 16 - 1) / this->local_backend_.HIP_block_size + 1),
                       dim3(this->local_backend_.HIP_block_size),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(nrow,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             this->mat_.val,
                                                                             cast_vec_diag->vec_);
            }
            else if(avg_nnz_per_row <= 256 || this->local_backend_.HIP_warp == 32)
            {
                kernel_csr_extract_diag<32>
                    <<<dim3((this->nrow_ * 32 - 1) / this->local_backend_.HIP_block_size + 1),
                       dim3(this->local_backend_.HIP_block_size),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(nrow,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             this->mat_.val,
                                                                             cast_vec_diag->vec_);
            }
            else
            {
                kernel_csr_extract_diag<64>
                    <<<dim3((this->nrow_ * 64 - 1) / this->local_backend_.HIP_block_size + 1),
                       dim3(this->local_backend_.HIP_block_size),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(nrow,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             this->mat_.val,
                                                                             cast_vec_diag->vec_);
            }
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ExtractInverseDiagonal(
        BaseVector<ValueType>* vec_inv_diag) const
    {
        if(this->nnz_ > 0)
        {
            assert(vec_inv_diag != NULL);

            HIPAcceleratorVector<ValueType>* cast_vec_inv_diag
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(vec_inv_diag);

            assert(cast_vec_inv_diag != NULL);
            assert(cast_vec_inv_diag->size_ == this->nrow_);

            int  nrow = this->nrow_;
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

            int* d_detect_zero_diag = NULL;
            allocate_hip(1, &d_detect_zero_diag);
            set_to_zero_hip(1, 1, d_detect_zero_diag);

            kernel_csr_extract_inv_diag<<<GridSize,
                                          BlockSize,
                                          0,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                nrow,
                this->mat_.row_offset,
                this->mat_.col,
                this->mat_.val,
                cast_vec_inv_diag->vec_,
                d_detect_zero_diag);

            int detect_zero_diag = 0;
            copy_d2h(1, d_detect_zero_diag, &detect_zero_diag);

            if(detect_zero_diag == 1)
            {
                LOG_VERBOSE_INFO(
                    2,
                    "*** warning: in HIPAcceleratorMatrixCSR::ExtractInverseDiagonal() a zero has "
                    "been detected on the diagonal. It has been replaced with one to avoid inf");
            }
            free_hip(&d_detect_zero_diag);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ExtractSubMatrix(int                    row_offset,
                                                              int                    col_offset,
                                                              int                    row_size,
                                                              int                    col_size,
                                                              BaseMatrix<ValueType>* mat) const
    {
        assert(mat != NULL);

        assert(row_offset >= 0);
        assert(col_offset >= 0);

        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        HIPAcceleratorMatrixCSR<ValueType>* cast_mat
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(mat);
        assert(cast_mat != NULL);

        PtrType mat_nnz = 0;

        PtrType* row_nnz = NULL;
        allocate_hip(row_size + 1, &row_nnz);

        // compute the nnz per row in the new matrix

        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(row_size / this->local_backend_.HIP_block_size + 1);

        kernel_csr_extract_submatrix_row_nnz<<<
            GridSize,
            BlockSize,
            0,
            HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->mat_.row_offset,
                                                                  this->mat_.col,
                                                                  this->mat_.val,
                                                                  row_offset,
                                                                  col_offset,
                                                                  row_size,
                                                                  col_size,
                                                                  row_nnz);

        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // compute the new nnz by reduction
        size_t rocprim_size;
        char*  rocprim_buffer = NULL;

        rocprim::exclusive_scan(NULL,
                                rocprim_size,
                                row_nnz,
                                row_nnz,
                                0,
                                row_size + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);

        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                row_nnz,
                                row_nnz,
                                0,
                                row_size + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&rocprim_buffer);

        copy_d2h(1, row_nnz + row_size, &mat_nnz);

        cast_mat->AllocateCSR(mat_nnz, row_size, col_size);

        // not empty submatrix
        if(mat_nnz > 0)
        {
            free_hip(&cast_mat->mat_.row_offset);
            cast_mat->mat_.row_offset = row_nnz;
            // copying the sub matrix

            kernel_csr_extract_submatrix_copy<<<
                GridSize,
                BlockSize,
                0,
                HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->mat_.row_offset,
                                                                      this->mat_.col,
                                                                      this->mat_.val,
                                                                      row_offset,
                                                                      col_offset,
                                                                      row_size,
                                                                      col_size,
                                                                      cast_mat->mat_.row_offset,
                                                                      cast_mat->mat_.col,
                                                                      cast_mat->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
        else
        {
            free_hip(&row_nnz);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ExtractL(BaseMatrix<ValueType>* L) const
    {
        assert(L != NULL);

        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        HIPAcceleratorMatrixCSR<ValueType>* cast_L
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(L);

        assert(cast_L != NULL);

        cast_L->Clear();

        // compute nnz per row
        int nrow = this->nrow_;

        allocate_hip(nrow + 1, &cast_L->mat_.row_offset);

        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        kernel_csr_slower_nnz_per_row<<<GridSize,
                                        BlockSize,
                                        0,
                                        HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
            nrow, this->mat_.row_offset, this->mat_.col, cast_L->mat_.row_offset);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // partial sum row_nnz to obtain row_offset vector
        size_t rocprim_size;
        char*  rocprim_buffer = NULL;

        rocprim::exclusive_scan(NULL,
                                rocprim_size,
                                cast_L->mat_.row_offset,
                                cast_L->mat_.row_offset,
                                0,
                                nrow + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);

        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                cast_L->mat_.row_offset,
                                cast_L->mat_.row_offset,
                                0,
                                nrow + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&rocprim_buffer);

        PtrType nnz_L;
        copy_d2h(1, cast_L->mat_.row_offset + nrow, &nnz_L);

        // allocate lower triangular part structure
        allocate_hip(nnz_L, &cast_L->mat_.col);
        allocate_hip(nnz_L, &cast_L->mat_.val);

        // fill lower triangular part
        kernel_csr_extract_l_triangular<<<GridSize,
                                          BlockSize,
                                          0,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
            nrow,
            this->mat_.row_offset,
            this->mat_.col,
            this->mat_.val,
            cast_L->mat_.row_offset,
            cast_L->mat_.col,
            cast_L->mat_.val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        cast_L->nrow_ = this->nrow_;
        cast_L->ncol_ = this->ncol_;
        cast_L->nnz_  = nnz_L;

        cast_L->ApplyAnalysis();

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ExtractLDiagonal(BaseMatrix<ValueType>* L) const
    {
        assert(L != NULL);

        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        HIPAcceleratorMatrixCSR<ValueType>* cast_L
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(L);

        assert(cast_L != NULL);

        cast_L->Clear();

        // compute nnz per row
        int nrow = this->nrow_;

        allocate_hip(nrow + 1, &cast_L->mat_.row_offset);

        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        kernel_csr_lower_nnz_per_row<<<GridSize,
                                       BlockSize,
                                       0,
                                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
            nrow, this->mat_.row_offset, this->mat_.col, cast_L->mat_.row_offset);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // partial sum row_nnz to obtain row_offset vector
        size_t rocprim_size;
        char*  rocprim_buffer = NULL;

        rocprim::exclusive_scan(NULL,
                                rocprim_size,
                                cast_L->mat_.row_offset,
                                cast_L->mat_.row_offset,
                                0,
                                nrow + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);

        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                cast_L->mat_.row_offset,
                                cast_L->mat_.row_offset,
                                0,
                                nrow + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&rocprim_buffer);

        PtrType nnz_L;
        copy_d2h(1, cast_L->mat_.row_offset + nrow, &nnz_L);

        // allocate lower triangular part structure
        allocate_hip(nnz_L, &cast_L->mat_.col);
        allocate_hip(nnz_L, &cast_L->mat_.val);

        // fill lower triangular part
        kernel_csr_extract_l_triangular<<<GridSize,
                                          BlockSize,
                                          0,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
            nrow,
            this->mat_.row_offset,
            this->mat_.col,
            this->mat_.val,
            cast_L->mat_.row_offset,
            cast_L->mat_.col,
            cast_L->mat_.val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        cast_L->nrow_ = this->nrow_;
        cast_L->ncol_ = this->ncol_;
        cast_L->nnz_  = nnz_L;

        cast_L->ApplyAnalysis();

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ExtractU(BaseMatrix<ValueType>* U) const
    {
        assert(U != NULL);

        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        HIPAcceleratorMatrixCSR<ValueType>* cast_U
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(U);

        assert(cast_U != NULL);

        cast_U->Clear();

        // compute nnz per row
        int nrow = this->nrow_;

        allocate_hip(nrow + 1, &cast_U->mat_.row_offset);

        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        kernel_csr_supper_nnz_per_row<<<GridSize,
                                        BlockSize,
                                        0,
                                        HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
            nrow, this->mat_.row_offset, this->mat_.col, cast_U->mat_.row_offset);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // partial sum row_nnz to obtain row_offset vector
        size_t rocprim_size;
        char*  rocprim_buffer = NULL;

        rocprim::exclusive_scan(NULL,
                                rocprim_size,
                                cast_U->mat_.row_offset,
                                cast_U->mat_.row_offset,
                                0,
                                nrow + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);

        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                cast_U->mat_.row_offset,
                                cast_U->mat_.row_offset,
                                0,
                                nrow + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&rocprim_buffer);

        PtrType nnz_U;
        copy_d2h(1, cast_U->mat_.row_offset + nrow, &nnz_U);

        // allocate lower triangular part structure
        allocate_hip(nnz_U, &cast_U->mat_.col);
        allocate_hip(nnz_U, &cast_U->mat_.val);

        // fill upper triangular part
        kernel_csr_extract_u_triangular<<<GridSize,
                                          BlockSize,
                                          0,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
            nrow,
            this->mat_.row_offset,
            this->mat_.col,
            this->mat_.val,
            cast_U->mat_.row_offset,
            cast_U->mat_.col,
            cast_U->mat_.val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        cast_U->nrow_ = this->nrow_;
        cast_U->ncol_ = this->ncol_;
        cast_U->nnz_  = nnz_U;

        cast_U->ApplyAnalysis();

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ExtractUDiagonal(BaseMatrix<ValueType>* U) const
    {
        assert(U != NULL);

        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        HIPAcceleratorMatrixCSR<ValueType>* cast_U
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(U);

        assert(cast_U != NULL);

        cast_U->Clear();

        // compute nnz per row
        int nrow = this->nrow_;

        allocate_hip(nrow + 1, &cast_U->mat_.row_offset);

        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

        kernel_csr_upper_nnz_per_row<<<GridSize,
                                       BlockSize,
                                       0,
                                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
            nrow, this->mat_.row_offset, this->mat_.col, cast_U->mat_.row_offset);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // partial sum row_nnz to obtain row_offset vector
        size_t rocprim_size;
        char*  rocprim_buffer = NULL;

        rocprim::exclusive_scan(NULL,
                                rocprim_size,
                                cast_U->mat_.row_offset,
                                cast_U->mat_.row_offset,
                                0,
                                nrow + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);

        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                cast_U->mat_.row_offset,
                                cast_U->mat_.row_offset,
                                0,
                                nrow + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&rocprim_buffer);

        PtrType nnz_U;
        copy_d2h(1, cast_U->mat_.row_offset + nrow, &nnz_U);

        // allocate lower triangular part structure
        allocate_hip(nnz_U, &cast_U->mat_.col);
        allocate_hip(nnz_U, &cast_U->mat_.val);

        // fill lower triangular part
        kernel_csr_extract_u_triangular<<<GridSize,
                                          BlockSize,
                                          0,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
            nrow,
            this->mat_.row_offset,
            this->mat_.col,
            this->mat_.val,
            cast_U->mat_.row_offset,
            cast_U->mat_.col,
            cast_U->mat_.val);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        cast_U->nrow_ = this->nrow_;
        cast_U->ncol_ = this->ncol_;
        cast_U->nnz_  = nnz_U;

        cast_U->ApplyAnalysis();

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::MaximalIndependentSet(
        int& size, BaseVector<int>* permutation) const
    {
        assert(permutation != NULL);

        HIPAcceleratorVector<int>* cast_perm
            = dynamic_cast<HIPAcceleratorVector<int>*>(permutation);

        assert(cast_perm != NULL);
        assert(this->nrow_ == this->ncol_);

        PtrType* h_row_offset = NULL;
        int*     h_col        = NULL;

        allocate_host(this->nrow_ + 1, &h_row_offset);
        allocate_host(this->nnz_, &h_col);

        copy_d2h(this->nrow_ + 1, this->mat_.row_offset, h_row_offset);
        copy_d2h(this->nnz_, this->mat_.col, h_col);

        int* mis = NULL;
        allocate_host(this->nrow_, &mis);
        memset(mis, 0, sizeof(int) * this->nrow_);

        size = 0;

        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            if(mis[ai] == 0)
            {
                // set the node
                mis[ai] = 1;
                ++size;

                // remove all nbh nodes (without diagonal)
                for(PtrType aj = h_row_offset[ai]; aj < h_row_offset[ai + 1]; ++aj)
                {
                    if(ai != h_col[aj])
                    {
                        mis[h_col[aj]] = -1;
                    }
                }
            }
        }

        int* h_perm = NULL;
        allocate_host(this->nrow_, &h_perm);

        int pos = 0;
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            if(mis[ai] == 1)
            {
                h_perm[ai] = pos;
                ++pos;
            }
            else
            {
                h_perm[ai] = size + ai - pos;
            }
        }

        // Check the permutation
        //
        //  for (int ai=0; ai<this->nrow_; ++ai) {
        //    assert( h_perm[ai] >= 0 );
        //    assert( h_perm[ai] < this->nrow_ );
        //  }

        cast_perm->Allocate(this->nrow_);
        copy_h2d(cast_perm->size_, h_perm, cast_perm->vec_);

        free_host(&h_row_offset);
        free_host(&h_col);

        free_host(&h_perm);
        free_host(&mis);

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::MultiColoring(int&             num_colors,
                                                           int**            size_colors,
                                                           BaseVector<int>* permutation) const
    {
        assert(permutation != NULL);

        HIPAcceleratorVector<int>* cast_perm
            = dynamic_cast<HIPAcceleratorVector<int>*>(permutation);

        assert(cast_perm != NULL);

        // node colors (init value = 0 i.e. no color)
        int*     color        = NULL;
        PtrType* h_row_offset = NULL;
        int*     h_col        = NULL;

        allocate_host(this->nrow_, &color);
        allocate_host(this->nrow_ + 1, &h_row_offset);
        allocate_host(this->nnz_, &h_col);

        copy_d2h(this->nrow_ + 1, this->mat_.row_offset, h_row_offset);
        copy_d2h(this->nnz_, this->mat_.col, h_col);

        memset(color, 0, this->nrow_ * sizeof(int));
        num_colors = 0;
        std::vector<bool> row_col;

        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            color[ai] = 1;
            row_col.clear();
            row_col.assign(num_colors + 2, false);

            for(PtrType aj = h_row_offset[ai]; aj < h_row_offset[ai + 1]; ++aj)
            {
                if(ai != h_col[aj])
                {
                    row_col[color[h_col[aj]]] = true;
                }
            }

            for(PtrType aj = h_row_offset[ai]; aj < h_row_offset[ai + 1]; ++aj)
            {
                if(row_col[color[ai]] == true)
                {
                    ++color[ai];
                }
            }

            if(color[ai] > num_colors)
            {
                num_colors = color[ai];
            }
        }

        free_host(&h_row_offset);
        free_host(&h_col);

        allocate_host(num_colors, size_colors);
        set_to_zero_host(num_colors, *size_colors);

        int* offsets_color = NULL;
        allocate_host(num_colors, &offsets_color);
        memset(offsets_color, 0, sizeof(int) * num_colors);

        for(int i = 0; i < this->nrow_; ++i)
        {
            ++(*size_colors)[color[i] - 1];
        }

        int total = 0;
        for(int i = 1; i < num_colors; ++i)
        {
            total += (*size_colors)[i - 1];
            offsets_color[i] = total;
            //   LOG_INFO("offsets = " << total);
        }

        int* h_perm = NULL;
        allocate_host(this->nrow_, &h_perm);

        for(int i = 0; i < this->nrow_; ++i)
        {
            h_perm[i] = offsets_color[color[i] - 1];
            ++offsets_color[color[i] - 1];
        }

        cast_perm->Allocate(this->nrow_);
        copy_h2d(cast_perm->size_, h_perm, cast_perm->vec_);

        free_host(&h_perm);
        free_host(&color);
        free_host(&offsets_color);

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::Scale(ValueType alpha)
    {
        if(this->nnz_ > 0)
        {
            assert(this->nnz_ <= std::numeric_limits<int>::max());

            rocblas_status status;
            status = rocblasTscal(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                  this->nnz_,
                                  &alpha,
                                  this->mat_.val,
                                  1);
            CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ScaleDiagonal(ValueType alpha)
    {
        if(this->nnz_ > 0)
        {
            int  nrow = this->nrow_;
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

            kernel_csr_scale_diagonal<<<GridSize,
                                        BlockSize,
                                        0,
                                        HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                nrow, this->mat_.row_offset, this->mat_.col, alpha, this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ScaleOffDiagonal(ValueType alpha)
    {
        if(this->nnz_ > 0)
        {
            int  nrow = this->nrow_;
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

            kernel_csr_scale_offdiagonal<<<GridSize,
                                           BlockSize,
                                           0,
                                           HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                nrow, this->mat_.row_offset, this->mat_.col, alpha, this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::AddScalarDiagonal(ValueType alpha)
    {
        if(this->nnz_ > 0)
        {
            int  nrow = this->nrow_;
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

            kernel_csr_add_diagonal<<<GridSize,
                                      BlockSize,
                                      0,
                                      HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                nrow, this->mat_.row_offset, this->mat_.col, alpha, this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::AddScalarOffDiagonal(ValueType alpha)
    {
        if(this->nnz_ > 0)
        {
            int  nrow = this->nrow_;
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

            kernel_csr_add_offdiagonal<<<GridSize,
                                         BlockSize,
                                         0,
                                         HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                nrow, this->mat_.row_offset, this->mat_.col, alpha, this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::AddScalar(ValueType alpha)
    {
        if(this->nnz_ > 0)
        {
            assert(this->nnz_ <= std::numeric_limits<int>::max());

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->nnz_ / this->local_backend_.HIP_block_size + 1);

            kernel_buffer_addscalar<<<GridSize,
                                      BlockSize,
                                      0,
                                      HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->nnz_, alpha, this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::DiagonalMatrixMultR(const BaseVector<ValueType>& diag)
    {
        const HIPAcceleratorVector<ValueType>* cast_diag
            = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&diag);

        assert(cast_diag != NULL);
        assert(cast_diag->size_ == this->ncol_);

        if(this->nnz_ > 0)
        {
            int  nrow = this->nrow_;
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

            kernel_csr_diagmatmult_r<<<GridSize,
                                       BlockSize,
                                       0,
                                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                nrow, this->mat_.row_offset, this->mat_.col, cast_diag->vec_, this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::DiagonalMatrixMultL(const BaseVector<ValueType>& diag)
    {
        const HIPAcceleratorVector<ValueType>* cast_diag
            = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&diag);

        assert(cast_diag != NULL);
        assert(cast_diag->size_ == this->ncol_);

        if(this->nnz_ > 0)
        {
            int  nrow = this->nrow_;
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

            kernel_csr_diagmatmult_l<<<GridSize,
                                       BlockSize,
                                       0,
                                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                nrow, this->mat_.row_offset, cast_diag->vec_, this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::MatMatMult(const BaseMatrix<ValueType>& A,
                                                        const BaseMatrix<ValueType>& B)
    {
        const HIPAcceleratorMatrixCSR<ValueType>* cast_mat_A
            = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&A);
        const HIPAcceleratorMatrixCSR<ValueType>* cast_mat_B
            = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&B);

        assert(cast_mat_A != NULL);
        assert(cast_mat_B != NULL);
        assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);
        assert(cast_mat_A->nrow_ >= 0);
        assert(cast_mat_B->ncol_ >= 0);
        assert(cast_mat_B->nrow_ >= 0);

        this->Clear();

        int m = cast_mat_A->nrow_;
        int n = cast_mat_B->ncol_;
        int k = cast_mat_B->nrow_;

        int nnzC = 0;

        ValueType alpha = static_cast<ValueType>(1);

        rocsparse_status status;

        size_t buffer_size = 0;

        assert(cast_mat_A->nnz_ <= std::numeric_limits<int>::max());
        assert(cast_mat_B->nnz_ <= std::numeric_limits<int>::max());

        status = rocsparseTcsrgemm_buffer_size(
            ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
            rocsparse_operation_none,
            rocsparse_operation_none,
            m,
            n,
            k,
            &alpha,
            cast_mat_A->mat_descr_,
            cast_mat_A->nnz_,
            cast_mat_A->mat_.row_offset,
            cast_mat_A->mat_.col,
            cast_mat_B->mat_descr_,
            cast_mat_B->nnz_,
            cast_mat_B->mat_.row_offset,
            cast_mat_B->mat_.col,
            this->mat_info_,
            &buffer_size);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        char* buffer = NULL;
        allocate_hip(buffer_size, &buffer);
        allocate_hip(m + 1, &this->mat_.row_offset);

        status = rocsparse_csrgemm_nnz(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       m,
                                       n,
                                       k,
                                       cast_mat_A->mat_descr_,
                                       cast_mat_A->nnz_,
                                       cast_mat_A->mat_.row_offset,
                                       cast_mat_A->mat_.col,
                                       cast_mat_B->mat_descr_,
                                       cast_mat_B->nnz_,
                                       cast_mat_B->mat_.row_offset,
                                       cast_mat_B->mat_.col,
                                       NULL,
                                       0,
                                       NULL,
                                       NULL,
                                       this->mat_descr_,
                                       this->mat_.row_offset,
                                       &nnzC,
                                       this->mat_info_,
                                       buffer);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(nnzC, &this->mat_.col);
        allocate_hip(nnzC, &this->mat_.val);

        this->nrow_ = m;
        this->ncol_ = n;
        this->nnz_  = nnzC;

        status = rocsparseTcsrgemm(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   m,
                                   n,
                                   k,
                                   &alpha,
                                   cast_mat_A->mat_descr_,
                                   cast_mat_A->nnz_,
                                   cast_mat_A->mat_.val,
                                   cast_mat_A->mat_.row_offset,
                                   cast_mat_A->mat_.col,
                                   cast_mat_B->mat_descr_,
                                   cast_mat_B->nnz_,
                                   cast_mat_B->mat_.val,
                                   cast_mat_B->mat_.row_offset,
                                   cast_mat_B->mat_.col,
                                   this->mat_descr_,
                                   this->mat_.val,
                                   this->mat_.row_offset,
                                   this->mat_.col,
                                   this->mat_info_,
                                   buffer);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        this->ApplyAnalysis();

        free_hip(&buffer);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::Gershgorin(ValueType& lambda_min,
                                                        ValueType& lambda_max) const
    {
        return false;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::MatrixAdd(const BaseMatrix<ValueType>& mat,
                                                       ValueType                    alpha,
                                                       ValueType                    beta,
                                                       bool                         structure)
    {
        const HIPAcceleratorMatrixCSR<ValueType>* cast_mat
            = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&mat);

        assert(cast_mat != NULL);
        assert(cast_mat->nrow_ == this->nrow_);
        assert(cast_mat->ncol_ == this->ncol_);
        assert(this->nnz_ >= 0);
        assert(cast_mat->nnz_ >= 0);

        if(structure == false)
        {
            int  nrow = this->nrow_;
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

            kernel_csr_add_csr_same_struct<<<GridSize,
                                             BlockSize,
                                             0,
                                             HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                nrow,
                this->mat_.row_offset,
                this->mat_.col,
                cast_mat->mat_.row_offset,
                cast_mat->mat_.col,
                cast_mat->mat_.val,
                alpha,
                beta,
                this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
        else
        {
            int        m          = this->nrow_;
            int        n          = this->ncol_;
            PtrType*   csrRowPtrC = NULL;
            int*       csrColC    = NULL;
            ValueType* csrValC    = NULL;
            PtrType    nnzC;

            allocate_hip(m + 1, &csrRowPtrC);

            rocsparse_status status;

            rocsparse_mat_descr desc_mat_C;

            status = rocsparse_create_mat_descr(&desc_mat_C);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            status = rocsparse_set_mat_index_base(desc_mat_C, rocsparse_index_base_zero);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            status = rocsparse_set_mat_type(desc_mat_C, rocsparse_matrix_type_general);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            status = rocsparse_set_pointer_mode(
                ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                rocsparse_pointer_mode_host);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            status = rocsparse_csrgeam_nnz(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                           m,
                                           n,
                                           this->mat_descr_,
                                           this->nnz_,
                                           this->mat_.row_offset,
                                           this->mat_.col,
                                           cast_mat->mat_descr_,
                                           cast_mat->nnz_,
                                           cast_mat->mat_.row_offset,
                                           cast_mat->mat_.col,
                                           desc_mat_C,
                                           csrRowPtrC,
                                           &nnzC);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            allocate_hip(nnzC, &csrColC);
            allocate_hip(nnzC, &csrValC);

            status = rocsparseTcsrgeam(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                       m,
                                       n,
                                       // A
                                       &alpha,
                                       this->mat_descr_,
                                       this->nnz_,
                                       this->mat_.val,
                                       this->mat_.row_offset,
                                       this->mat_.col,
                                       // B
                                       &beta,
                                       cast_mat->mat_descr_,
                                       cast_mat->nnz_,
                                       cast_mat->mat_.val,
                                       cast_mat->mat_.row_offset,
                                       cast_mat->mat_.col,
                                       // C
                                       desc_mat_C,
                                       csrValC,
                                       csrRowPtrC,
                                       csrColC);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            status = rocsparse_destroy_mat_descr(desc_mat_C);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            this->Clear();
            this->SetDataPtrCSR(&csrRowPtrC, &csrColC, &csrValC, nnzC, m, n);
        }

        this->ApplyAnalysis();

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::Compress(double drop_off)
    {
        if(this->nnz_ > 0)
        {
            HIPAcceleratorMatrixCSR<ValueType> tmp(this->local_backend_);

            tmp.CopyFrom(*this);

            int     nrow    = this->nrow_;
            PtrType mat_nnz = 0;

            int* row_offset = NULL;
            allocate_hip(nrow + 1, &row_offset);

            PtrType* mat_row_offset = NULL;
            allocate_hip(nrow + 1, &mat_row_offset);

            set_to_zero_hip(this->local_backend_.HIP_block_size, nrow + 1, row_offset);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

            kernel_csr_compress_count_nrow<<<GridSize,
                                             BlockSize,
                                             0,
                                             HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->mat_.row_offset, this->mat_.col, this->mat_.val, nrow, drop_off, row_offset);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            size_t rocprim_size;
            char*  rocprim_buffer = NULL;

            rocprim::exclusive_scan(NULL,
                                    rocprim_size,
                                    row_offset,
                                    mat_row_offset,
                                    0,
                                    nrow + 1,
                                    rocprim::plus<PtrType>(),
                                    HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            allocate_hip(rocprim_size, &rocprim_buffer);

            rocprim::exclusive_scan(rocprim_buffer,
                                    rocprim_size,
                                    row_offset,
                                    mat_row_offset,
                                    0,
                                    nrow + 1,
                                    rocprim::plus<PtrType>(),
                                    HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            free_hip(&rocprim_buffer);

            // get the new mat nnz
            copy_d2h(1, mat_row_offset + nrow, &mat_nnz);

            this->AllocateCSR(mat_nnz, nrow, this->ncol_);

            // copy row_offset
            copy_d2d(nrow + 1, mat_row_offset, this->mat_.row_offset);

            // copy col and val
            kernel_csr_compress_copy<<<GridSize,
                                       BlockSize,
                                       0,
                                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                tmp.mat_.row_offset,
                tmp.mat_.col,
                tmp.mat_.val,
                tmp.nrow_,
                drop_off,
                this->mat_.row_offset,
                this->mat_.col,
                this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            free_hip(&row_offset);
            free_hip(&mat_row_offset);
        }

        this->ApplyAnalysis();

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::Transpose(void)
    {
        if(this->nnz_ > 0)
        {
            HIPAcceleratorMatrixCSR<ValueType> tmp(this->local_backend_);

            tmp.CopyFrom(*this);
            tmp.Transpose(this);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::Transpose(BaseMatrix<ValueType>* T) const
    {
        assert(T != NULL);

        HIPAcceleratorMatrixCSR<ValueType>* cast_T
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(T);

        assert(cast_T != NULL);

        if(this->nnz_ > 0)
        {
            cast_T->Clear();
            cast_T->AllocateCSR(this->nnz_, this->ncol_, this->nrow_);

            size_t buffer_size;

            assert(this->nnz_ <= std::numeric_limits<int>::max());

            rocsparse_status status = rocsparse_csr2csc_buffer_size(
                ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                this->nrow_,
                this->ncol_,
                this->nnz_,
                this->mat_.row_offset,
                this->mat_.col,
                rocsparse_action_numeric,
                &buffer_size);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            char* buffer = NULL;
            allocate_hip(buffer_size, &buffer);

            status = rocsparseTcsr2csc(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                       this->nrow_,
                                       this->ncol_,
                                       this->nnz_,
                                       this->mat_.val,
                                       this->mat_.row_offset,
                                       this->mat_.col,
                                       cast_T->mat_.val,
                                       cast_T->mat_.col,
                                       cast_T->mat_.row_offset,
                                       rocsparse_action_numeric,
                                       rocsparse_index_base_zero,
                                       buffer);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            free_hip(&buffer);
        }

        cast_T->ApplyAnalysis();

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ReplaceColumnVector(int                          idx,
                                                                 const BaseVector<ValueType>& vec)
    {
        if(this->nnz_ > 0)
        {
            const HIPAcceleratorVector<ValueType>* cast_vec
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&vec);

            assert(cast_vec != NULL);
            assert(cast_vec->size_ == this->nrow_);

            PtrType*   row_offset = NULL;
            int*       col        = NULL;
            ValueType* val        = NULL;

            int nrow = this->nrow_;
            int ncol = this->ncol_;

            allocate_hip(nrow + 1, &row_offset);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

            kernel_csr_replace_column_vector_offset<<<
                GridSize,
                BlockSize,
                0,
                HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->mat_.row_offset, this->mat_.col, nrow, idx, cast_vec->vec_, row_offset);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            size_t rocprim_size;
            char*  rocprim_buffer = NULL;

            rocprim::exclusive_scan(NULL,
                                    rocprim_size,
                                    row_offset,
                                    row_offset,
                                    0,
                                    this->nrow_ + 1,
                                    rocprim::plus<PtrType>(),
                                    HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            allocate_hip(rocprim_size, &rocprim_buffer);

            rocprim::exclusive_scan(rocprim_buffer,
                                    rocprim_size,
                                    row_offset,
                                    row_offset,
                                    0,
                                    this->nrow_,
                                    rocprim::plus<PtrType>(),
                                    HIPSTREAM(this->local_backend_.HIP_stream_current));
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            free_hip(&rocprim_buffer);

            PtrType nnz;
            copy_d2h(1, row_offset + this->nrow_, &nnz);

            allocate_hip(nnz, &col);
            allocate_hip(nnz, &val);

            kernel_csr_replace_column_vector<<<
                GridSize,
                BlockSize,
                0,
                HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->mat_.row_offset,
                                                                      this->mat_.col,
                                                                      this->mat_.val,
                                                                      nrow,
                                                                      idx,
                                                                      cast_vec->vec_,
                                                                      row_offset,
                                                                      col,
                                                                      val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            this->Clear();
            this->SetDataPtrCSR(&row_offset, &col, &val, nnz, nrow, ncol);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ExtractColumnVector(int                    idx,
                                                                 BaseVector<ValueType>* vec) const
    {
        if(this->nnz_ > 0)
        {
            assert(vec != NULL);

            HIPAcceleratorVector<ValueType>* cast_vec
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(vec);

            assert(cast_vec != NULL);
            assert(cast_vec->size_ == this->nrow_);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->nrow_ / this->local_backend_.HIP_block_size + 1);

            kernel_csr_extract_column_vector<<<
                GridSize,
                BlockSize,
                0,
                HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->mat_.row_offset,
                                                                      this->mat_.col,
                                                                      this->mat_.val,
                                                                      this->nrow_,
                                                                      idx,
                                                                      cast_vec->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::ExtractRowVector(int                    idx,
                                                              BaseVector<ValueType>* vec) const
    {
        if(this->nnz_ > 0)
        {
            assert(vec != NULL);

            HIPAcceleratorVector<ValueType>* cast_vec
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(vec);

            assert(cast_vec != NULL);
            assert(cast_vec->size_ == this->ncol_);

            cast_vec->Zeros();

            // Get nnz of row idx
            PtrType nnz[2];

            copy_d2h(2, this->mat_.row_offset + idx, nnz);

            int row_nnz = nnz[1] - nnz[0];

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(row_nnz / this->local_backend_.HIP_block_size + 1);

            kernel_csr_extract_row_vector<<<GridSize,
                                            BlockSize,
                                            0,
                                            HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->mat_.row_offset,
                this->mat_.col,
                this->mat_.val,
                row_nnz,
                idx,
                cast_vec->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::AMGConnect(ValueType        eps,
                                                        BaseVector<int>* connections) const
    {
        assert(connections != NULL);

        HIPAcceleratorVector<int>* cast_conn
            = dynamic_cast<HIPAcceleratorVector<int>*>(connections);
        assert(cast_conn != NULL);

        cast_conn->Clear();
        cast_conn->Allocate(this->nnz_);

        ValueType eps2 = eps * eps;

        HIPAcceleratorVector<ValueType> vec_diag(this->local_backend_);
        vec_diag.Allocate(this->nrow_);

        this->ExtractDiagonal(&vec_diag);

        int avg_nnz_per_row = this->nnz_ / this->nrow_;

        if(avg_nnz_per_row <= 8)
        {
            kernel_csr_amg_connect<1>
                <<<dim3((this->nrow_ * 1 - 1) / this->local_backend_.HIP_block_size + 1),
                   dim3(this->local_backend_.HIP_block_size),
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         eps2,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         this->mat_.val,
                                                                         vec_diag.vec_,
                                                                         cast_conn->vec_);
        }
        else if(avg_nnz_per_row <= 16)
        {
            kernel_csr_amg_connect<2>
                <<<dim3((this->nrow_ * 2 - 1) / this->local_backend_.HIP_block_size + 1),
                   dim3(this->local_backend_.HIP_block_size),
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         eps2,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         this->mat_.val,
                                                                         vec_diag.vec_,
                                                                         cast_conn->vec_);
        }
        else if(avg_nnz_per_row <= 32)
        {
            kernel_csr_amg_connect<4>
                <<<dim3((this->nrow_ * 4 - 1) / this->local_backend_.HIP_block_size + 1),
                   dim3(this->local_backend_.HIP_block_size),
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         eps2,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         this->mat_.val,
                                                                         vec_diag.vec_,
                                                                         cast_conn->vec_);
        }
        else if(avg_nnz_per_row <= 64)
        {
            kernel_csr_amg_connect<8>
                <<<dim3((this->nrow_ * 8 - 1) / this->local_backend_.HIP_block_size + 1),
                   dim3(this->local_backend_.HIP_block_size),
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         eps2,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         this->mat_.val,
                                                                         vec_diag.vec_,
                                                                         cast_conn->vec_);
        }
        else if(avg_nnz_per_row <= 128)
        {
            kernel_csr_amg_connect<16>
                <<<dim3((this->nrow_ * 16 - 1) / this->local_backend_.HIP_block_size + 1),
                   dim3(this->local_backend_.HIP_block_size),
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         eps2,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         this->mat_.val,
                                                                         vec_diag.vec_,
                                                                         cast_conn->vec_);
        }
        else if(avg_nnz_per_row <= 256 || this->local_backend_.HIP_warp == 32)
        {
            kernel_csr_amg_connect<32>
                <<<dim3((this->nrow_ * 32 - 1) / this->local_backend_.HIP_block_size + 1),
                   dim3(this->local_backend_.HIP_block_size),
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         eps2,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         this->mat_.val,
                                                                         vec_diag.vec_,
                                                                         cast_conn->vec_);
        }
        else
        {
            kernel_csr_amg_connect<64>
                <<<dim3((this->nrow_ * 64 - 1) / this->local_backend_.HIP_block_size + 1),
                   dim3(this->local_backend_.HIP_block_size),
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         eps2,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         this->mat_.val,
                                                                         vec_diag.vec_,
                                                                         cast_conn->vec_);
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::AMGPMISAggregate(const BaseVector<int>& connections,
                                                              BaseVector<int>* aggregates) const
    {
        int64_t avg_nnz_per_row = this->nnz_ / this->nrow_;

        assert(aggregates != NULL);

        HIPAcceleratorVector<int>* cast_agg = dynamic_cast<HIPAcceleratorVector<int>*>(aggregates);
        const HIPAcceleratorVector<int>* cast_conn
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&connections);

        assert(cast_agg != NULL);
        assert(cast_conn != NULL);

        aggregates->Clear();
        aggregates->Allocate(this->nrow_);

        mis_tuple* tuples     = nullptr;
        mis_tuple* max_tuples = nullptr;
        allocate_hip(this->nrow_, &tuples);
        allocate_hip(this->nrow_, &max_tuples);

        bool  hdone = false;
        bool* ddone = nullptr;
        allocate_hip(1, &ddone);

        dim3 BlockSize(this->local_backend_.HIP_block_size);
        dim3 GridSize((this->nrow_ - 1) / this->local_backend_.HIP_block_size + 1);

        kernel_csr_amg_init_mis_tuples<<<GridSize,
                                         BlockSize,
                                         0,
                                         HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
            this->nrow_, this->mat_.row_offset, cast_conn->vec_, tuples);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        int iter = 0;
        while(!hdone)
        {
            hipMemcpy(max_tuples, tuples, sizeof(mis_tuple) * this->nrow_, hipMemcpyDeviceToDevice);

            if(avg_nnz_per_row <= 8)
            {
                kernel_csr_amg_find_max_mis_tuples_shared<256, 1>
                    <<<dim3((this->nrow_ * 1 - 1) / 256 + 1),
                       dim3(256),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             cast_conn->vec_,
                                                                             tuples,
                                                                             max_tuples,
                                                                             ddone);
            }
            else if(avg_nnz_per_row <= 16)
            {
                kernel_csr_amg_find_max_mis_tuples_shared<256, 2>
                    <<<dim3((this->nrow_ * 2 - 1) / 256 + 1),
                       dim3(256),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             cast_conn->vec_,
                                                                             tuples,
                                                                             max_tuples,
                                                                             ddone);
            }
            else if(avg_nnz_per_row <= 32)
            {
                kernel_csr_amg_find_max_mis_tuples_shared<256, 4>
                    <<<dim3((this->nrow_ * 4 - 1) / 256 + 1),
                       dim3(256),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             cast_conn->vec_,
                                                                             tuples,
                                                                             max_tuples,
                                                                             ddone);
            }
            else if(avg_nnz_per_row <= 64)
            {
                kernel_csr_amg_find_max_mis_tuples_shared<256, 8>
                    <<<dim3((this->nrow_ * 8 - 1) / 256 + 1),
                       dim3(256),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             cast_conn->vec_,
                                                                             tuples,
                                                                             max_tuples,
                                                                             ddone);
            }
            else if(avg_nnz_per_row <= 128)
            {
                kernel_csr_amg_find_max_mis_tuples_shared<256, 16>
                    <<<dim3((this->nrow_ * 16 - 1) / 256 + 1),
                       dim3(256),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             cast_conn->vec_,
                                                                             tuples,
                                                                             max_tuples,
                                                                             ddone);
            }
            else if(avg_nnz_per_row <= 256 || this->local_backend_.HIP_warp == 32)
            {
                kernel_csr_amg_find_max_mis_tuples_shared<256, 32>
                    <<<dim3((this->nrow_ * 32 - 1) / 256 + 1),
                       dim3(256),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             cast_conn->vec_,
                                                                             tuples,
                                                                             max_tuples,
                                                                             ddone);
            }
            else
            {
                kernel_csr_amg_find_max_mis_tuples_shared<256, 64>
                    <<<dim3((this->nrow_ * 64 - 1) / 256 + 1),
                       dim3(256),
                       0,
                       HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                             this->mat_.row_offset,
                                                                             this->mat_.col,
                                                                             cast_conn->vec_,
                                                                             tuples,
                                                                             max_tuples,
                                                                             ddone);
            }
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            kernel_csr_amg_update_mis_tuples<<<GridSize,
                                               BlockSize,
                                               0,
                                               HIPSTREAM(
                                                   this->local_backend_.HIP_stream_current)>>>(
                this->nrow_, max_tuples, tuples, cast_agg->vec_, ddone);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            copy_d2h(1, ddone, &hdone);

            iter++;

            if(iter > 10)
            {
                LOG_VERBOSE_INFO(2,
                                 "*** warning: HIPAcceleratorMatrixCSR::AMGPMISAggregate() Current "
                                 "number of iterations: "
                                     << iter);
            }
        }

        // exclusive scan on aggregates array
        size_t rocprim_size;
        char*  rocprim_buffer = NULL;

        rocprim::exclusive_scan(NULL,
                                rocprim_size,
                                cast_agg->vec_,
                                cast_agg->vec_,
                                0,
                                this->nrow_,
                                rocprim::plus<int>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);

        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                cast_agg->vec_,
                                cast_agg->vec_,
                                0,
                                this->nrow_,
                                rocprim::plus<int>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&rocprim_buffer);

        for(int k = 0; k < 2; k++)
        {
            hipMemcpy(max_tuples, tuples, sizeof(mis_tuple) * this->nrow_, hipMemcpyDeviceToDevice);

            kernel_csr_amg_update_aggregates<<<
                GridSize,
                BlockSize,
                0,
                HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                      this->mat_.row_offset,
                                                                      this->mat_.col,
                                                                      cast_conn->vec_,
                                                                      max_tuples,
                                                                      tuples,
                                                                      cast_agg->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        free_hip(&tuples);
        free_hip(&max_tuples);
        free_hip(&ddone);

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::AMGSmoothedAggregation(
        ValueType              relax,
        const BaseVector<int>& aggregates,
        const BaseVector<int>& connections,
        BaseMatrix<ValueType>* prolong,
        int                    lumping_strat) const
    {
        assert(prolong != NULL);

        const HIPAcceleratorVector<int>* cast_agg
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&aggregates);
        const HIPAcceleratorVector<int>* cast_conn
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&connections);
        HIPAcceleratorMatrixCSR<ValueType>* cast_prolong
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(prolong);

        assert(cast_agg != NULL);
        assert(cast_conn != NULL);
        assert(cast_prolong != NULL);

        PtrType*   prolong_row_offset = nullptr;
        int*       prolong_cols       = nullptr;
        ValueType* prolong_vals       = nullptr;

        // Allocate prolongation matrix row offset array
        allocate_hip(this->nrow_ + 1, &prolong_row_offset);

        // rocprim buffer
        size_t rocprim_size;
        char*  rocprim_buffer = NULL;

        // Obtain maximum entry in aggregation array
        rocprim::reduce(NULL,
                        rocprim_size,
                        cast_agg->vec_,
                        prolong_row_offset,
                        -2,
                        cast_agg->size_,
                        rocprim::maximum<int>(),
                        HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);

        rocprim::reduce(rocprim_buffer,
                        rocprim_size,
                        cast_agg->vec_,
                        prolong_row_offset,
                        -2,
                        cast_agg->size_,
                        rocprim::maximum<int>(),
                        HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&rocprim_buffer);
        rocprim_buffer = NULL;

        // Obtain sizes of P
        int prolong_ncol;
        int prolong_nrow = this->nrow_;
        copy_d2h(1, prolong_row_offset, &prolong_ncol);
        ++prolong_ncol;

        // Get maximum non-zeros per row to decided on size of unordered set
        kernel_calc_row_nnz<<<(this->nrow_ - 1) / 256 + 1,
                              256,
                              0,
                              HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
            this->nrow_, this->mat_.row_offset, prolong_row_offset + 1);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        rocprim::reduce(NULL,
                        rocprim_size,
                        prolong_row_offset + 1,
                        prolong_row_offset,
                        0,
                        this->nrow_,
                        rocprim::maximum<int>(),
                        HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);

        rocprim::reduce(rocprim_buffer,
                        rocprim_size,
                        prolong_row_offset + 1,
                        prolong_row_offset,
                        0,
                        this->nrow_,
                        rocprim::maximum<int>(),
                        HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&rocprim_buffer);
        rocprim_buffer = NULL;

        int max_row_nnz;
        copy_d2h(1, prolong_row_offset, &max_row_nnz);

        // Call corresponding kernel to compute non-zero entries per row of P
        if(max_row_nnz < 8)
        {
            kernel_csr_sa_prolong_nnz<256, 4, 8>
                <<<(this->nrow_ - 1) / (256 / 4) + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_conn->vec_,
                                                                         cast_agg->vec_,
                                                                         prolong_row_offset);
        }
        else if(max_row_nnz < 16)
        {
            kernel_csr_sa_prolong_nnz<256, 4, 16>
                <<<(this->nrow_ - 1) / (256 / 4) + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_conn->vec_,
                                                                         cast_agg->vec_,
                                                                         prolong_row_offset);
        }
        else if(max_row_nnz < 32)
        {
            kernel_csr_sa_prolong_nnz<256, 8, 32>
                <<<(this->nrow_ - 1) / (256 / 8) + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_conn->vec_,
                                                                         cast_agg->vec_,
                                                                         prolong_row_offset);
        }
        else if(max_row_nnz < 64)
        {
            kernel_csr_sa_prolong_nnz<256, 16, 64>
                <<<(this->nrow_ - 1) / (256 / 16) + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_conn->vec_,
                                                                         cast_agg->vec_,
                                                                         prolong_row_offset);
        }
        else if(max_row_nnz < 128)
        {
            kernel_csr_sa_prolong_nnz<256, 16, 128>
                <<<(this->nrow_ - 1) / (256 / 16) + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_conn->vec_,
                                                                         cast_agg->vec_,
                                                                         prolong_row_offset);
        }
        else if(max_row_nnz < 256)
        {
            kernel_csr_sa_prolong_nnz<256, 64, 256>
                <<<(this->nrow_ - 1) / (256 / 64) + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_conn->vec_,
                                                                         cast_agg->vec_,
                                                                         prolong_row_offset);
        }
        else if(max_row_nnz < 512)
        {
            kernel_csr_sa_prolong_nnz<256, 64, 512>
                <<<(this->nrow_ - 1) / (256 / 64) + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_conn->vec_,
                                                                         cast_agg->vec_,
                                                                         prolong_row_offset);
        }
        else if(max_row_nnz < 1024)
        {
            kernel_csr_sa_prolong_nnz<256, 64, 1024>
                <<<(this->nrow_ - 1) / (256 / 64) + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_conn->vec_,
                                                                         cast_agg->vec_,
                                                                         prolong_row_offset);
        }
        else if(max_row_nnz < 2048)
        {
            kernel_csr_sa_prolong_nnz<256, 64, 2048>
                <<<(this->nrow_ - 1) / (256 / 64) + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_conn->vec_,
                                                                         cast_agg->vec_,
                                                                         prolong_row_offset);
        }
        else if(max_row_nnz < 4096)
        {
            kernel_csr_sa_prolong_nnz<256, 64, 4096>
                <<<(this->nrow_ - 1) / (256 / 64) + 1,
                   256,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_conn->vec_,
                                                                         cast_agg->vec_,
                                                                         prolong_row_offset);
        }
        else if(max_row_nnz < 8192)
        {
            kernel_csr_sa_prolong_nnz<128, 64, 8192>
                <<<(this->nrow_ - 1) / (128 / 64) + 1,
                   128,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_conn->vec_,
                                                                         cast_agg->vec_,
                                                                         prolong_row_offset);
        }
        else if(max_row_nnz < 16384)
        {
            kernel_csr_sa_prolong_nnz<64, 64, 16384>
                <<<(this->nrow_ - 1) / (64 / 64) + 1,
                   64,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(this->nrow_,
                                                                         this->mat_.row_offset,
                                                                         this->mat_.col,
                                                                         cast_conn->vec_,
                                                                         cast_agg->vec_,
                                                                         prolong_row_offset);
        }
        else
        {
            // Fall back to host
            free_hip(&prolong_row_offset);

            return false;
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Obtain maximum row nnz to determine correct map size
        rocprim::reduce(NULL,
                        rocprim_size,
                        prolong_row_offset,
                        prolong_row_offset + this->nrow_,
                        0,
                        this->nrow_,
                        rocprim::maximum<int>(),
                        HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);

        rocprim::reduce(rocprim_buffer,
                        rocprim_size,
                        prolong_row_offset,
                        prolong_row_offset + this->nrow_,
                        0,
                        this->nrow_,
                        rocprim::maximum<int>(),
                        HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&rocprim_buffer);
        rocprim_buffer = NULL;

        copy_d2h(1, prolong_row_offset + this->nrow_, &max_row_nnz);

        // Perform exclusive scan on csr row pointer array
        rocprim::exclusive_scan(NULL,
                                rocprim_size,
                                prolong_row_offset,
                                prolong_row_offset,
                                0,
                                this->nrow_ + 1,
                                rocprim::plus<int>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);

        rocprim::exclusive_scan(rocprim_buffer,
                                rocprim_size,
                                prolong_row_offset,
                                prolong_row_offset,
                                0,
                                this->nrow_ + 1,
                                rocprim::plus<int>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&rocprim_buffer);
        rocprim_buffer = NULL;

        PtrType prolong_nnz;
        copy_d2h(1, prolong_row_offset + this->nrow_, &prolong_nnz);

        // Allocate prolongation matrix col indices and values arrays
        allocate_hip(prolong_nnz, &prolong_cols);
        allocate_hip(prolong_nnz, &prolong_vals);

        cast_prolong->Clear();
        cast_prolong->SetDataPtrCSR(&prolong_row_offset,
                                    &prolong_cols,
                                    &prolong_vals,
                                    prolong_nnz,
                                    prolong_nrow,
                                    prolong_ncol);

        // Call corresponding kernel to compute non-zero entries per row of P
        if(max_row_nnz < 8)
        {
            kernel_csr_sa_prolong_fill<128, 4, 8>
                <<<(this->nrow_ - 1) / (128 / 4) + 1,
                   128,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    relax,
                    lumping_strat,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    cast_conn->vec_,
                    cast_agg->vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val);
        }
        else if(max_row_nnz < 16)
        {
            kernel_csr_sa_prolong_fill<128, 8, 16>
                <<<(this->nrow_ - 1) / (128 / 8) + 1,
                   128,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    relax,
                    lumping_strat,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    cast_conn->vec_,
                    cast_agg->vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val);
        }
        else if(max_row_nnz < 32)
        {
            kernel_csr_sa_prolong_fill<128, 16, 32>
                <<<(this->nrow_ - 1) / (128 / 16) + 1,
                   128,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    relax,
                    lumping_strat,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    cast_conn->vec_,
                    cast_agg->vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val);
        }
        else if(max_row_nnz < 64)
        {
            kernel_csr_sa_prolong_fill<128, 32, 64>
                <<<(this->nrow_ - 1) / (128 / 32) + 1,
                   128,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    relax,
                    lumping_strat,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    cast_conn->vec_,
                    cast_agg->vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val);
        }
        else if(max_row_nnz < 128)
        {
            kernel_csr_sa_prolong_fill<128, 64, 128>
                <<<(this->nrow_ - 1) / (128 / 64) + 1,
                   128,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    relax,
                    lumping_strat,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    cast_conn->vec_,
                    cast_agg->vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val);
        }
        else if(max_row_nnz < 256)
        {
            kernel_csr_sa_prolong_fill<128, 64, 256>
                <<<(this->nrow_ - 1) / (128 / 64) + 1,
                   128,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    relax,
                    lumping_strat,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    cast_conn->vec_,
                    cast_agg->vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val);
        }
        else if(max_row_nnz < 512)
        {
            kernel_csr_sa_prolong_fill<128, 64, 512>
                <<<(this->nrow_ - 1) / (128 / 64) + 1,
                   128,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    relax,
                    lumping_strat,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    cast_conn->vec_,
                    cast_agg->vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val);
        }
        else if(max_row_nnz < 1024)
        {
            kernel_csr_sa_prolong_fill<128, 64, 1024>
                <<<(this->nrow_ - 1) / (128 / 64) + 1,
                   128,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    relax,
                    lumping_strat,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    cast_conn->vec_,
                    cast_agg->vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val);
        }
        else if(max_row_nnz < 2048)
        {
            kernel_csr_sa_prolong_fill<64, 64, 2048>
                <<<(this->nrow_ - 1) / (64 / 64) + 1,
                   64,
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_,
                    relax,
                    lumping_strat,
                    this->mat_.row_offset,
                    this->mat_.col,
                    this->mat_.val,
                    cast_conn->vec_,
                    cast_agg->vec_,
                    cast_prolong->mat_.row_offset,
                    cast_prolong->mat_.col,
                    cast_prolong->mat_.val);
        }
        else
        {
            // Fall back to host
            cast_prolong->Clear();

            return false;
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCSR<ValueType>::AMGAggregation(const BaseVector<int>& aggregates,
                                                            BaseMatrix<ValueType>* prolong) const
    {
        assert(prolong != NULL);

        const HIPAcceleratorVector<int>* cast_agg
            = dynamic_cast<const HIPAcceleratorVector<int>*>(&aggregates);
        HIPAcceleratorMatrixCSR<ValueType>* cast_prolong
            = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*>(prolong);

        assert(cast_agg != NULL);
        assert(cast_prolong != NULL);

        PtrType*   prolong_row_offset = nullptr;
        int*       prolong_cols       = nullptr;
        ValueType* prolong_vals       = nullptr;

        // Allocate prolongation matrix row offset array
        allocate_hip(this->nrow_ + 1, &prolong_row_offset);

        int* workspace = NULL;
        allocate_hip(256, &workspace);

        kernel_find_maximum_blockreduce<256>
            <<<dim3(256), dim3(256), 0, HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                cast_agg->size_, cast_agg->vec_, workspace);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        kernel_find_maximum_finalreduce<256>
            <<<dim3(1), dim3(256), 0, HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                workspace);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        int prolong_ncol = 0;
        int prolong_nrow = this->nrow_;
        copy_d2h(1, workspace, &prolong_ncol);
        free_hip(&workspace);

        kernel_csr_unsmoothed_prolong_nnz_per_row<256>
            <<<dim3((this->nrow_ - 1) / 256 + 1),
               dim3(256),
               0,
               HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                this->nrow_, cast_agg->vec_, prolong_row_offset);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Perform inclusive scan on csr row pointer array
        size_t rocprim_size;
        char*  rocprim_buffer = NULL;
        rocprim::inclusive_scan(nullptr,
                                rocprim_size,
                                prolong_row_offset,
                                prolong_row_offset,
                                this->nrow_ + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        allocate_hip(rocprim_size, &rocprim_buffer);

        rocprim::inclusive_scan(rocprim_buffer,
                                rocprim_size,
                                prolong_row_offset,
                                prolong_row_offset,
                                this->nrow_ + 1,
                                rocprim::plus<PtrType>(),
                                HIPSTREAM(this->local_backend_.HIP_stream_current));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        free_hip(&rocprim_buffer);

        PtrType prolong_nnz = 0;
        copy_d2h(1, prolong_row_offset + this->nrow_, &prolong_nnz);

        // Allocate prolongation matrix col indices and values arrays
        allocate_hip(prolong_nnz, &prolong_cols);
        allocate_hip(prolong_nnz, &prolong_vals);

        cast_prolong->Clear();
        cast_prolong->SetDataPtrCSR(&prolong_row_offset,
                                    &prolong_cols,
                                    &prolong_vals,
                                    prolong_nnz,
                                    prolong_nrow,
                                    prolong_ncol);

        if(prolong_nrow == prolong_nnz)
        {
            kernel_csr_unsmoothed_prolong_fill_simple<256>
                <<<dim3((this->nrow_ - 1) / 256 + 1),
                   dim3(256),
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_, cast_agg->vec_, prolong_cols, prolong_vals);
        }
        else
        {
            kernel_csr_unsmoothed_prolong_fill<256>
                <<<dim3((this->nrow_ - 1) / 256 + 1),
                   dim3(256),
                   0,
                   HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                    this->nrow_, cast_agg->vec_, prolong_row_offset, prolong_cols, prolong_vals);
        }
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // cast_prolong->Sort();

        return true;
    }

    template class HIPAcceleratorMatrixCSR<double>;
    template class HIPAcceleratorMatrixCSR<float>;
#ifdef SUPPORT_COMPLEX
    template class HIPAcceleratorMatrixCSR<std::complex<double>>;
    template class HIPAcceleratorMatrixCSR<std::complex<float>>;
#endif

} // namespace rocalution
