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

#include "host_matrix_csr.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"
#include "../matrix_formats_ind.hpp"
#include "host_conversion.hpp"
#include "host_io.hpp"
#include "host_matrix_bcsr.hpp"
#include "host_matrix_coo.hpp"
#include "host_matrix_dense.hpp"
#include "host_matrix_dia.hpp"
#include "host_matrix_ell.hpp"
#include "host_matrix_hyb.hpp"
#include "host_matrix_mcsr.hpp"
#include "host_vector.hpp"
#include "rocalution/utils/types.hpp"

#include <algorithm>
#include <complex>
#include <limits>
#include <map>
#include <math.h>
#include <numeric>
#include <string.h>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_set_nested(num) ;
#endif

namespace rocalution
{

    template <typename ValueType>
    HostMatrixCSR<ValueType>::HostMatrixCSR()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HostMatrixCSR<ValueType>::HostMatrixCSR(const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this, "HostMatrixCSR::HostMatrixCSR()", "constructor with local_backend");

        this->mat_.row_offset = NULL;
        this->mat_.col        = NULL;
        this->mat_.val        = NULL;
        this->set_backend(local_backend);

        this->L_diag_unit_ = false;
        this->U_diag_unit_ = false;
    }

    template <typename ValueType>
    HostMatrixCSR<ValueType>::~HostMatrixCSR()
    {
        log_debug(this, "HostMatrixCSR::~HostMatrixCSR()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::Clear(void)
    {
        free_host(&this->mat_.row_offset);
        free_host(&this->mat_.col);
        free_host(&this->mat_.val);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Zeros(void)
    {
        set_to_zero_host(this->nnz_, mat_.val);

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::Info(void) const
    {
        LOG_INFO(
            "HostMatrixCSR<ValueType>, OpenMP threads: " << this->local_backend_.OpenMP_threads);
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Check(void) const
    {
        bool sorted = true;

        if(this->nnz_ > 0)
        {
            assert(this->nrow_ > 0);
            assert(this->ncol_ > 0);

            assert(this->mat_.row_offset != NULL);
            assert(this->mat_.val != NULL);
            assert(this->mat_.col != NULL);

            // check nnz
            if((std::abs(this->nnz_) == std::numeric_limits<int>::infinity()) || // inf
               (this->nnz_ != this->nnz_))
            { // NaN
                LOG_VERBOSE_INFO(2, "*** error: Matrix CSR:Check - problems with matrix nnz");
                return false;
            }

            // nrow
            if((std::abs(this->nrow_) == std::numeric_limits<int>::infinity()) || // inf
               (this->nrow_ != this->nrow_))
            { // NaN
                LOG_VERBOSE_INFO(2, "*** error: Matrix CSR:Check - problems with matrix nrow");
                return false;
            }

            // ncol
            if((std::abs(this->ncol_) == std::numeric_limits<int>::infinity()) || // inf
               (this->ncol_ != this->ncol_))
            { // NaN
                LOG_VERBOSE_INFO(2, "*** error: Matrix CSR:Check - problems with matrix ncol");
                return false;
            }

            for(int ai = 0; ai < this->nrow_ + 1; ++ai)
            {
                int row = this->mat_.row_offset[ai];
                if((row < 0) || (row > this->nnz_))
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** error: Matrix CSR:Check - problems with matrix row offset pointers");
                    return false;
                }
            }

            for(int ai = 0; ai < this->nrow_; ++ai)
            {
                int s = this->mat_.col[this->mat_.row_offset[ai]];

                for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
                {
                    int col = this->mat_.col[aj];

                    if((col < 0) || (col > this->ncol_))
                    {
                        LOG_VERBOSE_INFO(
                            2, "*** error: Matrix CSR:Check - problems with matrix col values");
                        return false;
                    }

                    ValueType val = this->mat_.val[aj];
                    if((val == std::numeric_limits<ValueType>::infinity()) || (val != val))
                    {
                        LOG_VERBOSE_INFO(
                            2, "*** error: Matrix CSR:Check - problems with matrix values");
                        return false;
                    }

                    if((aj > this->mat_.row_offset[ai]) && (s >= col))
                    {
                        sorted = false;
                    }

                    s = this->mat_.col[aj];
                }
            }
        }
        else
        {
            assert(this->nnz_ == 0);
            assert(this->nrow_ >= 0);
            assert(this->ncol_ >= 0);

            if(this->nrow_ == 0 && this->ncol_ == 0)
            {
                assert(this->mat_.val == NULL);
                assert(this->mat_.col == NULL);
            }
        }

        if(sorted == false)
        {
            LOG_VERBOSE_INFO(2,
                             "*** warning: Matrix CSR:Check - the matrix has not sorted columns");
        }

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::AllocateCSR(int nnz, int nrow, int ncol)
    {
        assert(nnz >= 0);
        assert(ncol >= 0);
        assert(nrow >= 0);

        this->Clear();

        allocate_host(nrow + 1, &this->mat_.row_offset);
        allocate_host(nnz, &this->mat_.col);
        allocate_host(nnz, &this->mat_.val);

        set_to_zero_host(nrow + 1, mat_.row_offset);
        set_to_zero_host(nnz, mat_.col);
        set_to_zero_host(nnz, mat_.val);

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::SetDataPtrCSR(
        int** row_offset, int** col, ValueType** val, int nnz, int nrow, int ncol)
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

        this->mat_.row_offset = *row_offset;
        this->mat_.col        = *col;
        this->mat_.val        = *val;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LeaveDataPtrCSR(int** row_offset, int** col, ValueType** val)
    {
        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);
        assert(this->nnz_ >= 0);

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
    void HostMatrixCSR<ValueType>::CopyFromCSR(const int*       row_offsets,
                                               const int*       col,
                                               const ValueType* val)
    {
        assert(row_offsets != NULL);

        copy_h2h(this->nrow_ + 1, row_offsets, this->mat_.row_offset);

        if(this->nnz_ > 0)
        {
            assert(this->nrow_ > 0);
            assert(this->ncol_ > 0);
            assert(col != NULL);
            assert(val != NULL);

            copy_h2h(this->nnz_, col, this->mat_.col);
            copy_h2h(this->nnz_, val, this->mat_.val);
        }
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::CopyToCSR(int* row_offsets, int* col, ValueType* val) const
    {
        assert(row_offsets != NULL);

        copy_h2h(this->nrow_ + 1, this->mat_.row_offset, row_offsets);

        if(this->nnz_ > 0)
        {
            assert(this->nrow_ > 0);
            assert(this->ncol_ > 0);
            assert(col != NULL);
            assert(val != NULL);

            copy_h2h(this->nnz_, this->mat_.col, col);
            copy_h2h(this->nnz_, this->mat_.val, val);
        }
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType>& mat)
    {
        // copy only in the same format
        assert(this->GetMatFormat() == mat.GetMatFormat());
        assert(this->GetMatBlockDimension() == mat.GetMatBlockDimension());

        if(const HostMatrixCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
        {
            if(this->nnz_ == 0)
            {
                this->AllocateCSR(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            // Copy only if initialized
            if(cast_mat->mat_.row_offset != NULL)
            {
                copy_h2h(this->nrow_ + 1, cast_mat->mat_.row_offset, this->mat_.row_offset);
            }

            copy_h2h(this->nnz_, cast_mat->mat_.col, this->mat_.col);
            copy_h2h(this->nnz_, cast_mat->mat_.val, this->mat_.val);
        }
        else
        {
            // Host matrix knows only host matrices
            // -> dispatching
            mat.CopyTo(this);
        }
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::CopyTo(BaseMatrix<ValueType>* mat) const
    {
        mat->CopyFrom(*this);
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ReadFileCSR(const std::string& filename)
    {
        int nrow;
        int ncol;
        int nnz;

        int*       ptr = NULL;
        int*       col = NULL;
        ValueType* val = NULL;

        if(read_matrix_csr(nrow, ncol, nnz, &ptr, &col, &val, filename.c_str()) != true)
        {
            return false;
        }

        this->Clear();
        this->SetDataPtrCSR(&ptr, &col, &val, nnz, nrow, ncol);

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::CopyFromHostCSR(
        const int* row_offset, const int* col, const ValueType* val, int nnz, int nrow, int ncol)
    {
        assert(nnz >= 0);
        assert(ncol >= 0);
        assert(nrow >= 0);
        assert(row_offset != NULL);

        // Allocate matrix
        this->Clear();

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;

        allocate_host(nrow + 1, &this->mat_.row_offset);

        copy_h2h(this->nrow_ + 1, row_offset, this->mat_.row_offset);

        if(nnz > 0)
        {
            assert(col != NULL);
            assert(val != NULL);
        }

        allocate_host(nnz, &this->mat_.col);
        allocate_host(nnz, &this->mat_.val);

        copy_h2h(this->nnz_, col, this->mat_.col);
        copy_h2h(this->nnz_, val, this->mat_.val);
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::WriteFileCSR(const std::string& filename) const
    {
        if(write_matrix_csr(this->nrow_,
                            this->ncol_,
                            this->nnz_,
                            this->mat_.row_offset,
                            this->mat_.col,
                            this->mat_.val,
                            filename.c_str())
           != true)
        {
            return false;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // empty matrix is empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateCSR(mat.GetNnz(), mat.GetM(), mat.GetN());

            return true;
        }

        if(const HostMatrixCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
        {
            this->CopyFrom(*cast_mat);
            return true;
        }

        if(const HostMatrixBCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixBCSR<ValueType>*>(&mat))
        {
            this->Clear();

            int nrow = cast_mat->mat_.nrowb * cast_mat->mat_.blockdim;
            int ncol = cast_mat->mat_.ncolb * cast_mat->mat_.blockdim;
            int nnz  = cast_mat->mat_.nnzb * cast_mat->mat_.blockdim * cast_mat->mat_.blockdim;

            if(bcsr_to_csr(this->local_backend_.OpenMP_threads,
                           nnz,
                           nrow,
                           ncol,
                           cast_mat->mat_,
                           &this->mat_)
               == true)
            {
                this->nrow_ = nrow;
                this->ncol_ = ncol;
                this->nnz_  = nnz;

                return true;
            }
        }

        if(const HostMatrixCOO<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCOO<ValueType>*>(&mat))
        {
            this->Clear();

            if(coo_to_csr(this->local_backend_.OpenMP_threads,
                          cast_mat->nnz_,
                          cast_mat->nrow_,
                          cast_mat->ncol_,
                          cast_mat->mat_,
                          &this->mat_)
               == true)
            {
                this->nrow_ = cast_mat->nrow_;
                this->ncol_ = cast_mat->ncol_;
                this->nnz_  = cast_mat->nnz_;

                return true;
            }
        }

        if(const HostMatrixDENSE<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixDENSE<ValueType>*>(&mat))
        {
            this->Clear();
            int nnz = 0;

            if(dense_to_csr(this->local_backend_.OpenMP_threads,
                            cast_mat->nrow_,
                            cast_mat->ncol_,
                            cast_mat->mat_,
                            &this->mat_,
                            &nnz)
               == true)
            {
                this->nrow_ = cast_mat->nrow_;
                this->ncol_ = cast_mat->ncol_;
                this->nnz_  = nnz;

                return true;
            }
        }

        if(const HostMatrixDIA<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixDIA<ValueType>*>(&mat))
        {
            this->Clear();
            int nnz;

            if(dia_to_csr(this->local_backend_.OpenMP_threads,
                          cast_mat->nnz_,
                          cast_mat->nrow_,
                          cast_mat->ncol_,
                          cast_mat->mat_,
                          &this->mat_,
                          &nnz)
               == true)
            {
                this->nrow_ = cast_mat->nrow_;
                this->ncol_ = cast_mat->ncol_;
                this->nnz_  = nnz;

                return true;
            }
        }

        if(const HostMatrixELL<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixELL<ValueType>*>(&mat))
        {
            this->Clear();
            int nnz;

            if(ell_to_csr(this->local_backend_.OpenMP_threads,
                          cast_mat->nnz_,
                          cast_mat->nrow_,
                          cast_mat->ncol_,
                          cast_mat->mat_,
                          &this->mat_,
                          &nnz)
               == true)
            {
                this->nrow_ = cast_mat->nrow_;
                this->ncol_ = cast_mat->ncol_;
                this->nnz_  = nnz;

                return true;
            }
        }

        if(const HostMatrixMCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixMCSR<ValueType>*>(&mat))
        {
            this->Clear();

            if(mcsr_to_csr(this->local_backend_.OpenMP_threads,
                           cast_mat->nnz_,
                           cast_mat->nrow_,
                           cast_mat->ncol_,
                           cast_mat->mat_,
                           &this->mat_)
               == true)
            {
                this->nrow_ = cast_mat->nrow_;
                this->ncol_ = cast_mat->ncol_;
                this->nnz_  = cast_mat->nnz_;

                return true;
            }
        }

        if(const HostMatrixHYB<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixHYB<ValueType>*>(&mat))
        {
            this->Clear();
            int nnz;

            if(hyb_to_csr(this->local_backend_.OpenMP_threads,
                          cast_mat->nnz_,
                          cast_mat->nrow_,
                          cast_mat->ncol_,
                          cast_mat->ell_nnz_,
                          cast_mat->coo_nnz_,
                          cast_mat->mat_,
                          &this->mat_,
                          &nnz)
               == true)
            {
                this->nrow_ = cast_mat->nrow_;
                this->ncol_ = cast_mat->ncol_;
                this->nnz_  = nnz;

                return true;
            }
        }

        return false;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::Apply(const BaseVector<ValueType>& in,
                                         BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
        HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            ValueType sum     = static_cast<ValueType>(0);
            int       row_beg = this->mat_.row_offset[ai];
            int       row_end = this->mat_.row_offset[ai + 1];

            for(int aj = row_beg; aj < row_end; ++aj)
            {
                sum += this->mat_.val[aj] * cast_in->vec_[this->mat_.col[aj]];
            }

            cast_out->vec_[ai] = sum;
        }
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
                                            ValueType                    scalar,
                                            BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(in.GetSize() >= 0);
            assert(out->GetSize() >= 0);
            assert(in.GetSize() == this->ncol_);
            assert(out->GetSize() == this->nrow_);

            const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
            HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int ai = 0; ai < this->nrow_; ++ai)
            {
                for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
                {
                    cast_out->vec_[ai]
                        += scalar * this->mat_.val[aj] * cast_in->vec_[this->mat_.col[aj]];
                }
            }
        }
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractDiagonal(BaseVector<ValueType>* vec_diag) const
    {
        assert(vec_diag != NULL);
        assert(vec_diag->GetSize() == this->nrow_);

        HostVector<ValueType>* cast_vec_diag = dynamic_cast<HostVector<ValueType>*>(vec_diag);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai == this->mat_.col[aj])
                {
                    cast_vec_diag->vec_[ai] = this->mat_.val[aj];
                    break;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractInverseDiagonal(BaseVector<ValueType>* vec_inv_diag) const
    {
        assert(vec_inv_diag != NULL);
        assert(vec_inv_diag->GetSize() == this->nrow_);

        HostVector<ValueType>* cast_vec_inv_diag
            = dynamic_cast<HostVector<ValueType>*>(vec_inv_diag);

        int detect_zero_diag = 0;

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai == this->mat_.col[aj])
                {
                    if(this->mat_.val[aj] != static_cast<ValueType>(0))
                    {
                        cast_vec_inv_diag->vec_[ai]
                            = static_cast<ValueType>(1) / this->mat_.val[aj];
                    }
                    else
                    {
                        cast_vec_inv_diag->vec_[ai] = static_cast<ValueType>(1);
                        detect_zero_diag            = 1;
                    }

                    break;
                }
            }
        }

        if(detect_zero_diag == 1)
        {
            LOG_VERBOSE_INFO(
                2,
                "*** warning: in HostMatrixCSR::ExtractInverseDiagonal() a zero has been detected "
                "on the diagonal. It has been replaced with one to avoid inf");
        }
        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractSubMatrix(int                    row_offset,
                                                    int                    col_offset,
                                                    int                    row_size,
                                                    int                    col_size,
                                                    BaseMatrix<ValueType>* mat) const
    {
        assert(mat != NULL);

        assert(row_offset >= 0);
        assert(col_offset >= 0);

        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);

        HostMatrixCSR<ValueType>* cast_mat = dynamic_cast<HostMatrixCSR<ValueType>*>(mat);
        assert(cast_mat != NULL);

        int mat_nnz = 0;

        // use omp in local_matrix (higher level)

        //  _set_omp_backend_threads(this->local_backend_, this->nrow_);

        //#ifdef _OPENMP
        //#pragma omp parallel for reduction(+:mat_nnz)
        //#endif
        for(int ai = row_offset; ai < row_offset + row_size; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if((this->mat_.col[aj] >= col_offset)
                   && (this->mat_.col[aj] < col_offset + col_size))
                {
                    ++mat_nnz;
                }
            }
        }

        cast_mat->AllocateCSR(mat_nnz, row_size, col_size);

        // not empty submatrix
        if(mat_nnz > 0)
        {
            cast_mat->AllocateCSR(mat_nnz, row_size, col_size);

            int mat_row_offset           = 0;
            cast_mat->mat_.row_offset[0] = mat_row_offset;

            for(int ai = row_offset; ai < row_offset + row_size; ++ai)
            {
                for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
                {
                    if((this->mat_.col[aj] >= col_offset)
                       && (this->mat_.col[aj] < col_offset + col_size))
                    {
                        cast_mat->mat_.col[mat_row_offset] = this->mat_.col[aj] - col_offset;
                        cast_mat->mat_.val[mat_row_offset] = this->mat_.val[aj];
                        ++mat_row_offset;
                    }
                }

                cast_mat->mat_.row_offset[ai - row_offset + 1] = mat_row_offset;
            }

            cast_mat->mat_.row_offset[row_size] = mat_row_offset;
            assert(mat_row_offset == mat_nnz);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractU(BaseMatrix<ValueType>* U) const
    {
        assert(U != NULL);

        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        HostMatrixCSR<ValueType>* cast_U = dynamic_cast<HostMatrixCSR<ValueType>*>(U);

        assert(cast_U != NULL);

        // count nnz of upper triangular part
        int nnz_U = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : nnz_U)
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] > ai)
                {
                    ++nnz_U;
                }
            }
        }

        // allocate upper triangular part structure
        int*       row_offset = NULL;
        int*       col        = NULL;
        ValueType* val        = NULL;

        allocate_host(this->nrow_ + 1, &row_offset);
        allocate_host(nnz_U, &col);
        allocate_host(nnz_U, &val);

        // fill upper triangular part
        int nnz       = 0;
        row_offset[0] = 0;
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] > ai)
                {
                    col[nnz] = this->mat_.col[aj];
                    val[nnz] = this->mat_.val[aj];
                    ++nnz;
                }
            }

            row_offset[ai + 1] = nnz;
        }

        cast_U->Clear();
        cast_U->SetDataPtrCSR(&row_offset, &col, &val, nnz_U, this->nrow_, this->ncol_);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractUDiagonal(BaseMatrix<ValueType>* U) const
    {
        assert(U != NULL);

        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        HostMatrixCSR<ValueType>* cast_U = dynamic_cast<HostMatrixCSR<ValueType>*>(U);

        assert(cast_U != NULL);

        // count nnz of upper triangular part
        int nnz_U = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : nnz_U)
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] >= ai)
                {
                    ++nnz_U;
                }
            }
        }

        // allocate upper triangular part structure
        int*       row_offset = NULL;
        int*       col        = NULL;
        ValueType* val        = NULL;

        allocate_host(this->nrow_ + 1, &row_offset);
        allocate_host(nnz_U, &col);
        allocate_host(nnz_U, &val);

        // fill upper triangular part
        int nnz       = 0;
        row_offset[0] = 0;
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] >= ai)
                {
                    col[nnz] = this->mat_.col[aj];
                    val[nnz] = this->mat_.val[aj];
                    ++nnz;
                }
            }

            row_offset[ai + 1] = nnz;
        }

        cast_U->Clear();
        cast_U->SetDataPtrCSR(&row_offset, &col, &val, nnz_U, this->nrow_, this->ncol_);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractL(BaseMatrix<ValueType>* L) const
    {
        assert(L != NULL);

        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        HostMatrixCSR<ValueType>* cast_L = dynamic_cast<HostMatrixCSR<ValueType>*>(L);

        assert(cast_L != NULL);

        // count nnz of lower triangular part
        int nnz_L = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : nnz_L)
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] < ai)
                {
                    ++nnz_L;
                }
            }
        }

        // allocate lower triangular part structure
        int*       row_offset = NULL;
        int*       col        = NULL;
        ValueType* val        = NULL;

        allocate_host(this->nrow_ + 1, &row_offset);
        allocate_host(nnz_L, &col);
        allocate_host(nnz_L, &val);

        // fill lower triangular part
        int nnz       = 0;
        row_offset[0] = 0;
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] < ai)
                {
                    col[nnz] = this->mat_.col[aj];
                    val[nnz] = this->mat_.val[aj];
                    ++nnz;
                }
            }

            row_offset[ai + 1] = nnz;
        }

        cast_L->Clear();
        cast_L->SetDataPtrCSR(&row_offset, &col, &val, nnz_L, this->nrow_, this->ncol_);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractLDiagonal(BaseMatrix<ValueType>* L) const
    {
        assert(L != NULL);

        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        HostMatrixCSR<ValueType>* cast_L = dynamic_cast<HostMatrixCSR<ValueType>*>(L);

        assert(cast_L != NULL);

        // count nnz of lower triangular part
        int nnz_L = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : nnz_L)
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] <= ai)
                {
                    ++nnz_L;
                }
            }
        }

        // allocate lower triangular part structure
        int*       row_offset = NULL;
        int*       col        = NULL;
        ValueType* val        = NULL;

        allocate_host(this->nrow_ + 1, &row_offset);
        allocate_host(nnz_L, &col);
        allocate_host(nnz_L, &val);

        // fill lower triangular part
        int nnz       = 0;
        row_offset[0] = 0;
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] <= ai)
                {
                    col[nnz] = this->mat_.col[aj];
                    val[nnz] = this->mat_.val[aj];
                    ++nnz;
                }
            }

            row_offset[ai + 1] = nnz;
        }

        cast_L->Clear();
        cast_L->SetDataPtrCSR(&row_offset, &col, &val, nnz_L, this->nrow_, this->ncol_);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::LUSolve(const BaseVector<ValueType>& in,
                                           BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
        HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        // Solve L
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            cast_out->vec_[ai] = cast_in->vec_[ai];

            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] < ai)
                {
                    // under the diagonal
                    cast_out->vec_[ai] -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
                }
                else
                {
                    // CSR should be sorted
                    break;
                }
            }
        }

        // last elements should be the diagonal one (last)
        int diag_aj = this->nnz_ - 1;

        // Solve U
        for(int ai = this->nrow_ - 1; ai >= 0; --ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] > ai)
                {
                    // above the diagonal
                    cast_out->vec_[ai] -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
                }

                if(this->mat_.col[aj] == ai)
                {
                    diag_aj = aj;
                }
            }

            cast_out->vec_[ai] /= this->mat_.val[diag_aj];
        }

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LLAnalyse(void)
    {
        // do nothing
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LLAnalyseClear(void)
    {
        // do nothing
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LUAnalyse(void)
    {
        // do nothing
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LUAnalyseClear(void)
    {
        // do nothing
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::LLSolve(const BaseVector<ValueType>& in,
                                           BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
        HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        // Solve L
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            ValueType value    = cast_in->vec_[ai];
            int       diag_idx = this->mat_.row_offset[ai + 1] - 1;

            for(int aj = this->mat_.row_offset[ai]; aj < diag_idx; ++aj)
            {
                value -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
            }

            cast_out->vec_[ai] = value / this->mat_.val[diag_idx];
        }

        // Solve L^T
        for(int ai = this->nrow_ - 1; ai >= 0; --ai)
        {
            int       diag_idx = this->mat_.row_offset[ai + 1] - 1;
            ValueType value    = cast_out->vec_[ai] / this->mat_.val[diag_idx];

            for(int aj = this->mat_.row_offset[ai]; aj < diag_idx; ++aj)
            {
                cast_out->vec_[this->mat_.col[aj]] -= value * this->mat_.val[aj];
            }

            cast_out->vec_[ai] = value;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::LLSolve(const BaseVector<ValueType>& in,
                                           const BaseVector<ValueType>& inv_diag,
                                           BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);
        assert(inv_diag.GetSize() == this->nrow_ || inv_diag.GetSize() == this->ncol_);

        const HostVector<ValueType>* cast_in = dynamic_cast<const HostVector<ValueType>*>(&in);
        const HostVector<ValueType>* cast_diag
            = dynamic_cast<const HostVector<ValueType>*>(&inv_diag);
        HostVector<ValueType>* cast_out = dynamic_cast<HostVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        // Solve L
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            ValueType value    = cast_in->vec_[ai];
            int       diag_idx = this->mat_.row_offset[ai + 1] - 1;

            for(int aj = this->mat_.row_offset[ai]; aj < diag_idx; ++aj)
            {
                value -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
            }

            cast_out->vec_[ai] = value * cast_diag->vec_[ai];
        }

        // Solve L^T
        for(int ai = this->nrow_ - 1; ai >= 0; --ai)
        {
            int       diag_idx = this->mat_.row_offset[ai + 1] - 1;
            ValueType value    = cast_out->vec_[ai] * cast_diag->vec_[ai];

            for(int aj = this->mat_.row_offset[ai]; aj < diag_idx; ++aj)
            {
                cast_out->vec_[this->mat_.col[aj]] -= value * this->mat_.val[aj];
            }

            cast_out->vec_[ai] = value;
        }

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LAnalyse(bool diag_unit)
    {
        this->L_diag_unit_ = diag_unit;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LAnalyseClear(void)
    {
        // do nothing
        this->L_diag_unit_ = true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::LSolve(const BaseVector<ValueType>& in,
                                          BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
        HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        int diag_aj = 0;

        // Solve L
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            cast_out->vec_[ai] = cast_in->vec_[ai];

            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] < ai)
                {
                    // under the diagonal
                    cast_out->vec_[ai] -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
                }
                else
                {
                    // CSR should be sorted
                    if(this->L_diag_unit_ == false)
                    {
                        assert(this->mat_.col[aj] == ai);
                        diag_aj = aj;
                    }
                    break;
                }
            }

            if(this->L_diag_unit_ == false)
            {
                cast_out->vec_[ai] /= this->mat_.val[diag_aj];
            }
        }

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::UAnalyse(bool diag_unit)
    {
        this->U_diag_unit_ = diag_unit;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::UAnalyseClear(void)
    {
        // do nothing
        this->U_diag_unit_ = false;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::USolve(const BaseVector<ValueType>& in,
                                          BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
        HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        // last elements should the diagonal one (last)
        int diag_aj = this->nnz_ - 1;

        // Solve U
        for(int ai = this->nrow_ - 1; ai >= 0; --ai)
        {
            cast_out->vec_[ai] = cast_in->vec_[ai];

            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] > ai)
                {
                    // above the diagonal
                    cast_out->vec_[ai] -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
                }

                if(this->L_diag_unit_ == false)
                {
                    if(this->mat_.col[aj] == ai)
                    {
                        diag_aj = aj;
                    }
                }
            }

            if(this->L_diag_unit_ == false)
            {
                cast_out->vec_[ai] /= this->mat_.val[diag_aj];
            }
        }

        return true;
    }

    // Algorithm for ILU factorization is based on
    // Y. Saad, Iterative methods for sparse linear systems, 2nd edition, SIAM
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ILU0Factorize(void)
    {
        assert(this->nrow_ == this->ncol_);
        assert(this->nnz_ > 0);

        // pointer of upper part of each row
        int* diag_offset = NULL;
        int* nnz_entries = NULL;

        allocate_host(this->nrow_, &diag_offset);
        allocate_host(this->nrow_, &nnz_entries);

        set_to_zero_host(this->nrow_, nnz_entries);

        // ai = 0 to N loop over all rows
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            // ai-th row entries
            int row_start = this->mat_.row_offset[ai];
            int row_end   = this->mat_.row_offset[ai + 1];
            int j;

            // nnz position of ai-th row in mat_.val array
            for(j = row_start; j < row_end; ++j)
            {
                nnz_entries[this->mat_.col[j]] = j;
            }

            // loop over ai-th row nnz entries
            for(j = row_start; j < row_end; ++j)
            {
                // if nnz entry is in lower matrix
                if(this->mat_.col[j] < ai)
                {
                    int col_j  = this->mat_.col[j];
                    int diag_j = diag_offset[col_j];

                    if(this->mat_.val[diag_j] != static_cast<ValueType>(0))
                    {
                        // multiplication factor
                        this->mat_.val[j] = this->mat_.val[j] / this->mat_.val[diag_j];

                        // loop over upper offset pointer and do linear combination for nnz entry
                        for(int k = diag_j + 1; k < this->mat_.row_offset[col_j + 1]; ++k)
                        {
                            // if nnz at this position do linear combination
                            if(nnz_entries[this->mat_.col[k]] != 0)
                            {
                                this->mat_.val[nnz_entries[this->mat_.col[k]]]
                                    -= this->mat_.val[j] * this->mat_.val[k];
                            }
                        }
                    }
                }
                else
                {
                    break;
                }
            }

            // set diagonal pointer to diagonal element
            diag_offset[ai] = j;

            // clear nnz entries
            for(j = row_start; j < row_end; ++j)
            {
                nnz_entries[this->mat_.col[j]] = 0;
            }
        }

        free_host(&diag_offset);
        free_host(&nnz_entries);

        return true;
    }

    // Algorithm for ILUT factorization is based on
    // Y. Saad, Iterative methods for sparse linear systems, 2nd edition, SIAM
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ILUTFactorize(double t, int maxrow)
    {
        assert(this->nrow_ == this->ncol_);
        assert(this->nnz_ > 0);

        int nrow = this->nrow_;
        int ncol = this->ncol_;

        int*       row_offset  = NULL;
        int*       diag_offset = NULL;
        int*       nnz_entries = NULL;
        bool*      nnz_pos     = (bool*)malloc(nrow * sizeof(bool));
        ValueType* w           = NULL;

        allocate_host(nrow + 1, &row_offset);
        allocate_host(nrow, &diag_offset);
        allocate_host(nrow, &nnz_entries);
        allocate_host(nrow, &w);

        for(int i = 0; i < nrow; ++i)
        {
            w[i]           = 0.0;
            nnz_entries[i] = -1;
            nnz_pos[i]     = false;
            diag_offset[i] = 0;
        }

        // pre-allocate 1.5x nnz arrays for preconditioner matrix
        float      nnzA       = static_cast<float>(this->nnz_);
        size_t     alloc_size = static_cast<size_t>(nnzA * 1.5f);
        int*       col        = (int*)malloc(alloc_size * sizeof(int));
        ValueType* val        = (ValueType*)malloc(alloc_size * sizeof(ValueType));

        // initialize row_offset
        row_offset[0] = 0;
        int nnz       = 0;

        // loop over all rows
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            row_offset[ai + 1] = row_offset[ai];

            int    row_begin = this->mat_.row_offset[ai];
            int    row_end   = this->mat_.row_offset[ai + 1];
            double row_norm  = 0.0;

            // fill working array with ai-th row
            int m = 0;
            for(int aj = row_begin; aj < row_end; ++aj)
            {
                int idx        = this->mat_.col[aj];
                w[idx]         = this->mat_.val[aj];
                nnz_entries[m] = idx;
                nnz_pos[idx]   = true;

                row_norm += std::abs(this->mat_.val[aj]);
                ++m;
            }

            // threshold for dropping strategy
            double threshold = t * row_norm / (row_end - row_begin);

            for(int k = 0; k < nrow; ++k)
            {
                if(nnz_entries[k] == -1)
                {
                    break;
                }

                int aj = nnz_entries[k];

                // get smallest column index
                int sidx = k;
                for(int j = k + 1; j < nrow; ++j)
                {
                    if(nnz_entries[j] == -1)
                    {
                        break;
                    }

                    if(nnz_entries[j] < aj)
                    {
                        sidx = j;
                        aj   = nnz_entries[j];
                    }
                }

                aj = nnz_entries[k];

                // swap column index
                if(k != sidx)
                {
                    nnz_entries[k]    = nnz_entries[sidx];
                    nnz_entries[sidx] = aj;
                    aj                = nnz_entries[k];
                }

                // lower matrix part
                if(aj < ai)
                {
                    // if zero diagonal entry do nothing
                    if(val[diag_offset[aj]] == static_cast<ValueType>(0))
                    {
                        LOG_INFO("(ILUT) zero row");
                        continue;
                    }

                    w[aj] /= val[diag_offset[aj]];

                    // do linear combination with previous row
                    for(int l = diag_offset[aj] + 1; l < row_offset[aj + 1]; ++l)
                    {
                        int       idx    = col[l];
                        ValueType fillin = w[aj] * val[l];

                        // drop off strategy for fill in
                        if(nnz_pos[idx] == false)
                        {
                            if(std::abs(fillin) >= threshold)
                            {
                                nnz_entries[m] = idx;
                                nnz_pos[idx]   = true;
                                w[idx] -= fillin;
                                ++m;
                            }
                        }
                        else
                        {
                            w[idx] -= fillin;
                        }
                    }
                }
            }

            // fill ai-th row of preconditioner matrix
            for(int k = 0, num_lower = 0, num_upper = 0; k < nrow; ++k)
            {
                int aj = nnz_entries[k];

                if(aj == -1)
                {
                    break;
                }

                if(nnz_pos[aj] == false)
                {
                    break;
                }

                // lower part
                if(aj < ai && num_lower < maxrow)
                {
                    val[nnz] = w[aj];
                    col[nnz] = aj;

                    ++row_offset[ai + 1];
                    ++num_lower;
                    ++nnz;

                    // upper part
                }
                else if(aj > ai && num_upper < maxrow)
                {
                    val[nnz] = w[aj];
                    col[nnz] = aj;

                    ++row_offset[ai + 1];
                    ++num_upper;
                    ++nnz;

                    // diagonal part
                }
                else if(aj == ai)
                {
                    val[nnz] = w[aj];
                    col[nnz] = aj;

                    diag_offset[ai] = row_offset[ai + 1];

                    ++row_offset[ai + 1];
                    ++nnz;
                }

                // clear working arrays
                w[aj]          = static_cast<ValueType>(0);
                nnz_entries[k] = -1;
                nnz_pos[aj]    = false;
            }

            // resize preconditioner matrix if needed
            if(alloc_size < static_cast<size_t>(nnz + 2 * maxrow + 1))
            {
                alloc_size += static_cast<size_t>(nnzA * 1.5f);
                int*       col_tmp = (int*)realloc(col, alloc_size * sizeof(int));
                ValueType* val_tmp = (ValueType*)realloc(val, alloc_size * sizeof(ValueType));

                if(col_tmp == NULL || val_tmp == NULL)
                {
                    free(col);
                    free(val);

                    LOG_INFO("ILUTFactorize failed on realloc");
                    FATAL_ERROR(__FILE__, __LINE__);
                }
                else
                {
                    col = col_tmp;
                    val = val_tmp;
                }
            }
        }

        free_host(&w);
        free_host(&diag_offset);
        free_host(&nnz_entries);
        free(nnz_pos);

        // pinned memory
        int*       p_col = NULL;
        ValueType* p_val = NULL;

        allocate_host(nnz, &p_col);
        allocate_host(nnz, &p_val);

        copy_h2h(nnz, col, p_col);
        copy_h2h(nnz, val, p_val);

        free(col);
        free(val);

        this->Clear();
        this->SetDataPtrCSR(&row_offset, &p_col, &p_val, nnz, nrow, ncol);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ICFactorize(BaseVector<ValueType>* inv_diag)
    {
        assert(this->nrow_ == this->ncol_);
        assert(this->nnz_ > 0);

        assert(inv_diag != NULL);
        HostVector<ValueType>* cast_diag = dynamic_cast<HostVector<ValueType>*>(inv_diag);
        assert(cast_diag != NULL);

        cast_diag->Allocate(this->nrow_);

        int* diag_offset = NULL;
        int* nnz_entries = NULL;

        allocate_host(this->nrow_, &diag_offset);
        allocate_host(this->nrow_, &nnz_entries);

        set_to_zero_host(this->nrow_, nnz_entries);

        // i=0,..n
        for(int i = 0; i < this->nrow_; ++i)
        {
            int row_begin = this->mat_.row_offset[i];
            int row_end   = this->mat_.row_offset[i + 1];

            for(int j = row_begin; j < row_end; ++j)
            {
                nnz_entries[this->mat_.col[j]] = j;
            }

            ValueType sum = static_cast<ValueType>(0);

            bool has_diag = false;

            // j=0,..i
            int j;
            for(j = row_begin; j < row_end; ++j)
            {
                int       col_j = this->mat_.col[j];
                ValueType val_j = this->mat_.val[j];

                // Mark diagonal and skip row
                if(col_j == i)
                {
                    has_diag = true;
                    break;
                }

                // Skip upper triangular
                if(col_j > i)
                {
                    break;
                }

                int row_begin_j = this->mat_.row_offset[col_j];
                int row_diag_j  = diag_offset[col_j];

                ValueType local_sum = static_cast<ValueType>(0);
                ValueType inv_diag  = this->mat_.val[row_diag_j];

                // Check for numeric zero
                if(inv_diag == static_cast<ValueType>(0))
                {
                    LOG_INFO("IC breakdown: division by zero");
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                inv_diag = static_cast<ValueType>(1) / inv_diag;

                for(int k = row_begin_j; k < row_diag_j; ++k)
                {
                    int col_k = this->mat_.col[k];

                    if(nnz_entries[col_k] != 0)
                    {
                        int idx = nnz_entries[col_k];
                        local_sum += this->mat_.val[k] * this->mat_.val[idx];
                    }
                }

                val_j = (val_j - local_sum) * inv_diag;
                sum += val_j * val_j;

                this->mat_.val[j] = val_j;
            }

            if(!has_diag)
            {
                // Structural zero
                LOG_INFO("IC breakdown: structural zero diagonal");
                FATAL_ERROR(__FILE__, __LINE__);
            }

            // Process diagonal entry
            ValueType diag_entry = std::sqrt(std::abs(this->mat_.val[j] - sum));
            this->mat_.val[j]    = diag_entry;

            // Check for numerical zero
            if(diag_entry == static_cast<ValueType>(0))
            {
                LOG_INFO("IC breakdown: division by zero");
                FATAL_ERROR(__FILE__, __LINE__);
            }

            // Store inverse diagonal entry
            cast_diag->vec_[i] = static_cast<ValueType>(1) / diag_entry;

            // Store diagonal offset
            diag_offset[i] = j;

            // Clear nnz entries
            for(j = row_begin; j < row_end; ++j)
            {
                nnz_entries[this->mat_.col[j]] = 0;
            }
        }

        // Free temporary storage
        free_host(&diag_offset);
        free_host(&nnz_entries);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::MultiColoring(int&             num_colors,
                                                 int**            size_colors,
                                                 BaseVector<int>* permutation) const
    {
        assert(*size_colors == NULL);
        assert(permutation != NULL);
        HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
        assert(cast_perm != NULL);

        // node colors (init value = 0 i.e. no color)
        int* color = NULL;
        allocate_host(this->nrow_, &color);

        memset(color, 0, sizeof(int) * this->nrow_);
        num_colors = 0;
        std::vector<bool> row_col;

        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            color[ai] = 1;
            row_col.clear();
            row_col.reserve(num_colors + 2);
            row_col.assign(num_colors + 2, false);

            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai != this->mat_.col[aj])
                {
                    row_col[color[this->mat_.col[aj]]] = true;
                }
            }

            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
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

        cast_perm->Allocate(this->nrow_);

        for(int i = 0; i < permutation->GetSize(); ++i)
        {
            cast_perm->vec_[i] = offsets_color[color[i] - 1];
            ++offsets_color[color[i] - 1];
        }

        free_host(&color);
        free_host(&offsets_color);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::MaximalIndependentSet(int&             size,
                                                         BaseVector<int>* permutation) const
    {
        assert(permutation != NULL);
        assert(this->nrow_ == this->ncol_);

        HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
        assert(cast_perm != NULL);

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
                for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
                {
                    if(ai != this->mat_.col[aj])
                    {
                        mis[this->mat_.col[aj]] = -1;
                    }
                }
            }
        }

        cast_perm->Allocate(this->nrow_);

        int pos = 0;
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            if(mis[ai] == 1)
            {
                cast_perm->vec_[ai] = pos;
                ++pos;
            }
            else
            {
                cast_perm->vec_[ai] = size + ai - pos;
            }
        }

        // Check the permutation
        //
        //  for (int ai=0; ai<this->nrow_; ++ai) {
        //    assert( cast_perm->vec_[ai] >= 0 );
        //    assert( cast_perm->vec_[ai] < this->nrow_ );
        //  }

        free_host(&mis);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ZeroBlockPermutation(int&             size,
                                                        BaseVector<int>* permutation) const
    {
        assert(permutation != NULL);
        assert(permutation->GetSize() == this->nrow_);
        assert(permutation->GetSize() == this->ncol_);

        HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
        assert(cast_perm != NULL);

        size = 0;

        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai == this->mat_.col[aj])
                {
                    ++size;
                }
            }
        }

        int k_z  = size;
        int k_nz = 0;

        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            bool hit = false;

            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai == this->mat_.col[aj])
                {
                    cast_perm->vec_[ai] = k_nz;
                    ++k_nz;
                    hit = true;
                }
            }

            if(hit == false)
            {
                cast_perm->vec_[ai] = k_z;
                ++k_z;
            }
        }

        return true;
    }

    // following R.E.Bank and C.C.Douglas paper
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::SymbolicMatMatMult(const BaseMatrix<ValueType>& src)
    {
        const HostMatrixCSR<ValueType>* cast_mat
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&src);

        assert(cast_mat != NULL);
        assert(this->ncol_ == cast_mat->nrow_);

        std::vector<int>  row_offset;
        std::vector<int>* new_col = new std::vector<int>[this->nrow_];

        row_offset.resize(this->nrow_ + 1);

        row_offset[0] = 0;

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            // loop over the row
            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int ii = this->mat_.col[j];

                // loop corresponding row
                for(int k = cast_mat->mat_.row_offset[ii]; k < cast_mat->mat_.row_offset[ii + 1];
                    ++k)
                {
                    new_col[i].push_back(cast_mat->mat_.col[k]);
                }
            }

            std::sort(new_col[i].begin(), new_col[i].end());
            new_col[i].erase(std::unique(new_col[i].begin(), new_col[i].end()), new_col[i].end());

            row_offset[i + 1] = int(new_col[i].size());
        }

        for(int i = 0; i < this->nrow_; ++i)
        {
            row_offset[i + 1] += row_offset[i];
        }

        this->AllocateCSR(row_offset[this->nrow_], this->nrow_, this->ncol_);

        copy_h2h(this->nrow_ + 1, row_offset.data(), this->mat_.row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            int jj = 0;
            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                this->mat_.col[j] = new_col[i][jj];
                ++jj;
            }
        }

        //#ifdef _OPENMP
        //#pragma omp parallel for
        //#endif
        //  for (unsigned int i=0; i<this->nnz_; ++i)
        //    this->mat_.val[i] = static_cast<ValueType>(1);

        delete[] new_col;

        return true;
    }

    // ----------------------------------------------------------
    // original function prod(const spmat1 &A, const spmat2 &B)
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // sorting is added
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::MatMatMult(const BaseMatrix<ValueType>& A,
                                              const BaseMatrix<ValueType>& B)
    {
        assert((this != &A) && (this != &B));

        //  this->SymbolicMatMatMult(A, B);
        //  this->NumericMatMatMult (A, B);
        //
        //  return true;

        const HostMatrixCSR<ValueType>* cast_mat_A
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&A);
        const HostMatrixCSR<ValueType>* cast_mat_B
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&B);

        assert(cast_mat_A != NULL);
        assert(cast_mat_B != NULL);
        assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

        int n = cast_mat_A->nrow_;
        int m = cast_mat_B->ncol_;

        int* row_offset = NULL;
        allocate_host(n + 1, &row_offset);
        int*       col = NULL;
        ValueType* val = NULL;

        set_to_zero_host(n + 1, row_offset);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<int> marker(m, -1);

#ifdef _OPENMP
            int nt  = omp_get_num_threads();
            int tid = omp_get_thread_num();

            int chunk_size  = (n + nt - 1) / nt;
            int chunk_start = tid * chunk_size;
            int chunk_end   = std::min(n, chunk_start + chunk_size);
#else
            int chunk_start = 0;
            int chunk_end   = n;
#endif

            for(int ia = chunk_start; ia < chunk_end; ++ia)
            {
                for(int ja = cast_mat_A->mat_.row_offset[ia],
                        ea = cast_mat_A->mat_.row_offset[ia + 1];
                    ja < ea;
                    ++ja)
                {
                    int ca = cast_mat_A->mat_.col[ja];
                    for(int jb = cast_mat_B->mat_.row_offset[ca],
                            eb = cast_mat_B->mat_.row_offset[ca + 1];
                        jb < eb;
                        ++jb)
                    {
                        int cb = cast_mat_B->mat_.col[jb];

                        if(marker[cb] != ia)
                        {
                            marker[cb] = ia;
                            ++row_offset[ia + 1];
                        }
                    }
                }
            }

            std::fill(marker.begin(), marker.end(), -1);

#ifdef _OPENMP
#pragma omp barrier
#endif
#ifdef _OPENMP
#pragma omp single
#endif
            {
                for(int i = 1; i < n + 1; ++i)
                {
                    row_offset[i] += row_offset[i - 1];
                }

                allocate_host(row_offset[n], &col);
                allocate_host(row_offset[n], &val);
            }

            for(int ia = chunk_start; ia < chunk_end; ++ia)
            {
                int row_begin = row_offset[ia];
                int row_end   = row_begin;

                for(int ja = cast_mat_A->mat_.row_offset[ia],
                        ea = cast_mat_A->mat_.row_offset[ia + 1];
                    ja < ea;
                    ++ja)
                {
                    int       ca = cast_mat_A->mat_.col[ja];
                    ValueType va = cast_mat_A->mat_.val[ja];

                    for(int jb = cast_mat_B->mat_.row_offset[ca],
                            eb = cast_mat_B->mat_.row_offset[ca + 1];
                        jb < eb;
                        ++jb)
                    {
                        int       cb = cast_mat_B->mat_.col[jb];
                        ValueType vb = cast_mat_B->mat_.val[jb];

                        if(marker[cb] < row_begin)
                        {
                            marker[cb]   = row_end;
                            col[row_end] = cb;
                            val[row_end] = va * vb;
                            ++row_end;
                        }
                        else
                        {
                            val[marker[cb]] += va * vb;
                        }
                    }
                }
            }
        }

        this->SetDataPtrCSR(
            &row_offset, &col, &val, row_offset[n], cast_mat_A->nrow_, cast_mat_B->ncol_);

        // Sorting the col (per row)
        this->Sort();

        return true;
    }

    // following R.E.Bank and C.C.Douglas paper
    // this = A * B
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::SymbolicMatMatMult(const BaseMatrix<ValueType>& A,
                                                      const BaseMatrix<ValueType>& B)
    {
        const HostMatrixCSR<ValueType>* cast_mat_A
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&A);
        const HostMatrixCSR<ValueType>* cast_mat_B
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&B);

        assert(cast_mat_A != NULL);
        assert(cast_mat_B != NULL);
        assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

        std::vector<int>  row_offset;
        std::vector<int>* new_col = new std::vector<int>[cast_mat_A->nrow_];

        row_offset.resize(cast_mat_A->nrow_ + 1);

        row_offset[0] = 0;

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < cast_mat_A->nrow_; ++i)
        {
            // loop over the row
            for(int j = cast_mat_A->mat_.row_offset[i]; j < cast_mat_A->mat_.row_offset[i + 1]; ++j)
            {
                int ii = cast_mat_A->mat_.col[j];
                //      new_col[i].push_back(ii);

                // loop corresponding row
                for(int k = cast_mat_B->mat_.row_offset[ii];
                    k < cast_mat_B->mat_.row_offset[ii + 1];
                    ++k)
                {
                    new_col[i].push_back(cast_mat_B->mat_.col[k]);
                }
            }

            std::sort(new_col[i].begin(), new_col[i].end());
            new_col[i].erase(std::unique(new_col[i].begin(), new_col[i].end()), new_col[i].end());

            row_offset[i + 1] = int(new_col[i].size());
        }

        for(int i = 0; i < cast_mat_A->nrow_; ++i)
        {
            row_offset[i + 1] += row_offset[i];
        }

        this->AllocateCSR(row_offset[cast_mat_A->nrow_], cast_mat_A->nrow_, cast_mat_B->ncol_);

        copy_h2h(cast_mat_A->nrow_ + 1, row_offset.data(), this->mat_.row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < cast_mat_A->nrow_; ++i)
        {
            int jj = 0;
            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                this->mat_.col[j] = new_col[i][jj];
                ++jj;
            }
        }

        //#ifdef _OPENMP
        //#pragma omp parallel for
        //#endif
        //      for (unsigned int i=0; i<this->nnz_; ++i)
        //      this->mat_.val[i] = static_cast<ValueType>(1);

        delete[] new_col;

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::NumericMatMatMult(const BaseMatrix<ValueType>& A,
                                                     const BaseMatrix<ValueType>& B)
    {
        const HostMatrixCSR<ValueType>* cast_mat_A
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&A);
        const HostMatrixCSR<ValueType>* cast_mat_B
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&B);

        assert(cast_mat_A != NULL);
        assert(cast_mat_B != NULL);
        assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);
        assert(this->nrow_ == cast_mat_A->nrow_);
        assert(this->ncol_ == cast_mat_B->ncol_);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < cast_mat_A->nrow_; ++i)
        {
            // loop over the row
            for(int j = cast_mat_A->mat_.row_offset[i]; j < cast_mat_A->mat_.row_offset[i + 1]; ++j)
            {
                int ii = cast_mat_A->mat_.col[j];

                // loop corresponding row
                for(int k = cast_mat_B->mat_.row_offset[ii];
                    k < cast_mat_B->mat_.row_offset[ii + 1];
                    ++k)
                {
                    for(int p = this->mat_.row_offset[i]; p < this->mat_.row_offset[i + 1]; ++p)
                    {
                        if(cast_mat_B->mat_.col[k] == this->mat_.col[p])
                        {
                            this->mat_.val[p] += cast_mat_B->mat_.val[k] * cast_mat_A->mat_.val[j];
                            break;
                        }
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::SymbolicPower(int p)
    {
        assert(p > 1);

        // The optimal values for the firsts values

        if(p == 2)
        {
            this->SymbolicMatMatMult(*this);
        }

        if(p == 3)
        {
            HostMatrixCSR<ValueType> tmp(this->local_backend_);
            tmp.CopyFrom(*this);

            this->SymbolicPower(2);
            this->SymbolicMatMatMult(tmp);
        }

        if(p == 4)
        {
            this->SymbolicPower(2);
            this->SymbolicPower(2);
        }

        if(p == 5)
        {
            HostMatrixCSR<ValueType> tmp(this->local_backend_);
            tmp.CopyFrom(*this);

            this->SymbolicPower(4);
            this->SymbolicMatMatMult(tmp);
        }

        if(p == 6)
        {
            this->SymbolicPower(2);
            this->SymbolicPower(3);
        }

        if(p == 7)
        {
            HostMatrixCSR<ValueType> tmp(this->local_backend_);
            tmp.CopyFrom(*this);

            this->SymbolicPower(6);
            this->SymbolicMatMatMult(tmp);
        }

        if(p == 8)
        {
            HostMatrixCSR<ValueType> tmp(this->local_backend_);
            tmp.CopyFrom(*this);

            this->SymbolicPower(6);
            tmp.SymbolicPower(2);

            this->SymbolicMatMatMult(tmp);
        }

        if(p > 8)
        {
            HostMatrixCSR<ValueType> tmp(this->local_backend_);
            tmp.CopyFrom(*this);

            for(int i = 0; i < p; ++i)
            {
                this->SymbolicMatMatMult(tmp);
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ILUpFactorizeNumeric(int p, const BaseMatrix<ValueType>& mat)
    {
        const HostMatrixCSR<ValueType>* cast_mat
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat);

        assert(cast_mat != NULL);
        assert(cast_mat->nrow_ == this->nrow_);
        assert(cast_mat->ncol_ == this->ncol_);
        assert(this->nnz_ > 0);
        assert(cast_mat->nnz_ > 0);

        int*       row_offset = NULL;
        int*       ind_diag   = NULL;
        int*       levels     = NULL;
        ValueType* val        = NULL;

        allocate_host(cast_mat->nrow_ + 1, &row_offset);
        allocate_host(cast_mat->nrow_, &ind_diag);
        allocate_host(cast_mat->nnz_, &levels);
        allocate_host(cast_mat->nnz_, &val);

        int inf_level = 99999;
        int nnz       = 0;

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        // find diagonals
        for(int ai = 0; ai < cast_mat->nrow_; ++ai)
        {
            for(int aj = cast_mat->mat_.row_offset[ai]; aj < cast_mat->mat_.row_offset[ai + 1];
                ++aj)
            {
                if(ai == cast_mat->mat_.col[aj])
                {
                    ind_diag[ai] = aj;
                    break;
                }
            }
        }

        // init row_offset
        set_to_zero_host(cast_mat->nrow_ + 1, row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        // init inf levels
        for(int i = 0; i < cast_mat->nnz_; ++i)
        {
            levels[i] = inf_level;
        }

        set_to_zero_host(cast_mat->nnz_, val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        // fill levels and values
        for(int ai = 0; ai < cast_mat->nrow_; ++ai)
        {
            for(int aj = cast_mat->mat_.row_offset[ai]; aj < cast_mat->mat_.row_offset[ai + 1];
                ++aj)
            {
                for(int ajj = this->mat_.row_offset[ai]; ajj < this->mat_.row_offset[ai + 1]; ++ajj)
                {
                    if(cast_mat->mat_.col[aj] == this->mat_.col[ajj])
                    {
                        val[aj]    = this->mat_.val[ajj];
                        levels[aj] = 0;
                        break;
                    }
                }
            }
        }

        // ai = 1 to N
        for(int ai = 1; ai < cast_mat->nrow_; ++ai)
        {
            // ak = 1 to ai-1
            for(int ak = cast_mat->mat_.row_offset[ai]; ai > cast_mat->mat_.col[ak]; ++ak)
            {
                if(levels[ak] <= p)
                {
                    val[ak] /= val[ind_diag[cast_mat->mat_.col[ak]]];

                    // aj = ak+1 to N
                    for(int aj = ak + 1; aj < cast_mat->mat_.row_offset[ai + 1]; ++aj)
                    {
                        ValueType val_kj   = static_cast<ValueType>(0);
                        int       level_kj = inf_level;

                        // find a_k,j
                        for(int kj = cast_mat->mat_.row_offset[cast_mat->mat_.col[ak]];
                            kj < cast_mat->mat_.row_offset[cast_mat->mat_.col[ak] + 1];
                            ++kj)
                        {
                            if(cast_mat->mat_.col[aj] == cast_mat->mat_.col[kj])
                            {
                                level_kj = levels[kj];
                                val_kj   = val[kj];
                                break;
                            }
                        }

                        int lev = level_kj + levels[ak] + 1;

                        if(levels[aj] > lev)
                        {
                            levels[aj] = lev;
                        }

                        // a_i,j = a_i,j - a_i,k * a_k,j
                        val[aj] -= val[ak] * val_kj;
                    }
                }
            }

            for(int ak = cast_mat->mat_.row_offset[ai]; ak < cast_mat->mat_.row_offset[ai + 1];
                ++ak)
            {
                if(levels[ak] > p)
                {
                    levels[ak] = inf_level;
                    val[ak]    = static_cast<ValueType>(0);
                }
                else
                {
                    ++row_offset[ai + 1];
                }
            }
        }

        row_offset[0] = this->mat_.row_offset[0];
        row_offset[1] = this->mat_.row_offset[1];

        for(int i = 0; i < cast_mat->nrow_; ++i)
        {
            row_offset[i + 1] += row_offset[i];
        }

        nnz = row_offset[cast_mat->nrow_];

        this->AllocateCSR(nnz, cast_mat->nrow_, cast_mat->ncol_);

        int jj = 0;
        for(int i = 0; i < cast_mat->nrow_; ++i)
        {
            for(int j = cast_mat->mat_.row_offset[i]; j < cast_mat->mat_.row_offset[i + 1]; ++j)
            {
                if(levels[j] <= p)
                {
                    this->mat_.col[jj] = cast_mat->mat_.col[j];
                    this->mat_.val[jj] = val[j];
                    ++jj;
                }
            }
        }

        assert(jj == nnz);

        copy_h2h(this->nrow_ + 1, row_offset, this->mat_.row_offset);

        free_host(&row_offset);
        free_host(&ind_diag);
        free_host(&levels);
        free_host(&val);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::MatrixAdd(const BaseMatrix<ValueType>& mat,
                                             ValueType                    alpha,
                                             ValueType                    beta,
                                             bool                         structure)
    {
        const HostMatrixCSR<ValueType>* cast_mat
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat);

        assert(cast_mat != NULL);
        assert(cast_mat->nrow_ == this->nrow_);
        assert(cast_mat->ncol_ == this->ncol_);
        assert(this->nnz_ >= 0);
        assert(cast_mat->nnz_ >= 0);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

        // the structure is sub-set
        if(structure == false)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            // CSR should be sorted
            for(int ai = 0; ai < cast_mat->nrow_; ++ai)
            {
                int first_col = cast_mat->mat_.row_offset[ai];

                for(int ajj = this->mat_.row_offset[ai]; ajj < this->mat_.row_offset[ai + 1]; ++ajj)
                {
                    for(int aj = first_col; aj < cast_mat->mat_.row_offset[ai + 1]; ++aj)
                    {
                        if(cast_mat->mat_.col[aj] == this->mat_.col[ajj])
                        {
                            this->mat_.val[ajj]
                                = alpha * this->mat_.val[ajj] + beta * cast_mat->mat_.val[aj];
                            ++first_col;
                            break;
                        }
                    }
                }
            }
        }
        else
        {
            std::vector<int>  row_offset;
            std::vector<int>* new_col = new std::vector<int>[this->nrow_];

            HostMatrixCSR<ValueType> tmp(this->local_backend_);

            tmp.CopyFrom(*this);

            row_offset.resize(this->nrow_ + 1);

            row_offset[0] = 0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    new_col[i].push_back(this->mat_.col[j]);
                }

                for(int k = cast_mat->mat_.row_offset[i]; k < cast_mat->mat_.row_offset[i + 1]; ++k)
                {
                    new_col[i].push_back(cast_mat->mat_.col[k]);
                }

                std::sort(new_col[i].begin(), new_col[i].end());
                new_col[i].erase(std::unique(new_col[i].begin(), new_col[i].end()),
                                 new_col[i].end());

                row_offset[i + 1] = int(new_col[i].size());
            }

            for(int i = 0; i < this->nrow_; ++i)
            {
                row_offset[i + 1] += row_offset[i];
            }

            this->AllocateCSR(row_offset[this->nrow_], this->nrow_, this->ncol_);

            // copy structure
            copy_h2h(this->nrow_ + 1, row_offset.data(), this->mat_.row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                int jj = 0;
                for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    this->mat_.col[j] = new_col[i][jj];
                    ++jj;
                }
            }

#ifdef _OPENMP
#pragma omp parallel for
#endif
            // add values
            for(int i = 0; i < this->nrow_; ++i)
            {
                int Aj = tmp.mat_.row_offset[i];
                int Bj = cast_mat->mat_.row_offset[i];

                for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    for(int jj = Aj; jj < tmp.mat_.row_offset[i + 1]; ++jj)
                    {
                        if(this->mat_.col[j] == tmp.mat_.col[jj])
                        {
                            this->mat_.val[j] += alpha * tmp.mat_.val[jj];
                            ++Aj;
                            break;
                        }
                    }

                    for(int jj = Bj; jj < cast_mat->mat_.row_offset[i + 1]; ++jj)
                    {
                        if(this->mat_.col[j] == cast_mat->mat_.col[jj])
                        {
                            this->mat_.val[j] += beta * cast_mat->mat_.val[jj];
                            ++Bj;
                            break;
                        }
                    }
                }
            }
            delete[] new_col;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Gershgorin(ValueType& lambda_min, ValueType& lambda_max) const
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

        lambda_min = static_cast<ValueType>(0);
        lambda_max = static_cast<ValueType>(0);

        // TODO (parallel max, min)
        //#ifdef _OPENMP
        // #pragma omp parallel for
        //#endif

        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            ValueType sum  = static_cast<ValueType>(0);
            ValueType diag = static_cast<ValueType>(0);

            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai != this->mat_.col[aj])
                {
                    sum += std::abs(this->mat_.val[aj]);
                }
                else
                {
                    diag = this->mat_.val[aj];
                }
            }

            if(sum + diag > lambda_max)
            {
                lambda_max = sum + diag;
            }

            if(diag - sum < lambda_min)
            {
                lambda_min = diag - sum;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Scale(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nnz_; ++ai)
        {
            this->mat_.val[ai] *= alpha;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ScaleDiagonal(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai == this->mat_.col[aj])
                {
                    this->mat_.val[aj] *= alpha;
                    break;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ScaleOffDiagonal(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai != this->mat_.col[aj])
                {
                    this->mat_.val[aj] *= alpha;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AddScalar(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nnz_; ++ai)
        {
            this->mat_.val[ai] += alpha;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AddScalarDiagonal(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai == this->mat_.col[aj])
                {
                    this->mat_.val[aj] += alpha;
                    break;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AddScalarOffDiagonal(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai != this->mat_.col[aj])
                {
                    this->mat_.val[aj] += alpha;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::DiagonalMatrixMultR(const BaseVector<ValueType>& diag)
    {
        assert(diag.GetSize() == this->ncol_);

        const HostVector<ValueType>* cast_diag = dynamic_cast<const HostVector<ValueType>*>(&diag);
        assert(cast_diag != NULL);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                this->mat_.val[aj] *= cast_diag->vec_[this->mat_.col[aj]];
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::DiagonalMatrixMultL(const BaseVector<ValueType>& diag)
    {
        assert(diag.GetSize() == this->ncol_);

        const HostVector<ValueType>* cast_diag = dynamic_cast<const HostVector<ValueType>*>(&diag);
        assert(cast_diag != NULL);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                this->mat_.val[aj] *= cast_diag->vec_[ai];
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Compress(double drop_off)
    {
        if(this->nnz_ > 0)
        {
            std::vector<int> row_offset;

            HostMatrixCSR<ValueType> tmp(this->local_backend_);

            tmp.CopyFrom(*this);

            row_offset.resize(this->nrow_ + 1);

            row_offset[0] = 0;

            _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                row_offset[i + 1] = 0;

                for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    if((std::abs(this->mat_.val[j]) > drop_off) || (this->mat_.col[j] == i))
                    {
                        row_offset[i + 1] += 1;
                    }
                }
            }

            for(int i = 0; i < this->nrow_; ++i)
            {
                row_offset[i + 1] += row_offset[i];
            }

            this->AllocateCSR(row_offset[this->nrow_], this->nrow_, this->ncol_);

            copy_h2h(this->nrow_ + 1, row_offset.data(), this->mat_.row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                int jj = this->mat_.row_offset[i];

                for(int j = tmp.mat_.row_offset[i]; j < tmp.mat_.row_offset[i + 1]; ++j)
                {
                    if((std::abs(tmp.mat_.val[j]) > drop_off) || (tmp.mat_.col[j] == i))
                    {
                        this->mat_.col[jj] = tmp.mat_.col[j];
                        this->mat_.val[jj] = tmp.mat_.val[j];
                        ++jj;
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Transpose(void)
    {
        if(this->nnz_ > 0)
        {
            HostMatrixCSR<ValueType> tmp(this->local_backend_);

            tmp.CopyFrom(*this);
            tmp.Transpose(this);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Transpose(BaseMatrix<ValueType>* T) const
    {
        assert(T != NULL);

        HostMatrixCSR<ValueType>* cast_T = dynamic_cast<HostMatrixCSR<ValueType>*>(T);

        assert(cast_T != NULL);

        if(this->nnz_ > 0)
        {
            cast_T->Clear();
            cast_T->AllocateCSR(this->nnz_, this->ncol_, this->nrow_);

            for(int i = 0; i < cast_T->nnz_; ++i)
            {
                cast_T->mat_.row_offset[this->mat_.col[i] + 1] += 1;
            }

            for(int i = 0; i < cast_T->nrow_; ++i)
            {
                cast_T->mat_.row_offset[i + 1] += cast_T->mat_.row_offset[i];
            }

            for(int ai = 0; ai < cast_T->ncol_; ++ai)
            {
                for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
                {
                    int ind_col = this->mat_.col[aj];
                    int ind     = cast_T->mat_.row_offset[ind_col];

                    cast_T->mat_.col[ind] = ai;
                    cast_T->mat_.val[ind] = this->mat_.val[aj];

                    cast_T->mat_.row_offset[ind_col] += 1;
                }
            }

            int shift = 0;
            for(int i = 0; i < cast_T->nrow_; ++i)
            {
                int tmp                    = cast_T->mat_.row_offset[i];
                cast_T->mat_.row_offset[i] = shift;
                shift                      = tmp;
            }

            cast_T->mat_.row_offset[cast_T->nrow_] = shift;

            assert(this->nnz_ == shift);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Sort(void)
    {
        if(this->nnz_ > 0)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    for(int jj = this->mat_.row_offset[i]; jj < this->mat_.row_offset[i + 1] - 1;
                        ++jj)
                    {
                        if(this->mat_.col[jj] > this->mat_.col[jj + 1])
                        {
                            // swap elements
                            int       ind = this->mat_.col[jj];
                            ValueType val = this->mat_.val[jj];

                            this->mat_.col[jj] = this->mat_.col[jj + 1];
                            this->mat_.val[jj] = this->mat_.val[jj + 1];

                            this->mat_.col[jj + 1] = ind;
                            this->mat_.val[jj + 1] = val;
                        }
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Permute(const BaseVector<int>& permutation)
    {
        assert((permutation.GetSize() == this->nrow_) && (permutation.GetSize() == this->ncol_));

        if(this->nnz_ > 0)
        {
            const HostVector<int>* cast_perm = dynamic_cast<const HostVector<int>*>(&permutation);
            assert(cast_perm != NULL);

            _set_omp_backend_threads(this->local_backend_, this->nrow_);

            // Calculate nnz per row
            int* row_nnz = NULL;
            allocate_host(this->nrow_, &row_nnz);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                row_nnz[i] = this->mat_.row_offset[i + 1] - this->mat_.row_offset[i];
            }

            // Permute vector of nnz per row
            int* perm_row_nnz = NULL;
            allocate_host(this->nrow_, &perm_row_nnz);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                perm_row_nnz[cast_perm->vec_[i]] = row_nnz[i];
            }

            // Calculate new nnz
            int* perm_nnz = NULL;
            allocate_host<int>(this->nrow_ + 1, &perm_nnz);
            int sum = 0;

            for(int i = 0; i < this->nrow_; ++i)
            {
                perm_nnz[i] = sum;
                sum += perm_row_nnz[i];
            }

            perm_nnz[this->nrow_] = sum;

            // Permute rows
            int*       col = NULL;
            ValueType* val = NULL;
            allocate_host(this->nnz_, &col);
            allocate_host(this->nnz_, &val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                int permIndex = perm_nnz[cast_perm->vec_[i]];
                int prevIndex = this->mat_.row_offset[i];

                for(int j = 0; j < row_nnz[i]; ++j)
                {
                    col[permIndex + j] = this->mat_.col[prevIndex + j];
                    val[permIndex + j] = this->mat_.val[prevIndex + j];
                }
            }

// Permute columns
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                int row_index = perm_nnz[i];

                for(int j = 0; j < perm_row_nnz[i]; ++j)
                {
                    int k     = j - 1;
                    int aComp = col[row_index + j];
                    int comp  = cast_perm->vec_[aComp];
                    for(; k >= 0; --k)
                    {
                        if(this->mat_.col[row_index + k] > comp)
                        {
                            this->mat_.val[row_index + k + 1] = this->mat_.val[row_index + k];
                            this->mat_.col[row_index + k + 1] = this->mat_.col[row_index + k];
                        }
                        else
                        {
                            break;
                        }
                    }

                    this->mat_.val[row_index + k + 1] = val[row_index + j];
                    this->mat_.col[row_index + k + 1] = comp;
                }
            }

            free_host(&this->mat_.row_offset);
            this->mat_.row_offset = perm_nnz;
            free_host(&col);
            free_host(&val);
            free_host(&row_nnz);
            free_host(&perm_row_nnz);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::CMK(BaseVector<int>* permutation) const
    {
        assert(this->nnz_ > 0);
        assert(permutation != NULL);

        HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
        assert(cast_perm != NULL);

        cast_perm->Clear();
        cast_perm->Allocate(this->nrow_);

        int next   = 0;
        int head   = 0;
        int tmp    = 0;
        int test   = 1;
        int maxdeg = 0;

        int* nd         = NULL;
        int* marker     = NULL;
        int* levset     = NULL;
        int* nextlevset = NULL;

        allocate_host(this->nrow_, &nd);
        allocate_host(this->nrow_, &marker);
        allocate_host(this->nrow_, &levset);
        allocate_host(this->nrow_, &nextlevset);

        int qlength = 1;

        for(int k = 0; k < this->nrow_; ++k)
        {
            marker[k] = 0;
            nd[k]     = this->mat_.row_offset[k + 1] - this->mat_.row_offset[k] - 1;

            if(nd[k] > maxdeg)
            {
                maxdeg = nd[k];
            }
        }

        head               = this->mat_.col[0];
        levset[0]          = head;
        cast_perm->vec_[0] = 0;
        ++next;
        marker[head] = 1;

        while(next < this->nrow_)
        {
            int position = 0;

            for(int h = 0; h < qlength; ++h)
            {
                head = levset[h];

                for(int k = this->mat_.row_offset[head]; k < this->mat_.row_offset[head + 1]; ++k)
                {
                    tmp = this->mat_.col[k];

                    if((marker[tmp] == 0) && (tmp != head))
                    {
                        nextlevset[position] = tmp;
                        marker[tmp]          = 1;
                        cast_perm->vec_[tmp] = next;
                        ++next;
                        ++position;
                    }
                }
            }

            qlength = position;

            while(test == 1)
            {
                test = 0;

                for(int j = position - 1; j > 0; --j)
                {
                    if(nd[nextlevset[j]] < nd[nextlevset[j - 1]])
                    {
                        tmp               = nextlevset[j];
                        nextlevset[j]     = nextlevset[j - 1];
                        nextlevset[j - 1] = tmp;
                        test              = 1;
                    }
                }
            }

            for(int i = 0; i < position; ++i)
            {
                levset[i] = nextlevset[i];
            }

            if(qlength == 0)
            {
                for(int i = 0; i < this->nrow_; ++i)
                {
                    if(marker[i] == 0)
                    {
                        levset[0]          = i;
                        qlength            = 1;
                        cast_perm->vec_[i] = next;
                        marker[i]          = 1;
                        ++next;
                    }
                }
            }
        }

        free_host(&nd);
        free_host(&marker);
        free_host(&levset);
        free_host(&nextlevset);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RCMK(BaseVector<int>* permutation) const
    {
        HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
        assert(cast_perm != NULL);

        cast_perm->Clear();
        cast_perm->Allocate(this->nrow_);

        HostVector<int> tmp_perm(this->local_backend_);

        this->CMK(&tmp_perm);

        for(int i = 0; i < this->nrow_; ++i)
        {
            cast_perm->vec_[i] = this->nrow_ - tmp_perm.vec_[i] - 1;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ConnectivityOrder(BaseVector<int>* permutation) const
    {
        HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
        assert(cast_perm != NULL);

        cast_perm->Clear();
        cast_perm->Allocate(this->nrow_);

        std::multimap<int, int> map;

        for(int i = 0; i < this->nrow_; ++i)
        {
            map.insert(
                std::pair<int, int>(this->mat_.row_offset[i + 1] - this->mat_.row_offset[i], i));
        }

        std::multimap<int, int>::iterator it = map.begin();

        for(int i = 0; i < this->nrow_; ++i, ++it)
        {
            cast_perm->vec_[i] = it->second;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::CreateFromMap(const BaseVector<int>& map, int n, int m)
    {
        assert(map.GetSize() == n);

        const HostVector<int>* cast_map = dynamic_cast<const HostVector<int>*>(&map);

        assert(cast_map != NULL);

        int* row_nnz    = NULL;
        int* row_buffer = NULL;
        allocate_host(m, &row_nnz);
        allocate_host(m + 1, &row_buffer);

        set_to_zero_host(m, row_nnz);

        int nnz = 0;

        for(int i = 0; i < n; ++i)
        {
            assert(cast_map->vec_[i] < m);

            if(cast_map->vec_[i] < 0)
            {
                continue;
            }

            ++row_nnz[cast_map->vec_[i]];
            ++nnz;
        }

        this->Clear();
        this->AllocateCSR(nnz, m, n);

        this->mat_.row_offset[0] = 0;
        row_buffer[0]            = 0;

        for(int i = 0; i < m; ++i)
        {
            this->mat_.row_offset[i + 1] = this->mat_.row_offset[i] + row_nnz[i];
            row_buffer[i + 1]            = this->mat_.row_offset[i + 1];
        }

        for(int i = 0; i < nnz; ++i)
        {
            if(cast_map->vec_[i] < 0)
            {
                continue;
            }

            this->mat_.col[row_buffer[cast_map->vec_[i]]] = i;
            this->mat_.val[i]                             = static_cast<ValueType>(1);
            row_buffer[cast_map->vec_[i]]++;
        }

        assert(this->mat_.row_offset[m] == nnz);

        free_host(&row_nnz);
        free_host(&row_buffer);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::CreateFromMap(const BaseVector<int>& map,
                                                 int                    n,
                                                 int                    m,
                                                 BaseMatrix<ValueType>* pro)
    {
        assert(map.GetSize() == n);
        assert(pro != NULL);

        const HostVector<int>*    cast_map = dynamic_cast<const HostVector<int>*>(&map);
        HostMatrixCSR<ValueType>* cast_pro = dynamic_cast<HostMatrixCSR<ValueType>*>(pro);

        assert(cast_pro != NULL);
        assert(cast_map != NULL);

        // Build restriction operator
        this->CreateFromMap(map, n, m);

        // Build prolongation operator
        cast_pro->Clear();
        cast_pro->AllocateCSR(this->nnz_, n, m);

        int k = 0;

        for(int i = 0; i < n; ++i)
        {
            cast_pro->mat_.row_offset[i + 1] = cast_pro->mat_.row_offset[i];

            // Check for entry in i-th row
            if(cast_map->vec_[i] < 0)
            {
                continue;
            }

            assert(cast_map->vec_[i] < m);

            ++cast_pro->mat_.row_offset[i + 1];
            cast_pro->mat_.col[k] = cast_map->vec_[i];
            cast_pro->mat_.val[k] = static_cast<ValueType>(1);
            ++k;
        }

        return true;
    }

    // ----------------------------------------------------------
    // original function connect(const spmat &A,
    //                           float eps_strong)
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGConnect(ValueType eps, BaseVector<int>* connections) const
    {
        assert(connections != NULL);

        HostVector<int>* cast_conn = dynamic_cast<HostVector<int>*>(connections);
        assert(cast_conn != NULL);

        cast_conn->Clear();
        cast_conn->Allocate(this->nnz_);

        ValueType eps2 = eps * eps;

        HostVector<ValueType> vec_diag(this->local_backend_);
        vec_diag.Allocate(this->nrow_);
        this->ExtractDiagonal(&vec_diag);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            ValueType eps_dia_i = eps2 * vec_diag.vec_[i];

            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int       c = this->mat_.col[j];
                ValueType v = this->mat_.val[j];

                cast_conn->vec_[j] = (c != i) && (v * v > eps_dia_i * vec_diag.vec_[c]);
            }
        }

        return true;
    }

    // ----------------------------------------------------------
    // original function aggregates( const spmat &A,
    //                               const std::vector<char> &S )
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGAggregate(const BaseVector<int>& connections,
                                                BaseVector<int>*       aggregates) const
    {
        assert(aggregates != NULL);

        HostVector<int>*       cast_agg  = dynamic_cast<HostVector<int>*>(aggregates);
        const HostVector<int>* cast_conn = dynamic_cast<const HostVector<int>*>(&connections);

        assert(cast_agg != NULL);
        assert(cast_conn != NULL);

        aggregates->Clear();
        aggregates->Allocate(this->nrow_);

        const int undefined = -1;
        const int removed   = -2;

        // Remove nodes without neighbours
        int max_neib = 0;
        for(int i = 0; i < this->nrow_; ++i)
        {
            int j = this->mat_.row_offset[i];
            int e = this->mat_.row_offset[i + 1];

            max_neib = std::max(e - j, max_neib);

            int state = removed;
            for(; j < e; ++j)
            {
                if(cast_conn->vec_[j])
                {
                    state = undefined;
                    break;
                }
            }

            cast_agg->vec_[i] = state;
        }

        std::vector<int> neib;
        neib.reserve(max_neib);

        int last_g = -1;

        // Perform plain aggregation
        for(int i = 0; i < this->nrow_; ++i)
        {
            if(cast_agg->vec_[i] != undefined)
            {
                continue;
            }

            // The point is not adjacent to a core of any previous aggregate:
            // so its a seed of a new aggregate.
            cast_agg->vec_[i] = ++last_g;

            neib.clear();

            // Include its neighbors as well.
            for(int j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e; ++j)
            {
                int c = this->mat_.col[j];
                if(cast_conn->vec_[j] && cast_agg->vec_[c] != removed)
                {
                    cast_agg->vec_[c] = last_g;
                    neib.push_back(c);
                }
            }

            // Temporarily mark undefined points adjacent to the new aggregate as
            // belonging to the aggregate. If nobody claims them later, they will
            // stay here.
            for(typename std::vector<int>::const_iterator nb = neib.begin(); nb != neib.end(); ++nb)
            {
                for(int j = this->mat_.row_offset[*nb], e = this->mat_.row_offset[*nb + 1]; j < e;
                    ++j)
                {
                    if(cast_conn->vec_[j] && cast_agg->vec_[this->mat_.col[j]] == undefined)
                    {
                        cast_agg->vec_[this->mat_.col[j]] = last_g;
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGPMISAggregate(const BaseVector<int>& connections,
                                                    BaseVector<int>*       aggregates) const
    {
        assert(aggregates != NULL);

        HostVector<int>*       cast_agg  = dynamic_cast<HostVector<int>*>(aggregates);
        const HostVector<int>* cast_conn = dynamic_cast<const HostVector<int>*>(&connections);

        assert(cast_agg != NULL);
        assert(cast_conn != NULL);

        aggregates->Clear();
        aggregates->Allocate(this->nrow_);

        std::vector<mis_tuple> tuples(this->nrow_);
        std::vector<mis_tuple> max_tuples(this->nrow_);

        // Initialize tuples
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            int state = -2;

            int row_start = this->mat_.row_offset[i];
            int row_end   = this->mat_.row_offset[i + 1];
            for(int j = row_start; j < row_end; j++)
            {
                if(cast_conn->vec_[j] == 1)
                {
                    state = 0;
                    break;
                }
            }

            unsigned int hash = i;
            hash              = ((hash >> 16) ^ hash) * 0x45d9f3b;
            hash              = ((hash >> 16) ^ hash) * 0x45d9f3b;
            hash              = (hash >> 16) ^ hash;

            tuples[i].s = state;
            tuples[i].v = hash;
            tuples[i].i = i;
        }

        bool done = false;
        int  iter = 0;
        while(!done)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                max_tuples[i] = tuples[i];
            }

            for(int k = 0; k < 2; k++)
            {
                // Find max tuples
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
                for(int i = 0; i < this->nrow_; ++i)
                {
                    mis_tuple t_max = max_tuples[i];

                    int row_start = this->mat_.row_offset[t_max.i];
                    int row_end   = this->mat_.row_offset[t_max.i + 1];
                    for(int j = row_start; j < row_end; j++)
                    {
                        if(cast_conn->vec_[j] == 1)
                        {
                            int       c  = this->mat_.col[j];
                            mis_tuple tj = tuples[c];

                            // find lexographical maximum
                            if(tj.s > t_max.s)
                            {
                                t_max = tj;
                            }
                            else if(tj.s == t_max.s && (tj.v > t_max.v))
                            {
                                t_max = tj;
                            }
                        }
                    }

                    max_tuples[i] = t_max;
                }
            }

            done = true;

            // Update tuples
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                if(tuples[i].s == 0)
                {
                    mis_tuple t_max = max_tuples[i];

                    if(t_max.i == i)
                    {
                        tuples[i].s       = 1;
                        cast_agg->vec_[i] = 1;
                    }
                    else if(t_max.s == 1)
                    {
                        tuples[i].s       = -1;
                        cast_agg->vec_[i] = 0;
                    }
                    else
                    {
                        done = false;
                    }
                }
            }

            iter++;

            if(iter > 10)
            {
                LOG_VERBOSE_INFO(
                    2,
                    "*** warning: HostMatrixCSR::AMGPMISAggregate() Current number of iterations: "
                        << iter);
            }
        }

        // exclusive scan on aggregates array
        int sum = 0;
        for(int i = 0; i < this->nrow_; ++i)
        {
            int temp          = cast_agg->vec_[i];
            cast_agg->vec_[i] = sum;
            sum += temp;
        }

        for(int k = 0; k < 2; k++)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                max_tuples[i] = tuples[i];
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                mis_tuple t = max_tuples[i];

                assert(t.s != 0);

                if(t.s == -1)
                {
                    int row_start = this->mat_.row_offset[i];
                    int row_end   = this->mat_.row_offset[i + 1];

                    for(int j = row_start; j < row_end; j++)
                    {
                        if(cast_conn->vec_[j] == 1)
                        {
                            int c = this->mat_.col[j];

                            if(max_tuples[c].s == 1)
                            {
                                cast_agg->vec_[i] = cast_agg->vec_[c];
                                tuples[i].s       = 1;
                                break;
                            }
                        }
                    }
                }
                else if(t.s == -2)
                {
                    cast_agg->vec_[i] = -2;
                }
            }
        }

        return true;
    }

    // ----------------------------------------------------------
    // original function interp(const sparse::matrix<value_t,
    //                          index_t> &A, const params &prm)
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGSmoothedAggregation(ValueType              relax,
                                                          const BaseVector<int>& aggregates,
                                                          const BaseVector<int>& connections,
                                                          BaseMatrix<ValueType>* prolong,
                                                          int lumping_strat) const
    {
        assert(prolong != NULL);

        const HostVector<int>*    cast_agg     = dynamic_cast<const HostVector<int>*>(&aggregates);
        const HostVector<int>*    cast_conn    = dynamic_cast<const HostVector<int>*>(&connections);
        HostMatrixCSR<ValueType>* cast_prolong = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong);

        assert(cast_agg != NULL);
        assert(cast_conn != NULL);
        assert(cast_prolong != NULL);

        // Allocate
        cast_prolong->Clear();
        cast_prolong->AllocateCSR(this->nnz_, this->nrow_, this->ncol_);

        int ncol = 0;

        for(int i = 0; i < cast_agg->GetSize(); ++i)
        {
            if(cast_agg->vec_[i] > ncol)
            {
                ncol = cast_agg->vec_[i];
            }
        }

        ++ncol;

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<int> marker(ncol, -1);

#ifdef _OPENMP
            int nt  = omp_get_num_threads();
            int tid = omp_get_thread_num();

            int chunk_size  = (this->nrow_ + nt - 1) / nt;
            int chunk_start = tid * chunk_size;
            int chunk_end   = std::min(this->nrow_, chunk_start + chunk_size);
#else
            int chunk_start = 0;
            int chunk_end   = this->nrow_;
#endif

            // Count number of entries in prolong
            for(int i = chunk_start; i < chunk_end; ++i)
            {
                for(int j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e; ++j)
                {
                    int c = this->mat_.col[j];

                    if(c != i && !cast_conn->vec_[j])
                    {
                        continue;
                    }

                    int g = cast_agg->vec_[c];

                    if(g >= 0 && marker[g] != i)
                    {
                        marker[g] = i;
                        ++cast_prolong->mat_.row_offset[i + 1];
                    }
                }
            }

            std::fill(marker.begin(), marker.end(), -1);

#ifdef _OPENMP
#pragma omp barrier
#pragma omp single
#endif
            {
                int* row_offset = NULL;
                allocate_host(cast_prolong->nrow_ + 1, &row_offset);

                int*       col = NULL;
                ValueType* val = NULL;

                int nnz  = 0;
                int nrow = cast_prolong->nrow_;

                row_offset[0] = 0;
                for(int i = 1; i < nrow + 1; ++i)
                {
                    row_offset[i] = cast_prolong->mat_.row_offset[i] + row_offset[i - 1];
                }

                nnz = row_offset[nrow];

                allocate_host(nnz, &col);
                allocate_host(nnz, &val);

                cast_prolong->Clear();
                cast_prolong->SetDataPtrCSR(&row_offset, &col, &val, nnz, nrow, ncol);
            }

            // Fill the interpolation matrix.
            for(int i = chunk_start; i < chunk_end; ++i)
            {
                // Diagonal of the filtered matrix is original matrix diagonal plus (lumping_strat = 0) or
                // minus (lumping_strat = 1) its weak connections.
                ValueType dia = static_cast<ValueType>(0);
                for(int j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e; ++j)
                {
                    if(this->mat_.col[j] == i)
                    {
                        dia += this->mat_.val[j];
                    }
                    else if(!cast_conn->vec_[j])
                    {
                        if(lumping_strat == 0)
                        {
                            dia += this->mat_.val[j];
                        }
                        else
                        {
                            dia -= this->mat_.val[j];
                        }
                    }
                }

                dia = static_cast<ValueType>(1) / dia;

                int row_begin = cast_prolong->mat_.row_offset[i];
                int row_end   = row_begin;

                for(int j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e; ++j)
                {
                    int c = this->mat_.col[j];

                    // Skip weak couplings, ...
                    if(c != i && !cast_conn->vec_[j])
                    {
                        continue;
                    }

                    // ... and the ones not in any aggregate.
                    int g = cast_agg->vec_[c];
                    if(g < 0)
                    {
                        continue;
                    }

                    ValueType v = (c == i) ? static_cast<ValueType>(1) - relax
                                           : -relax * dia * this->mat_.val[j];

                    if(marker[g] < row_begin)
                    {
                        marker[g]                       = row_end;
                        cast_prolong->mat_.col[row_end] = g;
                        cast_prolong->mat_.val[row_end] = v;
                        ++row_end;
                    }
                    else
                    {
                        cast_prolong->mat_.val[marker[g]] += v;
                    }
                }
            }
        }

        cast_prolong->Sort();

        return true;
    }

    // ----------------------------------------------------------
    // original function interp(const sparse::matrix<value_t,
    //                          index_t> &A, const params &prm)
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGAggregation(const BaseVector<int>& aggregates,
                                                  BaseMatrix<ValueType>* prolong) const
    {
        assert(prolong != NULL);

        const HostVector<int>*    cast_agg     = dynamic_cast<const HostVector<int>*>(&aggregates);
        HostMatrixCSR<ValueType>* cast_prolong = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong);

        assert(cast_agg != NULL);
        assert(cast_prolong != NULL);

        int ncol = 0;

        for(int i = 0; i < cast_agg->GetSize(); ++i)
        {
            if(cast_agg->vec_[i] > ncol)
            {
                ncol = cast_agg->vec_[i];
            }
        }

        ++ncol;

        int* row_offset = NULL;
        allocate_host(this->nrow_ + 1, &row_offset);

        int*       col = NULL;
        ValueType* val = NULL;

        row_offset[0] = 0;
        for(int i = 0; i < this->nrow_; ++i)
        {
            if(cast_agg->vec_[i] >= 0)
            {
                row_offset[i + 1] = row_offset[i] + 1;
            }
            else
            {
                row_offset[i + 1] = row_offset[i];
            }
        }

        allocate_host(row_offset[this->nrow_], &col);
        allocate_host(row_offset[this->nrow_], &val);

        for(int i = 0, j = 0; i < this->nrow_; ++i)
        {
            if(cast_agg->vec_[i] >= 0)
            {
                col[j] = cast_agg->vec_[i];
                val[j] = 1.0;
                ++j;
            }
        }

        cast_prolong->Clear();
        cast_prolong->SetDataPtrCSR(
            &row_offset, &col, &val, row_offset[this->nrow_], this->nrow_, ncol);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::FSAI(int power, const BaseMatrix<ValueType>* pattern)
    {
        // Extract lower triangular matrix of A
        HostMatrixCSR<ValueType> L(this->local_backend_);

        const HostMatrixCSR<ValueType>* cast_pattern = NULL;
        if(pattern != NULL)
        {
            cast_pattern = dynamic_cast<const HostMatrixCSR<ValueType>*>(pattern);
            assert(cast_pattern != NULL);

            cast_pattern->ExtractLDiagonal(&L);
        }
        else if(power > 1)
        {
            HostMatrixCSR<ValueType> structure(this->local_backend_);
            structure.CopyFrom(*this);
            structure.SymbolicPower(power);
            structure.ExtractLDiagonal(&L);
        }
        else
        {
            this->ExtractLDiagonal(&L);
        }

        int        nnz        = L.nnz_;
        int        nrow       = L.nrow_;
        int        ncol       = L.ncol_;
        int*       row_offset = NULL;
        int*       col        = NULL;
        ValueType* val        = NULL;

        L.LeaveDataPtrCSR(&row_offset, &col, &val);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            // entries of ai-th row
            int nnz_row = row_offset[ai + 1] - row_offset[ai];

            if(nnz_row == 1)
            {
                int aj = this->mat_.row_offset[ai];
                if(this->mat_.col[aj] == ai)
                {
                    val[row_offset[ai]] = static_cast<ValueType>(1) / this->mat_.val[aj];
                }
            }
            else
            {
                // create submatrix taking only the lower tridiagonal part into account
                std::vector<ValueType> Asub(nnz_row * nnz_row, static_cast<ValueType>(0));

                for(int k = 0; k < nnz_row; ++k)
                {
                    int row_begin = this->mat_.row_offset[col[row_offset[ai] + k]];
                    int row_end   = this->mat_.row_offset[col[row_offset[ai] + k] + 1];

                    for(int aj = row_begin; aj < row_end; ++aj)
                    {
                        for(int j = 0; j < nnz_row; ++j)
                        {
                            int Asub_col = col[row_offset[ai] + j];

                            if(this->mat_.col[aj] < Asub_col)
                            {
                                break;
                            }

                            if(this->mat_.col[aj] == Asub_col)
                            {
                                Asub[j + k * nnz_row] = this->mat_.val[aj];
                                break;
                            }
                        }

                        if(this->mat_.col[aj] == ai)
                        {
                            break;
                        }
                    }
                }

                std::vector<ValueType> mk(nnz_row, static_cast<ValueType>(0));
                mk[nnz_row - 1] = static_cast<ValueType>(1);

                // compute inplace LU factorization of Asub
                for(int i = 0; i < nnz_row - 1; ++i)
                {
                    for(int k = i + 1; k < nnz_row; ++k)
                    {
                        Asub[i + k * nnz_row] /= Asub[i + i * nnz_row];

                        for(int j = i + 1; j < nnz_row; ++j)
                        {
                            Asub[j + k * nnz_row] -= Asub[i + k * nnz_row] * Asub[j + i * nnz_row];
                        }
                    }
                }

                // backward sweeps
                for(int i = nnz_row - 1; i >= 0; --i)
                {
                    mk[i] /= Asub[i + i * nnz_row];

                    for(int j = 0; j < i; ++j)
                    {
                        mk[j] -= mk[i] * Asub[i + j * nnz_row];
                    }
                }

                // update the preconditioner matrix with mk
                for(int aj = row_offset[ai], k = 0; aj < row_offset[ai + 1]; ++aj, ++k)
                {
                    val[aj] = mk[k];
                }
            }
        }

// Scaling
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < nrow; ++ai)
        {
            ValueType fac = sqrt(static_cast<ValueType>(1) / std::abs(val[row_offset[ai + 1] - 1]));

            for(int aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
            {
                val[aj] *= fac;
            }
        }

        this->Clear();
        this->SetDataPtrCSR(&row_offset, &col, &val, nnz, nrow, ncol);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::SPAI(void)
    {
        int nrow = this->nrow_;
        int nnz  = this->nnz_;

        ValueType* val = NULL;
        allocate_host(nnz, &val);

        HostMatrixCSR<ValueType> T(this->local_backend_);
        T.CopyFrom(*this);
        this->Transpose();

// Loop over each row to get J indexing vector
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < nrow; ++i)
        {
            int* J     = NULL;
            int  Jsize = this->mat_.row_offset[i + 1] - this->mat_.row_offset[i];
            allocate_host(Jsize, &J);
            std::vector<int> I;

            // Setup J = {j | m(j) != 0}
            for(int j = this->mat_.row_offset[i], idx = 0; j < this->mat_.row_offset[i + 1];
                ++j, ++idx)
            {
                J[idx] = this->mat_.col[j];
            }

            // Setup I = {i | row A(i,J) != 0}
            for(int idx = 0; idx < Jsize; ++idx)
            {
                for(int j = this->mat_.row_offset[J[idx]]; j < this->mat_.row_offset[J[idx] + 1];
                    ++j)
                {
                    if(std::find(I.begin(), I.end(), this->mat_.col[j]) == I.end())
                    {
                        I.push_back(this->mat_.col[j]);
                    }
                }
            }

            // Build dense matrix
            HostMatrixDENSE<ValueType> Asub(this->local_backend_);
            Asub.AllocateDENSE(int(I.size()), Jsize);

            for(int k = 0; k < Asub.nrow_; ++k)
            {
                for(int aj = T.mat_.row_offset[I[k]]; aj < T.mat_.row_offset[I[k] + 1]; ++aj)
                {
                    for(int j = 0; j < Jsize; ++j)
                    {
                        if(T.mat_.col[aj] == J[j])
                        {
                            Asub.mat_.val[DENSE_IND(k, j, Asub.nrow_, Asub.ncol_)] = T.mat_.val[aj];
                        }
                    }
                }
            }

            // QR Decomposition of dense submatrix Ak
            Asub.QRDecompose();

            // Solve least squares
            HostVector<ValueType> ek(this->local_backend_);
            HostVector<ValueType> mk(this->local_backend_);

            ek.Allocate(Asub.nrow_);
            mk.Allocate(Asub.ncol_);

            for(int j = 0; j < ek.GetSize(); ++j)
            {
                if(I[j] == i)
                {
                    ek.vec_[j] = 1.0;
                }
            }

            Asub.QRSolve(ek, &mk);

            // Write m_k into preconditioner matrix
            for(int j = 0; j < Jsize; ++j)
            {
                val[this->mat_.row_offset[i] + j] = mk.vec_[j];
            }

            // Clear index vectors
            I.clear();
            ek.Clear();
            mk.Clear();
            Asub.Clear();
            free_host(&J);
        }

        // Only reset value array since we keep the sparsity pattern of A
        free_host(&this->mat_.val);
        this->mat_.val = val;

        this->Transpose();

        return true;
    }

    // ----------------------------------------------------------
    // original functions:
    //   transfer_operators(const Matrix &A, const params &prm)
    //   connect(backend::crs<Val, Col, Ptr> const &A,
    //           float eps_strong,
    //           backend::crs<char, Col, Ptr> &S,
    //           std::vector<char> &cf)
    // ----------------------------------------------------------
    // Modified and adopted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adopted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSCoarsening(float             eps,
                                                BaseVector<int>*  CFmap,
                                                BaseVector<bool>* S) const
    {
        assert(CFmap != NULL);
        assert(S != NULL);

        HostVector<int>*  cast_cf = dynamic_cast<HostVector<int>*>(CFmap);
        HostVector<bool>* cast_S  = dynamic_cast<HostVector<bool>*>(S);

        assert(cast_cf != NULL);
        assert(cast_S != NULL);

        // Allocate CF mapping
        cast_cf->Clear();
        cast_cf->Allocate(this->nrow_);

        // Mark all vertices as undecided
        cast_cf->Zeros();

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

        // Array of strong couplings S
        int* S_row_offset = NULL;
        int* S_col        = NULL;

        allocate_host(this->nrow_ + 1, &S_row_offset);
        set_to_zero_host(this->nrow_ + 1, S_row_offset);

// Determine strong influences in matrix (Ruge Stuben approach)
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            // Determine minimum and maximum off-diagonal of the current row
            ValueType min_a_ik = static_cast<ValueType>(0);
            ValueType max_a_ik = static_cast<ValueType>(0);

            int row_begin = this->mat_.row_offset[i];
            int row_end   = this->mat_.row_offset[i + 1];

            // True, if the diagonal element is negative
            bool sign = false;

            // Determine diagonal sign and min/max
            for(int j = row_begin; j < row_end; ++j)
            {
                int       col = this->mat_.col[j];
                ValueType val = this->mat_.val[j];

                if(col == i)
                {
                    // Get diagonal entry sign
                    sign = val < static_cast<ValueType>(0);
                }
                else
                {
                    // Get min / max entries
                    min_a_ik = (min_a_ik < val) ? min_a_ik : val;
                    max_a_ik = (max_a_ik > val) ? max_a_ik : val;
                }
            }

            // Threshold to check for strength of connection
            ValueType cond = (sign ? max_a_ik : min_a_ik) * static_cast<ValueType>(eps);

            // Fill S
            for(int j = row_begin; j < row_end; ++j)
            {
                int       col = this->mat_.col[j];
                ValueType val = this->mat_.val[j];

                cast_S->vec_[j] = (col != i) && (val < cond);
            }

            // If cond is zero -> i is independent of other grid points
            if(cond == static_cast<ValueType>(0))
            {
                cast_cf->vec_[i] = 2;
            }
        }

        // Transpose S
        for(int i = 0; i < this->nnz_; ++i)
        {
            if(cast_S->vec_[i])
            {
                S_row_offset[this->mat_.col[i] + 1]++;
            }
        }

        for(int i = 0; i < this->nrow_; ++i)
        {
            S_row_offset[i + 1] += S_row_offset[i];
        }

        allocate_host(S_row_offset[this->nrow_], &S_col);

        for(int i = 0; i < this->nrow_; ++i)
        {
            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(cast_S->vec_[j])
                {
                    S_col[S_row_offset[this->mat_.col[j]]++] = i;
                }
            }
        }

        for(int i = this->nrow_; i > 0; --i)
        {
            S_row_offset[i] = S_row_offset[i - 1];
        }

        S_row_offset[0] = 0;

        // Split into C and F
        std::vector<int> lambda(this->nrow_);

        for(int i = 0; i < this->nrow_; ++i)
        {
            int temp = 0;
            for(int j = S_row_offset[i]; j < S_row_offset[i + 1]; ++j)
            {
                temp += (cast_cf->vec_[S_col[j]] == 0 ? 1 : 2);
            }

            lambda[i] = temp;
        }

        std::vector<int> ptr(this->nrow_ + 1, static_cast<int>(0));
        std::vector<int> cnt(this->nrow_, static_cast<int>(0));
        std::vector<int> i2n(this->nrow_);
        std::vector<int> n2i(this->nrow_);

        for(int i = 0; i < this->nrow_; ++i)
        {
            ptr[lambda[i] + 1]++;
        }

        for(unsigned int i = 1; i < ptr.size(); ++i)
        {
            ptr[i] += ptr[i - 1];
        }

        for(int i = 0; i < this->nrow_; ++i)
        {
            int lam  = lambda[i];
            int idx  = ptr[lam] + cnt[lam]++;
            i2n[idx] = i;
            n2i[i]   = idx;
        }

        for(int top = this->nrow_ - 1; top >= 0; --top)
        {
            int i   = i2n[top];
            int lam = lambda[i];

            if(lam == 0)
            {
                for(int ai = 0; ai < this->nrow_; ++ai)
                {
                    if(cast_cf->vec_[ai] == 0)
                    {
                        cast_cf->vec_[ai] = 1;
                    }
                }

                break;
            }

            cnt[lam]--;

            if(cast_cf->vec_[i] == 2)
            {
                continue;
            }

            assert(cast_cf->vec_[i] == 0);

            cast_cf->vec_[i] = 1;

            for(int j = S_row_offset[i]; j < S_row_offset[i + 1]; ++j)
            {
                int c = S_col[j];

                if(cast_cf->vec_[c] != 0)
                {
                    continue;
                }

                cast_cf->vec_[c] = 2;

                for(int jj = this->mat_.row_offset[c]; jj < this->mat_.row_offset[c + 1]; ++jj)
                {
                    if(!cast_S->vec_[jj])
                    {
                        continue;
                    }

                    int cc     = this->mat_.col[jj];
                    int lam_cc = lambda[cc];

                    if(cast_cf->vec_[cc] != 0 || lam_cc >= this->nrow_ - 1)
                    {
                        continue;
                    }

                    int old_pos = n2i[cc];
                    int new_pos = ptr[lam_cc] + cnt[lam_cc] - 1;

                    n2i[i2n[old_pos]] = new_pos;
                    n2i[i2n[new_pos]] = old_pos;

                    std::swap(i2n[old_pos], i2n[new_pos]);

                    --cnt[lam_cc];
                    ++cnt[lam_cc + 1];
                    ptr[lam_cc + 1] = ptr[lam_cc] + cnt[lam_cc];

                    ++lambda[cc];
                }
            }

            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(!cast_S->vec_[j])
                {
                    continue;
                }

                int c   = this->mat_.col[j];
                int lam = lambda[c];

                if(cast_cf->vec_[c] != 0 || lam == 0)
                {
                    continue;
                }

                int old_pos = n2i[c];
                int new_pos = ptr[lam];

                n2i[i2n[old_pos]] = new_pos;
                n2i[i2n[new_pos]] = old_pos;

                std::swap(i2n[old_pos], i2n[new_pos]);

                --cnt[lam];
                ++cnt[lam - 1];
                ++ptr[lam];
                --lambda[c];

                assert(ptr[lam - 1] == ptr[lam] - cnt[lam - 1]);
            }
        }

        // Clean up
        free_host(&S_row_offset);
        free_host(&S_col);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSPMISCoarsening(float             eps,
                                                    BaseVector<int>*  CFmap,
                                                    BaseVector<bool>* S) const
    {
        assert(CFmap != NULL);
        assert(S != NULL);

        HostVector<int>*  cast_cf = dynamic_cast<HostVector<int>*>(CFmap);
        HostVector<bool>* cast_S  = dynamic_cast<HostVector<bool>*>(S);

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
        HostVector<float> omega(this->local_backend_);
        omega.Allocate(this->nrow_);
        omega.SetRandomUniform(1234ULL, 0, 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        // Determine strong influences in the matrix
        for(int i = 0; i < this->nrow_; ++i)
        {
            // Determine minimum and maximum off-diagonal of the current row
            ValueType min_a_ik = static_cast<ValueType>(0);
            ValueType max_a_ik = static_cast<ValueType>(0);

            int row_begin = this->mat_.row_offset[i];
            int row_end   = this->mat_.row_offset[i + 1];

            // True, if the diagonal element is negative
            bool sign = false;

            // Determine diagonal sign and min/max
            for(int j = row_begin; j < row_end; ++j)
            {
                int       col = this->mat_.col[j];
                ValueType val = this->mat_.val[j];

                if(col == i)
                {
                    // Get diagonal entry sign
                    sign = val < static_cast<ValueType>(0);
                }
                else
                {
                    // Get min / max entries
                    min_a_ik = (min_a_ik < val) ? min_a_ik : val;
                    max_a_ik = (max_a_ik > val) ? max_a_ik : val;
                }
            }

            // Threshold to check for strength of connection
            ValueType cond = (sign ? max_a_ik : min_a_ik) * static_cast<ValueType>(eps);

            // Fill S
            for(int j = row_begin; j < row_end; ++j)
            {
                int       col = this->mat_.col[j];
                ValueType val = this->mat_.val[j];

                if(col != i && val < cond)
                {
                    // col is strongly connected to i
                    cast_S->vec_[j] = true;

#ifdef _OPENMP
#pragma omp atomic
#endif
                    // Increment omega, as it holds all strongly connected edges
                    // of vertex col.
                    // Additionally, omega holds a random number between 0 and 1 to
                    // distinguish neighbor points with equal number of strong
                    // connections.
                    omega.vec_[col] += 1.0f;
                }
            }
        }

        // Mark all vertices as undecided
        cast_cf->Zeros();

        bool* workspace = NULL;
        allocate_host(this->nrow_, &workspace);

        // Iteratively find coarse and fine vertices until all undecided vertices have
        // been marked (JPL approach)
        int iter = 0;

        while(true)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            // First, mark all vertices that have not been assigned yet, as coarse
            for(int i = 0; i < this->nrow_; ++i)
            {
                // workspace keeps track, whether a vertex has been marked coarse
                // during the current iteration, or not.
                workspace[i] = false;

                // Check only undecided vertices
                if(cast_cf->vec_[i] == 0)
                {
                    // If this vertex has an edge, it might be a coarse one
                    if(omega.vec_[i] >= 1.0f)
                    {
                        cast_cf->vec_[i] = 1;

                        // Keep in mind, that this vertex has been marked coarse in the
                        // current iteration
                        workspace[i] = true;
                    }
                    else
                    {
                        // This point does not influence any other points and thus is a
                        // fine point
                        cast_cf->vec_[i] = 2;
                    }
                }
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            // Now, correct previously marked vertices with respect to omega
            for(int i = 0; i < this->nrow_; ++i)
            {
                int row_begin = this->mat_.row_offset[i];
                int row_end   = this->mat_.row_offset[i + 1];

                // If this vertex has been marked coarse in the current iteration,
                // process it for further checks
                if(workspace[i])
                {
                    // Get the weight of the current row for comparison
                    float omega_row = omega.vec_[i];

                    // Loop over the full row to compare weights of other vertices that
                    // have been marked coarse in the current iteration
                    for(int j = row_begin; j < row_end; ++j)
                    {
                        // Process only vertices that are strongly connected
                        if(cast_S->vec_[j])
                        {
                            int col = this->mat_.col[j];

                            // If this vertex has been marked coarse in the current iteration,
                            // we need to check whether it is accepted as a coarse vertex or not.
                            if(workspace[col])
                            {
                                // Get the weight of the current vertex for comparison
                                float omega_col = omega.vec_[col];

                                if(omega_row > omega_col)
                                {
                                    // The diagonal entry has more edges and will remain
                                    // a coarse point, whereas this vertex gets reverted
                                    // back to undecided, for further processing.
                                    cast_cf->vec_[col] = 0;
                                }
                                else if(omega_row < omega_col)
                                {
                                    // The diagonal entry has fewer edges and gets
                                    // reverted back to undecided for further processing,
                                    // whereas this vertex stays
                                    // a coarse one.
                                    cast_cf->vec_[i] = 0;
                                }
                            }
                        }
                    }
                }
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            // Mark remaining edges of a coarse point to fine
            for(int i = 0; i < this->nrow_; ++i)
            {
                // Process only undecided vertices
                if(cast_cf->vec_[i] == 0)
                {
                    int row_begin = this->mat_.row_offset[i];
                    int row_end   = this->mat_.row_offset[i + 1];

                    // Loop over all edges of this undecided vertex
                    // and check, if there is a coarse point connected
                    for(int j = row_begin; j < row_end; ++j)
                    {
                        // Check, whether this edge is strongly connected to the vertex
                        if(cast_S->vec_[j])
                        {
                            int col = this->mat_.col[j];

                            // If this edge is coarse, our vertex must be fine
                            if(cast_cf->vec_[col] == 1)
                            {
                                cast_cf->vec_[i] = 2;
                                break;
                            }
                        }
                    }
                }
            }

            // Now, we need to check whether we have vertices left that are marked
            // undecided, in order to restart the loop
            bool undecided = false;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                // Check whether this vertex is undecided or not
                if(cast_cf->vec_[i] == 0)
                {
                    undecided = true;
                    i         = this->nrow_;
                }
            }

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
                                 "*** warning: HostMatrixCSR::RSPMISCoarsening() Current "
                                 "number of iterations: "
                                     << iter);
            }
        }

        free_host(&workspace);

        return true;
    }

    // ----------------------------------------------------------
    // original functions:
    //   cfsplit(backend::crs<Val, Col, Ptr> const &A,
    //           backend::crs<char, Col, Ptr> const &S,
    //           std::vector<char> &cf)
    // ----------------------------------------------------------
    // Modified and adopted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adopted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSDirectInterpolation(const BaseVector<int>&  CFmap,
                                                         const BaseVector<bool>& S,
                                                         BaseMatrix<ValueType>*  prolong,
                                                         BaseMatrix<ValueType>* restrict) const
    {
        assert(prolong != NULL);

        HostMatrixCSR<ValueType>* cast_prolong  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong);
        HostMatrixCSR<ValueType>* cast_restrict = dynamic_cast<HostMatrixCSR<ValueType>*>(restrict);

        const HostVector<int>*  cast_cf = dynamic_cast<const HostVector<int>*>(&CFmap);
        const HostVector<bool>* cast_S  = dynamic_cast<const HostVector<bool>*>(&S);

        assert(cast_prolong != NULL);
        assert(cast_cf != NULL);
        assert(cast_S != NULL);

        // Allocate
        cast_prolong->Clear();

        // Build coarsening operators
        int              nc = 0;
        std::vector<int> cidx(this->nrow_);

        for(int i = 0; i < this->nrow_; ++i)
        {
            if(cast_cf->vec_[i] == 1)
            {
                cidx[i] = nc++;
            }
        }

        std::vector<ValueType> Amin(this->nrow_);
        std::vector<ValueType> Amax(this->nrow_);

        // Allocate P row pointer array
        allocate_host(this->nrow_ + 1, &cast_prolong->mat_.row_offset);

        // Initialize P row pointer array with zeros
        set_to_zero_host(this->nrow_ + 1, cast_prolong->mat_.row_offset);

        cast_prolong->nrow_ = this->nrow_;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            if(cast_cf->vec_[i] == 1)
            {
                ++cast_prolong->mat_.row_offset[i + 1];
                continue;
            }

            ValueType amin = static_cast<ValueType>(0);
            ValueType amax = static_cast<ValueType>(0);

            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(!cast_S->vec_[j] || cast_cf->vec_[this->mat_.col[j]] != 1)
                {
                    continue;
                }

                amin = (amin < this->mat_.val[j]) ? amin : this->mat_.val[j];
                amax = (amax > this->mat_.val[j]) ? amax : this->mat_.val[j];
            }

            Amin[i] = amin = amin * static_cast<ValueType>(0.2);
            Amax[i] = amax = amax * static_cast<ValueType>(0.2);

            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(!cast_S->vec_[j] || cast_cf->vec_[this->mat_.col[j]] != 1)
                {
                    continue;
                }

                if(this->mat_.val[j] <= amin || this->mat_.val[j] >= amax)
                {
                    ++cast_prolong->mat_.row_offset[i + 1];
                }
            }
        }

        for(int i = 0; i < this->nrow_; ++i)
        {
            cast_prolong->mat_.row_offset[i + 1] += cast_prolong->mat_.row_offset[i];
        }

        int nnz_prolong = cast_prolong->mat_.row_offset[this->nrow_];

        allocate_host(nnz_prolong, &cast_prolong->mat_.col);
        allocate_host(nnz_prolong, &cast_prolong->mat_.val);

        cast_prolong->nnz_  = nnz_prolong;
        cast_prolong->ncol_ = nc;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            int row_head = cast_prolong->mat_.row_offset[i];

            if(cast_cf->vec_[i] == 1)
            {
                cast_prolong->mat_.col[row_head] = cidx[i];
                cast_prolong->mat_.val[row_head] = static_cast<ValueType>(1);
                continue;
            }

            ValueType diag  = static_cast<ValueType>(0);
            ValueType a_num = static_cast<ValueType>(0), a_den = static_cast<ValueType>(0);
            ValueType b_num = static_cast<ValueType>(0), b_den = static_cast<ValueType>(0);
            ValueType d_neg = static_cast<ValueType>(0), d_pos = static_cast<ValueType>(0);

            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int       c = this->mat_.col[j];
                ValueType v = this->mat_.val[j];

                if(c == i)
                {
                    diag = v;
                    continue;
                }

                if(v < static_cast<ValueType>(0))
                {
                    a_num += v;
                    if(cast_S->vec_[j] && cast_cf->vec_[c] == 1)
                    {
                        a_den += v;
                        if(v > Amin[i])
                        {
                            d_neg += v;
                        }
                    }
                }
                else
                {
                    b_num += v;
                    if(cast_S->vec_[j] && cast_cf->vec_[c] == 1)
                    {
                        b_den += v;
                        if(v < Amax[i])
                        {
                            d_pos += v;
                        }
                    }
                }
            }

            ValueType cf_neg = static_cast<ValueType>(1);
            ValueType cf_pos = static_cast<ValueType>(1);

            if(std::abs(a_den - d_neg) > 1e-32)
            {
                cf_neg = a_den / (a_den - d_neg);
            }

            if(std::abs(b_den - d_pos) > 1e-32)
            {
                cf_pos = b_den / (b_den - d_pos);
            }

            if(b_num > static_cast<ValueType>(0) && std::abs(b_den) < 1e-32)
            {
                diag += b_num;
            }

            ValueType alpha = std::abs(a_den) > 1e-32 ? -cf_neg * a_num / (diag * a_den)
                                                      : static_cast<ValueType>(0);
            ValueType beta  = std::abs(b_den) > 1e-32 ? -cf_pos * b_num / (diag * b_den)
                                                      : static_cast<ValueType>(0);

            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int       c = this->mat_.col[j];
                ValueType v = this->mat_.val[j];

                if(!cast_S->vec_[j] || cast_cf->vec_[c] != 1)
                {
                    continue;
                }

                if(v > Amin[i] && v < Amax[i])
                {
                    continue;
                }

                cast_prolong->mat_.col[row_head] = cidx[c];
                cast_prolong->mat_.val[row_head]
                    = (v < static_cast<ValueType>(0) ? alpha : beta) * v;
                ++row_head;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < cast_prolong->GetM(); ++i)
        {
            for(int j = cast_prolong->mat_.row_offset[i]; j < cast_prolong->mat_.row_offset[i + 1];
                ++j)
            {
                for(int jj = cast_prolong->mat_.row_offset[i];
                    jj < cast_prolong->mat_.row_offset[i + 1] - 1;
                    ++jj)
                {
                    if(cast_prolong->mat_.col[jj] > cast_prolong->mat_.col[jj + 1])
                    {
                        // swap elements
                        int       ind = cast_prolong->mat_.col[jj];
                        ValueType val = cast_prolong->mat_.val[jj];

                        cast_prolong->mat_.col[jj] = cast_prolong->mat_.col[jj + 1];
                        cast_prolong->mat_.val[jj] = cast_prolong->mat_.val[jj + 1];

                        cast_prolong->mat_.col[jj + 1] = ind;
                        cast_prolong->mat_.val[jj + 1] = val;
                    }
                }
            }
        }

        if(cast_restrict != NULL)
        {
            cast_prolong->Transpose(cast_restrict);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSExtPIInterpolation(const BaseVector<int>&  CFmap,
                                                        const BaseVector<bool>& S,
                                                        bool                    FF1,
                                                        float                   trunc,
                                                        BaseMatrix<ValueType>*  prolong,
                                                        BaseMatrix<ValueType>* restrict) const
    {
        assert(trunc >= 0.0f);
        assert(prolong != NULL);

        HostMatrixCSR<ValueType>* cast_prolong  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong);
        HostMatrixCSR<ValueType>* cast_restrict = dynamic_cast<HostMatrixCSR<ValueType>*>(restrict);

        const HostVector<int>*  cast_cf = dynamic_cast<const HostVector<int>*>(&CFmap);
        const HostVector<bool>* cast_S  = dynamic_cast<const HostVector<bool>*>(&S);

        assert(cast_prolong != NULL);
        assert(cast_cf != NULL);
        assert(cast_S != NULL);

        // Start with fresh P
        cast_prolong->Clear();

        // Allocate P row pointer array
        allocate_host(this->nrow_ + 1, &cast_prolong->mat_.row_offset);

        // We already know the number of rows of P
        cast_prolong->nrow_ = this->nrow_;

        // Temporary buffer
        int* workspace = NULL;
        allocate_host(this->nrow_ + 1, &workspace);
        set_to_zero_host(this->nrow_ + 1, workspace);

        // Determine nnz per row of P
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            // Coarse points generate a single entry
            if(cast_cf->vec_[i] == 1)
            {
                workspace[i + 1]                     = 1;
                cast_prolong->mat_.row_offset[i + 1] = 1;

                continue;
            }

            // Set, to discard duplicated column entries (no need to be ordered).
            std::unordered_set<int> set;

            // Row entry and exit points
            int row_begin = this->mat_.row_offset[i];
            int row_end   = this->mat_.row_offset[i + 1];

            // Loop over all columns of the i-th row, whereas each lane processes a column
            for(int j = row_begin; j < row_end; ++j)
            {
                // Skip points that do not influence the current point
                if(cast_S->vec_[j] == false)
                {
                    continue;
                }

                // Get the column index
                int col_j = this->mat_.col[j];

                // Skip diagonal entries (i does not influence itself)
                if(col_j == i)
                {
                    continue;
                }

                // Switch between coarse and fine points that influence the i-th point
                if(cast_cf->vec_[col_j] == 1)
                {
                    // This is a coarse point and thus contributes, count it for the row nnz
                    // We need to use a set here, to discard duplicates.
                    set.insert(col_j);
                }
                else
                {
                    // This is a fine point, check for strongly connected coarse points

                    // Row entry and exit of this fine point
                    int row_begin_j = this->mat_.row_offset[col_j];
                    int row_end_j   = this->mat_.row_offset[col_j + 1];

                    // Loop over all columns of the fine point
                    for(int k = row_begin_j; k < row_end_j; ++k)
                    {
                        // Skip points that do not influence the fine point
                        if(cast_S->vec_[k] == false)
                        {
                            continue;
                        }

                        // Get the column index
                        int col_k = this->mat_.col[k];

                        // Skip diagonal entries (the fine point does not influence itself)
                        if(col_k == col_j)
                        {
                            continue;
                        }

                        // Check whether k is a coarse point
                        if(cast_cf->vec_[col_k] == 1)
                        {
                            set.insert(col_k);

                            // FF1 limitation
                            if(FF1 == true)
                            {
                                break;
                            }
                        }
                    }
                }
            }

            // The number of entries of the set results in the nnz of row i
            cast_prolong->mat_.row_offset[i + 1] = set.size();
        }

        // Exclusive sum to obtain row pointers
        cast_prolong->mat_.row_offset[0] = 0;

        for(int i = 0; i < this->nrow_; ++i)
        {
            workspace[i + 1] += workspace[i];
            cast_prolong->mat_.row_offset[i + 1] += cast_prolong->mat_.row_offset[i];
        }

        cast_prolong->ncol_ = workspace[this->nrow_];
        cast_prolong->nnz_  = cast_prolong->mat_.row_offset[this->nrow_];

        // Allocate column and value arrays
        allocate_host(cast_prolong->nnz_, &cast_prolong->mat_.col);
        allocate_host(cast_prolong->nnz_, &cast_prolong->mat_.val);

        // Extract diagonal matrix entries
        HostVector<ValueType> diag(this->local_backend_);
        diag.Allocate(this->nrow_);

        this->ExtractDiagonal(&diag);

        // Fill column indices and values of P
        ValueType one  = static_cast<ValueType>(1);
        ValueType zero = static_cast<ValueType>(0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            // Coarse points generate a single entry
            if(cast_cf->vec_[i] == 1)
            {
                // Get index into P
                int idx = cast_prolong->mat_.row_offset[i];

                // Single entry in thie row (coarse point)
                cast_prolong->mat_.col[idx] = workspace[i];
                cast_prolong->mat_.val[idx] = one;

                continue;
            }

            // Hash table
            std::map<int, ValueType> table;

            // Fill the hash table according to the nnz pattern of P
            // This is identical to the nnz per row part

            // Row entry and exit points
            int row_begin = this->mat_.row_offset[i];
            int row_end   = this->mat_.row_offset[i + 1];

            // Loop over all columns of the i-th row, whereas each lane processes a column
            for(int k = row_begin; k < row_end; ++k)
            {
                // Skip points that do not influence the current point
                if(cast_S->vec_[k] == false)
                {
                    continue;
                }

                // Get the column index
                int col_ik = this->mat_.col[k];

                // Skip diagonal entries (i does not influence itself)
                if(col_ik == i)
                {
                    continue;
                }

                // Switch between coarse and fine points that influence the i-th point
                if(cast_cf->vec_[col_ik] == 1)
                {
                    // Explicitly create an entry in the hash table
                    table[col_ik] = zero;
                }
                else
                {
                    // This is a fine point, check for strongly connected coarse points

                    // Row entry and exit of this fine point
                    int row_begin_k = this->mat_.row_offset[col_ik];
                    int row_end_k   = this->mat_.row_offset[col_ik + 1];

                    // Loop over all columns of the fine point
                    for(int l = row_begin_k; l < row_end_k; ++l)
                    {
                        // Skip points that do not influence the fine point
                        if(cast_S->vec_[l] == false)
                        {
                            continue;
                        }

                        // Get the column index
                        int col_kl = this->mat_.col[l];

                        // Skip diagonal entries (the fine point does not influence itself)
                        if(col_kl == col_ik)
                        {
                            continue;
                        }

                        // Check whether l is a coarse point
                        if(cast_cf->vec_[col_kl] == 1)
                        {
                            // Explicitly create an entry in the hash table
                            table[col_kl] = zero;

                            // FF1 limitation
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
            ValueType val_ii = diag.vec_[i];

            // Sign of diagonal entry of i-th row
            bool pos_ii = val_ii >= zero;

            // Accumulators
            ValueType sum_k = zero;
            ValueType sum_n = zero;

            // Loop over all columns of the i-th row
            for(int k = row_begin; k < row_end; ++k)
            {
                // Get the column index
                int col_ik = this->mat_.col[k];

                // Skip diagonal entries (i does not influence itself)
                if(col_ik == i)
                {
                    continue;
                }

                // Get the column value
                ValueType val_ik = this->mat_.val[k];

                // Check, whether the k-th entry of the row is a fine point and strongly
                // connected to the i-th point (e.g. k \in F^S_i)
                if(cast_S->vec_[k] == true && cast_cf->vec_[col_ik] == 2)
                {
                    // Accumulator for the sum over l
                    ValueType sum_l = zero;

                    // Diagonal entry of k-th row
                    ValueType val_kk = diag.vec_[col_ik];

                    // Store a_ki, if present
                    ValueType val_ki = zero;

                    // Row entry and exit of this fine point
                    int row_begin_k = this->mat_.row_offset[col_ik];
                    int row_end_k   = this->mat_.row_offset[col_ik + 1];

                    // Loop over all columns of the fine point
                    for(int l = row_begin_k; l < row_end_k; ++l)
                    {
                        // Get the column index
                        int col_kl = this->mat_.col[l];

                        // Get the column value
                        ValueType val_kl = this->mat_.val[l];

                        // Sign of a_kl
                        bool pos_kl = val_kl >= zero;

                        // Differentiate between diagonal and off-diagonal
                        if(col_kl == i)
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
                        else if(cast_cf->vec_[col_kl] == 1)
                        {
                            // Check if sign is different from i-th row diagonal
                            if(pos_ii != pos_kl)
                            {
                                // Entry contributes only, if it is a coarse point
                                // and part of C^hat (e.g. we need to check the hash table)
                                if(table.find(col_kl) != table.end())
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
                    for(int l = row_begin_k; l < row_end_k; ++l)
                    {
                        // Get the column index
                        int col_kl = this->mat_.col[l];

                        // Only coarse points contribute
                        if(cast_cf->vec_[col_kl] != 1)
                        {
                            continue;
                        }

                        // Get the column value
                        ValueType val_kl = this->mat_.val[l];

                        // Compute the sign of a_kl
                        bool pos_kl = val_kl >= zero;

                        // Check for different sign
                        if(pos_kk != pos_kl)
                        {
                            if(table.find(col_kl) != table.end())
                            {
                                table[col_kl] += val_kl * sum_l;
                            }
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
                if(cast_cf->vec_[col_ik] == 1)
                {
                    // Check, whether col_ik is in C hat or not
                    if(table.find(col_ik) != table.end())
                    {
                        // Append a_ik to the sum of eq19
                        table[col_ik] += val_ik;

                        in_C_hat = true;
                    }
                }

                // If a_ik is not in C^hat and does not strongly influence i, it contributes
                // to sum_n
                if(in_C_hat == false && cast_S->vec_[k] == false)
                {
                    sum_n += val_ik;
                }
            }

            // Precompute a_ii tilde
            ValueType a_ii_tilde = -one / (val_ii + sum_n + sum_k);

            // Entry point into P
            int idx = cast_prolong->mat_.row_offset[i];

            // Finally, extract the numerical values from the hash table and fill P such
            // that the resulting matrix is sorted by columns
            for(auto it = table.begin(); it != table.end(); ++it)
            {
                cast_prolong->mat_.col[idx] = workspace[it->first];
                cast_prolong->mat_.val[idx] = a_ii_tilde * it->second;
                ++idx;
            }
        }

        // Free temporary buffer
        free_host(&workspace);

        // Transpose P to obtain R
        if(cast_restrict != NULL)
        {
            cast_prolong->Transpose(cast_restrict);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::InitialPairwiseAggregation(ValueType        beta,
                                                              int&             nc,
                                                              BaseVector<int>* G,
                                                              int&             Gsize,
                                                              int**            rG,
                                                              int&             rGsize,
                                                              int              ordering) const
    {
        assert(G != NULL);

        HostVector<int>* cast_G = dynamic_cast<HostVector<int>*>(G);

        assert(cast_G != NULL);

        // Initialize G
        for(int i = 0; i < cast_G->size_; ++i)
        {
            cast_G->vec_[i] = -2;
        }

        int  Usize    = 0;
        int* ind_diag = NULL;
        allocate_host(this->nrow_, &ind_diag);

        // Build U
        for(int i = 0; i < this->nrow_; ++i)
        {
            ValueType sum = static_cast<ValueType>(0);

            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(i != this->mat_.col[j])
                {
                    sum += std::abs(this->mat_.val[j]);
                }
                else
                {
                    ind_diag[i] = j;
                }
            }

            sum *= static_cast<ValueType>(5);

            if(this->mat_.val[ind_diag[i]] > sum)
            {
                cast_G->vec_[i] = -1;
                ++Usize;
            }
        }

        // Initialize rG and sizes
        Gsize  = 2;
        rGsize = this->nrow_ - Usize;
        allocate_host(Gsize * rGsize, rG);

        for(int i = 0; i < Gsize * rGsize; ++i)
        {
            (*rG)[i] = -1;
        }

        nc              = 0;
        ValueType betam = -beta;

        // Ordering
        HostVector<int> perm(this->local_backend_);

        switch(ordering)
        {
        case 0: // No ordering
            break;

        case 1: // Connectivity ordering
            this->ConnectivityOrder(&perm);
            break;

        case 2: // CMK
            this->CMK(&perm);
            break;

        case 3: // RCMK
            this->RCMK(&perm);
            break;

        case 4: // MIS
            int size;
            this->MaximalIndependentSet(size, &perm);
            break;

        case 5: // MultiColoring
            int  num_colors;
            int* size_colors = NULL;
            this->MultiColoring(num_colors, &size_colors, &perm);
            free_host(&size_colors);
            break;
        }

        // Main algorithm

        // While U != empty
        for(int k = 0; k < this->nrow_; ++k)
        {
            // Pick i according to the connectivity
            int i;
            if(ordering == 0)
            {
                i = k;
            }
            else
            {
                i = perm.vec_[k];
            }

            // Check if i in G
            if(cast_G->vec_[i] != -2)
            {
                continue;
            }

            // Mark i visited and fill G and rG
            cast_G->vec_[i] = nc;
            (*rG)[nc]       = i;

            // Determine minimum and maximum offdiagonal entry and j index
            ValueType min_a_ij = static_cast<ValueType>(0);
            ValueType max_a_ij = static_cast<ValueType>(0);
            int       min_j    = -1;
            bool      neg      = false;
            if(this->mat_.val[ind_diag[i]] < static_cast<ValueType>(0))
            {
                neg = true;
            }

            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int       col_j = this->mat_.col[j];
                ValueType val_j = this->mat_.val[j];

                if(neg == true)
                {
                    val_j *= static_cast<ValueType>(-1);
                }

                if(i == col_j)
                {
                    continue;
                }

                if(min_j == -1)
                {
                    max_a_ij = val_j;
                    if(cast_G->vec_[col_j] == -2)
                    {
                        min_j    = j;
                        min_a_ij = val_j;
                    }
                }

                if(val_j < min_a_ij && cast_G->vec_[col_j] == -2)
                {
                    min_a_ij = val_j;
                    min_j    = j;
                }

                if(val_j > max_a_ij)
                {
                    max_a_ij = val_j;
                }
            }

            // Fill G
            if(min_j != -1)
            {
                max_a_ij *= betam;

                int       col_j = this->mat_.col[min_j];
                ValueType val_j = this->mat_.val[min_j];

                if(neg == true)
                {
                    val_j *= static_cast<ValueType>(-1);
                }

                // Add j if conditions are fulfilled
                if(val_j < max_a_ij)
                {
                    cast_G->vec_[col_j] = nc;
                    (*rG)[rGsize + nc]  = col_j;
                }
            }

            ++nc;
        }

        free_host(&ind_diag);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::InitialPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                                              ValueType                    beta,
                                                              int&                         nc,
                                                              BaseVector<int>*             G,
                                                              int&                         Gsize,
                                                              int**                        rG,
                                                              int&                         rGsize,
                                                              int ordering) const
    {
        assert(G != NULL);

        HostVector<int>*                cast_G = dynamic_cast<HostVector<int>*>(G);
        const HostMatrixCSR<ValueType>* cast_mat
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat);

        assert(cast_G != NULL);
        assert(cast_mat != NULL);

        // Initialize G
        for(int i = 0; i < cast_G->size_; ++i)
        {
            cast_G->vec_[i] = -2;
        }

        int  Usize    = 0;
        int* ind_diag = NULL;
        allocate_host(this->nrow_, &ind_diag);

        // Build U
        for(int i = 0; i < this->nrow_; ++i)
        {
            ValueType sum = static_cast<ValueType>(0);

            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(i != this->mat_.col[j])
                {
                    sum += std::abs(this->mat_.val[j]);
                }
                else
                {
                    ind_diag[i] = j;
                }
            }

            if(cast_mat->nnz_ > 0)
            {
                for(int j = cast_mat->mat_.row_offset[i]; j < cast_mat->mat_.row_offset[i + 1]; ++j)
                {
                    sum += std::abs(cast_mat->mat_.val[j]);
                }
            }

            sum *= static_cast<ValueType>(5);

            if(this->mat_.val[ind_diag[i]] > sum)
            {
                cast_G->vec_[i] = -1;
                ++Usize;
            }
        }

        // Initialize rG and sizes
        Gsize  = 2;
        rGsize = this->nrow_ - Usize;
        allocate_host(Gsize * rGsize, rG);

        for(int i = 0; i < Gsize * rGsize; ++i)
        {
            (*rG)[i] = -1;
        }

        nc              = 0;
        ValueType betam = -beta;

        // Ordering
        HostVector<int> perm(this->local_backend_);

        switch(ordering)
        {
        case 0: // No ordering
            break;

        case 1: // Connectivity ordering
            this->ConnectivityOrder(&perm);
            break;

        case 2: // CMK
            this->CMK(&perm);
            break;

        case 3: // RCMK
            this->RCMK(&perm);
            break;

        case 4: // MIS
            int size;
            this->MaximalIndependentSet(size, &perm);
            break;

        case 5: // MultiColoring
            int  num_colors;
            int* size_colors = NULL;
            this->MultiColoring(num_colors, &size_colors, &perm);
            free_host(&size_colors);
            break;
        }

        // Main algorithm

        // While U != empty
        for(int k = 0; k < this->nrow_; ++k)
        {
            // Pick i according to the connectivity
            int i;
            if(ordering == 0)
            {
                i = k;
            }
            else
            {
                i = perm.vec_[k];
            }

            // Check if i in G
            if(cast_G->vec_[i] != -2)
            {
                continue;
            }

            // Mark i visited and fill G and rG
            cast_G->vec_[i] = nc;
            (*rG)[nc]       = i;

            // Determine minimum and maximum offdiagonal entry and j index
            ValueType min_a_ij = static_cast<ValueType>(0);
            ValueType max_a_ij = static_cast<ValueType>(0);
            int       min_j    = -1;
            bool      neg      = false;
            if(this->mat_.val[ind_diag[i]] < static_cast<ValueType>(0))
            {
                neg = true;
            }

            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int       col_j = this->mat_.col[j];
                ValueType val_j = this->mat_.val[j];

                if(neg == true)
                {
                    val_j *= static_cast<ValueType>(-1);
                }

                if(i == col_j)
                {
                    continue;
                }

                if(min_j == -1)
                {
                    max_a_ij = val_j;
                    if(cast_G->vec_[col_j] == -2)
                    {
                        min_j    = col_j;
                        min_a_ij = val_j;
                    }
                }

                if(val_j < min_a_ij && cast_G->vec_[col_j] == -2)
                {
                    min_a_ij = val_j;
                    min_j    = col_j;
                }

                if(val_j > max_a_ij)
                {
                    max_a_ij = val_j;
                }
            }

            if(cast_mat->nnz_ > 0)
            {
                for(int j = cast_mat->mat_.row_offset[i]; j < cast_mat->mat_.row_offset[i + 1]; ++j)
                {
                    ValueType val_j = cast_mat->mat_.val[j];

                    if(neg == true)
                    {
                        val_j *= static_cast<ValueType>(-1);
                    }

                    if(val_j > max_a_ij)
                    {
                        max_a_ij = val_j;
                    }
                }
            }

            // Fill G
            if(min_j != -1)
            {
                max_a_ij *= betam;

                for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    int       col_j = this->mat_.col[j];
                    ValueType val_j = this->mat_.val[j];

                    if(neg == true)
                    {
                        val_j *= static_cast<ValueType>(-1);
                    }

                    // Skip diagonal
                    if(i == col_j)
                    {
                        continue;
                    }

                    // Skip j which is not in U
                    if(cast_G->vec_[col_j] != -2)
                    {
                        continue;
                    }

                    // Add j if conditions are fulfilled
                    if(val_j < max_a_ij)
                    {
                        if(min_j == col_j)
                        {
                            cast_G->vec_[min_j] = nc;
                            (*rG)[rGsize + nc]  = min_j;
                            break;
                        }
                    }
                }
            }

            ++nc;
        }

        free_host(&ind_diag);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::FurtherPairwiseAggregation(ValueType        beta,
                                                              int&             nc,
                                                              BaseVector<int>* G,
                                                              int&             Gsize,
                                                              int**            rG,
                                                              int&             rGsize,
                                                              int              ordering) const
    {
        assert(G != NULL);
        HostVector<int>* cast_G = dynamic_cast<HostVector<int>*>(G);
        assert(cast_G != NULL);

        // Initialize G and inverse indexing for G
        Gsize *= 2;
        int  rGsizec = this->nrow_;
        int* rGc     = NULL;
        allocate_host(Gsize * rGsizec, &rGc);

        for(int i = 0; i < Gsize * rGsizec; ++i)
        {
            rGc[i] = -1;
        }

        for(int i = 0; i < cast_G->size_; ++i)
        {
            cast_G->vec_[i] = -1;
        }

        // Initialize U
        int* U = NULL;
        allocate_host(this->nrow_, &U);
        set_to_zero_host(this->nrow_, U);

        nc              = 0;
        ValueType betam = -beta;

        // Ordering
        HostVector<int> perm(this->local_backend_);

        switch(ordering)
        {
        case 0: // No ordering
            break;

        case 1: // Connectivity ordering
            this->ConnectivityOrder(&perm);
            break;

        case 2: // CMK
            this->CMK(&perm);
            break;

        case 3: // RCMK
            this->RCMK(&perm);
            break;

        case 4: // MIS
            int size;
            this->MaximalIndependentSet(size, &perm);
            break;

        case 5: // MultiColoring
            int  num_colors;
            int* size_colors = NULL;
            this->MultiColoring(num_colors, &size_colors, &perm);
            free_host(&size_colors);
            break;
        }

        // While U != empty
        for(int k = 0; k < this->nrow_; ++k)
        {
            // Pick i according to connectivity
            int i;
            if(ordering == 0)
            {
                i = k;
            }
            else
            {
                i = perm.vec_[k];
            }

            // Check if i in U
            if(U[i] == 1)
            {
                continue;
            }

            // Mark i visited
            U[i] = 1;

            // Fill G and rG
            for(int r = 0; r < Gsize / 2; ++r)
            {
                rGc[r * rGsizec + nc] = (*rG)[r * rGsize + i];

                if((*rG)[r * rGsize + i] >= 0)
                {
                    cast_G->vec_[(*rG)[r * rGsize + i]] = nc;
                }
            }

            // Get sign
            bool neg = false;
            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(i == this->mat_.col[j])
                {
                    if(this->mat_.val[j] < static_cast<ValueType>(0))
                    {
                        neg = true;
                    }

                    break;
                }
            }

            // Determine minimum and maximum offdiagonal entry and j index
            ValueType min_a_ij = static_cast<ValueType>(0);
            ValueType max_a_ij = static_cast<ValueType>(0);
            int       min_j    = -1;

            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int       col_j = this->mat_.col[j];
                ValueType val_j = this->mat_.val[j];

                if(neg == true)
                {
                    val_j *= static_cast<ValueType>(-1);
                }

                if(i == col_j)
                {
                    continue;
                }

                if(min_j == -1)
                {
                    max_a_ij = val_j;

                    if(U[col_j] == 0)
                    {
                        min_j    = j;
                        min_a_ij = max_a_ij;
                    }
                }

                if(val_j < min_a_ij && U[col_j] == 0)
                {
                    min_a_ij = val_j;
                    min_j    = j;
                }

                if(val_j < max_a_ij)
                {
                    max_a_ij = val_j;
                }
            }

            // Fill G
            if(min_j != -1)
            {
                max_a_ij *= betam;

                int       col_j = this->mat_.col[min_j];
                ValueType val_j = this->mat_.val[min_j];

                if(neg == true)
                {
                    val_j *= static_cast<ValueType>(-1);
                }

                // Add j if conditions are fulfilled
                if(val_j < max_a_ij)
                {
                    for(int r = 0; r < Gsize / 2; ++r)
                    {
                        rGc[(r + Gsize / 2) * rGsizec + nc] = (*rG)[r * rGsize + col_j];

                        if((*rG)[r * rGsize + col_j] >= 0)
                        {
                            cast_G->vec_[(*rG)[r * rGsize + col_j]] = nc;
                        }
                    }

                    U[col_j] = 1;
                }
            }

            ++nc;
        }

        free_host(&U);
        free_host(rG);

        (*rG)  = rGc;
        rGsize = rGsizec;

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::FurtherPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                                              ValueType                    beta,
                                                              int&                         nc,
                                                              BaseVector<int>*             G,
                                                              int&                         Gsize,
                                                              int**                        rG,
                                                              int&                         rGsize,
                                                              int ordering) const
    {
        assert(G != NULL);

        HostVector<int>*                cast_G = dynamic_cast<HostVector<int>*>(G);
        const HostMatrixCSR<ValueType>* cast_mat
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat);

        assert(cast_G != NULL);
        assert(cast_mat != NULL);

        // Initialize G and inverse indexing for G
        Gsize *= 2;
        int  rGsizec = this->nrow_;
        int* rGc     = NULL;
        allocate_host(Gsize * rGsizec, &rGc);

        for(int i = 0; i < Gsize * rGsizec; ++i)
        {
            rGc[i] = -1;
        }

        for(int i = 0; i < cast_G->size_; ++i)
        {
            cast_G->vec_[i] = -1;
        }

        // Initialize U
        int* U = NULL;
        allocate_host(this->nrow_, &U);
        set_to_zero_host(this->nrow_, U);

        nc              = 0;
        ValueType betam = -beta;

        // Ordering
        HostVector<int> perm(this->local_backend_);

        switch(ordering)
        {
        case 0: // No ordering
            break;

        case 1: // Connectivity ordering
            this->ConnectivityOrder(&perm);
            break;

        case 2: // CMK
            this->CMK(&perm);
            break;

        case 3: // RCMK
            this->RCMK(&perm);
            break;

        case 4: // MIS
            int size;
            this->MaximalIndependentSet(size, &perm);
            break;

        case 5: // MultiColoring
            int  num_colors;
            int* size_colors = NULL;
            this->MultiColoring(num_colors, &size_colors, &perm);
            free_host(&size_colors);
            break;
        }

        // While U != empty
        for(int k = 0; k < this->nrow_; ++k)
        {
            // Pick i according to connectivity
            int i;
            if(ordering == 0)
            {
                i = k;
            }
            else
            {
                i = perm.vec_[k];
            }

            // Check if i in U
            if(U[i] == 1)
            {
                continue;
            }

            // Mark i visited
            U[i] = 1;

            // Fill G and rG
            for(int r = 0; r < Gsize / 2; ++r)
            {
                rGc[r * rGsizec + nc] = (*rG)[r * rGsize + i];

                if((*rG)[r * rGsize + i] >= 0)
                {
                    cast_G->vec_[(*rG)[r * rGsize + i]] = nc;
                }
            }

            // Get sign
            bool neg = false;
            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(i == this->mat_.col[j])
                {
                    if(this->mat_.val[j] < static_cast<ValueType>(0))
                    {
                        neg = true;
                    }

                    break;
                }
            }

            // Determine minimum and maximum offdiagonal entry and j index
            ValueType min_a_ij = static_cast<ValueType>(0);
            ValueType max_a_ij = static_cast<ValueType>(0);
            int       min_j    = -1;

            for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int       col_j = this->mat_.col[j];
                ValueType val_j = this->mat_.val[j];

                if(neg == true)
                {
                    val_j *= static_cast<ValueType>(-1);
                }

                if(i == col_j)
                {
                    continue;
                }

                if(min_j == -1)
                {
                    max_a_ij = val_j;
                    if(U[col_j] == 0)
                    {
                        min_j    = col_j;
                        min_a_ij = max_a_ij;
                    }
                }

                if(val_j < min_a_ij && U[col_j] == 0)
                {
                    min_a_ij = val_j;
                    min_j    = col_j;
                }

                if(val_j < max_a_ij)
                {
                    max_a_ij = val_j;
                }
            }

            if(cast_mat->nnz_ > 0)
            {
                for(int j = cast_mat->mat_.row_offset[i]; j < cast_mat->mat_.row_offset[i + 1]; ++j)
                {
                    ValueType val_j = cast_mat->mat_.val[j];

                    if(neg == true)
                    {
                        val_j *= static_cast<ValueType>(-1);
                    }

                    if(val_j > max_a_ij)
                    {
                        max_a_ij = val_j;
                    }
                }
            }

            // Fill G
            if(min_j != -1)
            {
                max_a_ij *= betam;

                for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    int       col_j = this->mat_.col[j];
                    ValueType val_j = this->mat_.val[j];

                    if(neg == true)
                    {
                        val_j *= static_cast<ValueType>(-1);
                    }

                    // Skip diagonal
                    if(i == col_j)
                    {
                        continue;
                    }

                    // Skip j which is not in U
                    if(U[col_j] == 1)
                    {
                        continue;
                    }

                    // Add j if conditions are fulfilled
                    if(val_j < max_a_ij)
                    {
                        if(min_j == col_j)
                        {
                            for(int r = 0; r < Gsize / 2; ++r)
                            {
                                rGc[(r + Gsize / 2) * rGsizec + nc] = (*rG)[r * rGsize + min_j];

                                if((*rG)[r * rGsize + min_j] >= 0)
                                {
                                    cast_G->vec_[(*rG)[r * rGsize + min_j]] = nc;
                                }
                            }

                            U[min_j] = 1;
                            break;
                        }
                    }
                }
            }

            ++nc;
        }

        free_host(&U);
        free_host(rG);

        (*rG)  = rGc;
        rGsize = rGsizec;

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::CoarsenOperator(BaseMatrix<ValueType>* Ac,
                                                   int                    nrow,
                                                   int                    ncol,
                                                   const BaseVector<int>& G,
                                                   int                    Gsize,
                                                   const int*             rG,
                                                   int                    rGsize) const
    {
        assert(Ac != NULL);

        HostMatrixCSR<ValueType>* cast_Ac = dynamic_cast<HostMatrixCSR<ValueType>*>(Ac);
        const HostVector<int>*    cast_G  = dynamic_cast<const HostVector<int>*>(&G);

        assert(cast_Ac != NULL);
        assert(cast_G != NULL);

        // Allocate
        cast_Ac->Clear();

        int*       row_offset = NULL;
        int*       col        = NULL;
        ValueType* val        = NULL;

        allocate_host(nrow + 1, &row_offset);
        allocate_host(this->nnz_, &col);
        allocate_host(this->nnz_, &val);

        // Create P_ij: if i in G_j -> 1, else 0
        // (Ac)_kl = sum i in G_k (sum j in G_l (a_ij))) with k,l=1,...,nrow

        int* reverse_col = NULL;
        int* Gl          = NULL;
        int* erase       = NULL;

        int size = (nrow > ncol) ? nrow : ncol;

        allocate_host(size, &reverse_col);
        allocate_host(size, &Gl);
        allocate_host(size, &erase);

        for(int i = 0; i < size; ++i)
        {
            reverse_col[i] = -1;
        }

        set_to_zero_host(size, Gl);

        row_offset[0] = 0;

        for(int k = 0; k < nrow; ++k)
        {
            row_offset[k + 1] = row_offset[k];

            int m = 0;

            for(int r = 0; r < Gsize; ++r)
            {
                int i = rG[r * rGsize + k];

                if(i < 0)
                {
                    continue;
                }

                for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    int l = cast_G->vec_[this->mat_.col[j]];

                    if(l < 0)
                    {
                        continue;
                    }

                    if(Gl[l] == 0)
                    {
                        Gl[l]      = 1;
                        erase[m++] = l;

                        col[row_offset[k + 1]] = l;
                        val[row_offset[k + 1]] = this->mat_.val[j];
                        reverse_col[l]         = row_offset[k + 1];

                        ++row_offset[k + 1];
                    }
                    else
                    {
                        val[reverse_col[l]] += this->mat_.val[j];
                    }
                }
            }

            for(int j = 0; j < m; ++j)
            {
                Gl[erase[j]] = 0;
            }
        }

        free_host(&reverse_col);
        free_host(&Gl);
        free_host(&erase);

        int nnz = row_offset[nrow];

        int*       col_resized = NULL;
        ValueType* val_resized = NULL;

        allocate_host(nnz, &col_resized);
        allocate_host(nnz, &val_resized);

        // Resize
        copy_h2h(nnz, col, col_resized);
        copy_h2h(nnz, val, val_resized);

        free_host(&col);
        free_host(&val);

        // Allocate
        cast_Ac->Clear();
        cast_Ac->SetDataPtrCSR(&row_offset, &col_resized, &val_resized, nnz, nrow, nrow);

        return true;
    }

    template <typename T>
    int sgn(T val)
    {
        return (T(0) < val) - (val < T(0));
    }

    template <typename ValueType>
    bool
        HostMatrixCSR<ValueType>::Key(long int& row_key, long int& col_key, long int& val_key) const
    {
        row_key = 0;
        col_key = 0;
        val_key = 0;

        int row_sign = 1;
        int col_sign = 1;
        int val_sign = 1;

        int row_tmp = 0x12345678;
        int col_tmp = 0x23456789;
        int val_tmp = 0x34567890;

        int row_mask = 0x09876543;
        int col_mask = 0x98765432;
        int val_mask = 0x87654321;

        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            row_key += row_sign * row_tmp * (row_mask & this->mat_.row_offset[ai]);
            row_key  = row_key ^ (row_key >> 16);
            row_sign = sgn(row_tmp - (row_mask & this->mat_.row_offset[ai]));
            row_tmp  = row_mask & this->mat_.row_offset[ai];

            int row_beg = this->mat_.row_offset[ai];
            int row_end = this->mat_.row_offset[ai + 1];

            for(int aj = row_beg; aj < row_end; ++aj)
            {
                col_key += col_sign * col_tmp * (col_mask | this->mat_.col[aj]);
                col_key  = col_key ^ (col_key >> 16);
                col_sign = sgn(row_tmp - (col_mask | this->mat_.col[aj]));
                col_tmp  = col_mask | this->mat_.col[aj];

                double   double_val = std::abs(this->mat_.val[aj]);
                long int val        = 0;

                assert(sizeof(long int) == sizeof(double));

                memcpy(&val, &double_val, sizeof(long int));

                val_key += val_sign * val_tmp * (long int)(val_mask | val);
                val_key = val_key ^ (val_key >> 16);

                if(sgn(this->mat_.val[aj]) > 0)
                {
                    val_key = val_key ^ val;
                }
                else
                {
                    val_key = val_key | val;
                }

                val_sign = sgn(val_tmp - (long int)(val_mask | val));
                val_tmp  = val_mask | val;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ReplaceColumnVector(int idx, const BaseVector<ValueType>& vec)
    {
        assert(vec.GetSize() == this->nrow_);

        if(this->GetNnz() > 0)
        {
            const HostVector<ValueType>* cast_vec
                = dynamic_cast<const HostVector<ValueType>*>(&vec);
            assert(cast_vec != NULL);

            int*       row_offset = NULL;
            int*       col        = NULL;
            ValueType* val        = NULL;

            int nrow = this->nrow_;
            int ncol = this->ncol_;

            allocate_host(nrow + 1, &row_offset);
            row_offset[0] = 0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < nrow; ++i)
            {
                bool add = true;

                row_offset[i + 1] = this->mat_.row_offset[i + 1] - this->mat_.row_offset[i];

                for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    if(this->mat_.col[j] == idx)
                    {
                        add = false;
                        break;
                    }
                }

                if(add == true && cast_vec->vec_[i] != static_cast<ValueType>(0))
                {
                    ++row_offset[i + 1];
                }

                if(add == false && cast_vec->vec_[i] == static_cast<ValueType>(0))
                {
                    --row_offset[i + 1];
                }
            }

            for(int i = 0; i < nrow; ++i)
            {
                row_offset[i + 1] += row_offset[i];
            }

            int nnz = row_offset[nrow];

            allocate_host(nnz, &col);
            allocate_host(nnz, &val);

// Fill new CSR matrix
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < nrow; ++i)
            {
                int k = row_offset[i];
                int j = this->mat_.row_offset[i];

                for(; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    if(this->mat_.col[j] < idx)
                    {
                        col[k] = this->mat_.col[j];
                        val[k] = this->mat_.val[j];
                        ++k;
                    }
                    else
                    {
                        break;
                    }
                }

                if(cast_vec->vec_[i] != static_cast<ValueType>(0))
                {
                    col[k] = idx;
                    val[k] = cast_vec->vec_[i];
                    ++k;
                    ++j;
                }

                for(; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    if(this->mat_.col[j] > idx)
                    {
                        col[k] = this->mat_.col[j];
                        val[k] = this->mat_.val[j];
                        ++k;
                    }
                }
            }

            this->Clear();
            this->SetDataPtrCSR(&row_offset, &col, &val, row_offset[nrow], nrow, ncol);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractColumnVector(int idx, BaseVector<ValueType>* vec) const
    {
        assert(vec != NULL);
        assert(vec->GetSize() == this->nrow_);

        if(this->nnz_ > 0)
        {
            HostVector<ValueType>* cast_vec = dynamic_cast<HostVector<ValueType>*>(vec);
            assert(cast_vec != NULL);

            _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int ai = 0; ai < this->nrow_; ++ai)
            {
                // Initialize with zero
                cast_vec->vec_[ai] = static_cast<ValueType>(0);

                for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
                {
                    if(idx == this->mat_.col[aj])
                    {
                        cast_vec->vec_[ai] = this->mat_.val[aj];
                        break;
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ReplaceRowVector(int idx, const BaseVector<ValueType>& vec)
    {
        assert(vec.GetSize() == this->ncol_);

        if(this->nnz_ > 0)
        {
            const HostVector<ValueType>* cast_vec
                = dynamic_cast<const HostVector<ValueType>*>(&vec);
            assert(cast_vec != NULL);

            int*       row_offset = NULL;
            int*       col        = NULL;
            ValueType* val        = NULL;

            int nrow = this->nrow_;
            int ncol = this->ncol_;

            allocate_host(nrow + 1, &row_offset);
            row_offset[0] = 0;

            // Compute nnz of row idx
            int nnz_idx = 0;

            for(int i = 0; i < ncol; ++i)
            {
                if(cast_vec->vec_[i] != static_cast<ValueType>(0))
                {
                    ++nnz_idx;
                }
            }

            // Fill row_offset
            int shift = nnz_idx - this->mat_.row_offset[idx + 1] + this->mat_.row_offset[idx];

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < nrow + 1; ++i)
            {
                if(i < idx + 1)
                {
                    row_offset[i] = this->mat_.row_offset[i];
                }
                else
                {
                    row_offset[i] = this->mat_.row_offset[i] + shift;
                }
            }

            int nnz = row_offset[nrow];

            // Fill col and val
            allocate_host(nnz, &col);
            allocate_host(nnz, &val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < nrow; ++i)
            {
                // Rows before idx
                if(i < idx)
                {
                    for(int j = row_offset[i]; j < row_offset[i + 1]; ++j)
                    {
                        col[j] = this->mat_.col[j];
                        val[j] = this->mat_.val[j];
                    }

                    // Row == idx
                }
                else if(i == idx)
                {
                    int k = row_offset[i];

                    for(int j = 0; j < ncol; ++j)
                    {
                        if(cast_vec->vec_[j] != static_cast<ValueType>(0))
                        {
                            col[k] = j;
                            val[k] = cast_vec->vec_[j];
                            ++k;
                        }
                    }

                    // Rows after idx
                }
                else if(i > idx)
                {
                    int k = row_offset[i];

                    for(int j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                    {
                        col[k] = this->mat_.col[j];
                        val[k] = this->mat_.val[j];
                        ++k;
                    }
                }
            }

            this->Clear();
            this->SetDataPtrCSR(&row_offset, &col, &val, nnz, nrow, ncol);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractRowVector(int idx, BaseVector<ValueType>* vec) const
    {
        assert(vec != NULL);
        assert(vec->GetSize() == this->ncol_);

        if(this->nnz_ > 0)
        {
            HostVector<ValueType>* cast_vec = dynamic_cast<HostVector<ValueType>*>(vec);
            assert(cast_vec != NULL);

            _set_omp_backend_threads(this->local_backend_, this->nrow_);

            cast_vec->Zeros();

            for(int aj = this->mat_.row_offset[idx]; aj < this->mat_.row_offset[idx + 1]; ++aj)
            {
                cast_vec->vec_[this->mat_.col[aj]] = this->mat_.val[aj];
            }
        }

        return true;
    }

    template class HostMatrixCSR<double>;
    template class HostMatrixCSR<float>;
#ifdef SUPPORT_COMPLEX
    template class HostMatrixCSR<std::complex<double>>;
    template class HostMatrixCSR<std::complex<float>>;
#endif

} // namespace rocalution
