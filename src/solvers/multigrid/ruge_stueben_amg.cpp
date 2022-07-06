/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "ruge_stueben_amg.hpp"
#include "../../utils/def.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"

#include "../preconditioners/preconditioner.hpp"

#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    RugeStuebenAMG<OperatorType, VectorType, ValueType>::RugeStuebenAMG()
    {
        log_debug(this, "RugeStuebenAMG::RugeStuebenAMG()", "default constructor");

        // Parameter for strong couplings, for 3D problems 0.5 might work better
        this->eps_ = 0.25f;

        // Truncation coefficient
        this->trunc_ = 0.0f;

        // FF interpolation limiter
        this->FF1_ = false;

        // Coarsening strategy default
        this->coarsening_ = Greedy;

        // Interpolation type default
        this->interpolation_ = Direct;

        // Disable scaling
        this->scaling_ = false;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    RugeStuebenAMG<OperatorType, VectorType, ValueType>::~RugeStuebenAMG()
    {
        log_debug(this, "RugeStuebenAMG::RugeStuebenAMG()", "destructor");

        this->Clear();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void RugeStuebenAMG<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("AMG solver");

        if(this->build_ == true)
        {
            std::string coarsening;
            switch(this->coarsening_)
            {
            case Greedy:
                coarsening = "Greedy";
                break;
            case PMIS:
                coarsening = "PMIS";
                break;
            }

            std::string interpolation;
            switch(this->interpolation_)
            {
            case Direct:
                interpolation = "Direct";
                break;
            case ExtPI:
                interpolation = "Ext+i";
                break;
            }

            LOG_INFO("AMG number of levels " << this->levels_);
            LOG_INFO("AMG Ruge-Stuben using " << coarsening << " coarsening with " << interpolation
                                              << " interpolation");
            LOG_INFO("AMG coarsest operator size = " << this->op_level_[this->levels_ - 2]->GetM());
            LOG_INFO("AMG coarsest level nnz = " << this->op_level_[this->levels_ - 2]->GetNnz());
            LOG_INFO("AMG with smoother:");

            this->smoother_level_[0]->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void RugeStuebenAMG<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        assert(this->levels_ > 0);

        std::string coarsening;
        switch(this->coarsening_)
        {
        case Greedy:
            coarsening = "Greedy";
            break;
        case PMIS:
            coarsening = "PMIS";
            break;
        }

        std::string interpolation;
        switch(this->interpolation_)
        {
        case Direct:
            interpolation = "Direct";
            break;
        case ExtPI:
            interpolation = "Ext+i";
            break;
        }

        LOG_INFO("AMG solver starts");
        LOG_INFO("AMG number of levels " << this->levels_);
        LOG_INFO("AMG Ruge-Stuben using " << coarsening << " coarsening with " << interpolation
                                          << " interpolation");
        LOG_INFO("AMG coarsest operator size = " << this->op_level_[this->levels_ - 2]->GetM());
        LOG_INFO("AMG coarsest level nnz = " << this->op_level_[this->levels_ - 2]->GetNnz());
        LOG_INFO("AMG with smoother:");
        this->smoother_level_[0]->Print();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void RugeStuebenAMG<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        LOG_INFO("AMG ends");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void RugeStuebenAMG<OperatorType, VectorType, ValueType>::SetCouplingStrength(ValueType eps)
    {
        log_debug(this, "RugeStuebenAMG::SetCouplingStrength()", eps);

        this->eps_ = static_cast<float>(std::real(eps));
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void RugeStuebenAMG<OperatorType, VectorType, ValueType>::SetStrengthThreshold(float eps)
    {
        log_debug(this, "RugeStuebenAMG::SetStrengthThreshold()", eps);

        this->eps_ = eps;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void RugeStuebenAMG<OperatorType, VectorType, ValueType>::SetCoarseningStrategy(
        CoarseningStrategy strat)
    {
        log_debug(this, "RugeStuebenAMG::SetCoarseningStrategy()", strat);

        this->coarsening_ = strat;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void RugeStuebenAMG<OperatorType, VectorType, ValueType>::SetInterpolationType(
        InterpolationType type)
    {
        log_debug(this, "RugeStuebenAMG::SetInterpolationType()", type);

        this->interpolation_ = type;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void
        RugeStuebenAMG<OperatorType, VectorType, ValueType>::SetInterpolationTruncation(float trunc)
    {
        log_debug(this, "RugeStuebenAMG::SetInterpolationTruncation()", trunc);

        assert(this->build_ == false);
        assert(this->trunc_ >= 0.0f);

        this->trunc_ = trunc;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void RugeStuebenAMG<OperatorType, VectorType, ValueType>::SetInterpolationFF1Limit(bool FF1)
    {
        log_debug(this, "RugeStuebenAMG::SetInterpolationFF1Limit()", FF1);

        assert(this->build_ == false);

        this->FF1_ = FF1;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void RugeStuebenAMG<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "RugeStuebenAMG::ReBuildNumeric()", " #*# begin");

        assert(this->levels_ > 1);
        assert(this->build_);
        assert(this->op_ != NULL);

        this->op_level_[0]->Clear();
        this->op_level_[0]->ConvertToCSR();

        if(this->op_->GetFormat() != CSR)
        {
            OperatorType op_csr;
            op_csr.CloneFrom(*this->op_);
            op_csr.ConvertToCSR();

            // Create coarse operator
            OperatorType tmp;
            tmp.CloneBackend(*this->op_);
            this->op_level_[0]->CloneBackend(*this->op_);

            OperatorType* cast_res = dynamic_cast<OperatorType*>(this->restrict_op_level_[0]);
            OperatorType* cast_pro = dynamic_cast<OperatorType*>(this->prolong_op_level_[0]);
            assert(cast_res != NULL);
            assert(cast_pro != NULL);

            tmp.MatrixMult(*cast_res, op_csr);
            this->op_level_[0]->MatrixMult(tmp, *cast_pro);
        }
        else
        {
            // Create coarse operator
            OperatorType tmp;
            tmp.CloneBackend(*this->op_);
            this->op_level_[0]->CloneBackend(*this->op_);

            OperatorType* cast_res = dynamic_cast<OperatorType*>(this->restrict_op_level_[0]);
            OperatorType* cast_pro = dynamic_cast<OperatorType*>(this->prolong_op_level_[0]);
            assert(cast_res != NULL);
            assert(cast_pro != NULL);

            tmp.MatrixMult(*cast_res, *this->op_);
            this->op_level_[0]->MatrixMult(tmp, *cast_pro);
        }

        for(int i = 1; i < this->levels_ - 1; ++i)
        {
            this->op_level_[i]->Clear();
            this->op_level_[i]->ConvertToCSR();

            // Create coarse operator
            OperatorType tmp;
            tmp.CloneBackend(*this->op_);
            this->op_level_[i]->CloneBackend(*this->op_);

            OperatorType* cast_res = dynamic_cast<OperatorType*>(this->restrict_op_level_[i]);
            OperatorType* cast_pro = dynamic_cast<OperatorType*>(this->prolong_op_level_[i]);
            assert(cast_res != NULL);
            assert(cast_pro != NULL);

            if(i == this->levels_ - this->host_level_ - 1)
            {
                this->op_level_[i - 1]->MoveToHost();
            }

            tmp.MatrixMult(*cast_res, *this->op_level_[i - 1]);
            this->op_level_[i]->MatrixMult(tmp, *cast_pro);

            if(i == this->levels_ - this->host_level_ - 1)
            {
                this->op_level_[i - 1]->CloneBackend(*this->restrict_op_level_[i - 1]);
            }
        }

        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            if(i > 0)
            {
                this->smoother_level_[i]->ResetOperator(*this->op_level_[i - 1]);
            }
            else
            {
                this->smoother_level_[i]->ResetOperator(*this->op_);
            }

            this->smoother_level_[i]->ReBuildNumeric();
            this->smoother_level_[i]->Verbose(0);
        }

        this->solver_coarse_->ResetOperator(*this->op_level_[this->levels_ - 2]);
        this->solver_coarse_->ReBuildNumeric();
        this->solver_coarse_->Verbose(0);

        // Convert operator to op_format
        if(this->op_format_ != CSR)
        {
            for(int i = 0; i < this->levels_ - 1; ++i)
            {
                this->op_level_[i]->ConvertTo(this->op_format_, this->op_blockdim_);
            }
        }

        log_debug(this, "RugeStuebenAMG::ReBuildNumeric()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void RugeStuebenAMG<OperatorType, VectorType, ValueType>::Aggregate_(const OperatorType&  op,
                                                                         Operator<ValueType>* pro,
                                                                         Operator<ValueType>* res,
                                                                         OperatorType*     coarse,
                                                                         ParallelManager*  pm,
                                                                         LocalVector<int>* trans)
    {
        log_debug(this, "RugeStuebenAMG::Aggregate_()", (const void*&)op, pro, res, coarse);

        assert(pro != NULL);
        assert(res != NULL);
        assert(coarse != NULL);

        OperatorType* cast_res = dynamic_cast<OperatorType*>(res);
        OperatorType* cast_pro = dynamic_cast<OperatorType*>(pro);

        assert(cast_res != NULL);
        assert(cast_pro != NULL);

        // Determine C/F map and S
        LocalVector<int>  CFmap;
        LocalVector<bool> S;

        CFmap.CloneBackend(op);
        S.CloneBackend(op);

        switch(this->coarsening_)
        {
        case Greedy:
            op.RSCoarsening(this->eps_, &CFmap, &S);
            break;
        case PMIS:
            op.RSPMISCoarsening(this->eps_, &CFmap, &S);
            break;
        }

        // Create prolongation and restriction operators
        switch(this->interpolation_)
        {
        case Direct:
            op.RSDirectInterpolation(CFmap, S, cast_pro, cast_res);
            break;
        case ExtPI:
            op.RSExtPIInterpolation(CFmap, S, this->FF1_, this->trunc_, cast_pro, cast_res);
            break;
        }

        // Clean up
        CFmap.Clear();
        S.Clear();

        // Create coarse operator
        OperatorType tmp;
        tmp.CloneBackend(op);
        coarse->CloneBackend(op);

        tmp.MatrixMult(*cast_res, op);
        coarse->MatrixMult(tmp, *cast_pro);
    }

    template class RugeStuebenAMG<LocalMatrix<double>, LocalVector<double>, double>;
    template class RugeStuebenAMG<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class RugeStuebenAMG<LocalMatrix<std::complex<double>>,
                                  LocalVector<std::complex<double>>,
                                  std::complex<double>>;
    template class RugeStuebenAMG<LocalMatrix<std::complex<float>>,
                                  LocalVector<std::complex<float>>,
                                  std::complex<float>>;
#endif

} // namespace rocalution
