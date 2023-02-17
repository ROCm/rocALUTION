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

#include "smoothed_amg.hpp"
#include "../../utils/def.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"

#include "../../solvers/preconditioners/preconditioner.hpp"

#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"
#include "../../utils/time_functions.hpp"

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    SAAMG<OperatorType, VectorType, ValueType>::SAAMG()
    {
        log_debug(this, "SAAMG::SAAMG()", "default constructor");

        // parameter for strong couplings in smoothed aggregation
        this->eps_           = static_cast<ValueType>(0.01f);
        this->relax_         = static_cast<ValueType>(2.f / 3.f);
        this->strat_         = CoarseningStrategy::Greedy;
        this->lumping_strat_ = LumpingStrategy::AddWeakConnections;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    SAAMG<OperatorType, VectorType, ValueType>::~SAAMG()
    {
        log_debug(this, "SAAMG::SAAMG()", "destructor");

        this->Clear();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SAAMG<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("SAAMG solver");
        LOG_INFO("SAAMG number of levels " << this->levels_);
        if(this->strat_ == CoarseningStrategy::Greedy)
        {
            LOG_INFO("SAAMG using greedy smoothed aggregation");
        }
        else if(this->strat_ == CoarseningStrategy::PMIS)
        {
            LOG_INFO("SAAMG using PMIS smoothed aggregation");
        }
        if(this->lumping_strat_ == LumpingStrategy::AddWeakConnections)
        {
            LOG_INFO("SAAMG lumping strategy adds weak connections to diagonal in filter matrix");
        }
        else if(this->lumping_strat_ == LumpingStrategy::SubtractWeakConnections)
        {
            LOG_INFO(
                "SAAMG lumping strategy subtracts weak connections to diagonal in filter matrix");
        }
        LOG_INFO("SAAMG coarsest operator size = " << this->op_level_[this->levels_ - 2]->GetM());
        LOG_INFO("SAAMG coarsest level nnz = " << this->op_level_[this->levels_ - 2]->GetNnz());
        LOG_INFO("SAAMG with smoother:");
        this->smoother_level_[0]->Print();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SAAMG<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        assert(this->levels_ > 0);

        LOG_INFO("SAAMG solver starts");
        LOG_INFO("SAAMG number of levels " << this->levels_);
        if(this->strat_ == CoarseningStrategy::Greedy)
        {
            LOG_INFO("SAAMG using greedy smoothed aggregation");
        }
        else if(this->strat_ == CoarseningStrategy::PMIS)
        {
            LOG_INFO("SAAMG using PMIS smoothed aggregation");
        }
        if(this->lumping_strat_ == LumpingStrategy::AddWeakConnections)
        {
            LOG_INFO("SAAMG lumping strategy adds weak connections to diagonal in filter matrix");
        }
        else if(this->lumping_strat_ == LumpingStrategy::SubtractWeakConnections)
        {
            LOG_INFO(
                "SAAMG lumping strategy subtracts weak connections to diagonal in filter matrix");
        }
        LOG_INFO("SAAMG coarsest operator size = " << this->op_level_[this->levels_ - 2]->GetM());
        LOG_INFO("SAAMG coarsest level nnz = " << this->op_level_[this->levels_ - 2]->GetNnz());
        LOG_INFO("SAAMG with smoother:");
        this->smoother_level_[0]->Print();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SAAMG<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        LOG_INFO("SAAMG ends");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SAAMG<OperatorType, VectorType, ValueType>::SetInterpRelax(ValueType relax)
    {
        log_debug(this, "SAAMG::SetInterpRelax()", relax);

        this->relax_ = relax;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SAAMG<OperatorType, VectorType, ValueType>::SetCouplingStrength(ValueType eps)
    {
        log_debug(this, "SAAMG::SetCouplingStrength()", eps);

        this->eps_ = eps;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SAAMG<OperatorType, VectorType, ValueType>::SetCoarseningStrategy(CoarseningStrategy strat)
    {
        log_debug(this, "SAAMG::SetCoarseningStrategy()", strat);

        this->strat_ = strat;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SAAMG<OperatorType, VectorType, ValueType>::SetLumpingStrategy(
        LumpingStrategy lumping_strat)
    {
        log_debug(this, "SAAMG::SetLumpingStrategy()", lumping_strat);

        this->lumping_strat_ = lumping_strat;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SAAMG<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "SAAMG::ReBuildNumeric()", " #*# begin");

        assert(this->levels_ > 1);
        assert(this->build_);
        assert(this->op_ != NULL);

        // Create coarse operator
        this->op_level_[0]->Clear();
        this->op_level_[0]->ConvertToCSR();
        this->op_level_[0]->CloneBackend(*this->op_);

        assert(this->restrict_op_level_[0] != NULL);
        assert(this->prolong_op_level_[0] != NULL);

        if(this->op_->GetFormat() != CSR)
        {
            OperatorType op_csr;
            op_csr.CloneFrom(*this->op_);
            op_csr.ConvertToCSR();

            this->op_level_[0]->TripleMatrixProduct(
                *this->restrict_op_level_[0], op_csr, *this->prolong_op_level_[0]);
        }
        else
        {
            this->op_level_[0]->TripleMatrixProduct(
                *this->restrict_op_level_[0], *this->op_, *this->prolong_op_level_[0]);
        }

        for(int i = 1; i < this->levels_ - 1; ++i)
        {
            // Create coarse operator
            this->op_level_[i]->Clear();
            this->op_level_[i]->ConvertToCSR();
            this->op_level_[i]->CloneBackend(*this->op_);

            assert(this->restrict_op_level_[i] != NULL);
            assert(this->prolong_op_level_[i] != NULL);

            if(i == this->levels_ - this->host_level_ - 1)
            {
                this->op_level_[i - 1]->MoveToHost();
            }

            this->op_level_[i]->TripleMatrixProduct(
                *this->restrict_op_level_[i], *this->op_level_[i - 1], *this->prolong_op_level_[i]);

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
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SAAMG<OperatorType, VectorType, ValueType>::Aggregate_(const OperatorType& op,
                                                                OperatorType*       pro,
                                                                OperatorType*       res,
                                                                OperatorType*       coarse,
                                                                LocalVector<int>*   trans)
    {
        log_debug(this, "SAAMG::Aggregate_()", this->build_);

        assert(pro != NULL);
        assert(res != NULL);
        assert(coarse != NULL);

        LocalVector<int> connections;
        LocalVector<int> aggregates;

        connections.CloneBackend(op);
        aggregates.CloneBackend(op);

        ValueType eps = this->eps_;
        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            eps *= static_cast<ValueType>(0.5);
        }

        op.AMGConnect(eps, &connections);

        if(strat_ == CoarseningStrategy::Greedy)
        {
            op.AMGAggregate(connections, &aggregates);
        }
        else if(strat_ == CoarseningStrategy::PMIS)
        {
            op.AMGPMISAggregate(connections, &aggregates);
        }

        if(lumping_strat_ == LumpingStrategy::AddWeakConnections)
        {
            op.AMGSmoothedAggregation(this->relax_, aggregates, connections, pro, 0);
        }
        else if(lumping_strat_ == LumpingStrategy::SubtractWeakConnections)
        {
            op.AMGSmoothedAggregation(this->relax_, aggregates, connections, pro, 1);
        }

        // Free unused vectors
        connections.Clear();
        aggregates.Clear();

        // Transpose P to obtain R
        pro->Transpose(res);

        // Triple matrix product
        coarse->CloneBackend(op);
        coarse->TripleMatrixProduct(*res, op, *pro);
    }

    template class SAAMG<LocalMatrix<double>, LocalVector<double>, double>;
    template class SAAMG<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class SAAMG<LocalMatrix<std::complex<double>>,
                         LocalVector<std::complex<double>>,
                         std::complex<double>>;
    template class SAAMG<LocalMatrix<std::complex<float>>,
                         LocalVector<std::complex<float>>,
                         std::complex<float>>;
#endif

} // namespace rocalution
