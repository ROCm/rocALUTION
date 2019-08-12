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

#ifndef ROCALUTION_MULTIGRID_PAIRWISE_AMG_HPP_
#define ROCALUTION_MULTIGRID_PAIRWISE_AMG_HPP_

#include "../solver.hpp"
#include "base_amg.hpp"

namespace rocalution
{

    enum _aggregation_ordering
    {
        NoOrdering    = 0,
        Connectivity  = 1,
        CMK           = 2,
        RCMK          = 3,
        MIS           = 4,
        MultiColoring = 5
    };

    /** \ingroup solver_module
  * \class PairwiseAMG
  * \brief Pairwise Aggregation Algebraic MultiGrid Method
  * \details
  * The Pairwise Aggregation Algebraic MultiGrid method is based on a pairwise
  * aggregation matching scheme. It delivers very efficient building phase which is
  * suitable for Poisson-like equation. Most of the time it requires K-cycle for the
  * solving phase to provide low number of iterations.
  * \cite pairwiseamg
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class PairwiseAMG : public BaseAMG<OperatorType, VectorType, ValueType>
    {
    public:
        PairwiseAMG();
        virtual ~PairwiseAMG();

        virtual void Print(void) const;
        virtual void BuildHierarchy(void);
        virtual void ClearLocal(void);
        virtual void BuildSmoothers(void);

        /** \brief Set beta for pairwise aggregation */
        void SetBeta(ValueType beta);
        /** \brief Set re-ordering for aggregation */
        void SetOrdering(unsigned int ordering);
        /** \brief Set target coarsening factor */
        void SetCoarseningFactor(double factor);

        virtual void ReBuildNumeric(void);

    protected:
        /** \brief Constructs the prolongation, restriction and coarse operator */
        void Aggregate_(const OperatorType&  op,
                        Operator<ValueType>* pro,
                        Operator<ValueType>* res,
                        OperatorType*        coarse,
                        LocalVector<int>*    trans);

        /** \private */
        virtual void Aggregate_(const OperatorType&  op,
                                Operator<ValueType>* pro,
                                Operator<ValueType>* res,
                                OperatorType*        coarse);

        virtual void PrintStart_(void) const;
        virtual void PrintEnd_(void) const;

    private:
        // Beta factor
        ValueType beta_;

        // Target factor for coarsening ratio
        double coarsening_factor_;
        // Ordering for the aggregation scheme
        unsigned int aggregation_ordering_;

        // Transfer mapping
        LocalVector<int>** trans_level_;

        // Dimension of the coarse operators
        std::vector<int>  dim_level_;
        std::vector<int>  Gsize_level_;
        std::vector<int>  rGsize_level_;
        std::vector<int*> rG_level_;
    };

} // namespace rocalution

#endif // ROCALUTION_MULTIGRID_PAIRWISE_AMG_HPP_
