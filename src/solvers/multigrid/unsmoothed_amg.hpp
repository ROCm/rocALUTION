/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_UNSMOOTHED_AMG_HPP_
#define ROCALUTION_UNSMOOTHED_AMG_HPP_

#include "../solver.hpp"
#include "base_amg.hpp"

#include <vector>

namespace rocalution {

/** \ingroup solver_module
  * \class UAAMG
  * \brief Unsmoothed Aggregation Algebraic MultiGrid Method
  * \details
  * The Unsmoothed Aggregation Algebraic MultiGrid method is based on unsmoothed
  * aggregation based interpolation scheme.
  *
  * \tparam OperatorType
  * \tparam VectorType
  * \tparam ValueType
  */
template <class OperatorType, class VectorType, typename ValueType>
class UAAMG : public BaseAMG<OperatorType, VectorType, ValueType>
{
    public:
    UAAMG();
    virtual ~UAAMG();

    virtual void Print(void) const;
    virtual void BuildSmoothers(void);

    /** \brief Set coupling strength */
    void SetCouplingStrength(ValueType eps);
    /** \brief Set over-interpolation parameter for aggregation */
    void SetOverInterp(ValueType overInterp);

    virtual void ReBuildNumeric(void);

    protected:
    virtual void Aggregate_(const OperatorType& op,
                            Operator<ValueType>* pro,
                            Operator<ValueType>* res,
                            OperatorType* coarse);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    private:
    /** \brief Coupling strength */
    ValueType eps_;

    /** \brief Over-interpolation parameter for aggregation */
    ValueType over_interp_;
};

} // namespace rocalution

#endif // ROCALUTION_UNSMOOTHED_AMG_HPP_
