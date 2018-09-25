/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_MULTIGRID_RUGE_STUEBEN_AMG_HPP_
#define ROCALUTION_MULTIGRID_RUGE_STUEBEN_AMG_HPP_

#include "../solver.hpp"
#include "base_amg.hpp"

#include <vector>

namespace rocalution {

/** \ingroup solver_module
  * \class RugeStuebenAMG
  * \brief Ruge-Stueben Algebraic MultiGrid Method
  * \details
  * The Ruge-Stueben Algebraic MultiGrid method is based on the classic Ruge-Stueben
  * coarsening with direct interpolation.
  *
  * \tparam OperatorType
  * \tparam VectorType
  * \tparam ValueType
  */
template <class OperatorType, class VectorType, typename ValueType>
class RugeStuebenAMG : public BaseAMG<OperatorType, VectorType, ValueType>
{
    public:
    RugeStuebenAMG();
    virtual ~RugeStuebenAMG();

    virtual void Print(void) const;
    virtual void BuildSmoothers(void);

    /** \brief Set coupling strength */
    void SetCouplingStrength(ValueType eps);

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
};

} // namespace rocalution

#endif // ROCALUTION_MULTIGRID_RUGE_STUEBEN_AMG_HPP_
