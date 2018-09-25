/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_KRYLOV_FCG_HPP_
#define ROCALUTION_KRYLOV_FCG_HPP_

#include "../solver.hpp"

#include <vector>

namespace rocalution {

/** \ingroup solver_module
  * \class FCG
  * \brief Flexible Conjugate Gradient Method
  * \details
  * The Flexible Conjugate Gradient method is an iterative method for solving sparse
  * symmetric positive definite linear systems \f$Ax=b\f$. It is similar to the Conjugate
  * Gradient method with the only difference, that it allows the preconditioner
  * \f$M^{-1}\f$ to be not a constant operator. This can be especially helpful if the
  * operation \f$M^{-1}x\f$ is the result of another iterative process and not a constant
  * operator.
  *
  * \tparam OperatorType
  * \tparam VectorType
  * \tparam ValueType
  */
template <class OperatorType, class VectorType, typename ValueType>
class FCG : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    FCG();
    virtual ~FCG();

    virtual void Print(void) const;

    virtual void Build(void);
    virtual void ReBuildNumeric(void);
    virtual void Clear(void);

    protected:
    virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);
    virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    VectorType r_, w_, z_;
    VectorType p_, q_;
};

} // namespace rocalution

#endif // ROCALUTION_KRYLOV_FCG_HPP_
