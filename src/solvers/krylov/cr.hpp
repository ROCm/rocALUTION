/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_KRYLOV_CR_HPP_
#define ROCALUTION_KRYLOV_CR_HPP_

#include "../solver.hpp"

#include <vector>

namespace rocalution {

/** \ingroup solver_module
  * \class CR
  * \brief Conjugate Residual Method
  * \details
  * The Conjugate Residual method is an iterative method for solving sparse symmetric
  * semi-positive definite linear systems \f$Ax=b\f$. It is a Krylov subspace method and
  * differs from the much more popular Conjugate Gradient method that the system matrix
  * is not required to be positive definite. The method can be preconditioned where the
  * approximation should also be SPD or semi-positive definite.
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class CR : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    CR();
    virtual ~CR();

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
    VectorType r_, z_, t_;
    VectorType p_, q_, v_;
};

} // namespace rocalution

#endif // ROCALUTION_KRYLOV_CR_HPP_
