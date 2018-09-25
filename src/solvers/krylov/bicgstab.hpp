/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_KRYLOV_BICGSTAB_HPP_
#define ROCALUTION_KRYLOV_BICGSTAB_HPP_

#include "../solver.hpp"

namespace rocalution {

/** \ingroup solver_module
  * \class BiCGStab
  * \brief Bi-Conjugate Gradient Stabilized Method
  * \details
  * The Bi-Conjugate Gradient Stabilized method is a variation of CGS and solves sparse
  * (non) symmetric linear systems \f$Ax=b\f$.
  * 
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class BiCGStab : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    BiCGStab();
    virtual ~BiCGStab();

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
    VectorType r_;
    VectorType r0_;
    VectorType p_;
    VectorType q_;
    VectorType t_;
    VectorType v_;
    VectorType z_;
};

} // namespace rocalution

#endif // ROCALUTION_KRYLOV_BICGSTAB_HPP_
