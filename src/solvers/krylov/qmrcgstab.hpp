/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_KRYLOV_QMRCGSTAB_HPP_
#define ROCALUTION_KRYLOV_QMRCGSTAB_HPP_

#include "../solver.hpp"

#include <vector>

namespace rocalution {

/** \ingroup solver_module
  * \class QMRCGStab
  * \brief Quasi-Minimal Residual Conjugate Gradient Stabilized Method
  * \details
  * The Quasi-Minimal Residual Conjugate Gradient Stabilized method is a variant of the
  * Krylov subspace BiCGStab method for solving sparse (non) symmetric linear systems
  * \f$Ax=b\f$. For further details, see "Liu, X., Gu, T., Hang, X., and Sheng, Z.
  * A parallel version of QMRCGSTAB method for large linear systems in distributed
  * parallel environments. Applied Mathematics and Computation 172, 2 (2006), 744 - 752.
  * Special issue for The Beijing-HK Scientific Computing Meetings".
  *
  * \tparam OperatorType - can be LocalMatrix or GlobalMatrix
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class QMRCGStab : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    QMRCGStab();
    virtual ~QMRCGStab();

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
    VectorType r0_, r_;
    VectorType t_, p_;
    VectorType v_, d_;
    VectorType z_;
};

} // namespace rocalution

#endif // ROCALUTION_KRYLOV_QMRCGSTAB_HPP_
