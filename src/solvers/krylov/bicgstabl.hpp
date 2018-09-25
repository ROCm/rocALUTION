/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_KRYLOV_BICGSTABL_HPP_
#define ROCALUTION_KRYLOV_BICGSTABL_HPP_

#include "../solver.hpp"

#include <vector>

namespace rocalution {

/** \ingroup solver_module
  * \class BiCGStabl
  * \brief Bi-Conjugate Gradient Stabilized (l) Method
  * \details
  * The Bi-Conjugate Gradient Stabilized (l) method is a generalization of BiCGStab for
  * solving sparse (non) symmetric linear systems \f$Ax=b\f$. It minimizes residuals over
  * \f$l\f$-dimensional Krylov subspaces. The degree \f$l\f$ can be set with SetOrder().
  * For more details, see "G, G. L., Sleijpen, G., and Fokkema, D. Bicgstab(l) For Linear
  * Equations Involving Unsymmetric Matrices With Complex Spectrum, 1993".
  *
  * \tparam OperatorType - can be LocalMatrix or GlobalMatrix
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class BiCGStabl : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    BiCGStabl();
    virtual ~BiCGStabl();

    virtual void Print(void) const;

    virtual void Build(void);
    virtual void ReBuildNumeric(void);
    virtual void Clear(void);

    /** \brief Set the order */
    virtual void SetOrder(int l);

    protected:
    virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);
    virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    int l_;

    ValueType *gamma0_, *gamma1_, *gamma2_, *sigma_;
    ValueType** tau_;

    VectorType r0_, z_;
    VectorType **r_, **u_;
};

} // namespace rocalution

#endif // ROCALUTION_KRYLOV_BICGSTABL_HPP_
