/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_FGMRES_FGMRES_HPP_
#define ROCALUTION_FGMRES_FGMRES_HPP_

#include "../solver.hpp"

#include <vector>

namespace rocalution {

/** \ingroup solver_module
  * \class FGMRES
  * \brief Flexible Generalized Minimum Residual Method
  * \details
  * The Flexible Generalized Minimum Residual method (FGMRES) is a projection method for
  * solving sparse (non) symmetric linear systems \f$Ax=b\f$. It is similar to the GMRES
  * method with the only difference, the FGMRES is based on a window shifting of the
  * Krylov subspace and thus allows the preconditioner \f$M^{-1}\f$ to be not a constant
  * operator. This can be especially helpful if the operation \f$M^{-1}x\f$ is the result
  * of another iterative process and not a constant operator.
  *
  * The Krylov subspace basis
  * size can be set using SetBasisSize(). The default size is 30.
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class FGMRES : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    FGMRES();
    virtual ~FGMRES();

    virtual void Print(void) const;

    virtual void Build(void);
    virtual void ReBuildNumeric(void);
    virtual void Clear(void);

    /** \brief Set the size of the Krylov subspace basis */
    virtual void SetBasisSize(int size_basis);

    protected:
    virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);
    virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    /** \brief Generate Givens rotation */
    void GenerateGivensRotation_(ValueType dx, ValueType dy, ValueType& c, ValueType& s) const;
    /** \brief Apply Givens rotation */
    void ApplyGivensRotation_(ValueType c, ValueType s, ValueType& dx, ValueType& dy) const;

    private:
    VectorType** v_;
    VectorType** z_;

    ValueType* c_;
    ValueType* s_;
    ValueType* r_;
    ValueType* H_;

    int size_basis_;
};

} // namespace rocalution

#endif // ROCALUTION_FGMRES_FGMRES_HPP_
