/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_GMRES_GMRES_HPP_
#define ROCALUTION_GMRES_GMRES_HPP_

#include "../solver.hpp"

#include <vector>

namespace rocalution {

/** \ingroup solver_module
  * \class GMRES
  * \brief Generalized Minimum Residual Method
  * \details
  * The Generalized Minimum Residual method (GMRES) is a projection method for solving
  * sparse (non) symmetric linear systems \f$Ax=b\f$, based on restarting technique. The
  * solution is approximated in a Krylov subspace \f$\mathcal{K}=\mathcal{K}_{m}\f$ and
  * \f$\mathcal{L}=A\mathcal{K}_{m}\f$ with minimal residual, where \f$\mathcal{K}_{m}\f$
  * is the \f$m\f$-th Krylov subspace with \f$v_{1} = r_{0}/||r_{0}||_{2}\f$.
  *
  * The Krylov subspace basis size can be set using SetBasisSize(). The default size is
  * 30.
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class GMRES : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    GMRES();
    virtual ~GMRES();

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
    VectorType z_;

    ValueType* c_;
    ValueType* s_;
    ValueType* r_;
    ValueType* H_;

    int size_basis_;
};

} // namespace rocalution

#endif // ROCALUTION_GMRES_GMRES_HPP_
