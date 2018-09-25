/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_KRYLOV_CG_HPP_
#define ROCALUTION_KRYLOV_CG_HPP_

#include "../solver.hpp"

#include <vector>

namespace rocalution {

/** \ingroup solver_module
  * \class CG
  * \brief Conjugate Gradient Method
  * \details
  * The Conjugate Gradient method is the best known iterative method for solving sparse
  * symmetric positive definite linear systems \f$Ax=b\f$. It is based on orthogonal
  * projection onto the Krylov subspace \f$\mathcal{K}_{m}(r_{0}, A)\f$, where
  * \f$r_{0}\f$ is the initial residual.
  *
  * \tparam OperatorType
  * \tparam VectorType
  * \tparam ValueType
  */
template <class OperatorType, class VectorType, typename ValueType>
class CG : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    CG();
    virtual ~CG();

    virtual void Print(void) const;

    virtual void Build(void);

    virtual void BuildMoveToAcceleratorAsync(void);
    virtual void Sync(void);

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
    VectorType r_, z_;
    VectorType p_, q_;
};

} // namespace rocalution

#endif // ROCALUTION_KRYLOV_CG_HPP_
