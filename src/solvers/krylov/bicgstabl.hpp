/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_KRYLOV_BICGSTABL_HPP_
#define ROCALUTION_KRYLOV_BICGSTABL_HPP_

#include "../solver.hpp"

#include <vector>

namespace rocalution {

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

    /// Set the order
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
