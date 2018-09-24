/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_KRYLOV_FCG_HPP_
#define ROCALUTION_KRYLOV_FCG_HPP_

#include "../solver.hpp"

#include <vector>

namespace rocalution {

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
