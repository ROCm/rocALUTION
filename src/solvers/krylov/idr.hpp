/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_KRYLOV_IDR_HPP_
#define ROCALUTION_KRYLOV_IDR_HPP_

#include "../solver.hpp"

#include <vector>

namespace rocalution {

/// IDR(s) - Induced Dimension Reduction method, taken from "An Elegant IDR(s)
/// Variant that Efficiently Exploits Biorthogonality Properties" by Martin B.
/// van Gijzen and Peter Sonneveld, Delft University of Technology
template <class OperatorType, class VectorType, typename ValueType>
class IDR : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    IDR();
    virtual ~IDR();

    virtual void Print(void) const;

    virtual void Build(void);
    virtual void ReBuildNumeric(void);
    virtual void Clear(void);

    /// Set the size of the Shadow Space
    void SetShadowSpace(int s);
    /// Set random seed for ONB creation (seed must be greater than 0)
    void SetRandomSeed(unsigned long long seed);

    protected:
    virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);
    virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    int s_;
    unsigned long long seed_;

    ValueType kappa_;

    ValueType *c_;
    ValueType *f_;
    ValueType *M_;

    VectorType r_;
    VectorType v_;
    VectorType t_;

    VectorType **G_;
    VectorType **U_;
    VectorType **P_;
};

} // namespace rocalution

#endif // ROCALUTION_KRYLOV_IDR_HPP_
