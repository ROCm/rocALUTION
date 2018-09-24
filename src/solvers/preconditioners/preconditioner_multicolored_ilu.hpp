/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_PRECONDITIONER_MULTICOLORED_ILU_HPP_
#define ROCALUTION_PRECONDITIONER_MULTICOLORED_ILU_HPP_

#include "../solver.hpp"
#include "preconditioner.hpp"
#include "preconditioner_multicolored.hpp"
#include "../../base/local_vector.hpp"

#include <vector>

namespace rocalution {

/// ILU(p,q) preconditioner (see power(q)-pattern method, D. Lukarski "Parallel Sparse Linear
/// Algebra for Multi-core and Many-core Platforms - Parallel Solvers and Preconditioners",
/// PhD Thesis, 2012, KIT)
template <class OperatorType, class VectorType, typename ValueType>
class MultiColoredILU : public MultiColored<OperatorType, VectorType, ValueType>
{
    public:
    MultiColoredILU();
    virtual ~MultiColoredILU();

    virtual void Print(void) const;

    virtual void ReBuildNumeric(void);

    /// Initialize a multi-colored ILU(p,p+1) preconditioner
    virtual void Set(int p);

    /// Initialize a multi-colored ILU(p,q) preconditioner;
    /// level==true will perform the factorization with levels;
    /// level==false will perform the factorization only on the power(q)-pattern
    virtual void Set(int p, int q, bool level = true);

    protected:
    virtual void Build_Analyser_(void);
    virtual void Factorize_(void);
    virtual void PostAnalyse_(void);

    virtual void SolveL_(void);
    virtual void SolveD_(void);
    virtual void SolveR_(void);
    virtual void Solve_(const VectorType& rhs, VectorType* x);

    int q_;
    int p_;
    bool level_;
    int nnz_;
};

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_MULTICOLORED_ILU_HPP_
