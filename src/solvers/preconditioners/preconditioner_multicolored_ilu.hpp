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

/** \ingroup precond_module
  * \class MultiColoredILU
  * \brief Multi-Colored Incomplete LU Factorization Preconditioner
  * \details
  * Multi-Colored Incomplete LU Factorization based on p-levels based on ILU(p,q)
  * preconditioner power(q)-pattern method. Details can be found in ILU preconditioner
  * section.
  *
  * \tparam OperatorType
  * \tparam VectorType
  * \tparam ValueType
  */
template <class OperatorType, class VectorType, typename ValueType>
class MultiColoredILU : public MultiColored<OperatorType, VectorType, ValueType>
{
    public:
    MultiColoredILU();
    virtual ~MultiColoredILU();

    virtual void Print(void) const;

    virtual void ReBuildNumeric(void);

    /** \brief Initialize a multi-colored ILU(p, p+1) preconditioner */
    void Set(int p);

    /** \brief Initialize a multi-colored ILU(p, q) preconditioner
      * \details level = true will perform the factorization with levels <br>
      * level = false will perform the factorization only on the power(q)-pattern
      */
    void Set(int p, int q, bool level = true);

    protected:
    virtual void Build_Analyser_(void);
    virtual void Factorize_(void);
    virtual void PostAnalyse_(void);

    virtual void SolveL_(void);
    virtual void SolveD_(void);
    virtual void SolveR_(void);
    virtual void Solve_(const VectorType& rhs, VectorType* x);

    /** \brief power(q) pattern parameter */
    int q_;
    /** \brief p-levels parameter */
    int p_;
    /** \brief Perform factorization with levels or not */
    bool level_;
    /** \brief Number of non-zeros */
    int nnz_;
};

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_MULTICOLORED_ILU_HPP_
