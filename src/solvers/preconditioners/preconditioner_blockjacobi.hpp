/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_PRECONDITIONER_BLOCKJACOBI_HPP_
#define ROCALUTION_PRECONDITIONER_BLOCKJACOBI_HPP_

#include "preconditioner.hpp"

namespace rocalution {

/** \ingroup precond_module
  * \class BlockJacobi
  * \brief Block-Jacobi Preconditioner
  * \details
  * The Block-Jacobi preconditioner is designed to wrap any local preconditioner and
  * apply it in a global block fashion locally on each interior matrix.
  *
  * \tparam OperatorType
  * \tparam VectorType
  * \tparam ValueType
  */
template <class OperatorType, class VectorType, typename ValueType>
class BlockJacobi : public Preconditioner<OperatorType, VectorType, ValueType>
{
    public:
    BlockJacobi();
    virtual ~BlockJacobi();

    virtual void Print(void) const;

    /** \brief Set local preconditioner */
    void Set(Solver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>& precond);

    virtual void Solve(const VectorType& rhs, VectorType* x);

    virtual void SolveZeroSol(const VectorType& rhs, VectorType* x);

    virtual void Build(void);
    virtual void ReBuildNumeric(void);
    virtual void Clear(void);

    protected:
    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    Solver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>* local_precond_;
};

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_BLOCKJACOBI_HPP_
