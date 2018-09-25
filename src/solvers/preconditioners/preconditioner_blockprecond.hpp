/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_PRECONDITIONER_BLOCKPRECOND_HPP_
#define ROCALUTION_PRECONDITIONER_BLOCKPRECOND_HPP_

#include "../solver.hpp"
#include "preconditioner.hpp"
#include "../../base/local_vector.hpp"

#include <vector>

namespace rocalution {

/** \ingroup precond_module
  * \class BlockPreconditioner
  * \brief Block-Preconditioner
  * \details
  * When handling vector fields, typically one can try to use different preconditioners
  * and/or solvers for the different blocks. For such problems, the library provides a
  * block-type preconditioner. This preconditioner builds the following block-type matrix
  * \f[
  *   P = \begin{pmatrix}
  *         A_{d} & 0     & . & 0     \\
  *         B_{1} & B_{d} & . & 0     \\
  *         .     & .     & . & .     \\
  *         Z_{1} & Z_{2} & . & Z_{d}
  *       \end{pmatrix}
  * \f]
  * The solution of \f$P\f$ can be performed in two ways. It can be solved by
  * block-lower-triangular sweeps with inversion of the blocks \f$A_{d} \ldots Z_{d}\f$
  * and with a multiplication of the corresponding blocks. This is set by SetLSolver()
  * (which is the default solution scheme). Alternatively, it can be used only with an
  * inverse of the diagonal \f$A_{d} \ldots Z_{d}\f$ (Block-Jacobi type) by using
  * SetDiagonalSolver().
  *
  * \tparam OperatorType
  * \tparam VectorType
  * \tparam ValueType
  */
template <class OperatorType, class VectorType, typename ValueType>
class BlockPreconditioner : public Preconditioner<OperatorType, VectorType, ValueType>
{
    public:
    BlockPreconditioner();
    virtual ~BlockPreconditioner();

    virtual void Print(void) const;
    virtual void Clear(void);

    /** \brief Set number, size and diagonal solver */
    void Set(int n, const int* size, Solver<OperatorType, VectorType, ValueType>** D_solver);

    /** \brief Set diagonal solver mode */
    void SetDiagonalSolver(void);
    /** \brief Set lower triangular sweep mode */
    void SetLSolver(void);

    /** \brief Set external last block matrix */
    void SetExternalLastMatrix(const OperatorType& mat);

    /** \brief Set permutation vector */
    virtual void SetPermutation(const LocalVector<int>& perm);

    virtual void Build(void);

    virtual void Solve(const VectorType& rhs, VectorType* x);

    protected:
    /** \brief The operator decomposition */
    OperatorType*** A_block_;
    /** \brief The operator of the last block */
    OperatorType* A_last_;

    /** \brief The precond matrix in CSR or not */
    bool op_mat_format_;
    /** \brief The precond matrix format */
    unsigned int precond_mat_format_;

    /** \brief Solution vector of each block */
    VectorType** x_block_;
    /** \brief Temporary vector objects */
    VectorType** tmp_block_;
    /** \brief Solution vector */
    VectorType x_;

    /** \brief Number of blocks */
    int num_blocks_;
    /** \brief Block sizes */
    int* block_sizes_;

    /** \brief Diagonal solvers */
    Solver<OperatorType, VectorType, ValueType>** D_solver_;

    /** \brief Flag if diagonal solves enabled */
    bool diag_solve_;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);
};

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_BLOCKPRECOND_HPP_
