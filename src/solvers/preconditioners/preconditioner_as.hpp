/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_PRECONDITIONER_AS_HPP_
#define ROCALUTION_PRECONDITIONER_AS_HPP_

#include "preconditioner.hpp"

namespace rocalution {

/** \ingroup precond_module
  * \class AS
  * \brief Additive Schwarz Preconditioner
  * \details
  * The Additive Schwarz preconditioner relies on a preconditioning technique, where the
  * linear system \f$Ax=b\f$ can be decomposed into small sub-problems based on
  * \f$A_{i} = R_{i}^{T}AR_{i}\f$, where \f$R_{i}\f$ are restriction operators. Those
  * restriction operators produce sub-matrices wich overlap. This leads to contributions
  * from two preconditioners on the overlapped area which are scaled by \f$1/2\f$.
  *
  * \tparam OperatorType
  * \tparam VectorType
  * \tparam ValueType
  */
template <class OperatorType, class VectorType, typename ValueType>
class AS : public Preconditioner<OperatorType, VectorType, ValueType>
{
    public:
    AS();
    virtual ~AS();

    virtual void Print(void) const;

    /** \brief Set number of blocks, overlap and array of preconditioners */
    void Set(int nb, int overlap, Solver<OperatorType, VectorType, ValueType>** preconds);

    virtual void Solve(const VectorType& rhs, VectorType* x);

    virtual void Build(void);
    virtual void Clear(void);

    protected:
    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    /** \brief Number of blocks */
    int num_blocks_; /**< Number of blocks */
    /** \brief Overlap */
    int overlap_;
    /** \brief Position */
    int* pos_;
    /** \brief Sizes including overlap */
    int* sizes_;

    /** \brief Preconditioner for each block */
    Solver<OperatorType, VectorType, ValueType>** local_precond_;

    /** \brief Local operator */
    OperatorType** local_mat_;
    /** \brief r */
    VectorType** r_;
    /** \brief z */
    VectorType** z_;
    /** \brief weights */
    VectorType weight_;
};

/** \ingroup precond_module
  * \class RAS
  * \brief Restricted Additive Schwarz Preconditioner
  * \details
  * The Restricted Additive Schwarz preconditioner relies on a preconditioning technique,
  * where the linear system \f$Ax=b\f$ can be decomposed into small sub-problems based on
  * \f$A_{i} = R_{i}^{T}AR_{i}\f$, where \f$R_{i}\f$ are restriction operators. The RAS
  * method is a mixture of block Jacobi and the AS scheme. In this case, the sub-matrices
  * contain overlapped areas from other blocks, too.
  *
  * \tparam OperatorType
  * \tparam VectorType
  * \tparam ValueType
  */
template <class OperatorType, class VectorType, typename ValueType>
class RAS : public AS<OperatorType, VectorType, ValueType>
{
    public:
    RAS();
    virtual ~RAS();

    virtual void Print(void) const;

    virtual void Solve(const VectorType& rhs, VectorType* x);
};

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_AS_HPP_
