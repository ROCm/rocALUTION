/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_MULTIGRID_HPP_
#define ROCALUTION_MULTIGRID_HPP_

#include "base_multigrid.hpp"

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
class MultiGrid : public BaseMultiGrid<OperatorType, VectorType, ValueType>
{
    public:
    MultiGrid();
    virtual ~MultiGrid();

    /// Set thre restriction method by operator for each level
    virtual void SetRestrictOperator(OperatorType** op);

    /// Set the prolongation operator for each level
    virtual void SetProlongOperator(OperatorType** op);

    /// Set the operator for each level
    virtual void SetOperatorHierarchy(OperatorType** op);
};

} // namespace rocalution

#endif // ROCALUTION_MULTIGRID_HPP_
