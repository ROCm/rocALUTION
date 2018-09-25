/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_MULTIGRID_HPP_
#define ROCALUTION_MULTIGRID_HPP_

#include "base_multigrid.hpp"

namespace rocalution {

/** \ingroup solver_module
  * \class MultiGrid
  * \brief MultiGrid Method
  * \details
  * The MultiGrid method can be used with external data, such as externally computed
  * restriction, prolongation and operator hierarchy.
  *
  * \tparam OperatorType - can be LocalMatrix or GlobalMatrix
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class MultiGrid : public BaseMultiGrid<OperatorType, VectorType, ValueType>
{
    public:
    MultiGrid();
    virtual ~MultiGrid();

    virtual void SetRestrictOperator(OperatorType** op);
    virtual void SetProlongOperator(OperatorType** op);
    virtual void SetOperatorHierarchy(OperatorType** op);
};

} // namespace rocalution

#endif // ROCALUTION_MULTIGRID_HPP_
