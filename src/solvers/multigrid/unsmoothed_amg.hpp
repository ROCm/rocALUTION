/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_UNSMOOTHED_AMG_HPP_
#define ROCALUTION_UNSMOOTHED_AMG_HPP_

#include "../solver.hpp"
#include "base_amg.hpp"

#include <vector>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
class UAAMG : public BaseAMG<OperatorType, VectorType, ValueType> {

public:

  UAAMG();
  virtual ~UAAMG();

  virtual void Print(void) const;

  /// Build UAAMG smoothers
  virtual void BuildSmoothers(void);

  /// Sets coupling strength
  virtual void SetCouplingStrength(ValueType eps);
  /// Sets over-interpolation parameter for aggregation
  virtual void SetOverInterp(ValueType overInterp);

  /// Rebuild coarser operators with previous intergrid operators
  virtual void ReBuildNumeric(void);

protected:

  /// Constructs the prolongation, restriction and coarse operator
  virtual void Aggregate(const OperatorType &op,
                         Operator<ValueType> *pro,
                         Operator<ValueType> *res,
                         OperatorType *coarse);

  virtual void PrintStart_(void) const;
  virtual void PrintEnd_(void) const;

private:

  /// Coupling strength
  ValueType eps_;

  /// Over-interpolation parameter for aggregation
  ValueType over_interp_;

};

} // namespace rocalution

#endif // ROCALUTION_UNSMOOTHED_AMG_HPP_
