/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_SMOOTHED_AMG_HPP_
#define ROCALUTION_SMOOTHED_AMG_HPP_

#include "../solver.hpp"
#include "base_amg.hpp"

#include <vector>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
class SAAMG : public BaseAMG<OperatorType, VectorType, ValueType> {

public:

  SAAMG();
  virtual ~SAAMG();

  virtual void Print(void) const;

  /// Build SAAMG smoothers
  virtual void BuildSmoothers(void);

  /// Sets coupling strength
  virtual void SetCouplingStrength(const ValueType eps);
  /// Sets the relaxation parameter for smoothed aggregation
  virtual void SetInterpRelax(const ValueType relax);

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

  /// Relaxation parameter for smoothed aggregation
  ValueType relax_;
};

} // namespace rocalution

#endif // ROCALUTION_SMOOTHED_AMG_HPP_
