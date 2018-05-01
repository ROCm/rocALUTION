#ifndef ROCALUTION_AMG_HPP_
#define ROCALUTION_AMG_HPP_

#include "../solver.hpp"
#include "base_amg.hpp"

#include <vector>

namespace rocalution {

enum _interp {
  Aggregation,
  SmoothedAggregation
};

template <class OperatorType, class VectorType, typename ValueType>
class AMG : public BaseAMG<OperatorType, VectorType, ValueType> {

public:

  AMG();
  virtual ~AMG();

  virtual void Print(void) const;

  /// Build AMG smoothers
  virtual void BuildSmoothers(void);

  /// Sets coupling strength
  virtual void SetCouplingStrength(const ValueType eps);
  /// Sets the interpolation type
  virtual void SetInterpolation(_interp interpType);
  /// Sets the relaxation parameter for smoothed aggregation
  virtual void SetInterpRelax(const ValueType relax);
  /// Sets over-interpolation parameter for aggregation
  virtual void SetOverInterp(const ValueType overInterp);

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

  /// Over-interpolation parameter for aggregation
  ValueType over_interp_;

  /// interpolation type for grid transfer operators
  _interp interp_type_;

};


}

#endif // ROCALUTION_AMG_HPP_
