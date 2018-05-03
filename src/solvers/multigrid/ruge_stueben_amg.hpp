#ifndef ROCALUTION_MULTIGRID_RUGE_STUEBEN_AMG_HPP_
#define ROCALUTION_MULTIGRID_RUGE_STUEBEN_AMG_HPP_

#include "../solver.hpp"
#include "base_amg.hpp"

#include <vector>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
class RugeStuebenAMG : public BaseAMG<OperatorType, VectorType, ValueType> {

public:

  RugeStuebenAMG();
  virtual ~RugeStuebenAMG();

  virtual void Print(void) const;

  /// Build AMG smoothers
  virtual void BuildSmoothers(void);

  /// Sets coupling strength
  virtual void SetCouplingStrength(const ValueType eps);

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

  /// Coupling coefficient
  ValueType eps_;

};


}

#endif // ROCALUTION_MULTIGRID_RUGE_STUEBEN_AMG_HPP_
