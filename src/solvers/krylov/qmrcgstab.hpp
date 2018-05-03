#ifndef ROCALUTION_KRYLOV_QMRCGSTAB_HPP_
#define ROCALUTION_KRYLOV_QMRCGSTAB_HPP_

#include "../solver.hpp"

#include <vector>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
class QMRCGStab : public IterativeLinearSolver<OperatorType, VectorType, ValueType> {

public:

  QMRCGStab();
  virtual ~QMRCGStab();

  virtual void Print(void) const;

  virtual void Build(void);
  virtual void ReBuildNumeric(void);
  virtual void Clear(void);

protected:

  virtual void SolveNonPrecond_(const VectorType &rhs,
                                VectorType *x);
  virtual void SolvePrecond_(const VectorType &rhs,
                             VectorType *x);

  virtual void PrintStart_(void) const;
  virtual void PrintEnd_(void) const;

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);

private:

  VectorType r0_, r_;
  VectorType t_, p_;
  VectorType v_, d_;
  VectorType z_;

};


}

#endif // ROCALUTION_KRYLOV_QMRCGSTAB_HPP_
