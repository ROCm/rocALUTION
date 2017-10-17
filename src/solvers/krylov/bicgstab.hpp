#ifndef PARALUTION_KRYLOV_BICGSTAB_HPP_
#define PARALUTION_KRYLOV_BICGSTAB_HPP_

#include "../solver.hpp"

#include <vector>

namespace paralution {

template <class OperatorType, class VectorType, typename ValueType>
class BiCGStab : public IterativeLinearSolver<OperatorType, VectorType, ValueType> {

public:

  BiCGStab();
  virtual ~BiCGStab();

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

  VectorType r_, z_, q_;
  VectorType p_, v_, r0_, s_, t_;

};


}

#endif // PARALUTION_KRYLOV_BICGSTAB_HPP_
