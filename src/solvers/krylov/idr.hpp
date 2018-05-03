#ifndef ROCALUTION_KRYLOV_IDR_HPP_
#define ROCALUTION_KRYLOV_IDR_HPP_

#include "../solver.hpp"

#include <vector>

namespace rocalution {

/// IDR(s) - Induced Dimension Reduction method, taken from "An Elegant IDR(s)
/// Variant that Efficiently Exploits Biorthogonality Properties" by Martin B.
/// van Gijzen and Peter Sonneveld, Delft University of Technology
template <class OperatorType, class VectorType, typename ValueType>
class IDR : public IterativeLinearSolver<OperatorType, VectorType, ValueType> {

public:

  IDR();
  virtual ~IDR();

  virtual void Print(void) const;

  virtual void Build(void);
  virtual void ReBuildNumeric(void);
  virtual void Clear(void);

  /// Set the size of the Shadow Space
  virtual void SetShadowSpace(const int s);

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

  int s_;

  ValueType kappa_;

  ValueType *fhost_, *Mhost_;

  VectorType r_, v_, z_, t_;
  VectorType **g_, **u_, **P_;

};


}

#endif // ROCALUTION_KRYLOV_IDR_HPP_
