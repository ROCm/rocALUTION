#ifndef PARALUTION_GMRES_GMRES_HPP_
#define PARALUTION_GMRES_GMRES_HPP_

#include "../solver.hpp"

#include <vector>

namespace paralution {

template <class OperatorType, class VectorType, typename ValueType>
class GMRES : public IterativeLinearSolver<OperatorType, VectorType, ValueType> {

public:

  GMRES();
  virtual ~GMRES();

  virtual void Print(void) const;

  virtual void Build(void);
  virtual void ReBuildNumeric(void);
  virtual void Clear(void);

  /// Set the size of the Krylov-space basis
  virtual void SetBasisSize(const int size_basis);

protected:

  virtual void SolveNonPrecond_(const VectorType &rhs,
                                VectorType *x);
  virtual void SolvePrecond_(const VectorType &rhs,
                             VectorType *x);

  virtual void PrintStart_(void) const;
  virtual void PrintEnd_(void) const;

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);

  void GenerateGivensRotation_(const ValueType &x, const ValueType &y,
                                     ValueType &c,       ValueType &s) const;

  void ApplyGivensRotation_(const ValueType &c, const ValueType &s,
                                  ValueType &x,       ValueType &y) const;

  void BackSubstitute_(std::vector<ValueType> &g,
                       const std::vector<ValueType> &H,
                       int k) const;

private:

  VectorType z_, w_;
  VectorType **v_;

  int size_basis_;

};


}

#endif // PARALUTION_GMRES_GMRES_HPP_
