/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_MIXED_PRECISION_HPP_
#define ROCALUTION_MIXED_PRECISION_HPP_

#include "solver.hpp"

namespace rocalution {

template <class OperatorTypeH, class VectorTypeH, typename ValueTypeH,
          class OperatorTypeL, class VectorTypeL, typename ValueTypeL>
class MixedPrecisionDC : public IterativeLinearSolver<OperatorTypeH, VectorTypeH, ValueTypeH> {
  
public:

  MixedPrecisionDC();
  virtual ~MixedPrecisionDC();

  virtual void Print(void) const;

  void Set(Solver<OperatorTypeL, VectorTypeL, ValueTypeL> &Solver_L);

  virtual void Build(void);
  virtual void ReBuildNumeric(void);
  virtual void Clear(void);

protected:

  virtual void SolveNonPrecond_(const VectorTypeH &rhs,
                                VectorTypeH *x);
  virtual void SolvePrecond_(const VectorTypeH &rhs,
                             VectorTypeH *x);

  virtual void PrintStart_(void) const;
  virtual void PrintEnd_(void) const;

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);

private:

  Solver<OperatorTypeL, VectorTypeL, ValueTypeL> *Solver_L_;

  VectorTypeH r_h_;
  VectorTypeL r_l_;

  VectorTypeH *x_h_;
  VectorTypeL d_l_;
  VectorTypeH d_h_;

  const OperatorTypeH *op_h_;
  OperatorTypeL *op_l_;

};

} // namespace rocalution

#endif // ROCALUTION_MIXED_PRECISION_HPP_
