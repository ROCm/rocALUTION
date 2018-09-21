/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_DIRECT_INVERSION_HPP_
#define ROCALUTION_DIRECT_INVERSION_HPP_

#include "../solver.hpp"

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
class Inversion : public DirectLinearSolver<OperatorType, VectorType, ValueType> {

public:

  Inversion();
  virtual ~Inversion();

  virtual void Print(void) const;

  virtual void Build(void);
  virtual void Clear(void);

protected:

  virtual void Solve_(const VectorType &rhs, VectorType *x);

  virtual void PrintStart_(void) const;
  virtual void PrintEnd_(void) const;

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);

private:

  OperatorType inverse_;

};

} // namespace rocalution

#endif // ROCALUTION_DIRECT_INVERSION_HPP_
