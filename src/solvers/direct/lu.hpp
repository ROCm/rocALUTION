/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_DIRECT_LU_HPP_
#define ROCALUTION_DIRECT_LU_HPP_

#include "../solver.hpp"

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
class LU : public DirectLinearSolver<OperatorType, VectorType, ValueType> {

public:

  LU();
  virtual ~LU();

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

  OperatorType lu_;

};

} // namespace rocalution

#endif // ROCALUTION_DIRECT_LU_HPP_
