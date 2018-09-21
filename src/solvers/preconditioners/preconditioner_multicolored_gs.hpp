/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_PRECONDITIONER_MULTICOLORED_GS_HPP_
#define ROCALUTION_PRECONDITIONER_MULTICOLORED_GS_HPP_

#include "../solver.hpp"
#include "preconditioner.hpp"
#include "preconditioner_multicolored.hpp"
#include "../../base/local_vector.hpp"

#include <vector>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
class MultiColoredSGS : public MultiColored<OperatorType, VectorType, ValueType> {

public:

  MultiColoredSGS();
  virtual ~MultiColoredSGS();

  virtual void Print(void) const;  

  virtual void ReBuildNumeric(void);

  /// Set the relaxation parameter for the SOR/SSOR scheme
  virtual void SetRelaxation(const ValueType omega);

protected:

  virtual void PostAnalyse_(void);

  virtual void SolveL_(void);
  virtual void SolveD_(void);
  virtual void SolveR_(void);
  virtual void Solve_(const VectorType &rhs,
                      VectorType *x);  

  ValueType omega_;

};

template <class OperatorType, class VectorType, typename ValueType>
class MultiColoredGS : public MultiColoredSGS<OperatorType, VectorType, ValueType> {

public:

  MultiColoredGS();
  virtual ~MultiColoredGS();

  virtual void Print(void) const;  

protected:

  virtual void PostAnalyse_(void);

  virtual void SolveL_(void);
  virtual void SolveD_(void);
  virtual void SolveR_(void);
  virtual void Solve_(const VectorType &rhs,
                      VectorType *x);

};

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_MULTICOLORED_GS_HPP_
