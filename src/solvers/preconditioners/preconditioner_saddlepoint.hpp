#ifndef PARALUTION_PRECONDITIONER_SADDLEPOINT_HPP_
#define PARALUTION_PRECONDITIONER_SADDLEPOINT_HPP_

#include "../solver.hpp"
#include "preconditioner.hpp"
#include "../../base/local_vector.hpp"

#include <vector>

namespace paralution {

template <class OperatorType, class VectorType, typename ValueType>
class DiagJacobiSaddlePointPrecond : public Preconditioner<OperatorType, VectorType, ValueType> {

public:

  DiagJacobiSaddlePointPrecond();
  virtual ~DiagJacobiSaddlePointPrecond();

  virtual void Print(void) const;
  virtual void Clear(void);

  virtual void Set(Solver<OperatorType, VectorType, ValueType> &K_Solver,
                   Solver<OperatorType, VectorType, ValueType> &S_Solver);

  virtual void Build(void);

  virtual void Solve(const VectorType &rhs,
                     VectorType *x);

protected:

  /// A_ is decomposed into \f$ [K_, F_; E_, 0] \f$, where

  OperatorType A_;
  OperatorType K_, S_;

  /// The sizes of the K_ matrix
  int K_nrow_, K_nnz_;

  /// Keep the precond matrix in CSR or not
  bool op_mat_format_;
  /// Precond matrix format
  unsigned int precond_mat_format_;

  VectorType x_;
  VectorType x_1_;
  VectorType x_2_;
  VectorType x_1tmp_;

  VectorType rhs_;
  VectorType rhs_1_;
  VectorType rhs_2_;

  Solver<OperatorType, VectorType, ValueType> *K_solver_;
  Solver<OperatorType, VectorType, ValueType> *S_solver_;

  LocalVector<int> permutation_;
  int size_;

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);

};


}

#endif // PARALUTION_PRECONDITIONER_SADDLEPOINT_HPP_
