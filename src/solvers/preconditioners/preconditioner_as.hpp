#ifndef ROCALUTION_PRECONDITIONER_AS_HPP_
#define ROCALUTION_PRECONDITIONER_AS_HPP_

#include "preconditioner.hpp"

namespace rocalution {

/// AS preconditioner
template <class OperatorType, class VectorType, typename ValueType>
class AS : public Preconditioner<OperatorType, VectorType, ValueType> {

public:

  AS();
  virtual ~AS();

  virtual void Print(void) const;
  virtual void Set(const int nb, const int overlap,
                   Solver<OperatorType, VectorType, ValueType> **preconds);
  
  virtual void Solve(const VectorType &rhs,
                     VectorType *x);


  virtual void Build(void);
  virtual void Clear(void);

protected:

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void) ;

  int num_blocks_;
  int overlap_;
  int *pos_;
  int *sizes_; // with overlap

  Solver<OperatorType, VectorType, ValueType> **local_precond_;

  OperatorType **local_mat_;
  VectorType **r_;
  VectorType **z_;
  VectorType weight_;

};

/// AS preconditioner
template <class OperatorType, class VectorType, typename ValueType>
class RAS : public AS<OperatorType, VectorType, ValueType> {

public:

  RAS();
  virtual ~RAS();

  virtual void Print(void) const;

  virtual void Solve(const VectorType &rhs,
                     VectorType *x);

};


}

#endif // ROCALUTION_PRECONDITIONER_AS_HPP_
