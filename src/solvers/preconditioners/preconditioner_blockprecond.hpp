#ifndef ROCALUTION_PRECONDITIONER_BLOCKPRECOND_HPP_
#define ROCALUTION_PRECONDITIONER_BLOCKPRECOND_HPP_

#include "../solver.hpp"
#include "preconditioner.hpp"
#include "../../base/local_vector.hpp"

#include <vector>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
class BlockPreconditioner : public Preconditioner<OperatorType, VectorType, ValueType> {

public:

  BlockPreconditioner();
  virtual ~BlockPreconditioner();

  virtual void Print(void) const;  
  virtual void Clear(void);  

  virtual void Set(const int n,
                   const int *size,
                   Solver<OperatorType, VectorType, ValueType> **D_solver);

  virtual void SetDiagonalSolver(void);
  virtual void SetLSolver(void);

  virtual void SetExternalLastMatrix(const OperatorType &mat);
  
  virtual void SetPermutation(const LocalVector<int> &perm);

  virtual void Build(void);

  virtual void Solve(const VectorType &rhs,
                     VectorType *x);

protected:

  // The operator decomposition
  OperatorType ***A_block_;
  OperatorType *A_last_;

  /// Keep the precond matrix in CSR or not
  bool op_mat_format_; 
  /// Precond matrix format
  unsigned int precond_mat_format_;

  VectorType **x_block_;  
  VectorType **tmp_block_;  
  VectorType x_;

  int num_blocks_;
  int *block_sizes_;

  Solver<OperatorType, VectorType, ValueType> **D_solver_;


  bool diag_solve_;

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);
  

};


}

#endif // ROCALUTION_PRECONDITIONER_BLOCKPRECOND_HPP_
