#ifndef PARALUTION_PRECONDITIONER_MULTIELIMINATION_HPP_
#define PARALUTION_PRECONDITIONER_MULTIELIMINATION_HPP_

#include "../solver.hpp"
#include "preconditioner.hpp"
#include "../../base/local_vector.hpp"

#include <vector>

namespace paralution {

/// MultiElimination (I)LU factorization (see 12.5.1 Multi-Elimination ILU from 
/// "Iterative Methods for Sparse Linear Systems", 2nd Edition, Yousef Saad);
/// The ME-ILU preconditioner is build recursively 
template <class OperatorType, class VectorType, typename ValueType>
class MultiElimination : public Preconditioner<OperatorType, VectorType, ValueType> {

public:

  MultiElimination();
  virtual ~MultiElimination();

  /// Returns the size of the first (diagonal) block of the preconditioner
  inline int get_size_diag_block(void) const { return this->size_; }

  /// Return the depth of the current level
  inline int get_level(void) const { return this->level_; }

  virtual void Print(void) const;
  virtual void Clear(void);

  /// Initialize (recursively) ME-ILU with level (defines the depth of recursion);
  /// AA_Solvers - defines the last-block solver;
  /// drop_off - defines drop-off tolerance
  virtual void Set(Solver<OperatorType, VectorType, ValueType> &AA_Solver,
                   const int level, const double drop_off = double(0.0));

  /// Set a specific matrix type of the decomposed block matrices;
  /// if not set, CSR matrix format will be used
  virtual void SetPrecondMatrixFormat(const unsigned int mat_format);

  virtual void Build(void);

  virtual void Solve(const VectorType &rhs,
                     VectorType *x);

protected:

  /// A_ is decomposed into \f$ [D_, F_; E_, C_] \f$, where
  /// \f$ AA = C - E D^{-1} F\f$;
  /// \f$ E = E D^{-1} \f$;
  OperatorType A_;
  /// Diagonal part
  OperatorType D_;
  OperatorType E_,F_,C_;
  /// \f$ AA = C - E D^{-1} F\f$;
  OperatorType AA_;

  /// The sizes of the AA_ matrix
  int AA_nrow_, AA_nnz_;

  /// Keep the precond matrix in CSR or not
  bool op_mat_format_;
  /// Precond matrix format
  unsigned int precond_mat_format_;

  VectorType x_;
  VectorType x_1_;
  VectorType x_2_;

  VectorType rhs_;
  VectorType rhs_1_;
  VectorType rhs_2_;

  MultiElimination<OperatorType, VectorType, ValueType> *AA_me_;
  Solver<OperatorType, VectorType, ValueType> *AA_solver_;

  bool diag_solver_init_;
  int level_;
  double drop_off_;

  VectorType inv_vec_D_;
  VectorType vec_D_;
  LocalVector<int> permutation_;
  int size_;

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);

};


}

#endif // PARALUTION_PRECONDITIONER_MULTIELIMINATION_HPP_
