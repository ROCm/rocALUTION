/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_SOLVER_HPP_
#define ROCALUTION_SOLVER_HPP_

#include "iter_ctrl.hpp"
#include "../base/base_rocalution.hpp"
#include "../base/local_vector.hpp"

namespace rocalution {

/// The base class for all solvers and preconditioners
template <class OperatorType, class VectorType, typename ValueType>
class Solver : public RocalutionObj {

public:

  Solver();
  virtual ~Solver();

  /// Set the Operator of the solver
  void SetOperator(const OperatorType &op);

  /// Reset the operator; see ReBuildNumeric
  virtual void ResetOperator(const OperatorType &op);

  /// Print information about the solver
  virtual void Print(void) const = 0;

  /// Solve Operator x = rhs
  virtual void Solve(const VectorType &rhs,
                     VectorType *x) = 0;

  /// Solve Operator x = rhs;
  /// but set first the init x = 0
  virtual void SolveZeroSol(const VectorType &rhs,
                            VectorType *x);

  /// Clear (free all local data) the solver
  virtual void Clear(void);

  /// Build the solver (data allocation, structure computation, 
  /// numerical computation)
  virtual void Build(void);

  virtual void BuildMoveToAcceleratorAsync(void);
  virtual void Sync(void);

  /// Rebuild only with numerical computation (no allocation or data 
  /// structure computation)
  virtual void ReBuildNumeric(void);

  /// Move all the data (i.e. move the solver) to the host
  virtual void MoveToHost(void);
  /// Move all the data (i.e. move the solver) to the accelerator
  virtual void MoveToAccelerator(void);

  /// Provide verbose output of the solver:
  /// verb == 0 no output
  /// verb == 1 print info about the solver (start,end);
  /// verb == 2 print (iter, residual) via iteration control;
  virtual void Verbose(const int verb=1);

protected:

  /// Pointer to the operator
  const OperatorType *op_;

  /// Pointer to the defined preconditioner
  Solver<OperatorType, VectorType, ValueType> *precond_;

  /// Flag == true after building the solver (e.g. Build())
  bool build_;

  /// Permutation vector (used if the solver performs 
  /// permutation/re-ordering techniques)
  LocalVector<int> permutation_;

  /// Verbose flag
  /// verb == 0 no output
  /// verb == 1 print info about the solver (start,end);
  /// verb == 2 print (iter, residual) via iteration control;
  int verb_;

  /// Print starting msg of the solver
  virtual void PrintStart_(void) const = 0;
  /// Print ending msg of the solver
  virtual void PrintEnd_(void) const = 0;

  /// Move all local data to the host
  virtual void MoveToHostLocalData_(void) = 0;
  /// Move all local data to the accelerator
  virtual void MoveToAcceleratorLocalData_(void) = 0;

};

/// Base class for all linear (iterative) solvers
template <class OperatorType, class VectorType, typename ValueType>
class IterativeLinearSolver : public Solver<OperatorType, VectorType, ValueType> {

public:

  IterativeLinearSolver();
  virtual ~IterativeLinearSolver();

  /// Initialize the solver with absolute/relative/divergence 
  /// tolerance and maximum number of iterations
  void Init(const double abs_tol,
            const double rel_tol,
            const double div_tol,
            const int max_iter);

  /// Initialize the solver with absolute/relative/divergence 
  /// tolerance and minimum/maximum number of iterations
  void Init(const double abs_tol,
            const double rel_tol,
            const double div_tol,
            const int min_iter,
            const int max_iter);

  /// Set the minimum number of iterations
  void InitMinIter(const int min_iter);

  /// Set the maximum number of iterations
  void InitMaxIter(const int max_iter);

  /// Set the absolute/relative/divergence tolerance
  void InitTol(const double abs,
               const double rel,
               const double div);

  /// Set the residual norm to L1, L2 or Inf norm
  /// resnorm == 1 L1 Norm
  /// resnorm == 2 L2 Norm (default)
  /// resnorm == 3 Inf Norm
  void SetResidualNorm(const int resnorm);

  /// Record the residual history
  void RecordResidualHistory(void);

  /// Write the history to file
  void RecordHistory(const std::string filename) const;

  virtual void Verbose(const int verb=1);

  virtual void Solve(const VectorType &rhs,
                     VectorType *x);

  /// Set a preconditioner of the linear solver
  virtual void SetPreconditioner(Solver<OperatorType, VectorType, ValueType> &precond);

  /// Return the iteration count
  virtual int GetIterationCount(void);

  /// Return the current residual
  virtual double GetCurrentResidual(void);

  /// Return the current status
  virtual int GetSolverStatus(void);

  /// Return absolute maximum index of residual vector when using Linf norm
  virtual int GetAmaxResidualIndex(void);

protected:

  /// Iteration control (monitor)
  IterationControl iter_ctrl_;

  /// Non-preconditioner solution procedure
  virtual void SolveNonPrecond_(const VectorType &rhs,
                                VectorType *x) = 0;

  /// Preconditioned solution procedure
  virtual void SolvePrecond_(const VectorType &rhs,
                             VectorType *x) = 0;

  /// Residual norm
  /// res_norm = 1 L1 Norm
  /// res_norm = 2 L2 Norm
  /// res_norm = 3 Linf Norm
  int res_norm_;

  /// Absolute maximum index of residual vector when using Linf norm
  int index_;

  /// Computes the vector norm
  ValueType Norm(const VectorType &vec);

};

/// Fixed-point iteration \f$x_{k+1}=x_k-\omega M^{-1} (A x_k - b)\f$,
/// where the solution of \f$M^{-1}\f$ is provide by solver via SetPreconditioner()
template <class OperatorType, class VectorType, typename ValueType>
class FixedPoint : public IterativeLinearSolver<OperatorType, VectorType, ValueType> {

public:

  FixedPoint();
  virtual ~FixedPoint();

  virtual void Print(void) const;

  virtual void ReBuildNumeric(void);

  /// Set a relaxation parameter of the iterative solver
  virtual void SetRelaxation(const ValueType omega);

  virtual void Build(void);

  virtual void Clear(void);

protected:

  ValueType omega_;
  VectorType x_old_;
  VectorType x_res_;


  virtual void SolveNonPrecond_(const VectorType &rhs,
                                VectorType *x);
  virtual void SolvePrecond_(const VectorType &rhs,
                                VectorType *x);

  virtual void PrintStart_(void) const;
  virtual void PrintEnd_(void) const;

  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);

};

/// Base class for all linear (direct) solvers
template <class OperatorType, class VectorType, typename ValueType>
class DirectLinearSolver : public Solver<OperatorType, VectorType, ValueType> {

public:

  DirectLinearSolver();
  virtual ~DirectLinearSolver();

  virtual void Verbose(const int verb=1);

  virtual void Solve(const VectorType &rhs,
                     VectorType *x);

protected:

  /// Solution procedure
  virtual void Solve_(const VectorType &rhs, VectorType *x) = 0;

};

} // namespace rocalution

#endif // ROCALUTION_SOLVER_HPP_
