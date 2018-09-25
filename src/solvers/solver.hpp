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

/** \ingroup solver_module
  * \class Solver
  * \brief Base class for all solvers and preconditioners
  *
  * \tparam OperatorType
  * \tparam VectorType
  * \tparam ValueType
  */
template <class OperatorType, class VectorType, typename ValueType>
class Solver : public RocalutionObj
{
    public:
    Solver();
    virtual ~Solver();

    /** \brief Set the Operator of the solver */
    void SetOperator(const OperatorType& op);

    /** \brief Reset the operator; see ReBuildNumeric() */
    virtual void ResetOperator(const OperatorType& op);

    /** \brief Print information about the solver */
    virtual void Print(void) const = 0;

    /** \brief Solve Operator x = rhs */
    virtual void Solve(const VectorType& rhs, VectorType* x) = 0;

    /** \brief Solve Operator x = rhs, setting initial x = 0 */
    virtual void SolveZeroSol(const VectorType& rhs, VectorType* x);

    /** \brief Clear (free all local data) the solver */
    virtual void Clear(void);

    /** \brief Build the solver (data allocation, structure and numerical computation) */
    virtual void Build(void);

    /** \brief Build the solver and move it to the accelerator asynchronously */
    virtual void BuildMoveToAcceleratorAsync(void);

    /** \brief Synchronize the solver */
    virtual void Sync(void);

    /** \brief Rebuild the solver only with numerical computation (no allocation or data
      * structure computation)
      */
    virtual void ReBuildNumeric(void);

    /** \brief Move all data (i.e. move the solver) to the host */
    virtual void MoveToHost(void);
    /** \brief Move all data (i.e. move the solver) to the accelerator */
    virtual void MoveToAccelerator(void);

    /** \brief Provide verbose output of the solver
      * \details
      * verb = 0 no output <br>
      * verb = 1 print info about the solver (start, end); <br>
      * verb = 2 print (iter, residual) via iteration control;
      */
    virtual void Verbose(int verb = 1);

    protected:
    /** \brief Pointer to the operator */
    const OperatorType* op_;

    /** \brief Pointer to the defined preconditioner */
    Solver<OperatorType, VectorType, ValueType>* precond_;

    /** \brief Flag == true after building the solver (e.g. Build()) */
    bool build_;

    /** \brief Permutation vector (used if the solver performs permutation/re-ordering
      * techniques)
      */
    LocalVector<int> permutation_;

    /** \brief Verbose flag */
    int verb_;

    /** \brief Print starting message of the solver */
    virtual void PrintStart_(void) const = 0;
    /** \brief Print ending message of the solver */
    virtual void PrintEnd_(void) const = 0;

    /** \brief Move all local data to the host */
    virtual void MoveToHostLocalData_(void) = 0;
    /** \brief Move all local data to the accelerator */
    virtual void MoveToAcceleratorLocalData_(void) = 0;
};

/// Base class for all linear (iterative) solvers
template <class OperatorType, class VectorType, typename ValueType>
class IterativeLinearSolver : public Solver<OperatorType, VectorType, ValueType>
{
    public:
    IterativeLinearSolver();
    virtual ~IterativeLinearSolver();

    /// Initialize the solver with absolute/relative/divergence
    /// tolerance and maximum number of iterations
    void Init(double abs_tol, double rel_tol, double div_tol, int max_iter);

    /// Initialize the solver with absolute/relative/divergence
    /// tolerance and minimum/maximum number of iterations
    void Init(double abs_tol, double rel_tol, double div_tol, int min_iter, int max_iter);

    /// Set the minimum number of iterations
    void InitMinIter(int min_iter);

    /// Set the maximum number of iterations
    void InitMaxIter(int max_iter);

    /// Set the absolute/relative/divergence tolerance
    void InitTol(double abs, double rel, double div);

    /// Set the residual norm to L1, L2 or Inf norm
    /// resnorm == 1 L1 Norm
    /// resnorm == 2 L2 Norm (default)
    /// resnorm == 3 Inf Norm
    void SetResidualNorm(int resnorm);

    /// Record the residual history
    void RecordResidualHistory(void);

    /// Write the history to file
    void RecordHistory(const std::string filename) const;

    virtual void Verbose(int verb = 1);

    virtual void Solve(const VectorType& rhs, VectorType* x);

    /// Set a preconditioner of the linear solver
    virtual void SetPreconditioner(Solver<OperatorType, VectorType, ValueType>& precond);

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
    virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x) = 0;

    /// Preconditioned solution procedure
    virtual void SolvePrecond_(const VectorType& rhs, VectorType* x) = 0;

    /// Residual norm
    /// res_norm = 1 L1 Norm
    /// res_norm = 2 L2 Norm
    /// res_norm = 3 Linf Norm
    int res_norm_;

    /// Absolute maximum index of residual vector when using Linf norm
    int index_;

    /// Computes the vector norm
    ValueType Norm(const VectorType& vec);
};

/// Fixed-point iteration \f$x_{k+1}=x_k-\omega M^{-1} (A x_k - b)\f$,
/// where the solution of \f$M^{-1}\f$ is provide by solver via SetPreconditioner()
template <class OperatorType, class VectorType, typename ValueType>
class FixedPoint : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    FixedPoint();
    virtual ~FixedPoint();

    virtual void Print(void) const;

    virtual void ReBuildNumeric(void);

    /// Set a relaxation parameter of the iterative solver
    void SetRelaxation(ValueType omega);

    virtual void Build(void);

    virtual void Clear(void);

    protected:
    ValueType omega_;
    VectorType x_old_;
    VectorType x_res_;

    virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);
    virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);
};

/// Base class for all linear (direct) solvers
template <class OperatorType, class VectorType, typename ValueType>
class DirectLinearSolver : public Solver<OperatorType, VectorType, ValueType>
{
    public:
    DirectLinearSolver();
    virtual ~DirectLinearSolver();

    virtual void Verbose(int verb = 1);

    virtual void Solve(const VectorType& rhs, VectorType* x);

    protected:
    /// Solution procedure
    virtual void Solve_(const VectorType& rhs, VectorType* x) = 0;
};

} // namespace rocalution

#endif // ROCALUTION_SOLVER_HPP_
