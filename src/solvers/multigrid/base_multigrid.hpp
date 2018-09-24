/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_BASE_MULTIGRID_HPP_
#define ROCALUTION_BASE_MULTIGRID_HPP_

#include "../solver.hpp"
#include "../../base/operator.hpp"

namespace rocalution {

enum _cycle
{
    Vcycle = 0,
    Wcycle = 1,
    Kcycle = 2,
    Fcycle = 3
};

template <class OperatorType, class VectorType, typename ValueType>
class BaseMultiGrid : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    BaseMultiGrid();
    virtual ~BaseMultiGrid();

    virtual void Print(void) const;

    /// disabled function
    virtual void SetPreconditioner(Solver<OperatorType, VectorType, ValueType>& precond);

    /// Set the smoother for each level
    virtual void SetSolver(Solver<OperatorType, VectorType, ValueType>& solver);

    /// Set the smoother for each level
    virtual void SetSmoother(IterativeLinearSolver<OperatorType, VectorType, ValueType>** smoother);

    /// Set the number of pre-smoothing steps
    virtual void SetSmootherPreIter(int iter);

    /// Set the number of post-smoothing steps
    virtual void SetSmootherPostIter(int iter);

    /// Set thre restriction method by operator for each level
    virtual void SetRestrictOperator(OperatorType** op) = 0;

    /// Set the prolongation operator for each level
    virtual void SetProlongOperator(OperatorType** op) = 0;

    /// Set the operator for each level
    virtual void SetOperatorHierarchy(OperatorType** op) = 0;

    /// Enable/disable scaling of intergrid transfers
    virtual void SetScaling(bool scaling);

    /// Force computation of coarser levels on the host backend
    virtual void SetHostLevels(int levels);

    /// Set the MultiGrid Cycle (default: Vcycle)
    virtual void SetCycle(unsigned int cycle);

    /// Set the MultiGrid Kcycle on all levels or only on finest level
    virtual void SetKcycleFull(bool kcycle_full);

    /// Set the depth of the multigrid solver
    virtual void InitLevels(int levels);

    /// Called by default the V-cycle
    virtual void Solve(const VectorType& rhs, VectorType* x);

    virtual void Build(void);
    virtual void Clear(void);

    protected:
    // Restricts from level 'level' to 'level-1'
    virtual void Restrict_(const VectorType& fine, VectorType* coarse, int level);

    // Prolongs from level 'level' to 'level+1'
    virtual void Prolong_(const VectorType& coarse, VectorType* fine, int level);

    void Vcycle_(const VectorType& rhs, VectorType* x);
    void Wcycle_(const VectorType& rhs, VectorType* x);
    void Fcycle_(const VectorType& rhs, VectorType* x);
    void Kcycle_(const VectorType& rhs, VectorType* x);

    /// disabled function
    virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);

    /// disabled function
    virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);
    virtual void MoveHostLevels(void);

    int levels_;
    int host_level_;
    int current_level_;
    bool scaling_;
    int iter_pre_smooth_;
    int iter_post_smooth_;
    unsigned int cycle_;
    bool kcycle_full_;

    double res_norm_;

    OperatorType** op_level_;

    Operator<ValueType>** restrict_op_level_;
    Operator<ValueType>** prolong_op_level_;

    VectorType** d_level_;
    VectorType** r_level_;
    VectorType** t_level_;
    VectorType** s_level_;
    VectorType** p_level_;
    VectorType** q_level_;
    VectorType** k_level_;
    VectorType** l_level_;

    Solver<OperatorType, VectorType, ValueType>* solver_coarse_;
    IterativeLinearSolver<OperatorType, VectorType, ValueType>** smoother_level_;
};

} // namespace rocalution

#endif // ROCALUTION_BASE_MULTIGRID_HPP_
