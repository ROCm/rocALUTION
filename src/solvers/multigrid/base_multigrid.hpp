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

/** \ingroup solver_module
  * \class BaseMultiGrid
  * \brief Base class for all multigrid solvers
  *
  * \tparam OperatorType
  * \tparam VectorType
  * \tparam ValueType
  */
template <class OperatorType, class VectorType, typename ValueType>
class BaseMultiGrid : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    BaseMultiGrid();
    virtual ~BaseMultiGrid();

    virtual void Print(void) const;

    /** \private */
    virtual void SetPreconditioner(Solver<OperatorType, VectorType, ValueType>& precond);

    /** \brief Set the coarse grid solver */
    void SetSolver(Solver<OperatorType, VectorType, ValueType>& solver);

    /** \brief Set the smoother for each level */
    void SetSmoother(IterativeLinearSolver<OperatorType, VectorType, ValueType>** smoother);

    /** \brief Set the number of pre-smoothing steps */
    void SetSmootherPreIter(int iter);

    /** \brief Set the number of post-smoothing steps */
    void SetSmootherPostIter(int iter);

    /** \brief Set the restriction operator for each level */
    virtual void SetRestrictOperator(OperatorType** op) = 0;

    /** \brief Set the prolongation operator for each level */
    virtual void SetProlongOperator(OperatorType** op) = 0;

    /** \brief Set the operator for each level */
    virtual void SetOperatorHierarchy(OperatorType** op) = 0;

    /** \brief Enable/disable scaling of intergrid transfers */
    void SetScaling(bool scaling);

    /** \brief Force computation of coarser levels on the host backend */
    void SetHostLevels(int levels);

    /** \brief Set the MultiGrid Cycle (default: Vcycle) */
    void SetCycle(unsigned int cycle);

    /** \brief Set the MultiGrid Kcycle on all levels or only on finest level */
    void SetKcycleFull(bool kcycle_full);

    /** \brief Set the depth of the multigrid solver */
    void InitLevels(int levels);

    virtual void Solve(const VectorType& rhs, VectorType* x);

    virtual void Build(void);
    virtual void Clear(void);

    protected:
    /** \brief Restricts from level 'level' to 'level-1' */
    void Restrict_(const VectorType& fine, VectorType* coarse, int level);

    /** \brief Prolongs from level 'level' to 'level+1' */
    void Prolong_(const VectorType& coarse, VectorType* fine, int level);

    /** \brief V-cycle */
    void Vcycle_(const VectorType& rhs, VectorType* x);
    /** \brief W-cycle */
    void Wcycle_(const VectorType& rhs, VectorType* x);
    /** \brief F-cycle */
    void Fcycle_(const VectorType& rhs, VectorType* x);
    /** \brief K-cycle */
    void Kcycle_(const VectorType& rhs, VectorType* x);

    /** \private */
    virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);

    /** \private */
    virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    /** \brief Move all level data to the host */
    void MoveHostLevels_(void);

    /** \brief Number of levels in the hierarchy */
    int levels_;
    /** \brief Host levels */
    int host_level_;
    /** \brief Current level */
    int current_level_;
    /** \brief Intergrid transfer scaling */
    bool scaling_;
    /** \brief Number of pre-smoothing steps */
    int iter_pre_smooth_;
    /** \brief Number of post-smoothing steps */
    int iter_post_smooth_;
    /** \brief Cycle type */
    unsigned int cycle_;
    /** \brief K-cycle type */
    bool kcycle_full_;

    /** \brief Residual norm */
    double res_norm_;

    /** \brief Operator hierarchy */
    OperatorType** op_level_;

    /** \brief Restriction operator hierarchy */
    Operator<ValueType>** restrict_op_level_;
    /** \brief Prolongation operator hierarchy */
    Operator<ValueType>** prolong_op_level_;

    VectorType** d_level_; /**< \private */
    VectorType** r_level_; /**< \private */
    VectorType** t_level_; /**< \private */
    VectorType** s_level_; /**< \private */
    VectorType** p_level_; /**< \private */
    VectorType** q_level_; /**< \private */
    VectorType** k_level_; /**< \private */
    VectorType** l_level_; /**< \private */

    /** \brief Coarse grid solver */
    Solver<OperatorType, VectorType, ValueType>* solver_coarse_;
    /** \brief Smoother for each level */
    IterativeLinearSolver<OperatorType, VectorType, ValueType>** smoother_level_;
};

} // namespace rocalution

#endif // ROCALUTION_BASE_MULTIGRID_HPP_
