#ifndef PARALUTION_MULTIGRID_GLOBAL_PAIRWISE_AMG_HPP_
#define PARALUTION_MULTIGRID_GLOBAL_PAIRWISE_AMG_HPP_

#include "../solver.hpp"
#include "base_amg.hpp"
#include "pairwise_amg.hpp"

namespace paralution {

template <class OperatorType, class VectorType, typename ValueType>
class GlobalPairwiseAMG : public BaseAMG<OperatorType, VectorType, ValueType> {

public:

  GlobalPairwiseAMG();
  virtual ~GlobalPairwiseAMG();

  virtual void Print(void) const;

  /// Creates AMG hierarchy
  virtual void BuildHierarchy(void);
  virtual void ClearLocal(void);

  /// Sets beta for pairwise aggregation
  virtual void SetBeta(const ValueType beta);
  /// Sets reordering for aggregation
  virtual void SetOrdering(const _aggregation_ordering ordering);
  /// Sets target coarsening factor
  virtual void SetCoarseningFactor(const double factor);

  virtual void ReBuildNumeric(void);

protected:

  /// Constructs the prolongation, restriction and coarse operator
  virtual void Aggregate(const OperatorType &op,
                         Operator<ValueType> *pro,
                         Operator<ValueType> *res,
                         OperatorType *coarse,
                         ParallelManager *pm,
                         LocalVector<int> *trans);

  /// disabled function
  virtual void Aggregate(const OperatorType &op,
                         Operator<ValueType> *pro,
                         Operator<ValueType> *res,
                         OperatorType *coarse);

  virtual void PrintStart_(void) const;
  virtual void PrintEnd_(void) const;

private:

  /// Beta factor
  ValueType beta_;

  /// Target factor for coarsening ratio
  double coarsening_factor_;
  /// Ordering for the aggregation scheme
  int aggregation_ordering_;

  /// Parallel Manager for coarser levels
  ParallelManager **pm_level_;

  /// Transfer mapping
  LocalVector<int> **trans_level_;

  /// Dimension of the coarse operators
  std::vector<int> dim_level_;
  std::vector<int> Gsize_level_;
  std::vector<int> rGsize_level_;
  std::vector<int*> rG_level_;

};


}

#endif // PARALUTION_MULTIGRID_GLOBAL_PAIRWISE_AMG_HPP_
