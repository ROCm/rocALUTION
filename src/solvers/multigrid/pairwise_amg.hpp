#ifndef ROCALUTION_MULTIGRID_PAIRWISE_AMG_HPP_
#define ROCALUTION_MULTIGRID_PAIRWISE_AMG_HPP_

#include "../solver.hpp"
#include "base_amg.hpp"

namespace rocalution {

enum _aggregation_ordering {
  NoOrdering = 0,
  Connectivity = 1,
  CMK = 2,
  RCMK = 3,
  MIS = 4,
  MultiColoring = 5
};

template <class OperatorType, class VectorType, typename ValueType>
class PairwiseAMG : public BaseAMG<OperatorType, VectorType, ValueType> {

public:

  PairwiseAMG();
  virtual ~PairwiseAMG();

  virtual void Print(void) const;

  /// Creates AMG hierarchy
  virtual void BuildHierarchy(void);
  virtual void ClearLocal(void);

  /// Build AMG smoothers
  virtual void BuildSmoothers(void);

  /// Sets beta for pairwise aggregation
  virtual void SetBeta(const ValueType beta);
  /// Sets reordering for aggregation
  virtual void SetOrdering(unsigned int ordering);
  /// Sets target coarsening factor
  virtual void SetCoarseningFactor(const double factor);

  /// Rebuild coarser operators with previous intergrid operators
  virtual void ReBuildNumeric(void);

protected:

  /// Constructs the prolongation, restriction and coarse operator
  virtual void Aggregate(const OperatorType &op,
                         Operator<ValueType> *pro,
                         Operator<ValueType> *res,
                         OperatorType *coarse,
                         LocalVector<int> *trans);

  /// disabled function
  virtual void Aggregate(const OperatorType &op,
                         Operator<ValueType> *pro,
                         Operator<ValueType> *res,
                         OperatorType *coarse);

  virtual void PrintStart_(void) const;
  virtual void PrintEnd_(void) const;

private:

  // Beta factor
  ValueType beta_;

  /// Target factor for coarsening ratio
  double coarsening_factor_;
  /// Ordering for the aggregation scheme
  unsigned int aggregation_ordering_;

  /// Transfer mapping
  LocalVector<int> **trans_level_;

  /// Dimension of the coarse operators
  std::vector<int> dim_level_;
  std::vector<int> Gsize_level_;
  std::vector<int> rGsize_level_;
  std::vector<int*> rG_level_;

};


}

#endif // ROCALUTION_MULTIGRID_PAIRWISE_AMG_HPP_
