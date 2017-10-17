#include "../../utils/def.hpp"
#include "../../utils/types.hpp"
#include "pairwise_amg.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"

#include "../preconditioners/preconditioner.hpp"

#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"
#include "../../utils/allocate_free.hpp"

#include <list>
#include <complex>

namespace paralution {

template <class OperatorType, class VectorType, typename ValueType>
PairwiseAMG<OperatorType, VectorType, ValueType>::PairwiseAMG() {

  LOG_DEBUG(this, "PairwiseAMG::PairwiseAMG()",
            "default constructor");

  this->beta_ = ValueType(0.25f);
  this->coarse_size_ = 300;

  // target coarsening factor
  this->coarsening_factor_ = 4.0f;

  // number of pre- and post-smoothing steps
  this->iter_pre_smooth_ = 2;
  this->iter_post_smooth_ = 2;

  // set K-cycle to default
  this->cycle_ = 2;

  // disable scaling
  this->scaling_ = false;

  // set default ordering to connectivity ordering
  this->aggregation_ordering_ = 1;

}

template <class OperatorType, class VectorType, typename ValueType>
PairwiseAMG<OperatorType, VectorType, ValueType>::~PairwiseAMG() {

  LOG_DEBUG(this, "PairwiseAMG::PairwiseAMG()",
            "destructor");

  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::Print(void) const {

  LOG_INFO("AMG solver");
  LOG_INFO("AMG number of levels " << this->levels_);
  LOG_INFO("AMG using pairwise aggregation");
  LOG_INFO("AMG coarsest operator size = " << this->op_level_[this->levels_-2]->get_nrow());
  LOG_INFO("AMG coarsest level nnz = " << this->op_level_[this->levels_ - 2]->get_nnz());
  LOG_INFO("AMG with smoother:");
  this->smoother_level_[0]->Print();
  
}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::PrintStart_(void) const {

  assert(this->levels_ > 0);

  LOG_INFO("AMG solver starts");
  LOG_INFO("AMG number of levels " << this->levels_);
  LOG_INFO("AMG using pairwise aggregation");
  LOG_INFO("AMG coarsest operator size = " << this->op_level_[this->levels_-2]->get_nrow());
  LOG_INFO("AMG coarsest level nnz = " << this->op_level_[this->levels_ - 2]->get_nnz());
  LOG_INFO("AMG with smoother:");
  this->smoother_level_[0]->Print();

}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::PrintEnd_(void) const {

    LOG_INFO("AMG ends");

}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::SetBeta(const ValueType beta) {

  LOG_DEBUG(this, "PairwiseAMG::SetBeta()",
            beta);

  this->beta_ = beta;

}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::SetCoarseningFactor(const double factor) {

  LOG_DEBUG(this, "PairwiseAMG::SetCoarseningFactor()",
            factor);

  this->coarsening_factor_ = factor;

}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::SetOrdering(const _aggregation_ordering ordering) {

  LOG_DEBUG(this, "PairwiseAMG::SetOrdering()",
            ordering);

  assert(ordering >= 0 && ordering <= 5);

  this->aggregation_ordering_ = ordering;

}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::BuildHierarchy(void) {

  LOG_DEBUG(this, "PairwiseAMG::BuildHierarchy()",
            " #*# begin");

  if (this->hierarchy_ == false) {

    assert(this->build_ == false);
    this->hierarchy_ = true;

    // AMG will use operators for inter grid transfers
    assert(this->op_ != NULL);
    assert(this->coarse_size_ > 0);

    if (this->op_->get_nrow() <= (IndexType2) this->coarse_size_) {
      LOG_INFO("Problem size too small for AMG, use Krylov solver instead");
      FATAL_ERROR(__FILE__, __LINE__);
    }

    // Lists for the building procedure
    std::list<OperatorType*> op_list_;
    std::list<OperatorType*> restrict_list_;
    std::list<OperatorType*> prolong_list_;
    std::list<LocalVector<int>*> trans_list_;

    this->levels_ = 1;

    // Build finest hierarchy
    op_list_.push_back(new OperatorType);
    restrict_list_.push_back(new OperatorType);
    prolong_list_.push_back(new OperatorType);
    trans_list_.push_back(new LocalVector<int>);

    op_list_.back()->CloneBackend(*this->op_);
    restrict_list_.back()->CloneBackend(*this->op_);
    prolong_list_.back()->CloneBackend(*this->op_);
    trans_list_.back()->CloneBackend(*this->op_);

    // Create prolongation and restriction operators
    this->Aggregate(*this->op_, prolong_list_.back(), restrict_list_.back(), op_list_.back(), trans_list_.back());

    ++this->levels_;

    while(op_list_.back()->get_nrow() > (IndexType2) this->coarse_size_) {

      // Add new list elements
      restrict_list_.push_back(new OperatorType);
      prolong_list_.push_back(new OperatorType);
      OperatorType *prev_op_ = op_list_.back();
      op_list_.push_back(new OperatorType);
      trans_list_.push_back(new LocalVector<int>);

      op_list_.back()->CloneBackend(*this->op_);
      restrict_list_.back()->CloneBackend(*this->op_);
      prolong_list_.back()->CloneBackend(*this->op_);
      trans_list_.back()->CloneBackend(*this->op_);

      this->Aggregate(*prev_op_, prolong_list_.back(), restrict_list_.back(), op_list_.back(), trans_list_.back());

      ++this->levels_;

    }

    // Allocate data structures
    this->op_level_ = new OperatorType*[this->levels_-1];
    this->restrict_op_level_ = new Operator<ValueType>*[this->levels_-1];
    this->prolong_op_level_ = new Operator<ValueType>*[this->levels_-1];
    this->trans_level_ = new LocalVector<int>*[this->levels_-1];

    typename std::list<OperatorType*>::iterator op_it  = op_list_.begin();
    typename std::list<OperatorType*>::iterator pro_it = prolong_list_.begin();
    typename std::list<OperatorType*>::iterator res_it = restrict_list_.begin();
    typename std::list<LocalVector<int>*>::iterator trans_it = trans_list_.begin();

    for (int i=0; i<this->levels_-1; ++i) {

      this->op_level_[i] = *op_it;
      this->op_level_[i]->Sort();
      ++op_it;

      this->restrict_op_level_[i] = *res_it;
      ++res_it;

      this->prolong_op_level_[i] = *pro_it;
      ++pro_it;

      this->trans_level_[i] = *trans_it;
      ++trans_it;

    }

  }

  LOG_DEBUG(this, "PairwiseAMG::BuildHierarchy()",
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::BuildSmoothers(void) {

  LOG_DEBUG(this, "PairwiseAMG::BuildSmoothers()",
            " #*# begin");

  // Smoother for each level
  FixedPoint<OperatorType, VectorType, ValueType > **sm = NULL;
  sm = new FixedPoint<OperatorType, VectorType, ValueType >* [this->levels_-1];

  this->smoother_level_ = new IterativeLinearSolver<OperatorType, VectorType, ValueType>*[this->levels_-1];
  this->sm_default_ = new Solver<OperatorType, VectorType, ValueType>*[this->levels_-1];

  Jacobi<OperatorType, VectorType, ValueType > **jac = NULL;
  jac = new Jacobi<OperatorType, VectorType, ValueType >* [this->levels_-1];

  for (int i=0; i<this->levels_-1; ++i) {
    sm[i] = new FixedPoint<OperatorType, VectorType, ValueType >;
    jac[i] = new Jacobi<OperatorType, VectorType, ValueType >;

    sm[i]->SetRelaxation(ValueType(0.67f));
    sm[i]->SetPreconditioner(*jac[i]);
    sm[i]->Verbose(0);
    this->smoother_level_[i] = sm[i];
    this->sm_default_[i] = jac[i];
  }

  delete[] jac;
  delete[] sm;

}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::ReBuildNumeric(void) {

  LOG_DEBUG(this, "PairwiseAMG::ReBuildNumeric()",
            " #*# begin");

  assert(this->levels_ > 1);
  assert(this->build_ == true);
  assert(this->op_ != NULL);

  this->op_level_[0]->Clear();
  this->op_level_[0]->CloneBackend(*this->op_);
  this->op_level_[0]->ConvertToCSR();

  this->trans_level_[0]->CloneBackend(*this->op_);

  this->op_->CoarsenOperator(this->op_level_[0], this->dim_level_[0], this->dim_level_[0],
                             *this->trans_level_[0], this->Gsize_level_[0], this->rG_level_[0], this->rGsize_level_[0]);

  for (int i=1; i<this->levels_-1; ++i) {

    this->op_level_[i]->Clear();
    this->op_level_[i]->ConvertToCSR();

    this->trans_level_[i]->CloneBackend(*this->op_level_[i]);

    if (i == this->levels_ - this->host_level_ - 1)
      this->op_level_[i-1]->MoveToHost();

    this->op_level_[i-1]->CoarsenOperator(this->op_level_[i], this->dim_level_[i],
                                          this->dim_level_[i], *this->trans_level_[i], this->Gsize_level_[i],
                                          this->rG_level_[i], this->rGsize_level_[i]);

    if (i == this->levels_ - this->host_level_ - 1)
      this->op_level_[i-1]->CloneBackend(*this->restrict_op_level_[i-1]);

  }

  this->smoother_level_[0]->ResetOperator(*this->op_);
  this->smoother_level_[0]->ReBuildNumeric();
  this->smoother_level_[0]->Verbose(0);

  for (int i=1; i<this->levels_-1; ++i) {

    this->smoother_level_[i]->ResetOperator(*this->op_level_[i-1]);
    this->smoother_level_[i]->ReBuildNumeric();
    this->smoother_level_[i]->Verbose(0);

  }

  this->solver_coarse_->ResetOperator(*this->op_level_[this->levels_-2]);
  this->solver_coarse_->ReBuildNumeric();
  this->solver_coarse_->Verbose(0);

  // Convert operator to op_format
  if (this->op_format_ != CSR)
    for (int i=0; i<this->levels_-1;++i)
      this->op_level_[i]->ConvertTo(this->op_format_);

}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::ClearLocal(void) {

  LOG_DEBUG(this, "PairwiseAMG::ClearLocal()",
            this->build_);

  if (this->build_ == true) {

    for (int i=0; i<this->levels_-1; ++i) {

      delete this->trans_level_[i];

      free_host(&this->rG_level_[i]);

    }

    delete[] this->trans_level_;

    this->dim_level_.clear();
    this->Gsize_level_.clear();
    this->rGsize_level_.clear();
    this->rG_level_.clear();

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::Aggregate(const OperatorType &op,
                                                                 Operator<ValueType> *pro,
                                                                 Operator<ValueType> *res,
                                                                 OperatorType *coarse,
                                                                 LocalVector<int> *trans) {

  LOG_DEBUG(this, "PairwiseAMG::Aggregate()",
            this->build_);

  assert(pro    != NULL);
  assert(res    != NULL);
  assert(coarse != NULL);

  OperatorType *cast_res = dynamic_cast<OperatorType*>(res);
  OperatorType *cast_pro = dynamic_cast<OperatorType*>(pro);

  assert(cast_res != NULL);
  assert(cast_pro != NULL);

  int nc;
  int *rG = NULL;
  int Gsize;
  int rGsize;

  // Allocate transfer mapping for current level
  trans->Allocate("transfer map", op.get_local_nrow());

  op.InitialPairwiseAggregation(this->beta_, nc, trans, Gsize, &rG, rGsize, this->aggregation_ordering_);
  op.CoarsenOperator(coarse, nc, nc, *trans, Gsize, rG, rGsize);

  int cycle = 0;

  while (double(op.get_nrow()) / double(coarse->get_nrow()) < this->coarsening_factor_) {

    coarse->FurtherPairwiseAggregation(this->beta_, nc, trans, Gsize, &rG, rGsize, this->aggregation_ordering_);
    op.CoarsenOperator(coarse, nc, nc, *trans, Gsize, rG, rGsize);

    ++cycle;

    if (cycle > 8)
      LOG_VERBOSE_INFO(2, "*** warning: PairwiseAMG::Build() Coarsening cannot obtain satisfying coarsening factor");

  }

  cast_res->CreateFromMap(*trans, op.get_local_nrow(), nc, cast_pro);

  // Store data for possible coarse operator rebuild
  this->dim_level_.push_back(nc);
  this->Gsize_level_.push_back(Gsize);
  this->rGsize_level_.push_back(rGsize);
  this->rG_level_.push_back(rG);

}

// disabled function
template <class OperatorType, class VectorType, typename ValueType>
void PairwiseAMG<OperatorType, VectorType, ValueType>::Aggregate(const OperatorType &op,
                                                                 Operator<ValueType> *pro,
                                                                 Operator<ValueType> *res,
                                                                 OperatorType *coarse) {

  FATAL_ERROR(__FILE__, __LINE__);

}


template class PairwiseAMG< LocalMatrix<double>, LocalVector<double>, double >;
template class PairwiseAMG< LocalMatrix<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class PairwiseAMG< LocalMatrix<std::complex<double> >, LocalVector<std::complex<double> >, std::complex<double> >;
template class PairwiseAMG< LocalMatrix<std::complex<float> >,  LocalVector<std::complex<float> >,  std::complex<float> >;
#endif

}
