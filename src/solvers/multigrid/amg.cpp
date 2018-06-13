#include "../../utils/def.hpp"
#include "amg.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"

#include "../../solvers/preconditioners/preconditioner_multicolored_gs.hpp"

#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
AMG<OperatorType, VectorType, ValueType>::AMG() {

  LOG_DEBUG(this, "AMG::AMG()",
            "default constructor");

  // parameter for strong couplings in smoothed aggregation
  this->eps_   = ValueType(0.01f);
  this->relax_ = ValueType(2.0f/3.0f);
  this->over_interp_ = ValueType(1.5f);
  this->interp_type_ = SmoothedAggregation;

}

template <class OperatorType, class VectorType, typename ValueType>
AMG<OperatorType, VectorType, ValueType>::~AMG() {

  LOG_DEBUG(this, "AMG::AMG()",
            "destructor");

  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void AMG<OperatorType, VectorType, ValueType>::Print(void) const {

  LOG_INFO("AMG solver");

  LOG_INFO("AMG number of levels " << this->levels_);

  switch(this->interp_type_) {
    case Aggregation:
      LOG_INFO("AMG using aggregation interpolation");
      break;
    case SmoothedAggregation:
      LOG_INFO("AMG using smoothed aggregation interpolation");
      break;
  }

  LOG_INFO("AMG coarsest operator size = " << this->op_level_[this->levels_-2]->GetM());
  LOG_INFO("AMG coarsest level nnz = " <<this->op_level_[this->levels_-2]->GetNnz());
  LOG_INFO("AMG with smoother:");
  this->smoother_level_[0]->Print();
  
}

template <class OperatorType, class VectorType, typename ValueType>
void AMG<OperatorType, VectorType, ValueType>::PrintStart_(void) const {

  assert(this->levels_ > 0);

  LOG_INFO("AMG solver starts");
  LOG_INFO("AMG number of levels " << this->levels_);

  switch(this->interp_type_) {
    case Aggregation:
      LOG_INFO("AMG using aggregation interpolation");
      break;
    case SmoothedAggregation:
      LOG_INFO("AMG using smoothed aggregation interpolation");
      break;
  }

  LOG_INFO("AMG coarsest operator size = " << this->op_level_[this->levels_-2]->GetM());
  LOG_INFO("AMG coarsest level nnz = " <<this->op_level_[this->levels_-2]->GetNnz());
  LOG_INFO("AMG with smoother:");
  this->smoother_level_[0]->Print();

}

template <class OperatorType, class VectorType, typename ValueType>
void AMG<OperatorType, VectorType, ValueType>::PrintEnd_(void) const {

    LOG_INFO("AMG ends");

}

template <class OperatorType, class VectorType, typename ValueType>
void AMG<OperatorType, VectorType, ValueType>::SetInterpolation(unsigned int interpType) {

  this->interp_type_ = interpType;

}

template <class OperatorType, class VectorType, typename ValueType>
void AMG<OperatorType, VectorType, ValueType>::SetInterpRelax(const ValueType relax) {

  LOG_DEBUG(this, "AMG::SetInterpRelax()",
            relax);

  this->relax_ = relax;

}

template <class OperatorType, class VectorType, typename ValueType>
void AMG<OperatorType, VectorType, ValueType>::SetOverInterp(const ValueType overInterp) {

  LOG_DEBUG(this, "AMG::SetOverInterp()",
            overInterp);

  this->over_interp_ = overInterp;

}

template <class OperatorType, class VectorType, typename ValueType>
void AMG<OperatorType, VectorType, ValueType>::SetCouplingStrength(const ValueType eps) {

  LOG_DEBUG(this, "AMG::SetCouplingStrength()",
            eps);

  this->eps_ = eps;

}

template <class OperatorType, class VectorType, typename ValueType>
void AMG<OperatorType, VectorType, ValueType>::BuildSmoothers(void) {

  LOG_DEBUG(this, "AMG::BuildSmoothers()",
            " #*# begin");

  // Smoother for each level
  this->smoother_level_ = new IterativeLinearSolver<OperatorType, VectorType, ValueType>*[this->levels_-1];
  this->sm_default_ = new Solver<OperatorType, VectorType, ValueType>*[this->levels_-1];

  for (int i=0; i<this->levels_-1; ++i) {
    FixedPoint<OperatorType, VectorType, ValueType> *sm =
        new FixedPoint<OperatorType, VectorType, ValueType>;
    MultiColoredGS<OperatorType, VectorType, ValueType> *gs =
        new MultiColoredGS<OperatorType, VectorType, ValueType>;

    gs->SetPrecondMatrixFormat(this->sm_format_);
    sm->SetRelaxation(ValueType(1.3f));
    sm->SetPreconditioner(*gs);
    sm->Verbose(0);

    this->smoother_level_[i] = sm;
    this->sm_default_[i] = gs;
  }

}

template <class OperatorType, class VectorType, typename ValueType>
void AMG<OperatorType, VectorType, ValueType>::ReBuildNumeric(void) {

  LOG_DEBUG(this, "AMG::ReBuildNumeric()",
            " #*# begin");

  assert(this->levels_ > 1);
  assert(this->build_);
  assert(this->op_ != NULL);

  this->op_level_[0]->Clear();
  this->op_level_[0]->ConvertToCSR();

  if (this->op_->GetFormat() != CSR) {
    OperatorType op_csr;
    op_csr.CloneFrom(*this->op_);
    op_csr.ConvertToCSR();

    // Create coarse operator
    OperatorType tmp;
    tmp.CloneBackend(*this->op_);
    this->op_level_[0]->CloneBackend(*this->op_);

    OperatorType *cast_res = dynamic_cast<OperatorType*>(this->restrict_op_level_[0]);
    OperatorType *cast_pro = dynamic_cast<OperatorType*>(this->prolong_op_level_[0]);
    assert(cast_res != NULL);
    assert(cast_pro != NULL);

    tmp.MatrixMult(*cast_res, op_csr);
    this->op_level_[0]->MatrixMult(tmp, *cast_pro);

  } else {

    // Create coarse operator
    OperatorType tmp;
    tmp.CloneBackend(*this->op_);
    this->op_level_[0]->CloneBackend(*this->op_);

    OperatorType *cast_res = dynamic_cast<OperatorType*>(this->restrict_op_level_[0]);
    OperatorType *cast_pro = dynamic_cast<OperatorType*>(this->prolong_op_level_[0]);
    assert(cast_res != NULL);
    assert(cast_pro != NULL);

    tmp.MatrixMult(*cast_res, *this->op_);
    this->op_level_[0]->MatrixMult(tmp, *cast_pro);

  }

  for (int i=1; i<this->levels_-1; ++i) {

    this->op_level_[i]->Clear();
    this->op_level_[i]->ConvertToCSR();

    // Create coarse operator
    OperatorType tmp;
    tmp.CloneBackend(*this->op_);
    this->op_level_[i]->CloneBackend(*this->op_);

    OperatorType *cast_res = dynamic_cast<OperatorType*>(this->restrict_op_level_[i]);
    OperatorType *cast_pro = dynamic_cast<OperatorType*>(this->prolong_op_level_[i]);
    assert(cast_res != NULL);
    assert(cast_pro != NULL);

    if (i == this->levels_ - this->host_level_ - 1)
      this->op_level_[i-1]->MoveToHost();

    tmp.MatrixMult(*cast_res, *this->op_level_[i-1]);
    this->op_level_[i]->MatrixMult(tmp, *cast_pro);

    if (i == this->levels_ - this->host_level_ - 1)
      this->op_level_[i-1]->CloneBackend(*this->restrict_op_level_[i-1]);

  }

  for (int i=0; i<this->levels_-1; ++i) {

    if (i > 0)
      this->smoother_level_[i]->ResetOperator(*this->op_level_[i-1]);
    else
      this->smoother_level_[i]->ResetOperator(*this->op_);


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
void AMG<OperatorType, VectorType, ValueType>::Aggregate(const OperatorType &op,
                                                         Operator<ValueType> *pro,
                                                         Operator<ValueType> *res,
                                                         OperatorType *coarse) {

  LOG_DEBUG(this, "AMG::Aggregate()",
            this->build_);

  assert(pro    != NULL);
  assert(res    != NULL);
  assert(coarse != NULL);

  OperatorType *cast_res = dynamic_cast<OperatorType*>(res);
  OperatorType *cast_pro = dynamic_cast<OperatorType*>(pro);

  assert(cast_res != NULL);
  assert(cast_pro != NULL);

  LocalVector<int> connections;
  LocalVector<int> aggregates;

  connections.CloneBackend(op);
  aggregates.CloneBackend(op);

  ValueType eps = this->eps_;
  for (int i=0; i<this->levels_-1; ++i)
    eps *= ValueType(0.5);

  op.AMGConnect(eps, &connections);
  op.AMGAggregate(connections, &aggregates);

  switch(this->interp_type_) {

    case Aggregation:
      op.AMGAggregation(aggregates, cast_pro, cast_res);
      break;

    case SmoothedAggregation:
      op.AMGSmoothedAggregation(this->relax_, aggregates, connections, cast_pro, cast_res);
      break;

    default:
      LOG_INFO("Aggregation type not valid");
      FATAL_ERROR(__FILE__, __LINE__);

  }

  // Free unused vectors
  connections.Clear();
  aggregates.Clear();

  OperatorType tmp;
  tmp.CloneBackend(op);
  coarse->CloneBackend(op);

  tmp.MatrixMult(*cast_res, op);
  coarse->MatrixMult(tmp, *cast_pro);

  if (this->interp_type_ == Aggregation && this->over_interp_ > ValueType(1.0))
    coarse->Scale(ValueType(1.0)/this->over_interp_);

}


template class AMG< LocalMatrix<double>, LocalVector<double>, double >;
template class AMG< LocalMatrix<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class AMG< LocalMatrix<std::complex<double> >, LocalVector<std::complex<double> >, std::complex<double> >;
template class AMG< LocalMatrix<std::complex<float> >,  LocalVector<std::complex<float> >,  std::complex<float> >;
#endif

}
