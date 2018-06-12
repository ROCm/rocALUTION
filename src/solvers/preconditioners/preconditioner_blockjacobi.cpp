#include "../../utils/def.hpp"
#include "preconditioner_blockjacobi.hpp"
#include "../solver.hpp"
#include "../../base/global_matrix.hpp"
#include "../../base/local_matrix.hpp"

#include "../../base/global_vector.hpp"
#include "../../base/local_vector.hpp"

#include "../../utils/log.hpp"

#include "preconditioner.hpp"

#include <math.h>
#include <complex>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
BlockJacobi<OperatorType, VectorType, ValueType>::BlockJacobi() {

  LOG_DEBUG(this, "BlockJacobi::BlockJacobi()",
            "default constructor");

  this->local_precond_ = NULL;

}

template <class OperatorType, class VectorType, typename ValueType>
BlockJacobi<OperatorType, VectorType, ValueType>::~BlockJacobi() {

  LOG_DEBUG(this, "BlockJacobi::~BlockJacobi()",
            "destructor");

  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockJacobi<OperatorType, VectorType, ValueType>::Print(void) const {

  LOG_INFO("BlockJacobi preconditioner");

  this->local_precond_->Print();

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockJacobi<OperatorType, VectorType, ValueType>::Init(Solver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> &precond) {

  LOG_DEBUG(this, "BlockJacobi::Init()",
            this->build_);

  assert(this->local_precond_ == NULL);
  assert(this->build_ == false);

  this->local_precond_ = &precond;

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockJacobi<OperatorType, VectorType, ValueType>::Build(void) {

  LOG_DEBUG(this, "BlockJacobi::Build()",
            this->build_ <<
            " #*# begin");

  if (this->build_ == true)
    this->Clear();

  assert(this->build_ == false);
  this->build_ = true;

  assert(this->op_ != NULL);
  assert(this->local_precond_ != NULL);

  this->local_precond_->SetOperator(this->op_->GetInterior());
  this->local_precond_->Build();

  LOG_DEBUG(this, "BlockJacobi::Build()",
            this->build_ <<
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockJacobi<OperatorType, VectorType, ValueType>::ReBuildNumeric(void) {

  LOG_DEBUG(this, "BlockJacobi::ReBuildNumeric()",
            this->build_);

  if (this->build_ == true) {

    this->local_precond_->ReBuildNumeric();

  } else {

    this->Clear();
    this->Build();

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockJacobi<OperatorType, VectorType, ValueType>::Clear(void) {

  LOG_DEBUG(this, "BlockJacobi::Clear()",
            this->build_);

  if (this->local_precond_ != NULL)
    this->local_precond_->Clear();

  this->local_precond_ = NULL;

  this->build_ = false;

  LOG_DEBUG(this, "BlockJacobi::Build()",
            this->build_ <<
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockJacobi<OperatorType, VectorType, ValueType>::Solve(const VectorType &rhs,
                                                                   VectorType *x) {

  LOG_DEBUG(this, "BlockJacobi::Solve()",
            " #*# begin");

  this->local_precond_->Solve(rhs.GetInterior(), &x->GetInterior());

  LOG_DEBUG(this, "BlockJacobi::Solve()",
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockJacobi<OperatorType, VectorType, ValueType>::SolveZeroSol(const VectorType &rhs,
                                                                    VectorType *x) {

  LOG_DEBUG(this, "BlockJacobi::SolveZeroSol()",
            " #*# begin");

  this->local_precond_->SolveZeroSol(rhs.GetInterior(), &x->GetInterior());

  LOG_DEBUG(this, "BlockJacobi::SolveZeroSol()",
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockJacobi<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  LOG_DEBUG(this, "BlockJacobi::MoveToHostLocalData_()",
            this->build_);

  this->local_precond_->MoveToHost();

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockJacobi<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  LOG_DEBUG(this, "BlockJacobi::MoveToAcceleratorLocalData_()",
            this->build_);

  this->local_precond_->MoveToAccelerator();

}


template class BlockJacobi< GlobalMatrix<double>, GlobalVector<double>, double >;
template class BlockJacobi< GlobalMatrix<float>,  GlobalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class BlockJacobi< GlobalMatrix<std::complex<double> >, GlobalVector<std::complex<double> >, std::complex<double> >;
template class BlockJacobi< GlobalMatrix<std::complex<float> >,  GlobalVector<std::complex<float> >, std::complex<float> >;
#endif

};
