#include "../../utils/def.hpp"
#include "inversion.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"

#include "../../utils/log.hpp"

#include <math.h>
#include <complex>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
Inversion<OperatorType, VectorType, ValueType>::Inversion() {

  LOG_DEBUG(this, "Inversion::Inversion()",
            "default constructor");

}

template <class OperatorType, class VectorType, typename ValueType>
Inversion<OperatorType, VectorType, ValueType>::~Inversion() {

  LOG_DEBUG(this, "Inversion::~Inversion()",
            "destructor");

  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void Inversion<OperatorType, VectorType, ValueType>::Print(void) const {

  LOG_INFO("Inversion solver");

}


template <class OperatorType, class VectorType, typename ValueType>
void Inversion<OperatorType, VectorType, ValueType>::PrintStart_(void) const {

  LOG_INFO("Inversion direct solver starts");

}

template <class OperatorType, class VectorType, typename ValueType>
void Inversion<OperatorType, VectorType, ValueType>::PrintEnd_(void) const {

  LOG_INFO("Inversion ends");

}

template <class OperatorType, class VectorType, typename ValueType>
void Inversion<OperatorType, VectorType, ValueType>::Build(void) {

  LOG_DEBUG(this, "Inversion::Build()",
            this->build_ <<
            " #*# begin");

  if (this->build_ == true)
    this->Clear();

  assert(this->build_ == false);
  this->build_ = true;

  assert(this->op_ != NULL);
  assert(this->op_->get_nrow() == this->op_->get_ncol());
  assert(this->op_->get_nrow() > 0);

  this->inverse_.CloneFrom(*this->op_);
  this->inverse_.Invert();

  LOG_DEBUG(this, "Inversion::Build()",
            this->build_ <<
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void Inversion<OperatorType, VectorType, ValueType>::Clear(void) {

  LOG_DEBUG(this, "Inversion::Clear()",
            this->build_);

  if (this->build_ == true) {

    this->inverse_.Clear();
    this->build_ = false;

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void Inversion<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  LOG_DEBUG(this, "Inversion::MoveToHostLocalData_()",
            this->build_);

  if (this->build_ == true)
    this->inverse_.MoveToHost();

}

template <class OperatorType, class VectorType, typename ValueType>
void Inversion<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  LOG_DEBUG(this, "Inversion::MoveToAcceleratorLocalData_()",
            this->build_);

  if (this->build_ == true)
    this->inverse_.MoveToAccelerator();

}

template <class OperatorType, class VectorType, typename ValueType>
void Inversion<OperatorType, VectorType, ValueType>::Solve_(const VectorType &rhs, VectorType *x) {

  LOG_DEBUG(this, "Inversion::Solve_()",
            " #*# begin");

  assert(x != NULL);
  assert(x != &rhs);
  assert(&this->inverse_ != NULL);
  assert(this->build_ == true);

  this->inverse_.Apply(rhs, x);

  LOG_DEBUG(this, "Inversion::Solve_()",
            " #*# end");

}


template class Inversion< LocalMatrix<double>, LocalVector<double>, double >;
template class Inversion< LocalMatrix<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class Inversion< LocalMatrix<std::complex<double> >,  LocalVector<std::complex<double> >, std::complex<double> >;
template class Inversion< LocalMatrix<std::complex<float> >,  LocalVector<std::complex<float> >, std::complex<float> >;
#endif

}
