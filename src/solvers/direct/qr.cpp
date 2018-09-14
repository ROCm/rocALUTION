#include "../../utils/def.hpp"
#include "qr.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"

#include "../../utils/log.hpp"

#include <math.h>
#include <complex>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
QR<OperatorType, VectorType, ValueType>::QR() {

  log_debug(this, "QR::QR()");

}

template <class OperatorType, class VectorType, typename ValueType>
QR<OperatorType, VectorType, ValueType>::~QR() {

  log_debug(this, "QR::~QR()");

  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void QR<OperatorType, VectorType, ValueType>::Print(void) const {

  LOG_INFO("QR solver");

}


template <class OperatorType, class VectorType, typename ValueType>
void QR<OperatorType, VectorType, ValueType>::PrintStart_(void) const {

  LOG_INFO("QR direct solver starts");

}

template <class OperatorType, class VectorType, typename ValueType>
void QR<OperatorType, VectorType, ValueType>::PrintEnd_(void) const {

  LOG_INFO("QR ends");

}

template <class OperatorType, class VectorType, typename ValueType>
void QR<OperatorType, VectorType, ValueType>::Build(void) {

  log_debug(this, "QR::Build()",
            this->build_,
            " #*# begin");

  if (this->build_ == true)
    this->Clear();

  assert(this->build_ == false);
  this->build_ = true;

  assert(this->op_ != NULL);
  assert(this->op_->GetM() == this->op_->GetN());
  assert(this->op_->GetM() > 0);

  this->qr_.CloneFrom(*this->op_);
  this->qr_.QRDecompose();

  log_debug(this, "QR::Build()",
            this->build_,
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void QR<OperatorType, VectorType, ValueType>::Clear(void) {

  log_debug(this, "QR::Clear()",
            this->build_);

  if (this->build_ == true) {

    this->qr_.Clear();
    this->build_ = false;

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void QR<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  log_debug(this, "QR::MoveToHostLocalData_()",
            this->build_);

  if (this->build_ == true)
    this->qr_.MoveToHost();

}

template <class OperatorType, class VectorType, typename ValueType>
void QR<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  log_debug(this, "QR::MoveToAcceleratorLocalData_()",
            this->build_);

  if (this->build_ == true)
    this->qr_.MoveToAccelerator();

}

template <class OperatorType, class VectorType, typename ValueType>
void QR<OperatorType, VectorType, ValueType>::Solve_(const VectorType &rhs, VectorType *x) {

  log_debug(this, "QR::Solve_()",
            " #*# begin",
            (const void*&)rhs,
            x);

  assert(x != NULL);
  assert(x != &rhs);
  assert(this->build_ == true);

  this->qr_.QRSolve(rhs, x);

  log_debug(this, "QR::Solve_()",
            " #*# end");

}


template class QR< LocalMatrix<double>, LocalVector<double>, double >;
template class QR< LocalMatrix<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class QR< LocalMatrix<std::complex<double> >,  LocalVector<std::complex<double> >, std::complex<double> >;
template class QR< LocalMatrix<std::complex<float> >,  LocalVector<std::complex<float> >, std::complex<float> >;
#endif

}
