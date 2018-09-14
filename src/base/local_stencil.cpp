#include "../utils/def.hpp"
#include "local_stencil.hpp"
#include "local_vector.hpp"
#include "stencil_types.hpp"
#include "host/host_stencil_laplace2d.hpp"
#include "host/host_vector.hpp"

#include "../utils/log.hpp"

#include <complex>

namespace rocalution {

template <typename ValueType>
LocalStencil<ValueType>::LocalStencil() {

  log_debug(this, "LocalStencil::LocalStencil()");

  this->object_name_ = "";

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
LocalStencil<ValueType>::~LocalStencil() {

  log_debug(this, "LocalStencil::~LocalStencil()");

  delete this->stencil_;

}

template <typename ValueType>
LocalStencil<ValueType>::LocalStencil(unsigned int type) {

  log_debug(this, "LocalStencil::LocalStencil()", type);

  assert(type == Laplace2D); // the only one at the moment

  this->object_name_ = _stencil_type_names[type];

  this->stencil_host_ = new HostStencilLaplace2D<ValueType>(this->local_backend_);
  this->stencil_ = this->stencil_host_;

}

template <typename ValueType>
int LocalStencil<ValueType>::GetNDim(void) const {

  return this->stencil_->GetNDim();
}

template <typename ValueType>
IndexType2 LocalStencil<ValueType>::GetNnz(void) const {

  return this->stencil_->GetNnz();
}

template <typename ValueType>
IndexType2 LocalStencil<ValueType>::GetM(void) const {

  return this->stencil_->GetM();
}

template <typename ValueType>
IndexType2 LocalStencil<ValueType>::GetN(void) const {

  return this->stencil_->GetN();
}

template <typename ValueType>
void LocalStencil<ValueType>::Info(void) const {

  this->stencil_->Info();

}

template <typename ValueType>
void LocalStencil<ValueType>::Clear(void) {


  log_debug(this, "LocalStencil::Clear()");

  this->stencil_->SetGrid(0);


}

template <typename ValueType>
void LocalStencil<ValueType>::SetGrid(const int size) {

  log_debug(this, "LocalStencil::SetGrid()",
            size);

  assert (size >= 0);

  this->stencil_->SetGrid(size);

}


template <typename ValueType>
void LocalStencil<ValueType>::Apply(const LocalVector<ValueType> &in, LocalVector<ValueType> *out) const {

  log_debug(this, "LocalStencil::Apply()", (const void*&)in, out);

  assert(out != NULL);

  assert( ( (this->stencil_ == this->stencil_host_)  && (in.vector_ == in.vector_host_) && (out->vector_ == out->vector_host_)) ||
          ( (this->stencil_ == this->stencil_accel_) && (in.vector_ == in.vector_accel_) && (out->vector_ == out->vector_accel_)) );

  this->stencil_->Apply(*in.vector_, out->vector_);
  
}

template <typename ValueType>
void LocalStencil<ValueType>::ApplyAdd(const LocalVector<ValueType> &in, const ValueType scalar, 
                                       LocalVector<ValueType> *out) const {

  log_debug(this, "LocalStencil::ApplyAdd()", (const void*&)in, scalar, out);

  assert(out != NULL);

  assert( ( (this->stencil_ == this->stencil_host_)  && (in.vector_ == in.vector_host_) && (out->vector_ == out->vector_host_)) ||
          ( (this->stencil_ == this->stencil_accel_) && (in.vector_ == in.vector_accel_) && (out->vector_ == out->vector_accel_)) );

  this->stencil_->Apply(*in.vector_, out->vector_);

}

template <typename ValueType>
void LocalStencil<ValueType>::MoveToAccelerator(void) {

  LOG_INFO("The function is not implemented (yet)!");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void LocalStencil<ValueType>::MoveToHost(void) {

  LOG_INFO("The function is not implemented (yet)!");
  FATAL_ERROR(__FILE__, __LINE__);

}


template class LocalStencil<double>;
template class LocalStencil<float>;
#ifdef SUPPORT_COMPLEX
template class LocalStencil<std::complex<double> >;
template class LocalStencil<std::complex<float> >;
#endif

}
