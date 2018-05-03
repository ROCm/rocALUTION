#include "../utils/def.hpp"
#include "base_stencil.hpp"
#include "base_vector.hpp"
#include "backend_manager.hpp"
#include "../utils/log.hpp"

#include <stdlib.h>
#include <complex>

namespace rocalution {

template <typename ValueType>
BaseStencil<ValueType>::BaseStencil() {

  LOG_DEBUG(this, "BaseStencil::BaseStencil()",
            "default constructor");

  this->ndim_ = 0;
  this->size_ = 0;

}

template <typename ValueType>
BaseStencil<ValueType>::~BaseStencil() {

  LOG_DEBUG(this, "BaseStencil::~BaseStencil()",
            "default destructor");

}

template <typename ValueType>
int BaseStencil<ValueType>::get_nrow(void) const {

  int dim = 1;

  if (this->get_ndim() > 0) {

    for (int i=0; i<ndim_; ++i)
      dim *= this->size_;

  }

  return dim;

} 

template <typename ValueType>
int BaseStencil<ValueType>::get_ncol(void) const {

  return this->get_nrow(); 
} 

template <typename ValueType>
int BaseStencil<ValueType>::get_ndim(void) const {

  return this->ndim_; 
} 

template <typename ValueType>
void BaseStencil<ValueType>::set_backend(const Rocalution_Backend_Descriptor local_backend) {

  this->local_backend_ = local_backend;

}

template <typename ValueType>
void BaseStencil<ValueType>::SetGrid(const int size) {

  assert(size >= 0);
  this->size_ = size;

}





template <typename ValueType>
HostStencil<ValueType>::HostStencil() {
}

template <typename ValueType>
HostStencil<ValueType>::~HostStencil() {
}



template <typename ValueType>
AcceleratorStencil<ValueType>::AcceleratorStencil() {
}

template <typename ValueType>
AcceleratorStencil<ValueType>::~AcceleratorStencil() {
}


template <typename ValueType>
HIPAcceleratorStencil<ValueType>::HIPAcceleratorStencil() {
}

template <typename ValueType>
HIPAcceleratorStencil<ValueType>::~HIPAcceleratorStencil() {
}


template class BaseStencil<double>;
template class BaseStencil<float>;
#ifdef SUPPORT_COMPLEX
template class BaseStencil<std::complex<double> >;
template class BaseStencil<std::complex<float> >;
#endif
template class BaseStencil<int>;

template class HostStencil<double>;
template class HostStencil<float>;
#ifdef SUPPORT_COMPLEX
template class HostStencil<std::complex<double> >;
template class HostStencil<std::complex<float> >;
#endif
template class HostStencil<int>;

template class AcceleratorStencil<double>;
template class AcceleratorStencil<float>;
#ifdef SUPPORT_COMPLEX
template class AcceleratorStencil<std::complex<double> >;
template class AcceleratorStencil<std::complex<float> >;
#endif
template class AcceleratorStencil<int>;

template class HIPAcceleratorStencil<double>;
template class HIPAcceleratorStencil<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorStencil<std::complex<double> >;
template class HIPAcceleratorStencil<std::complex<float> >;
#endif
template class HIPAcceleratorStencil<int>;

}