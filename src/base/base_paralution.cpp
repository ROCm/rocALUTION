#include "../utils/def.hpp"
#include "base_paralution.hpp"
#include "parallel_manager.hpp"
#include "../utils/log.hpp"

#include <assert.h>
namespace paralution {

/// Global obj tracking structure
Paralution_Object_Data Paralution_Object_Data_Tracking;

ParalutionObj::ParalutionObj() {

  LOG_DEBUG(this, "ParalutionObj::ParalutionObj()",
              "default constructor");

#ifndef OBJ_TRACKING_OFF

  this->global_obj_id = _paralution_add_obj(this); 

#else 

  this->global_obj_id = 0;

#endif

}

ParalutionObj::~ParalutionObj() { 

  LOG_DEBUG(this, "ParalutionObj::ParalutionObj()",
              "default destructor");

#ifndef OBJ_TRACKING_OFF

  bool status = false ;
  status = _paralution_del_obj(this, this->global_obj_id);

  if (status != true) {
    LOG_INFO("Error: PARALUTION tracking problem");
    FATAL_ERROR(__FILE__, __LINE__);
  }

#else 

  // nothing

#endif

};

template <typename ValueType>
BaseParalution<ValueType>::BaseParalution() {

  LOG_DEBUG(this, "BaseParalution::BaseParalution()",
              "default constructor");

  // copy the backend description
  this->local_backend_ = *_get_backend_descriptor();

  this->asyncf = false;
  
  assert(_get_backend_descriptor()->init == true);

}

template <typename ValueType>
BaseParalution<ValueType>::BaseParalution(const BaseParalution<ValueType> &src) {

  LOG_DEBUG(this, "BaseParalution::BaseParalution()",
            "copy constructor");

  LOG_INFO("no copy constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
BaseParalution<ValueType>::~BaseParalution() {

  LOG_DEBUG(this, "BaseParalution::~BaseParalution()",
            "default destructor");

}

template<typename ValueType>
BaseParalution<ValueType>& BaseParalution<ValueType>::operator=(const BaseParalution<ValueType> &src) {

  LOG_DEBUG(this, "BaseParalution::operator=()",
            "");

  LOG_INFO("no overloaded operator=()");
  FATAL_ERROR(__FILE__, __LINE__);

}

template<typename ValueType>
void BaseParalution<ValueType>::CloneBackend(const BaseParalution<ValueType> &src) {

  LOG_DEBUG(this, "BaseParalution::CloneBackend()",
            "with the same ValueType");


  assert(this != &src);

  this->local_backend_ = src.local_backend_; 
  this->pm_ = src.pm_;

  if (src.is_host()) {

    // move to host
    this->MoveToHost();
    
  } else {

    assert(src.is_accel());

    // move to accelerator
    this->MoveToAccelerator();

  }

}


template <typename ValueType>
template <typename ValueType2>
void BaseParalution<ValueType>::CloneBackend(const BaseParalution<ValueType2> &src) {

  LOG_DEBUG(this, "BaseParalution::CloneBackend()",
            "with different ValueType");


  this->local_backend_ = src.local_backend_; 
  this->pm_ = src.pm_;

  if (src.is_host()) {

    // move to host
    this->MoveToHost();
    
  } else {

    assert(src.is_accel());

    // move to accelerator
    this->MoveToAccelerator();

  }

}

template<typename ValueType>
void BaseParalution<ValueType>::MoveToAcceleratorAsync(void) {

  // default call
  this->MoveToAccelerator();

}

template<typename ValueType>
void BaseParalution<ValueType>::MoveToHostAsync(void) {

  // default call
  this->MoveToHost();

}

template<typename ValueType>
void BaseParalution<ValueType>::Sync(void) {

  _paralution_sync();
  this->asyncf = false;

}


template class BaseParalution<double>;
template class BaseParalution<float>;
#ifdef SUPPORT_COMPLEX
template class BaseParalution<std::complex<double> >;
template class BaseParalution<std::complex<float> >;
#endif
template class BaseParalution<int>;

template void BaseParalution<int>::CloneBackend(const BaseParalution<double> &src);
template void BaseParalution<int>::CloneBackend(const BaseParalution<float> &src);

template void BaseParalution<float>::CloneBackend(const BaseParalution<double> &src);
template void BaseParalution<double>::CloneBackend(const BaseParalution<float> &src);

#ifdef SUPPORT_COMPLEX
template void BaseParalution<int>::CloneBackend(const BaseParalution<std::complex<double> > &src);
template void BaseParalution<int>::CloneBackend(const BaseParalution<std::complex<float> > &src);

template void BaseParalution<std::complex<float> >::CloneBackend(const BaseParalution<std::complex<double> > &src);
template void BaseParalution<std::complex<double> >::CloneBackend(const BaseParalution<std::complex<float> > &src);

#endif

}
