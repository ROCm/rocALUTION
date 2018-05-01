#include "../utils/def.hpp"
#include "base_rocalution.hpp"
#include "parallel_manager.hpp"
#include "../utils/log.hpp"

#include <assert.h>
namespace rocalution {

/// Global obj tracking structure
Rocalution_Object_Data Rocalution_Object_Data_Tracking;

RocalutionObj::RocalutionObj() {

  LOG_DEBUG(this, "RocalutionObj::RocalutionObj()",
              "default constructor");

#ifndef OBJ_TRACKING_OFF

  this->global_obj_id = _rocalution_add_obj(this); 

#else 

  this->global_obj_id = 0;

#endif

}

RocalutionObj::~RocalutionObj() { 

  LOG_DEBUG(this, "RocalutionObj::RocalutionObj()",
              "default destructor");

#ifndef OBJ_TRACKING_OFF

  bool status = false ;
  status = _rocalution_del_obj(this, this->global_obj_id);

  if (status != true) {
    LOG_INFO("Error: rocALUTION tracking problem");
    FATAL_ERROR(__FILE__, __LINE__);
  }

#else 

  // nothing

#endif

};

template <typename ValueType>
BaseRocalution<ValueType>::BaseRocalution() {

  LOG_DEBUG(this, "BaseRocalution::BaseRocalution()",
              "default constructor");

  // copy the backend description
  this->local_backend_ = *_get_backend_descriptor();

  this->asyncf = false;
  
  assert(_get_backend_descriptor()->init == true);

}

template <typename ValueType>
BaseRocalution<ValueType>::BaseRocalution(const BaseRocalution<ValueType> &src) {

  LOG_DEBUG(this, "BaseRocalution::BaseRocalution()",
            "copy constructor");

  LOG_INFO("no copy constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
BaseRocalution<ValueType>::~BaseRocalution() {

  LOG_DEBUG(this, "BaseRocalution::~BaseRocalution()",
            "default destructor");

}

template<typename ValueType>
BaseRocalution<ValueType>& BaseRocalution<ValueType>::operator=(const BaseRocalution<ValueType> &src) {

  LOG_DEBUG(this, "BaseRocalution::operator=()",
            "");

  LOG_INFO("no overloaded operator=()");
  FATAL_ERROR(__FILE__, __LINE__);

}

template<typename ValueType>
void BaseRocalution<ValueType>::CloneBackend(const BaseRocalution<ValueType> &src) {

  LOG_DEBUG(this, "BaseRocalution::CloneBackend()",
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
void BaseRocalution<ValueType>::CloneBackend(const BaseRocalution<ValueType2> &src) {

  LOG_DEBUG(this, "BaseRocalution::CloneBackend()",
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
void BaseRocalution<ValueType>::MoveToAcceleratorAsync(void) {

  // default call
  this->MoveToAccelerator();

}

template<typename ValueType>
void BaseRocalution<ValueType>::MoveToHostAsync(void) {

  // default call
  this->MoveToHost();

}

template<typename ValueType>
void BaseRocalution<ValueType>::Sync(void) {

  _rocalution_sync();
  this->asyncf = false;

}


template class BaseRocalution<double>;
template class BaseRocalution<float>;
#ifdef SUPPORT_COMPLEX
template class BaseRocalution<std::complex<double> >;
template class BaseRocalution<std::complex<float> >;
#endif
template class BaseRocalution<int>;

template void BaseRocalution<int>::CloneBackend(const BaseRocalution<double> &src);
template void BaseRocalution<int>::CloneBackend(const BaseRocalution<float> &src);

template void BaseRocalution<float>::CloneBackend(const BaseRocalution<double> &src);
template void BaseRocalution<double>::CloneBackend(const BaseRocalution<float> &src);

#ifdef SUPPORT_COMPLEX
template void BaseRocalution<int>::CloneBackend(const BaseRocalution<std::complex<double> > &src);
template void BaseRocalution<int>::CloneBackend(const BaseRocalution<std::complex<float> > &src);

template void BaseRocalution<std::complex<float> >::CloneBackend(const BaseRocalution<std::complex<double> > &src);
template void BaseRocalution<std::complex<double> >::CloneBackend(const BaseRocalution<std::complex<float> > &src);

#endif

}
