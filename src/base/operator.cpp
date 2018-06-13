#include "../utils/def.hpp"
#include "operator.hpp"
#include "vector.hpp"
#include "global_vector.hpp"
#include "local_vector.hpp"
#include "../utils/log.hpp"

#include <complex>

namespace rocalution {

template <typename ValueType>
Operator<ValueType>::Operator() {

  LOG_DEBUG(this, "Operator::Operator()",
            "default constructor");

  this->object_name_ = "";

}

template <typename ValueType>
Operator<ValueType>::~Operator() {

  LOG_DEBUG(this, "Operator::~Operator()",
            "default destructor");

}

template <typename ValueType>
int Operator<ValueType>::get_local_nrow(void) const {
 
  return IndexTypeToInt(this->get_nrow());

}

template <typename ValueType>
int Operator<ValueType>::get_local_ncol(void) const {
 
  return IndexTypeToInt(this->get_ncol());

}

template <typename ValueType>
int Operator<ValueType>::get_local_nnz(void) const {
 
  return IndexTypeToInt(this->get_nnz());

}

template <typename ValueType>
int Operator<ValueType>::get_ghost_nrow(void) const {
 
  return 0;
}

template <typename ValueType>
int Operator<ValueType>::get_ghost_ncol(void) const {
 
  return 0;

}

template <typename ValueType>
int Operator<ValueType>::get_ghost_nnz(void) const {
 
  return 0;

}

template <typename ValueType>
void Operator<ValueType>::Apply(const GlobalVector<ValueType> &in, GlobalVector<ValueType> *out) const {

  LOG_INFO("Operator<ValueType>::Apply(const GlobalVector<ValueType> &in, GlobalVector<ValueType> *out)");
  LOG_INFO("Mismatched types:");
  this->Info();
  in.Info();
  out->Info();
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void Operator<ValueType>::Apply(const LocalVector<ValueType> &in, LocalVector<ValueType> *out) const {

  LOG_INFO("Operator<ValueType>::Apply(const LocalVector<ValueType> &in, LocalVector<ValueType> *out)");
  LOG_INFO("Mismatched types:");
  this->Info();
  in.Info();
  out->Info();
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void Operator<ValueType>::ApplyAdd(const GlobalVector<ValueType> &in, const ValueType scalar, GlobalVector<ValueType> *out) const {

  LOG_INFO("Operator<ValueType>::ApplyAdd(const GlobalVector<ValueType> &in, const ValueType scalar, GlobalVector<ValueType> *out)");
  LOG_INFO("Mismatched types:");
  this->Info();
  in.Info();
  out->Info();
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void Operator<ValueType>::ApplyAdd(const LocalVector<ValueType> &in, const ValueType scalar, LocalVector<ValueType> *out) const {

  LOG_INFO("Operator<ValueType>::ApplyAdd(const LocalVector<ValueType> &in, const ValueType scalar, LocalVector<ValueType> *out)");
  LOG_INFO("Mismatched types:");
  this->Info();
  in.Info();
  out->Info();
  FATAL_ERROR(__FILE__, __LINE__); 

}


template class Operator<double>;
template class Operator<float>;
#ifdef SUPPORT_COMPLEX
template class Operator<std::complex<double> >;
template class Operator<std::complex<float> >;
#endif
template class Operator<int>;

}
