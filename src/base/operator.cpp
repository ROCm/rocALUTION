/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

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

  log_debug(this, "Operator::Operator()");

  this->object_name_ = "";

}

template <typename ValueType>
Operator<ValueType>::~Operator() {

  log_debug(this, "Operator::~Operator()");
}

template <typename ValueType>
int Operator<ValueType>::GetLocalM(void) const {
 
  return IndexTypeToInt(this->GetM());

}

template <typename ValueType>
int Operator<ValueType>::GetLocalN(void) const {
 
  return IndexTypeToInt(this->GetN());

}

template <typename ValueType>
int Operator<ValueType>::GetLocalNnz(void) const {
 
  return IndexTypeToInt(this->GetNnz());

}

template <typename ValueType>
int Operator<ValueType>::GetGhostM(void) const {
 
  return 0;
}

template <typename ValueType>
int Operator<ValueType>::GetGhostN(void) const {
 
  return 0;

}

template <typename ValueType>
int Operator<ValueType>::GetGhostNnz(void) const {
 
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

} // namespace rocalution
