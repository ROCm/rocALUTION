#include "def.hpp"
#include "math_functions.hpp"

#include <stdlib.h>
#include <math.h>
#include <limits>

namespace paralution {

float paralution_abs(const float val) {

  return std::fabs(val);

}

double paralution_abs(const double val) {

  return std::fabs(val);

}

float paralution_abs(const std::complex<float> val) {

  return std::abs(val);

}

double paralution_abs(const std::complex<double> val) {

  return std::abs(val);

}

int paralution_abs(const int val) {

  return abs(val);

}

template <typename ValueType>
ValueType paralution_eps(void) {

  return std::numeric_limits<ValueType>::epsilon();

}

template <typename ValueType>
bool operator<(const std::complex<ValueType> &lhs, const std::complex<ValueType> &rhs) {

  if (&lhs == &rhs)
    return false;

  assert(lhs.imag() == rhs.imag() && lhs.imag() == ValueType(0.0));

  return lhs.real() < rhs.real();

}

template <typename ValueType>
bool operator>(const std::complex<ValueType> &lhs, const std::complex<ValueType> &rhs) {

  if (&lhs == &rhs)
    return false;

  assert(lhs.imag() == rhs.imag() && lhs.imag() == ValueType(0.0));

  return lhs.real() > rhs.real();

}

template <typename ValueType>
bool operator<=(const std::complex<ValueType> &lhs, const std::complex<ValueType> &rhs) {

  if (&lhs == &rhs)
    return true;

  assert(lhs.imag() == rhs.imag() && lhs.imag() == ValueType(0.0));

  return lhs.real() <= rhs.real();

}

template <typename ValueType>
bool operator>=(const std::complex<ValueType> &lhs, const std::complex<ValueType> &rhs) {

  if (&lhs == &rhs)
    return true;

  assert(lhs.imag() == rhs.imag() && lhs.imag() == ValueType(0.0));

  return lhs.real() >= rhs.real();

}

template double paralution_eps(void);
template float  paralution_eps(void);
template std::complex<double> paralution_eps(void);
template std::complex<float>  paralution_eps(void);

template bool operator<(const std::complex<float> &lhs, const std::complex<float> &rhs);
template bool operator<(const std::complex<double> &lhs, const std::complex<double> &rhs);

template bool operator>(const std::complex<float> &lhs, const std::complex<float> &rhs);
template bool operator>(const std::complex<double> &lhs, const std::complex<double> &rhs);

template bool operator<=(const std::complex<float> &lhs, const std::complex<float> &rhs);
template bool operator<=(const std::complex<double> &lhs, const std::complex<double> &rhs);

template bool operator>=(const std::complex<float> &lhs, const std::complex<float> &rhs);
template bool operator>=(const std::complex<double> &lhs, const std::complex<double> &rhs);

}
