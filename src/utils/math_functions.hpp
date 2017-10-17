#ifndef PARALUTION_UTILS_MATH_FUNCTIONS_HPP_
#define PARALUTION_UTILS_MATH_FUNCTIONS_HPP_

#include <complex>

namespace paralution {

/// Return absolute float value
float paralution_abs(const float val);
/// Return absolute double value
double paralution_abs(const double val);
/// Return absolute float value
float paralution_abs(const std::complex<float> val);
/// Return absolute double value
double paralution_abs(const std::complex<double> val);
/// Return absolute int value
int paralution_abs(const int val);

/// Return smallest positive floating point number
template <typename ValueType>
ValueType paralution_eps(void);

/// Overloaded < operator for complex numbers
template <typename ValueType>
bool operator<(const std::complex<ValueType> &lhs, const std::complex<ValueType> &rhs);
/// Overloaded > operator for complex numbers
template <typename ValueType>
bool operator>(const std::complex<ValueType> &lhs, const std::complex<ValueType> &rhs);
/// Overloaded <= operator for complex numbers
template <typename ValueType>
bool operator<=(const std::complex<ValueType> &lhs, const std::complex<ValueType> &rhs);
/// Overloaded >= operator for complex numbers
template <typename ValueType>
bool operator>=(const std::complex<ValueType> &lhs, const std::complex<ValueType> &rhs);

}

#endif // PARALUTION_UTILS_MATH_FUNCTIONS_HPP_
