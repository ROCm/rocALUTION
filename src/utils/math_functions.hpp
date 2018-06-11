#ifndef ROCALUTION_UTILS_MATH_FUNCTIONS_HPP_
#define ROCALUTION_UTILS_MATH_FUNCTIONS_HPP_

#include <complex>

namespace rocalution {

/// Return absolute float value
float rocalution_abs(const float& val);
/// Return absolute double value
double rocalution_abs(const double& val);
/// Return absolute float value
float rocalution_abs(const std::complex<float>& val);
/// Return absolute double value
double rocalution_abs(const std::complex<double>& val);
/// Return absolute int value
int rocalution_abs(const int& val);

/// Return smallest positive floating point number
template <typename ValueType>
ValueType rocalution_eps(void);

/// Overloaded < operator for complex numbers
template <typename ValueType>
bool operator<(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs);
/// Overloaded > operator for complex numbers
template <typename ValueType>
bool operator>(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs);
/// Overloaded <= operator for complex numbers
template <typename ValueType>
bool operator<=(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs);
/// Overloaded >= operator for complex numbers
template <typename ValueType>
bool operator>=(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs);

} // namespace rocalution

#endif // ROCALUTION_UTILS_MATH_FUNCTIONS_HPP_
