#include "def.hpp"
#include "math_functions.hpp"

#include <stdlib.h>
#include <math.h>
#include <limits>

namespace rocalution {

float rocalution_abs(const float& val) { return std::fabs(val); }

double rocalution_abs(const double& val) { return std::fabs(val); }

float rocalution_abs(const std::complex<float>& val) { return std::abs(val); }

double rocalution_abs(const std::complex<double>& val) { return std::abs(val); }

int rocalution_abs(const int& val) { return abs(val); }

template <typename ValueType>
ValueType rocalution_eps(void)
{
    return std::numeric_limits<ValueType>::epsilon();
}

template <typename ValueType>
bool operator<(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs)
{
    if(&lhs == &rhs)
    {
        return false;
    }

    assert(lhs.imag() == rhs.imag() && lhs.imag() == static_cast<ValueType>(0));

    return lhs.real() < rhs.real();
}

template <typename ValueType>
bool operator>(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs)
{
    if(&lhs == &rhs)
    {
        return false;
    }

    assert(lhs.imag() == rhs.imag() && lhs.imag() == static_cast<ValueType>(0));

    return lhs.real() > rhs.real();
}

template <typename ValueType>
bool operator<=(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs)
{
    if(&lhs == &rhs)
    {
        return true;
    }

    assert(lhs.imag() == rhs.imag() && lhs.imag() == static_cast<ValueType>(0));

    return lhs.real() <= rhs.real();
}

template <typename ValueType>
bool operator>=(const std::complex<ValueType>& lhs, const std::complex<ValueType>& rhs)
{
    if(&lhs == &rhs)
    {
        return true;
    }

    assert(lhs.imag() == rhs.imag() && lhs.imag() == static_cast<ValueType>(0));

    return lhs.real() >= rhs.real();
}

template double rocalution_eps(void);
template float rocalution_eps(void);
template std::complex<double> rocalution_eps(void);
template std::complex<float> rocalution_eps(void);

template bool operator<(const std::complex<float>& lhs, const std::complex<float>& rhs);
template bool operator<(const std::complex<double>& lhs, const std::complex<double>& rhs);

template bool operator>(const std::complex<float>& lhs, const std::complex<float>& rhs);
template bool operator>(const std::complex<double>& lhs, const std::complex<double>& rhs);

template bool operator<=(const std::complex<float>& lhs, const std::complex<float>& rhs);
template bool operator<=(const std::complex<double>& lhs, const std::complex<double>& rhs);

template bool operator>=(const std::complex<float>& lhs, const std::complex<float>& rhs);
template bool operator>=(const std::complex<double>& lhs, const std::complex<double>& rhs);

} // namespace rocalution
