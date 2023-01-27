/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_UTILS_MATH_FUNCTIONS_HPP_
#define ROCALUTION_UTILS_MATH_FUNCTIONS_HPP_

#include <complex>

namespace rocalution
{
    /// Return double value
    double rocalution_double(const int& val);
    /// Return double value
    double rocalution_double(const int64_t& val);
    /// Return double value
    double rocalution_double(const float& val);
    /// Return double value
    double rocalution_double(const double& val);
    /// Return double value
    double rocalution_double(const std::complex<float>& val);
    /// Return double value
    double rocalution_double(const std::complex<double>& val);

    /// Return conjugate complex
    inline float rocalution_conj(const float& val)
    {
        return val;
    }
    /// Return conjugate complex
    inline double rocalution_conj(const double& val)
    {
        return val;
    }
    /// Return conjugate complex
    inline std::complex<float> rocalution_conj(const std::complex<float>& val)
    {
        return std::conj(val);
    }
    /// Return conjugate complex
    inline std::complex<double> rocalution_conj(const std::complex<double>& val)
    {
        return std::conj(val);
    }

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
