/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

#ifndef ROCALUTION_HIP_COMPLEX_HPP_
#define ROCALUTION_HIP_COMPLEX_HPP_

#include <hip/hip_runtime.h>

#ifdef SUPPORT_COMPLEX
#include <hip/hip_complex.h>
#endif

namespace rocalution {

#ifdef SUPPORT_COMPLEX
__device__ static __inline__ hipDoubleComplex operator+(const hipDoubleComplex& lhs,
                                                        const hipDoubleComplex& rhs)
{
    return hipCadd(lhs, rhs);
}

__device__ static __inline__ hipFloatComplex operator+(const hipFloatComplex& lhs,
                                                       const hipFloatComplex& rhs)
{
    return hipCaddf(lhs, rhs);
}

__device__ static __inline__ hipDoubleComplex operator*(const hipDoubleComplex& lhs,
                                                        const hipDoubleComplex& rhs)
{
    return hipCmul(lhs, rhs);
}

__device__ static __inline__ hipFloatComplex operator*(const hipFloatComplex& lhs,
                                                       const hipFloatComplex& rhs)
{
    return hipCmulf(lhs, rhs);
}

__device__ static __inline__ hipDoubleComplex operator/(const hipDoubleComplex& lhs,
                                                        const hipDoubleComplex& rhs)
{
    return hipCdiv(lhs, rhs);
}

__device__ static __inline__ hipFloatComplex operator/(const hipFloatComplex& lhs,
                                                       const hipFloatComplex& rhs)
{
    return hipCdivf(lhs, rhs);
}
#endif

__device__ static __inline__ double hip_abs(double val) { return abs(val); }
__device__ static __inline__ float hip_abs(float val) { return abs(val); }
#ifdef SUPPORT_COMPLEX
__device__ static __inline__ double hip_abs(const hipDoubleComplex& val) { return hipCabs(val); }
__device__ static __inline__ float hip_abs(const hipFloatComplex& val) { return hipCabsf(val); }
#endif

__device__ static __inline__ double hip_pow(double val, double exp) { return pow(val, exp); }

__device__ static __inline__ float hip_pow(float val, double exp) { return powf(val, exp); }

__device__ static __inline__ void make_ValueType(float& val, const float& scalar) { val = scalar; }

__device__ static __inline__ void make_ValueType(double& val, const double& scalar)
{
    val = scalar;
}

#ifdef SUPPORT_COMPLEX
__device__ static __inline__ void make_ValueType(hipFloatComplex& val, const float& scalar)
{
    val = make_hipFloatComplex(scalar, 0.0f);
}

__device__ static __inline__ void make_ValueType(hipDoubleComplex& val, const double& scalar)
{
    val = make_hipDoubleComplex(scalar, 0.0);
}
#endif

__device__ static __inline__ void make_ValueType(int& val, const int& scalar) { val = scalar; }

__device__ static __inline__ void atomic_add_hip(float* address, float val)
{
    atomicAdd(address, val);
}

__device__ static __inline__ void atomic_add_hip(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old             = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old     = atomicCAS(
            address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while(assumed != old);
}

#ifdef SUPPORT_COMPLEX
__device__ static __inline__ void atomic_add_hip(hipFloatComplex* address, hipFloatComplex val)
{
    atomic_add_hip((float*)address, val.x);
    atomic_add_hip((float*)address + 1, val.y);
}

__device__ static __inline__ void atomic_add_hip(hipDoubleComplex* address, hipDoubleComplex val)
{
    atomic_add_hip((double*)address, val.x);
    atomic_add_hip((double*)address + 1, val.y);
}
#endif

} // namespace rocalution

#endif // ROCALUTION_HIP_COMPLEX_HPP_
