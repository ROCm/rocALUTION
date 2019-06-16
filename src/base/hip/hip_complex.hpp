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

namespace rocalution
{

    __device__ static __inline__ double hip_abs(double val)
    {
        return abs(val);
    }
    __device__ static __inline__ float hip_abs(float val)
    {
        return abs(val);
    }

    __device__ static __inline__ void make_ValueType(float& val, const float& scalar)
    {
        val = scalar;
    }
    __device__ static __inline__ void make_ValueType(double& val, const double& scalar)
    {
        val = scalar;
    }
    __device__ static __inline__ void make_ValueType(int& val, const int& scalar)
    {
        val = scalar;
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_COMPLEX_HPP_
