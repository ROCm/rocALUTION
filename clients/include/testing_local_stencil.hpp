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

#pragma once
#ifndef TESTING_LOCAL_STENCIL_HPP
#define TESTING_LOCAL_STENCIL_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution.hpp>

using namespace rocalution;

template <typename T>
void testing_local_stencil_bad_args(void)
{
    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    LocalStencil<T> stn(Laplace2D);
    LocalVector<T>  vec;

    // Apply
    {
        LocalVector<T>* null_vec = nullptr;
        ASSERT_DEATH(stn.Apply(vec, null_vec), ".*Assertion.*out != (NULL|__null)*");
    }

    // ApplyAdd
    {
        LocalVector<T>* null_vec = nullptr;
        ASSERT_DEATH(stn.ApplyAdd(vec, 1.0, null_vec), ".*Assertion.*out != (NULL|__null)*");
    }

    // Stop rocALUTION
    stop_rocalution();
}

#endif // TESTING_LOCAL_STENCIL_HPP
