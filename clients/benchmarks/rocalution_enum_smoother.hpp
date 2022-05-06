/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include "rocalution/rocalution.hpp"
#include "utility.hpp"

#include <cstring>

struct rocalution_enum_smoother
{

#define LIST_ROCALUTION_ENUM_SMOOTHER \
    ENUM_SMOOTHER(FSAI)               \
    ENUM_SMOOTHER(ILU)

    //
    //
    //
#define ENUM_SMOOTHER(x_) x_,

    typedef enum rocalution_enum_smoother__ : int
    {
        LIST_ROCALUTION_ENUM_SMOOTHER
    } value_type;

    static constexpr value_type  all[]{LIST_ROCALUTION_ENUM_SMOOTHER};
    static constexpr std::size_t size = countof(all);

#undef ENUM_SMOOTHER

    //
    //
    //
#define ENUM_SMOOTHER(x_) #x_,

    static constexpr const char* names[size]{LIST_ROCALUTION_ENUM_SMOOTHER};

#undef ENUM_SMOOTHER

    bool is_invalid() const;
    rocalution_enum_smoother();
    rocalution_enum_smoother& operator()(const char* name_);
    rocalution_enum_smoother(const char* name_);

    value_type value{};
};
