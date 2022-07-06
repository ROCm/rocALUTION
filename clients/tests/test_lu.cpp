/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_lu.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, unsigned int, std::string> lu_tuple;

int          lu_size[]        = {7, 16, 21};
unsigned int lu_format[]      = {1, 2, 3, 4, 5, 6, 7};
std::string  lu_matrix_type[] = {"Laplacian2D"};

class parameterized_lu : public testing::TestWithParam<lu_tuple>
{
protected:
    parameterized_lu() {}
    virtual ~parameterized_lu() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_lu_arguments(lu_tuple tup)
{
    Arguments arg;
    arg.size        = std::get<0>(tup);
    arg.format      = std::get<1>(tup);
    arg.matrix_type = std::get<2>(tup);
    return arg;
}

TEST_P(parameterized_lu, lu_float)
{
    Arguments arg = setup_lu_arguments(GetParam());
    ASSERT_EQ(testing_lu<float>(arg), true);
}

TEST_P(parameterized_lu, lu_double)
{
    Arguments arg = setup_lu_arguments(GetParam());
    ASSERT_EQ(testing_lu<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(lu,
                        parameterized_lu,
                        testing::Combine(testing::ValuesIn(lu_size),
                                         testing::ValuesIn(lu_format),
                                         testing::ValuesIn(lu_matrix_type)));
