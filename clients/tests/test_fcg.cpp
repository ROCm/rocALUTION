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

#include "testing_fcg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, unsigned int> fcg_tuple;

int         fcg_size[]    = {7, 63};
std::string fcg_precond[] = {"None",
                             "Chebyshev",
                             "FSAI",
                             "SPAI",
                             "TNS",
                             "Jacobi",
                             //                             "SGS",
                             //                             "ILU",
                             "ILUT",
                             //                             "IC",
                             "MCSGS"}; //,
//                             "MCILU"};
unsigned int fcg_format[] = {1, 2, 4, 5, 6, 7};

class parameterized_fcg : public testing::TestWithParam<fcg_tuple>
{
protected:
    parameterized_fcg() {}
    virtual ~parameterized_fcg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_fcg_arguments(fcg_tuple tup)
{
    Arguments arg;
    arg.size    = std::get<0>(tup);
    arg.precond = std::get<1>(tup);
    arg.format  = std::get<2>(tup);
    return arg;
}

TEST_P(parameterized_fcg, fcg_float)
{
    Arguments arg = setup_fcg_arguments(GetParam());
    ASSERT_EQ(testing_fcg<float>(arg), true);
}

TEST_P(parameterized_fcg, fcg_double)
{
    Arguments arg = setup_fcg_arguments(GetParam());
    ASSERT_EQ(testing_fcg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(fcg,
                        parameterized_fcg,
                        testing::Combine(testing::ValuesIn(fcg_size),
                                         testing::ValuesIn(fcg_precond),
                                         testing::ValuesIn(fcg_format)));
