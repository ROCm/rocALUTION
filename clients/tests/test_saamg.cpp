/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_saamg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::
    tuple<int, int, int, std::string, std::string, std::string, unsigned int, int, int, int>
        saamg_tuple;

int          saamg_size[]             = {22, 63, 134, 207};
int          saamg_pre_iter[]         = {2};
int          saamg_post_iter[]        = {2};
std::string  saamg_smoother[]         = {"FSAI", "SPAI"};
std::string  saamg_coarsening_strat[] = {"Greedy", "PMIS"};
std::string  saamg_matrix_type[]      = {"Laplacian2D"};
unsigned int saamg_format[]           = {1, 6};
int          saamg_cycle[]            = {2};
int          saamg_scaling[]          = {1};
int          saamg_rebuildnumeric[]   = {0, 1};

class parameterized_saamg : public testing::TestWithParam<saamg_tuple>
{
protected:
    parameterized_saamg() {}
    virtual ~parameterized_saamg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_saamg_arguments(saamg_tuple tup)
{
    Arguments arg;
    arg.size                = std::get<0>(tup);
    arg.pre_smooth          = std::get<1>(tup);
    arg.post_smooth         = std::get<2>(tup);
    arg.smoother            = std::get<3>(tup);
    arg.coarsening_strategy = std::get<4>(tup);
    arg.matrix_type         = std::get<5>(tup);
    arg.format              = std::get<6>(tup);
    arg.cycle               = std::get<7>(tup);
    arg.ordering            = std::get<8>(tup);
    arg.rebuildnumeric      = std::get<9>(tup);

    return arg;
}

TEST_P(parameterized_saamg, saamg_float)
{
    Arguments arg = setup_saamg_arguments(GetParam());
    ASSERT_EQ(testing_saamg<float>(arg), true);
}

TEST_P(parameterized_saamg, saamg_double)
{
    Arguments arg = setup_saamg_arguments(GetParam());
    ASSERT_EQ(testing_saamg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(saamg,
                        parameterized_saamg,
                        testing::Combine(testing::ValuesIn(saamg_size),
                                         testing::ValuesIn(saamg_pre_iter),
                                         testing::ValuesIn(saamg_post_iter),
                                         testing::ValuesIn(saamg_smoother),
                                         testing::ValuesIn(saamg_coarsening_strat),
                                         testing::ValuesIn(saamg_matrix_type),
                                         testing::ValuesIn(saamg_format),
                                         testing::ValuesIn(saamg_cycle),
                                         testing::ValuesIn(saamg_scaling),
                                         testing::ValuesIn(saamg_rebuildnumeric)));
