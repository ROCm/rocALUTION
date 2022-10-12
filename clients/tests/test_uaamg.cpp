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

#include "testing_uaamg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::
    tuple<int, int, int, std::string, std::string, std::string, unsigned int, int, int, int>
        uaamg_tuple;

int          uaamg_size[]             = {22, 63, 134, 207};
int          uaamg_pre_iter[]         = {2};
int          uaamg_post_iter[]        = {2};
std::string  uaamg_smoother[]         = {"FSAI" /*, "ILU"*/};
std::string  uaamg_coarsening_strat[] = {"Greedy", "PMIS"};
std::string  uaamg_matrix_type[]      = {"Laplacian2D", "Laplacian3D"};
unsigned int uaamg_format[]           = {1, 6};
int          uaamg_cycle[]            = {2};
int          uaamg_scaling[]          = {1};
int          uaamg_rebuildnumeric[]   = {0, 1};

class parameterized_uaamg : public testing::TestWithParam<uaamg_tuple>
{
protected:
    parameterized_uaamg() {}
    virtual ~parameterized_uaamg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_uaamg_arguments(uaamg_tuple tup)
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

std::string get_arch_name()
{
    set_device_rocalution(device);
    init_rocalution();

    std::string arch = rocalution::get_arch_rocalution();

    stop_rocalution();
    return arch;
}

bool is_windows()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) || defined(__WIN64)
    return true;
#else
    return false;
#endif
}

TEST_P(parameterized_uaamg, uaamg_float)
{
    Arguments arg = setup_uaamg_arguments(GetParam());
    ASSERT_EQ(testing_uaamg<float>(arg), true);
}

TEST_P(parameterized_uaamg, uaamg_double)
{
    Arguments arg = setup_uaamg_arguments(GetParam());
    if(get_arch_name() == "gfx1102")
    {
        if((!is_windows() && arg.size == 207 && arg.matrix_type == "Laplacian3D"
            && (arg.format == 6 || (arg.format == 1 && arg.rebuildnumeric == 1)))
           || (is_windows() && arg.size == 207 && arg.matrix_type == "Laplacian3D"
               && arg.format == 6))
        {
            std::cout << "Test has been disabled on your machine's configuration" << std::endl;
            return;
        }
    }
    ASSERT_EQ(testing_uaamg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(uaamg,
                        parameterized_uaamg,
                        testing::Combine(testing::ValuesIn(uaamg_size),
                                         testing::ValuesIn(uaamg_pre_iter),
                                         testing::ValuesIn(uaamg_post_iter),
                                         testing::ValuesIn(uaamg_smoother),
                                         testing::ValuesIn(uaamg_coarsening_strat),
                                         testing::ValuesIn(uaamg_matrix_type),
                                         testing::ValuesIn(uaamg_format),
                                         testing::ValuesIn(uaamg_cycle),
                                         testing::ValuesIn(uaamg_scaling),
                                         testing::ValuesIn(uaamg_rebuildnumeric)));
