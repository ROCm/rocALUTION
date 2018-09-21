/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_uaamg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, int, int, unsigned int, int, int> uaamg_tuple;

int uaamg_size[] = {63, 134};
std::string uaamg_smoother[] = {"FSAI", "ILU"};
int uaamg_pre_iter[] = {1, 2};
int uaamg_post_iter[] = {1, 2};
int uaamg_cycle[] = {0, 2};
int uaamg_scaling[] = {0, 1};

unsigned int uaamg_format[] = {1, 6};

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
    arg.size        = std::get<0>(tup);
    arg.smoother    = std::get<1>(tup);
    arg.format      = std::get<2>(tup);
    arg.pre_smooth  = std::get<3>(tup);
    arg.post_smooth = std::get<4>(tup);
    arg.cycle       = std::get<5>(tup);
    arg.ordering    = std::get<6>(tup);
    return arg;
}

TEST_P(parameterized_uaamg, uaamg_float)
{
    Arguments arg = setup_uaamg_arguments(GetParam());
    ASSERT_EQ(testing_uaamg<float>(arg), true);
}

TEST_P(parameterized_uaamg, uaamg_double)
{
    Arguments arg = setup_uaamg_arguments(GetParam());
    ASSERT_EQ(testing_uaamg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(uaamg,
                        parameterized_uaamg,
                        testing::Combine(testing::ValuesIn(uaamg_size),
                                         testing::ValuesIn(uaamg_smoother),
                                         testing::ValuesIn(uaamg_pre_iter),
                                         testing::ValuesIn(uaamg_post_iter),
                                         testing::ValuesIn(uaamg_format),
                                         testing::ValuesIn(uaamg_cycle),
                                         testing::ValuesIn(uaamg_scaling)));
