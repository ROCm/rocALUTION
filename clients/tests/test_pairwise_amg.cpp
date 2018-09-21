/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_pairwise_amg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, int, int, unsigned int, int> pwamg_tuple;

int pwamg_size[] = {63, 134};
std::string pwamg_smoother[] = {"Jacobi", "MCILU"};
int pwamg_pre_iter[] = {1, 2};
int pwamg_post_iter[] = {1, 2};
int pwamg_ordering[] = {0, 1, 2, 3, 4, 5};

unsigned int pwamg_format[] = {1, 7};

class parameterized_pairwise_amg : public testing::TestWithParam<pwamg_tuple>
{
    protected:
    parameterized_pairwise_amg() {}
    virtual ~parameterized_pairwise_amg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_pwamg_arguments(pwamg_tuple tup)
{
    Arguments arg;
    arg.size        = std::get<0>(tup);
    arg.smoother    = std::get<1>(tup);
    arg.format      = std::get<2>(tup);
    arg.pre_smooth  = std::get<3>(tup);
    arg.post_smooth = std::get<4>(tup);
    arg.ordering    = std::get<5>(tup);
    return arg;
}

TEST_P(parameterized_pairwise_amg, pairwise_amg_float)
{
    Arguments arg = setup_pwamg_arguments(GetParam());
    ASSERT_EQ(testing_pairwise_amg<float>(arg), true);
}

TEST_P(parameterized_pairwise_amg, pairwise_amg_double)
{
    Arguments arg = setup_pwamg_arguments(GetParam());
    ASSERT_EQ(testing_pairwise_amg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(pairwise_amg,
                        parameterized_pairwise_amg,
                        testing::Combine(testing::ValuesIn(pwamg_size),
                                         testing::ValuesIn(pwamg_smoother),
                                         testing::ValuesIn(pwamg_pre_iter),
                                         testing::ValuesIn(pwamg_post_iter),
                                         testing::ValuesIn(pwamg_format),
                                         testing::ValuesIn(pwamg_ordering)));
