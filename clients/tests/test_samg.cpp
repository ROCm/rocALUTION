/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_samg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, int, int, unsigned int, int, int, unsigned int> samg_tuple;

int samg_size[] = {63, 134};
std::string samg_smoother[] = {"FSAI", "SPAI"};
int samg_pre_iter[] = {1, 2};
int samg_post_iter[] = {1, 2};
int samg_cycle[] = {0, 2};
int samg_scaling[] = {0, 1};
unsigned int samg_aggr[] = {0, 1};

unsigned int samg_format[] = {1, 6};

class parameterized_samg : public testing::TestWithParam<samg_tuple>
{
    protected:
    parameterized_samg() {}
    virtual ~parameterized_samg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_samg_arguments(samg_tuple tup)
{
    Arguments arg;
    arg.size        = std::get<0>(tup);
    arg.smoother    = std::get<1>(tup);
    arg.format      = std::get<2>(tup);
    arg.pre_smooth  = std::get<3>(tup);
    arg.post_smooth = std::get<4>(tup);
    arg.cycle       = std::get<5>(tup);
    arg.ordering    = std::get<6>(tup);
    arg.aggr        = std::get<7>(tup);
    return arg;
}

TEST_P(parameterized_samg, samg_float)
{
    Arguments arg = setup_samg_arguments(GetParam());
    ASSERT_EQ(testing_samg<float>(arg), true);
}

TEST_P(parameterized_samg, samg_double)
{
    Arguments arg = setup_samg_arguments(GetParam());
    ASSERT_EQ(testing_samg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(samg,
                        parameterized_samg,
                        testing::Combine(testing::ValuesIn(samg_size),
                                         testing::ValuesIn(samg_smoother),
                                         testing::ValuesIn(samg_pre_iter),
                                         testing::ValuesIn(samg_post_iter),
                                         testing::ValuesIn(samg_format),
                                         testing::ValuesIn(samg_cycle),
                                         testing::ValuesIn(samg_scaling),
                                         testing::ValuesIn(samg_aggr)));
