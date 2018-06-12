/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_ruge_stueben_amg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, int, int, unsigned int, int, int> rsamg_tuple;

int rsamg_size[] = {63, 134};
std::string rsamg_smoother[] = {"ILU", "MCGS"};
int rsamg_pre_iter[] = {1, 2};
int rsamg_post_iter[] = {1, 2};
int rsamg_cycle[] = {0, 1};
int rsamg_scaling[] = {0, 1};

unsigned int rsamg_format[] = {1, 7};

class parameterized_ruge_stueben_amg : public testing::TestWithParam<rsamg_tuple>
{
    protected:
    parameterized_ruge_stueben_amg() {}
    virtual ~parameterized_ruge_stueben_amg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_rsamg_arguments(rsamg_tuple tup)
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

TEST_P(parameterized_ruge_stueben_amg, ruge_stueben_amg_float)
{
    Arguments arg = setup_rsamg_arguments(GetParam());
    ASSERT_EQ(testing_ruge_stueben_amg<float>(arg), true);
}

TEST_P(parameterized_ruge_stueben_amg, ruge_stueben_amg_double)
{
    Arguments arg = setup_rsamg_arguments(GetParam());
    ASSERT_EQ(testing_ruge_stueben_amg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(ruge_stueben_amg,
                        parameterized_ruge_stueben_amg,
                        testing::Combine(testing::ValuesIn(rsamg_size),
                                         testing::ValuesIn(rsamg_smoother),
                                         testing::ValuesIn(rsamg_pre_iter),
                                         testing::ValuesIn(rsamg_post_iter),
                                         testing::ValuesIn(rsamg_format),
                                         testing::ValuesIn(rsamg_cycle),
                                         testing::ValuesIn(rsamg_scaling)));
