/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_saamg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, int, int, unsigned int, int, int> saamg_tuple;

int saamg_size[] = {63, 134};
std::string saamg_smoother[] = {"FSAI", "SPAI"};
int saamg_pre_iter[] = {1, 2};
int saamg_post_iter[] = {1, 2};
int saamg_cycle[] = {0, 2};
int saamg_scaling[] = {0, 1};

unsigned int saamg_format[] = {1, 6};

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
    arg.size        = std::get<0>(tup);
    arg.smoother    = std::get<1>(tup);
    arg.format      = std::get<2>(tup);
    arg.pre_smooth  = std::get<3>(tup);
    arg.post_smooth = std::get<4>(tup);
    arg.cycle       = std::get<5>(tup);
    arg.ordering    = std::get<6>(tup);
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
                                         testing::ValuesIn(saamg_smoother),
                                         testing::ValuesIn(saamg_pre_iter),
                                         testing::ValuesIn(saamg_post_iter),
                                         testing::ValuesIn(saamg_format),
                                         testing::ValuesIn(saamg_cycle),
                                         testing::ValuesIn(saamg_scaling)));
