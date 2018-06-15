/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_bicgstabl.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, unsigned int, int> bicgstabl_tuple;

int bicgstabl_size[] = {7, 63};
std::string bicgstabl_precond[] = {"None", "SPAI", "TNS", "Jacobi", "GS", "ILU", "ILUT", "MCGS", "MCILU"};
unsigned int bicgstabl_format[] = {1, 2, 4, 5, 6, 7};
int bicgstabl_level[] = {1, 2, 4};

class parameterized_bicgstabl : public testing::TestWithParam<bicgstabl_tuple>
{
    protected:
    parameterized_bicgstabl() {}
    virtual ~parameterized_bicgstabl() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_bicgstabl_arguments(bicgstabl_tuple tup)
{
    Arguments arg;
    arg.size       = std::get<0>(tup);
    arg.precond    = std::get<1>(tup);
    arg.format     = std::get<2>(tup);
    arg.index      = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_bicgstabl, bicgstabl_float)
{
    Arguments arg = setup_bicgstabl_arguments(GetParam());
    ASSERT_EQ(testing_bicgstabl<float>(arg), true);
}

TEST_P(parameterized_bicgstabl, bicgstabl_double)
{
    Arguments arg = setup_bicgstabl_arguments(GetParam());
    ASSERT_EQ(testing_bicgstabl<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(bicgstabl,
                        parameterized_bicgstabl,
                        testing::Combine(testing::ValuesIn(bicgstabl_size),
                                         testing::ValuesIn(bicgstabl_precond),
                                         testing::ValuesIn(bicgstabl_format),
                                         testing::ValuesIn(bicgstabl_level)));
