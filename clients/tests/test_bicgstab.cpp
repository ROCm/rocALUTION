/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_bicgstab.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, unsigned int> bicgstab_tuple;

int bicgstab_size[] = {7, 63};
std::string bicgstab_precond[] = {"None", "Chebyshev", "SPAI", "TNS", "Jacobi", "GS", "ILU", "ILUT", "MCGS", "MCILU"};
unsigned int bicgstab_format[] = {1, 2, 4, 5, 6, 7};

class parameterized_bicgstab : public testing::TestWithParam<bicgstab_tuple>
{
    protected:
    parameterized_bicgstab() {}
    virtual ~parameterized_bicgstab() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_bicgstab_arguments(bicgstab_tuple tup)
{
    Arguments arg;
    arg.size       = std::get<0>(tup);
    arg.precond    = std::get<1>(tup);
    arg.format     = std::get<2>(tup);
    return arg;
}

TEST_P(parameterized_bicgstab, bicgstab_float)
{
    Arguments arg = setup_bicgstab_arguments(GetParam());
    ASSERT_EQ(testing_bicgstab<float>(arg), true);
}

TEST_P(parameterized_bicgstab, bicgstab_double)
{
    Arguments arg = setup_bicgstab_arguments(GetParam());
    ASSERT_EQ(testing_bicgstab<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(bicgstab,
                        parameterized_bicgstab,
                        testing::Combine(testing::ValuesIn(bicgstab_size),
                                         testing::ValuesIn(bicgstab_precond),
                                         testing::ValuesIn(bicgstab_format)));
