/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_idr.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, unsigned int, int> idr_tuple;

int idr_size[] = {7, 63};
std::string idr_precond[] = {"None", "SPAI", "GS", "ILU", "ILUT", "MCGS", "MCILU"};
unsigned int idr_format[] = {1, 2, 4, 5, 6, 7};
int idr_level[] = {1, 2, 4};

class parameterized_idr : public testing::TestWithParam<idr_tuple>
{
    protected:
    parameterized_idr() {}
    virtual ~parameterized_idr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_idr_arguments(idr_tuple tup)
{
    Arguments arg;
    arg.size       = std::get<0>(tup);
    arg.precond    = std::get<1>(tup);
    arg.format     = std::get<2>(tup);
    arg.index      = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_idr, idr_float)
{
    Arguments arg = setup_idr_arguments(GetParam());
    ASSERT_EQ(testing_idr<float>(arg), true);
}

TEST_P(parameterized_idr, idr_double)
{
    Arguments arg = setup_idr_arguments(GetParam());
    ASSERT_EQ(testing_idr<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(idr,
                        parameterized_idr,
                        testing::Combine(testing::ValuesIn(idr_size),
                                         testing::ValuesIn(idr_precond),
                                         testing::ValuesIn(idr_format),
                                         testing::ValuesIn(idr_level)));
