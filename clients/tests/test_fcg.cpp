/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_fcg.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, unsigned int> fcg_tuple;

int fcg_size[] = {7, 63};
std::string fcg_precond[] = {"None", "Chebyshev", "FSAI", "SPAI", "TNS", "Jacobi", "SGS", "ILU", "ILUT", "IC", "MCSGS", "MCILU"};
unsigned int fcg_format[] = {1, 2, 4, 5, 6, 7};

class parameterized_fcg : public testing::TestWithParam<fcg_tuple>
{
    protected:
    parameterized_fcg() {}
    virtual ~parameterized_fcg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_fcg_arguments(fcg_tuple tup)
{
    Arguments arg;
    arg.size       = std::get<0>(tup);
    arg.precond    = std::get<1>(tup);
    arg.format     = std::get<2>(tup);
    return arg;
}

TEST_P(parameterized_fcg, krylov_float)
{
    Arguments arg = setup_fcg_arguments(GetParam());
    ASSERT_EQ(testing_fcg<float>(arg), true);
}

TEST_P(parameterized_fcg, krylov_double)
{
    Arguments arg = setup_fcg_arguments(GetParam());
    ASSERT_EQ(testing_fcg<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(fcg,
                        parameterized_fcg,
                        testing::Combine(testing::ValuesIn(fcg_size),
                                         testing::ValuesIn(fcg_precond),
                                         testing::ValuesIn(fcg_format)));
