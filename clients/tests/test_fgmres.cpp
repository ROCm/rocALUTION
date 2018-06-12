/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_fgmres.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, int, std::string, unsigned int> fgmres_tuple;

int fgmres_size[] = {7, 63};
int fgmres_basis[] = {20, 60};
std::string fgmres_precond[] = {"None", "Chebyshev", "SPAI", "TNS", "Jacobi", "GS", "ILU", "ILUT", "MCGS", "MCILU"};
unsigned int fgmres_format[] = {1, 2, 4, 5, 6, 7};

class parameterized_fgmres : public testing::TestWithParam<fgmres_tuple>
{
    protected:
    parameterized_fgmres() {}
    virtual ~parameterized_fgmres() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_fgmres_arguments(fgmres_tuple tup)
{
    Arguments arg;
    arg.size       = std::get<0>(tup);
    arg.index      = std::get<1>(tup);
    arg.precond    = std::get<2>(tup);
    arg.format     = std::get<3>(tup);
    return arg;
}

TEST_P(parameterized_fgmres, krylov_float)
{
    Arguments arg = setup_fgmres_arguments(GetParam());
    ASSERT_EQ(testing_fgmres<float>(arg), true);
}

TEST_P(parameterized_fgmres, krylov_double)
{
    Arguments arg = setup_fgmres_arguments(GetParam());
    ASSERT_EQ(testing_fgmres<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(fgmres,
                        parameterized_fgmres,
                        testing::Combine(testing::ValuesIn(fgmres_size),
                                         testing::ValuesIn(fgmres_basis),
                                         testing::ValuesIn(fgmres_precond),
                                         testing::ValuesIn(fgmres_format)));
