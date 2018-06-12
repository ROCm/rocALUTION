/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_cr.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, unsigned int> cr_tuple;

int cr_size[] = {7, 63};
std::string cr_precond[] = {"None", "Chebyshev", "FSAI", "SPAI", "TNS", "Jacobi", "SGS", "ILU", "ILUT", "IC", "MCSGS", "MCILU"};
unsigned int cr_format[] = {1, 2, 4, 5, 6, 7};

class parameterized_cr : public testing::TestWithParam<cr_tuple>
{
    protected:
    parameterized_cr() {}
    virtual ~parameterized_cr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_cr_arguments(cr_tuple tup)
{
    Arguments arg;
    arg.size       = std::get<0>(tup);
    arg.precond    = std::get<1>(tup);
    arg.format     = std::get<2>(tup);
    return arg;
}
/* TODO there _MIGHT_ be some issue with float accuracy
TEST_P(parameterized_cr, cr_float)
{
    Arguments arg = setup_cr_arguments(GetParam());
    ASSERT_EQ(testing_cr<float>(arg), true);
}
*/
TEST_P(parameterized_cr, cr_double)
{
    Arguments arg = setup_cr_arguments(GetParam());
    ASSERT_EQ(testing_cr<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(cr,
                        parameterized_cr,
                        testing::Combine(testing::ValuesIn(cr_size),
                                         testing::ValuesIn(cr_precond),
                                         testing::ValuesIn(cr_format)));
