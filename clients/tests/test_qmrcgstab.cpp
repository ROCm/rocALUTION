/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_qmrcgstab.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, std::string, unsigned int> qmrcgstab_tuple;

int qmrcgstab_size[] = {7, 63};
std::string qmrcgstab_precond[] = {"None", "Chebyshev", "SPAI", "TNS", "Jacobi", "GS", "ILU", "ILUT", "MCGS", "MCILU"};
unsigned int qmrcgstab_format[] = {1, 2, 4, 5, 6, 7};

class parameterized_qmrcgstab : public testing::TestWithParam<qmrcgstab_tuple>
{
    protected:
    parameterized_qmrcgstab() {}
    virtual ~parameterized_qmrcgstab() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_qmrcgstab_arguments(qmrcgstab_tuple tup)
{
    Arguments arg;
    arg.size       = std::get<0>(tup);
    arg.precond    = std::get<1>(tup);
    arg.format     = std::get<2>(tup);
    return arg;
}

TEST_P(parameterized_qmrcgstab, qmrcgstab_float)
{
    Arguments arg = setup_qmrcgstab_arguments(GetParam());
    ASSERT_EQ(testing_qmrcgstab<float>(arg), true);
}

TEST_P(parameterized_qmrcgstab, qmrcgstab_double)
{
    Arguments arg = setup_qmrcgstab_arguments(GetParam());
    ASSERT_EQ(testing_qmrcgstab<double>(arg), true);
}

INSTANTIATE_TEST_CASE_P(qmrcgstab,
                        parameterized_qmrcgstab,
                        testing::Combine(testing::ValuesIn(qmrcgstab_size),
                                         testing::ValuesIn(qmrcgstab_precond),
                                         testing::ValuesIn(qmrcgstab_format)));
