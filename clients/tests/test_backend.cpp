/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_backend.hpp"
#include "utility.hpp"

#include <gtest/gtest.h>

typedef std::tuple<int, int, int, int, bool, int, bool> backend_tuple;

int backend_rank[] = {-1, 0, 13};
int backend_dev_node[] = {-1, 0, 13};
int backend_dev[] = {-1, 0, 13};
int backend_omp_threads[] = {-1, 0, 8};
bool backend_affinity[] = {true, false};
int backend_omp_threshold[] = {-1, 0, 20000};
bool backend_disable_acc[] = {true, false};

class parameterized_backend : public testing::TestWithParam<backend_tuple>
{
    protected:
    parameterized_backend() {}
    virtual ~parameterized_backend() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_backend_arguments(backend_tuple tup)
{
    Arguments arg;
    arg.rank          = std::get<0>(tup);
    arg.dev_per_node  = std::get<1>(tup);
    arg.dev           = std::get<2>(tup);
    arg.omp_nthreads  = std::get<3>(tup);
    arg.omp_affinity  = std::get<4>(tup);
    arg.omp_threshold = std::get<5>(tup);
    arg.use_acc       = std::get<6>(tup);
    return arg;
}

TEST(backend_init_order, backend)
{
    testing_backend_init_order();
}

TEST_P(parameterized_backend, backend)
{
    Arguments arg = setup_backend_arguments(GetParam());
    testing_backend(arg);
}

INSTANTIATE_TEST_CASE_P(backend,
                        parameterized_backend,
                        testing::Combine(testing::ValuesIn(backend_rank),
                                         testing::ValuesIn(backend_dev_node),
                                         testing::ValuesIn(backend_dev),
                                         testing::ValuesIn(backend_omp_threads),
                                         testing::ValuesIn(backend_affinity),
                                         testing::ValuesIn(backend_omp_threshold),
                                         testing::ValuesIn(backend_disable_acc)));
