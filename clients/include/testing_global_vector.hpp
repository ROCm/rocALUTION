/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_GLOBAL_VECTOR_HPP
#define TESTING_GLOBAL_VECTOR_HPP

#include "utility.hpp"

#include <rocalution.hpp>
#include <gtest/gtest.h>

using namespace rocalution;

template <typename T>
void testing_global_vector_bad_args(void)
{
    int safe_size = 100;

    // Initialize rocALUTION
    init_rocalution();

    GlobalVector<T> vec;

    // SetDataPtr
    {
        T* null_data = nullptr;
        ASSERT_DEATH(vec.SetDataPtr(nullptr, "", safe_size), ".*Assertion.*ptr != NULL*");
        ASSERT_DEATH(vec.SetDataPtr(&null_data, "", safe_size), ".*Assertion.*ptr != NULL*");
    }

    // LeaveDataPtr
    {
        T* data = nullptr;
        allocate_host(safe_size, &data);
        ASSERT_DEATH(vec.LeaveDataPtr(&data), ".*Assertion.*ptr == NULL*");
        free_host(&data);
    }

    // Stop rocALUTION
    stop_rocalution();
}

#endif // TESTING_GLOBAL_VECTOR_HPP
