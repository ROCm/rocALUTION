/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_LOCAL_VECTOR_HPP
#define TESTING_LOCAL_VECTOR_HPP

#include "utility.hpp"

#include <rocalution.hpp>
#include <gtest/gtest.h>

using namespace rocalution;

template <typename T>
void testing_local_vector_bad_args(void)
{
    int safe_size = 100;

    // Initialize rocALUTION
    init_rocalution();

    // LocalVector object
    LocalVector<T> vec;

    // SetDataPtr
    {
        T* null_ptr = nullptr;
        ASSERT_DEATH(vec.SetDataPtr(nullptr, "", safe_size), ".*Assertion.*ptr != NULL*");
        ASSERT_DEATH(vec.SetDataPtr(&null_ptr, "", safe_size), ".*Assertion.*ptr != NULL*");
    }

    // LeaveDataPtr
    {
        T* vdata = nullptr;
        allocate_host(safe_size, &vdata);
        ASSERT_DEATH(vec.LeaveDataPtr(&vdata), ".*Assertion.*ptr == NULL*");
        free_host(&vdata);
    }

    // CopyFromData
    {
        T* null_ptr = nullptr;
        ASSERT_DEATH(vec.CopyFromData(null_ptr), ".*Assertion.*data != NULL*");
    }

    // CopyToData
    {
        T* null_ptr = nullptr;
        ASSERT_DEATH(vec.CopyToData(null_ptr), ".*Assertion.*data != NULL*");
    }

    // SetIndexArray
    {
        int* null_int = nullptr;
        ASSERT_DEATH(vec.SetIndexArray(safe_size, null_int), ".*Assertion.*index != NULL*");
    }

    // GetIndexValues
    {
        T* null_T = nullptr;
        ASSERT_DEATH(vec.GetIndexValues(null_T), ".*Assertion.*values != NULL*");
    }

    // SetIndexValues
    {
        T* null_T = nullptr;
        ASSERT_DEATH(vec.SetIndexValues(null_T), ".*Assertion.*values != NULL*");
    }

    // GetContinuousValues
    {
        T* null_T = nullptr;
        ASSERT_DEATH(vec.GetContinuousValues(0, 0, null_T), ".*Assertion.*values != NULL*");
    }

    // SetContinuousValues
    {
        T* null_T = nullptr;
        ASSERT_DEATH(vec.SetContinuousValues(0, 0, null_T), ".*Assertion.*values != NULL*");
    }

    // ExtractCoarseMapping
    {
        int* null_int = nullptr;
        int* vint = nullptr;
        allocate_host(safe_size, &vint);
        ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, null_int, 0, vint, vint), ".*Assertion.*index != NULL*");
        ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, vint, 0, null_int, vint), ".*Assertion.*size != NULL*");
        ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, vint, 0, vint, null_int), ".*Assertion.*map != NULL*");
        free_host(&vint);
    }

    // ExtractCoarseBoundary
    {
        int* null_int = nullptr;
        int* vint = nullptr;
        allocate_host(safe_size, &vint);
        ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, null_int, 0, vint, vint), ".*Assertion.*index != NULL*");
        ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, vint, 0, null_int, vint), ".*Assertion.*size != NULL*");
        ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, vint, 0, vint, null_int), ".*Assertion.*boundary != NULL*");
        free_host(&vint);
    }

    // Stop rocALUTION
    stop_rocalution();
}

#endif // TESTING_LOCAL_VECTOR_HPP
