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

    LocalVector<T> vec1;
    LocalVector<T> vec2;
    LocalVector<int> vec_int;

    int* vint = NULL;
    T* vdata = NULL;

    allocate_host(safe_size, &vint);
    allocate_host(safe_size, &vdata);

    // SetDataPtr, LeaveDataPtr
    {
        T* null_ptr = nullptr;

        ASSERT_DEATH(vec1.SetDataPtr(nullptr, "", safe_size), ".*Assertion.*ptr != NULL*");
        ASSERT_DEATH(vec1.SetDataPtr(&null_ptr, "", safe_size), ".*Assertion.*ptr != NULL*");
        ASSERT_DEATH(vec1.LeaveDataPtr(&vdata), ".*Assertion.*ptr == NULL*");
    }

    // CopyFrom, CopyFromAsync, CopyFromPermute
    {
        LocalVector<T> *null_ptr = nullptr;
        ASSERT_DEATH(vec1.CopyFrom(*null_ptr), ".*Assertion.*src != NULL*");
        ASSERT_DEATH(vec1.CopyFromAsync(*null_ptr), ".*Assertion.*src != NULL*");
        ASSERT_DEATH(vec1.CopyFrom(*null_ptr, 0, 0, safe_size), ".*Assertion.*src != NULL*");
    }

    // CopyFromFloat, CopyFromDouble
    {
        LocalVector<float> *null_ptr_float = nullptr;
        LocalVector<double> *null_ptr_double = nullptr;
        ASSERT_DEATH(vec1.CopyFromFloat(*null_ptr_float), ".*Assertion.*src != NULL*");
        ASSERT_DEATH(vec1.CopyFromDouble(*null_ptr_double), ".*Assertion.*src != NULL*");
    }

    // CopyFromPermute, CopyFromPermuteBackward, Permute, PermuteBackward
    // Restriction, Prolongation
    {
        LocalVector<T> *null_ptr = nullptr;
        LocalVector<int> *null_ptr_int = nullptr;
        ASSERT_DEATH(vec1.CopyFromPermute(*null_ptr, vec_int), ".*Assertion.*src != NULL*");
        ASSERT_DEATH(vec1.CopyFromPermute(vec2, *null_ptr_int), ".*Assertion.*permutation != NULL*");
        ASSERT_DEATH(vec1.CopyFromPermuteBackward(*null_ptr, vec_int), ".*Assertion.*src != NULL*");
        ASSERT_DEATH(vec1.CopyFromPermuteBackward(vec2, *null_ptr_int), ".*Assertion.*permutation != NULL*");
        ASSERT_DEATH(vec1.Permute(*null_ptr_int), ".*Assertion.*permutation != NULL*");
        ASSERT_DEATH(vec1.PermuteBackward(*null_ptr_int), ".*Assertion.*permutation != NULL*");
        ASSERT_DEATH(vec1.Restriction(*null_ptr, vec_int), ".*Assertion.*vec_fine != NULL*");
        ASSERT_DEATH(vec1.Restriction(vec2, *null_ptr_int), ".*Assertion.*map != NULL*");
        ASSERT_DEATH(vec1.Prolongation(*null_ptr, vec_int), ".*Assertion.*vec_coarse != NULL*");
        ASSERT_DEATH(vec1.Prolongation(vec2, *null_ptr_int), ".*Assertion.*map != NULL*");
    }

    // CopyFromData, CopyToData
    {
        T* null_ptr = nullptr;
        ASSERT_DEATH(vec1.CopyFromData(null_ptr), ".*Assertion.*data != NULL*");
        ASSERT_DEATH(vec1.CopyToData(null_ptr), ".*Assertion.*data != NULL*");
    }

    // CloneFrom
    {
        LocalVector<T> *null_ptr = nullptr;
        ASSERT_DEATH(vec1.CloneFrom(*null_ptr), ".*Assertion.*src != NULL*");
    }

    // AddScale, ScaleAdd, ScaleAddScale, ScaleAdd2
    {
        LocalVector<T> *null_ptr = nullptr;
        ASSERT_DEATH(vec1.AddScale(*null_ptr, 1.0), ".*Assertion.*x != NULL*");
        ASSERT_DEATH(vec1.ScaleAdd(1.0, *null_ptr), ".*Assertion.*x != NULL*");
        ASSERT_DEATH(vec1.ScaleAddScale(1.0, *null_ptr, 1.0), ".*Assertion.*x != NULL*");
        ASSERT_DEATH(vec1.ScaleAddScale(1.0, *null_ptr, 1.0, 0, 0, safe_size), ".*Assertion.*x != NULL*");
        ASSERT_DEATH(vec1.ScaleAdd2(1.0, *null_ptr, 1.0, vec2, 1.0), ".*Assertion.*x != NULL*");
        ASSERT_DEATH(vec1.ScaleAdd2(1.0, vec2, 1.0, *null_ptr, 1.0), ".*Assertion.*y != NULL*");
    }

    // ExclusiveScan, Dot, DotNonConj, PointWiseMult
    {
        LocalVector<T> *null_ptr = nullptr;
        ASSERT_DEATH(vec1.ExclusiveScan(*null_ptr), ".*Assertion.*x != NULL*");
        ASSERT_DEATH(vec1.Dot(*null_ptr), ".*Assertion.*x != NULL*");
        ASSERT_DEATH(vec1.DotNonConj(*null_ptr), ".*Assertion.*x != NULL*");
        ASSERT_DEATH(vec1.PointWiseMult(*null_ptr), ".*Assertion.*x != NULL*");
        ASSERT_DEATH(vec1.PointWiseMult(*null_ptr, vec2), ".*Assertion.*x != NULL*");
        ASSERT_DEATH(vec1.PointWiseMult(vec2, *null_ptr), ".*Assertion.*y != NULL*");
    }

    // SetIndexArray, GetIndexValues, SetIndexValues
    // GetContinuousValues, SetContinuousValues
    {
        int* null_int = nullptr;
        T* null_T = nullptr;
        ASSERT_DEATH(vec1.SetIndexArray(safe_size, null_int), ".*Assertion.*index != NULL*");
        ASSERT_DEATH(vec1.GetIndexValues(null_T), ".*Assertion.*values != NULL*");
        ASSERT_DEATH(vec1.SetIndexValues(null_T), ".*Assertion.*values != NULL*");
        ASSERT_DEATH(vec1.GetContinuousValues(0, 0, null_T), ".*Assertion.*values != NULL*");
        ASSERT_DEATH(vec1.SetContinuousValues(0, 0, null_T), ".*Assertion.*values != NULL*");
    }

    // ExtractCoarseMapping, ExtractCoarseBoundary
    {
        int* null_int = nullptr;
        ASSERT_DEATH(vec1.ExtractCoarseMapping(0, 0, null_int, 0, vint, vint), ".*Assertion.*index != NULL*");
        ASSERT_DEATH(vec1.ExtractCoarseMapping(0, 0, vint, 0, null_int, vint), ".*Assertion.*size != NULL*");
        ASSERT_DEATH(vec1.ExtractCoarseMapping(0, 0, vint, 0, vint, null_int), ".*Assertion.*map != NULL*");
        ASSERT_DEATH(vec1.ExtractCoarseBoundary(0, 0, null_int, 0, vint, vint), ".*Assertion.*index != NULL*");
        ASSERT_DEATH(vec1.ExtractCoarseBoundary(0, 0, vint, 0, null_int, vint), ".*Assertion.*size != NULL*");
        ASSERT_DEATH(vec1.ExtractCoarseBoundary(0, 0, vint, 0, vint, null_int), ".*Assertion.*boundary != NULL*");
    }

    free_host(&vint);
    free_host(&vdata);

    // Stop rocALUTION
    stop_rocalution();
}

#endif // TESTING_LOCAL_VECTOR_HPP
