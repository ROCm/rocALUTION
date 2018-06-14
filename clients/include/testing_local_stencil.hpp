/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_LOCAL_STENCIL_HPP
#define TESTING_LOCAL_STENCIL_HPP

#include "utility.hpp"

#include <rocalution.hpp>
#include <gtest/gtest.h>

using namespace rocalution;

template <typename T>
void testing_local_stencil_bad_args(void)
{
    // Initialize rocALUTION
    init_rocalution();

    LocalStencil<T> stn(Laplace2D);
    LocalVector<T> vec;

    // Apply
    {
        LocalVector<T> *null_vec = nullptr;
        ASSERT_DEATH(stn.Apply(vec, null_vec), ".*Assertion.*out != NULL*");
    }

    // ApplyAdd
    {
        LocalVector<T> *null_vec = nullptr;
        ASSERT_DEATH(stn.ApplyAdd(vec, 1.0, null_vec), ".*Assertion.*out != NULL*");
    }

    // Stop rocALUTION
    stop_rocalution();
}

#endif // TESTING_LOCAL_STENCIL_HPP
