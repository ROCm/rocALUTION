/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef TESTING_PARALLEL_MANAGER_HPP
#define TESTING_PARALLEL_MANAGER_HPP

#include "utility.hpp"

#include <rocalution.hpp>
#include <gtest/gtest.h>

using namespace rocalution;

template <typename T>
void testing_parallel_manager_bad_args(void)
{
    int safe_size = 100;

    // Initialize rocALUTION
    init_rocalution();

    ParallelManager pm;

    int* idata = nullptr;
    allocate_host(safe_size, &idata);

    // SetMPICommunicator
    {
        void* null_ptr = nullptr;
        ASSERT_DEATH(pm.SetMPICommunicator(null_ptr), ".*Assertion.*comm != NULL*");
    }

    // SetBoundaryIndex
    {
        int* null_int = nullptr;
        ASSERT_DEATH(pm.SetBoundaryIndex(safe_size, null_int), ".*Assertion.*index != NULL*");
    }

    // SetReceivers
    {
        int* null_int = nullptr;
        ASSERT_DEATH(pm.SetReceivers(safe_size, null_int, idata), ".*Assertion.*recvs != NULL*");
        ASSERT_DEATH(pm.SetReceivers(safe_size, idata, null_int), ".*Assertion.*recv_offset != NULL*");
    }

    // SetSenders
    {
        int* null_int = nullptr;
        ASSERT_DEATH(pm.SetSenders(safe_size, null_int, idata), ".*Assertion.*sends != NULL*");
        ASSERT_DEATH(pm.SetSenders(safe_size, idata, null_int), ".*Assertion.*send_offset != NULL*");
    }

    free_host(&idata);

    // Stop rocALUTION
    stop_rocalution();
}

#endif // TESTING_PARALLEL_MANAGER_HPP
