/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_BACKEND_HPP
#define TESTING_BACKEND_HPP

#include <rocalution.hpp>

using namespace rocalution;

void testing_backend_bad_arg(void)
{
    init_rocalution();
    stop_rocalution();
}

#endif // TESTING_BACKEND_HPP
