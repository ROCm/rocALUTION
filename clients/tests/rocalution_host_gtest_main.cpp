/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <stdexcept>
#include <rocalution.hpp>

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    rocalution::init_rocalution();
    rocalution::info_rocalution();
    rocalution::stop_rocalution();

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
