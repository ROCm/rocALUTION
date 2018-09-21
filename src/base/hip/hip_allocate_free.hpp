/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HIP_ALLOCATE_FREE_HPP_
#define ROCALUTION_HIP_ALLOCATE_FREE_HPP_

#include <iostream>

namespace rocalution {

template <typename DataType>
void allocate_hip(int size, DataType** ptr);

template <typename DataType>
void free_hip(DataType** ptr);

template <typename DataType>
void set_to_zero_hip(int blocksize, int size, DataType* ptr);

template <typename DataType>
void set_to_one_hip(int blocksize, int size, DataType* ptr);

} // namespace rocalution

#endif // ROCALUTION_HIP_ALLOCATE_FREE_HPP_
