/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef TESTING_LOCAL_VECTOR_HPP
#define TESTING_LOCAL_VECTOR_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

template <typename T>
void testing_local_vector_bad_args(void)
{
    int safe_size = 100;

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    // LocalVector object
    LocalVector<T> vec;

    // SetDataPtr
    {
        T* null_ptr = nullptr;
        ASSERT_DEATH(vec.SetDataPtr(nullptr, "", safe_size), ".*Assertion.*ptr != (NULL|__null)*");
        ASSERT_DEATH(vec.SetDataPtr(&null_ptr, "", safe_size),
                     ".*Assertion.*ptr != (NULL|__null)*");
    }

    // LeaveDataPtr
    {
        T* vdata = nullptr;
        allocate_host(safe_size, &vdata);
        ASSERT_DEATH(vec.LeaveDataPtr(&vdata), ".*Assertion.*ptr == (NULL|__null)*");
        free_host(&vdata);
    }

    // CopyFromData
    {
        T* null_ptr = nullptr;
        ASSERT_DEATH(vec.CopyFromData(null_ptr), ".*Assertion.*data != (NULL|__null)*");
    }

    // CopyToData
    {
        T* null_ptr = nullptr;
        ASSERT_DEATH(vec.CopyToData(null_ptr), ".*Assertion.*data != (NULL|__null)*");
    }

    // GetContinuousValues
    {
        vec.Allocate("", safe_size);
        T* null_T = nullptr;
        ASSERT_DEATH(vec.GetContinuousValues(0, safe_size, null_T),
                     ".*Assertion.*values != (NULL|__null)*");
    }

    // ExtractCoarseMapping
    {
        int* null_int = nullptr;
        int* vint     = nullptr;
        allocate_host(safe_size, &vint);
        ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, null_int, 0, vint, vint),
                     ".*Assertion.*index != (NULL|__null)*");
        ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, vint, 0, null_int, vint),
                     ".*Assertion.*size != (NULL|__null)*");
        ASSERT_DEATH(vec.ExtractCoarseMapping(0, 0, vint, 0, vint, null_int),
                     ".*Assertion.*map != (NULL|__null)*");
        free_host(&vint);
    }

    // ExtractCoarseBoundary
    {
        int* null_int = nullptr;
        int* vint     = nullptr;
        allocate_host(safe_size, &vint);
        ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, null_int, 0, vint, vint),
                     ".*Assertion.*index != (NULL|__null)*");
        ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, vint, 0, null_int, vint),
                     ".*Assertion.*size != (NULL|__null)*");
        ASSERT_DEATH(vec.ExtractCoarseBoundary(0, 0, vint, 0, vint, null_int),
                     ".*Assertion.*boundary != (NULL|__null)*");
        free_host(&vint);
    }

    // Stop rocALUTION
    stop_rocalution();
}

template <typename T>
LocalVector<T> getTestVector(int size = 10)
{
    // Create a LocalVector
    LocalVector<T> vec;
    vec.Allocate("TestVector", size);

    // Fill the source vector with some values
    for(int i = 0; i < size; ++i)
    {
        vec[i] = static_cast<T>(i + 1);
    }

    return vec;
}

template <typename T>
void testing_get_interior()
{
    // Get a LocalVector
    LocalVector<T> vec = getTestVector<T>();

    // Call GetInterior
    LocalVector<T>& interior = vec.GetInterior();

    // Validate that the returned reference is the same as the original object
    EXPECT_EQ(&interior, &vec);

    // Validate that the size and properties of the vector remain unchanged
    EXPECT_EQ(interior.GetSize(), vec.GetSize());

    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], interior[i]);
    }
}

template <typename T>
void testing_const_get_interior()
{
    // Get a LocalVector
    const LocalVector<T> vec = getTestVector<T>();

    // Call GetInterior
    const LocalVector<T>& interior = vec.GetInterior();

    // Validate that the returned reference is the same as the original object
    EXPECT_EQ(&interior, &vec);

    // Validate that the size and properties of the vector remain unchanged
    EXPECT_EQ(interior.GetSize(), vec.GetSize());

    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], interior[i]);
    }
}

template <typename T>
void testing_check()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>();

    // Check the vector
    EXPECT_NO_THROW(vec.Check());
}

template <typename T>
void testing_zeros()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>();

    vec.Zeros();
    // Validate that all elements are set to zero
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>(0.0));
    }
}

template <typename T>
void testing_ones()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>();

    vec.Ones();
    // Validate that all elements are set to one
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>(1.0));
    }
}

template <typename T>
void testing_set_values()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>();

    // Set values in the vector
    T value = static_cast<T>(5.0);
    vec.SetValues(value);

    // Validate that all elements are set to the specified value
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], value);
    }
}

template <typename T>
void testing_set_random_uniform()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>();

    // Set random values in the vector
    T min = static_cast<T>(0.0);
    T max = static_cast<T>(1.0);
    vec.SetRandomUniform(min, max);

    // Validate that all elements are within the specified range
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_GE(vec[i], min);
        EXPECT_LE(vec[i], max);
    }
}

template <typename T>
void testing_set_random_normal()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(1000);

    // Set random values in the vector
    T mean = static_cast<T>(0.0);
    T std  = static_cast<T>(1.0);
    vec.SetRandomNormal(mean, std);

    // Validate that the mean of all elements is close to the specified mean
    T sum = 0.0;
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        sum += vec[i];
    }
    T calculated_mean = sum / vec.GetSize();
    EXPECT_NEAR(calculated_mean, mean, 3 * std);

    // Validate that the standard deviation of all elements is close to the specified std
    T sum_sq = 0.0;
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        sum_sq += (vec[i] - calculated_mean) * (vec[i] - calculated_mean);
    }
    T calculated_std = std::sqrt(sum_sq / vec.GetSize());
    EXPECT_NEAR(calculated_std, std, 3 * std);
}

template <typename T>
void testing_copy_from()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Create another LocalVector to copy from
    LocalVector<T> src_vec = getTestVector<T>(5);

    // Copy values from the source vector
    EXPECT_NO_THROW(vec.CopyFrom(src_vec));

    // Validate the copied values
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], src_vec[i]);
    }
}

template <typename T>
void testing_clone_from()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Clone the vector
    LocalVector<T> cloned_vec;
    EXPECT_NO_THROW(cloned_vec.CloneFrom(vec));

    // Validate the cloned values
    for(int i = 0; i < cloned_vec.GetSize(); ++i)
    {
        EXPECT_EQ(cloned_vec[i], vec[i]);
    }
}

template <typename T>
void testing_binary()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Define the filename for the test
    std::string filename = "test_vector.bin";

    // Write the vector to a binary file
    EXPECT_NO_THROW(vec.WriteFileBinary(filename));

    // Read the vector from the binary file
    LocalVector<T> read_vec;
    EXPECT_NO_THROW(read_vec.ReadFileBinary(filename));

    // Validate the read values
    for(int i = 0; i < read_vec.GetSize(); ++i)
    {
        EXPECT_EQ(read_vec[i], vec[i]);
    }

    // Clean up the temporary file
    std::remove(filename.c_str());
}

template <typename T>
void testing_add_scale()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> v = getTestVector<T>(5);
    LocalVector<T> x = getTestVector<T>(5);

    // Define a scaling factor
    T alpha = static_cast<T>(2.0);

    // Perform the AddScale operation
    EXPECT_NO_THROW(v.AddScale(x, alpha));

    // Validate the result
    for(int i = 0; i < v.GetSize(); ++i)
    {
        EXPECT_EQ(v[i], static_cast<T>((i + 1) * (1 + alpha)));
    }
}

template <typename T>
void testing_scale_add()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> v = getTestVector<T>(5);
    LocalVector<T> x = getTestVector<T>(5);

    // Define a scaling factor
    T alpha = static_cast<T>(2.0);

    // v = 1 initially
    v.Ones();

    // Perform the v = alpha * v + x
    EXPECT_NO_THROW(v.ScaleAdd(alpha, x));

    // Validate the result
    for(int i = 0; i < v.GetSize(); ++i)
    {
        EXPECT_EQ(v[i], alpha + x[i]);
    }
}

template <typename T>
void testing_scale_add_2()
{
    LocalVector<T> v = getTestVector<T>(3);
    LocalVector<T> x = getTestVector<T>(3);
    LocalVector<T> y = getTestVector<T>(3);

    // Fill x and y with test values
    x[0] = static_cast<T>(1);
    x[1] = static_cast<T>(2);
    x[2] = static_cast<T>(3);
    y[0] = static_cast<T>(4);
    y[1] = static_cast<T>(5);
    y[2] = static_cast<T>(6);

    // v = 1 initially
    v.Ones();

    // v.ScaleAdd2(alpha, x, beta, y, gamma)
    T alpha = static_cast<T>(2);
    T beta  = static_cast<T>(3);
    T gamma = static_cast<T>(3);
    v.ScaleAdd2(alpha, x, beta, y, gamma);

    // v[i] = alpha * v[i] + beta * x[i] + gamma * y[i]
    for(int i = 0; i < v.GetSize(); ++i)
    {
        EXPECT_EQ(v[i], alpha + beta * x[i] + gamma * y[i]);
    }
}

template <typename T>
void testing_scale()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Define a scaling factor
    T alpha = static_cast<T>(2.0);

    // Perform the Scale operation
    EXPECT_NO_THROW(vec.Scale(alpha));

    // Validate the result
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>((i + 1) * alpha));
    }
}

template <typename T>
void testing_dot()
{
    // Create two LocalVectors and allocate some size
    LocalVector<T> vec1 = getTestVector<T>(5);
    LocalVector<T> vec2 = getTestVector<T>(5);

    // Perform the Dot product
    T result = 0.0;
    EXPECT_NO_THROW(result = vec1.Dot(vec2));

    // Validate the result
    T result2 = 0.0;
    for(int i = 0; i < vec1.GetSize(); ++i)
    {
        result2 += vec1[i] * vec2[i];
    }
    EXPECT_EQ(result, result2);
}

template <typename T>
void testing_dot_non_conj()
{
    // Create two LocalVectors and allocate some size
    LocalVector<T> vec1 = getTestVector<T>(5);
    LocalVector<T> vec2 = getTestVector<T>(5);

    // Perform the Dot product without conjugation
    T result = 0.0;
    EXPECT_NO_THROW(result = vec1.DotNonConj(vec2));

    // Validate the result
    T result2 = 0.0;
    for(int i = 0; i < vec1.GetSize(); ++i)
    {
        result2 += vec1[i] * vec2[i];
    }
    EXPECT_EQ(result, result2);
}

template <typename T>
void testing_norm()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Perform the Norm operation
    T result = 0.0;
    EXPECT_NO_THROW(result = vec.Norm());

    // Validate the result
    T result2 = 0.0;
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        result2 += vec[i] * vec[i];
    }
    EXPECT_NEAR(result * result, result2, 1e-6 * std::abs(result2));
}

template <typename T>
void testing_reduce()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Perform the Reduce operation
    T result = 0.0;
    EXPECT_NO_THROW(result = vec.Reduce());

    // Validate the result
    T result2 = 0.0;
    for(int i = 0; i < vec.GetSize(); ++i)
    {
        result2 += vec[i];
    }
    EXPECT_EQ(result, result2);
}

template <typename T>
void testing_point_wise_mult()
{
    // Create two LocalVectors and allocate some size
    LocalVector<T> vec1 = getTestVector<T>(5);
    LocalVector<T> vec2 = getTestVector<T>(5);

    // Perform the PointwiseMult operation
    EXPECT_NO_THROW(vec1.PointWiseMult(vec2));

    // Validate the result
    for(int i = 0; i < vec1.GetSize(); ++i)
    {
        EXPECT_EQ(vec1[i], static_cast<T>((i + 1) * (i + 1)));
    }
}

template <typename T>
void testing_point_wise_mult_2()
{
    // Create two LocalVectors and allocate some size
    LocalVector<T> vec1 = getTestVector<T>(5);
    LocalVector<T> vec2 = getTestVector<T>(5);
    LocalVector<T> vec3 = getTestVector<T>(5);

    // Perform the PointwiseMult operation
    EXPECT_NO_THROW(vec1.PointWiseMult(vec2, vec3));

    // Validate the result
    for(int i = 0; i < vec1.GetSize(); ++i)
    {
        EXPECT_EQ(vec1[i], static_cast<T>((i + 1) * (i + 1)));
    }
}

template <typename T>
void testing_leave_data_ptr()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Pointer to data (initialized to NULL)
    T* data_ptr = nullptr;

    // Attempt to leave the data pointer
    EXPECT_NO_THROW(vec.LeaveDataPtr(&data_ptr));

    // Ensure the pointer is not NULL after the call
    EXPECT_NE(data_ptr, nullptr);

    // Ensure the vector is cleared
    EXPECT_EQ(vec.GetSize(), 0);
}

template <typename T>
void testing_sync()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Perform a synchronization operation
    EXPECT_NO_THROW(vec.Sync());

    // Validate that the vector is still accessible and contains the correct values
    EXPECT_EQ(vec.GetSize(), 5);
    EXPECT_EQ(vec[0], static_cast<T>(1.0));
    EXPECT_EQ(vec[1], static_cast<T>(2.0));
    EXPECT_EQ(vec[2], static_cast<T>(3.0));
    EXPECT_EQ(vec[3], static_cast<T>(4.0));
    EXPECT_EQ(vec[4], static_cast<T>(5.0));
}

template <typename T>
void testing_copy_from_data()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Pointer to data
    T data_ptr[5] = {0, 2, 4, 8, 16};

    // Attempt to copy from the data pointer
    EXPECT_NO_THROW(vec.CopyFromData(data_ptr));

    EXPECT_EQ(vec.GetSize(), 5);
    EXPECT_EQ(vec[0], static_cast<T>(0.0));
    EXPECT_EQ(vec[1], static_cast<T>(2.0));
    EXPECT_EQ(vec[2], static_cast<T>(4.0));
    EXPECT_EQ(vec[3], static_cast<T>(8.0));
    EXPECT_EQ(vec[4], static_cast<T>(16.0));
}

template <typename T>
void testing_copy_to_host_data()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Pointer to host data
    T host_data_ptr[5];

    // Attempt to copy to the host data pointer
    EXPECT_NO_THROW(vec.CopyToHostData(host_data_ptr));

    for(int i = 0; i < vec.GetSize(); ++i)
    {
        EXPECT_EQ(host_data_ptr[i], static_cast<T>(i + 1));
    }
}

template <typename T>
void testing_restriction()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Create a mapping vector (map fine indices to coarse indices)
    LocalVector<int> map;
    map.Allocate("MappingVector", 5);
    map[0] = 0;
    map[1] = 0;
    map[2] = 1;
    map[3] = 1;
    map[4] = 2;

    // Create a LocalVector to store the restriction
    LocalVector<T> restricted_vec = getTestVector<T>(5);

    // Perform the Restriction operation
    EXPECT_NO_THROW(restricted_vec.Restriction(vec, map));

    // Validate the result size (should match the number of unique coarse indices, here 3)
    EXPECT_EQ(restricted_vec.GetSize(), 5);

    // Optionally, check the restricted values
    EXPECT_EQ(restricted_vec[0], static_cast<T>(vec[0] + vec[1])); // sum of fine indices 0 and 1
    EXPECT_EQ(restricted_vec[1], static_cast<T>(vec[2] + vec[3])); // sum of fine indices 2 and 3
    EXPECT_EQ(restricted_vec[2], static_cast<T>(vec[4])); // fine index 4
}

template <typename T>
void testing_copy_from_with_offsets()
{
    // Create a source LocalVector and allocate some size
    LocalVector<T> src_vec = getTestVector<T>(5); // [1, 2, 3, 4, 5]

    // Create a destination LocalVector and allocate some size
    LocalVector<T> dst_vec;
    dst_vec.Allocate("DstVector", 5);

    // Fill destination with zeros
    dst_vec.Zeros();

    // Copy 3 elements from src_vec[1..3] to dst_vec[2..4]
    int64_t src_offset = 1;
    int64_t dst_offset = 2;
    int64_t size       = 3;

    EXPECT_NO_THROW(dst_vec.CopyFrom(src_vec, src_offset, dst_offset, size));

    // Validate the copied values
    // dst_vec should now be [0, 0, 2, 3, 4]
    EXPECT_EQ(dst_vec[0], static_cast<T>(0));
    EXPECT_EQ(dst_vec[1], static_cast<T>(0));
    EXPECT_EQ(dst_vec[2], static_cast<T>(2));
    EXPECT_EQ(dst_vec[3], static_cast<T>(3));
    EXPECT_EQ(dst_vec[4], static_cast<T>(4));
}

template <typename T>
void testing_set_data_ptr_null()
{
    // Create a LocalVector
    LocalVector<T> vec;

    // Pointer to data (initialized to NULL)
    T* data_ptr = NULL;

    // Attempt to set the data pointer
    EXPECT_NO_THROW(vec.SetDataPtr(&data_ptr, "TestVector", 0));

    // Ensure the pointer is still NULL after the call
    EXPECT_EQ(data_ptr, nullptr);

    // Ensure the vector is cleared
    EXPECT_EQ(vec.GetSize(), 0);
}

template <typename T>
void testing_info()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec;
    vec.Allocate("TestVector", 10);

    // Redirect standard output to a stringstream to capture the Info output
    std::stringstream output_stream;
    std::streambuf*   original_cout_buffer = std::cout.rdbuf();
    std::cout.rdbuf(output_stream.rdbuf());

    // Call the Info method
    vec.Info();

    // Restore the original standard output buffer
    std::cout.rdbuf(original_cout_buffer);

    // Get the captured output
    std::string output = output_stream.str();

    // Validate the output contains expected information
    EXPECT_NE(output.find("name=TestVector"), std::string::npos); // Check vector name
    EXPECT_NE(output.find("size=10"), std::string::npos); // Check vector size
    EXPECT_NE(output.find("prec=" + std::to_string(sizeof(T) * 8) + "bit"),
              std::string::npos); // Check data type
}

template <typename T>
void testing_sort()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec;
    vec.Allocate("TestVector", 5);

    LocalVector<T> sorted_vec;
    sorted_vec.Allocate("TestVector", 5);

    // Fill the vector with unsorted values
    vec[0] = static_cast<T>(3.0);
    vec[1] = static_cast<T>(1.0);
    vec[2] = static_cast<T>(5.0);
    vec[3] = static_cast<T>(2.0);
    vec[4] = static_cast<T>(4.0);

    // Call the Sort method
    EXPECT_NO_THROW(vec.Sort(&sorted_vec, nullptr));

    // Validate that the vector is sorted in ascending order
    EXPECT_EQ(sorted_vec[0], static_cast<T>(1.0));
    EXPECT_EQ(sorted_vec[1], static_cast<T>(2.0));
    EXPECT_EQ(sorted_vec[2], static_cast<T>(3.0));
    EXPECT_EQ(sorted_vec[3], static_cast<T>(4.0));
    EXPECT_EQ(sorted_vec[4], static_cast<T>(5.0));
}

template <typename T>
void testing_permute_backward()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Create a permutation vector
    LocalVector<int> permutation;
    permutation.Allocate("PermutationVector", 5);
    permutation[0] = 4; // Move element at index 4 to index 0
    permutation[1] = 2; // Move element at index 2 to index 1
    permutation[2] = 0; // Move element at index 0 to index 2
    permutation[3] = 3; // Keep element at index 3 in place
    permutation[4] = 1; // Move element at index 1 to index 4

    // Call the PermuteBackward method
    EXPECT_NO_THROW(vec.PermuteBackward(permutation));

    // Validate the permuted vector
    EXPECT_EQ(vec[0], static_cast<T>(5.0)); // Element at index 4
    EXPECT_EQ(vec[1], static_cast<T>(3.0)); // Element at index 2
    EXPECT_EQ(vec[2], static_cast<T>(1.0)); // Element at index 0
    EXPECT_EQ(vec[3], static_cast<T>(4.0)); // Element at index 3
    EXPECT_EQ(vec[4], static_cast<T>(2.0)); // Element at index 1
}

template <typename T>
void testing_inclusive_sum()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> input_vec;
    input_vec.Allocate("InputVector", 5);

    // Fill the input vector with values
    input_vec[0] = static_cast<T>(1.0);
    input_vec[1] = static_cast<T>(2.0);
    input_vec[2] = static_cast<T>(3.0);
    input_vec[3] = static_cast<T>(4.0);
    input_vec[4] = static_cast<T>(5.0);

    // Create a LocalVector to store the result
    LocalVector<T> result_vec;
    result_vec.Allocate("ResultVector", 5);

    // Call the InclusiveSum method
    EXPECT_NO_THROW(result_vec.InclusiveSum(input_vec));

    // Validate the result
    EXPECT_EQ(result_vec[0], static_cast<T>(1.0)); // 1
    EXPECT_EQ(result_vec[1], static_cast<T>(3.0)); // 1 + 2
    EXPECT_EQ(result_vec[2], static_cast<T>(6.0)); // 1 + 2 + 3
    EXPECT_EQ(result_vec[3], static_cast<T>(10.0)); // 1 + 2 + 3 + 4
    EXPECT_EQ(result_vec[4], static_cast<T>(15.0)); // 1 + 2 + 3 + 4 + 5
}

template <typename T>
void testing_power()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>();

    // Apply the Power method with power = 2.0
    EXPECT_NO_THROW(vec.Power(2.0));

    // Validate the results
    for(int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>((i + 1) * (i + 1))); // Squared values
    }

    // Apply the Power method with power = 0.5 (square root)
    EXPECT_NO_THROW(vec.Power(0.5));

    // Validate the results
    for(int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>(i + 1)); // Original values restored
    }
}

template <typename T>
void testing_scale_add_scale()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Create another LocalVector for the operation
    LocalVector<T> x;
    x.Allocate("XVector", 5);

    // Fill the second vector with values
    x[0] = 5.0f;
    x[1] = 4.0f;
    x[2] = 3.0f;
    x[3] = 2.0f;
    x[4] = 1.0f;

    // Define scaling factors
    T alpha = 2.0f;
    T beta  = 3.0f;

    // Perform the ScaleAddScale operation
    EXPECT_NO_THROW(vec.ScaleAddScale(alpha, x, beta));

    // Validate the result
    EXPECT_EQ(vec[0], static_cast<T>(alpha * 1.0 + beta * 5.0)); // 2 * 1 + 3 * 5 = 17
    EXPECT_EQ(vec[1], static_cast<T>(alpha * 2.0 + beta * 4.0)); // 2 * 2 + 3 * 4 = 16
    EXPECT_EQ(vec[2], static_cast<T>(alpha * 3.0 + beta * 3.0)); // 2 * 3 + 3 * 3 = 15
    EXPECT_EQ(vec[3], static_cast<T>(alpha * 4.0 + beta * 2.0)); // 2 * 4 + 3 * 2 = 14
    EXPECT_EQ(vec[4], static_cast<T>(alpha * 5.0 + beta * 1.0)); // 2 * 5 + 3 * 1 = 13
}

template <typename T>
void testing_operator_index_const()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Create a const reference to the vector
    const LocalVector<T>& const_vec = vec;

    // Validate access to elements using the const operator[]
    EXPECT_EQ(const_vec[0], static_cast<T>(1.0)); // First element
    EXPECT_EQ(const_vec[1], static_cast<T>(2.0)); // Second element
    EXPECT_EQ(const_vec[2], static_cast<T>(3.0)); // Third element
    EXPECT_EQ(const_vec[3], static_cast<T>(4.0)); // Fourth element
    EXPECT_EQ(const_vec[4], static_cast<T>(5.0)); // Fifth element
}

template <typename T>
void testing_read_file_ascii()
{
    // Create a temporary ASCII file with test data
    std::string   filename = "test_vector.txt";
    std::ofstream file(filename);
    ASSERT_TRUE(file.is_open());
    file << "1.0\n";
    file << "2.0\n";
    file << "3.0\n";
    file << "4.0\n";
    file << "5.0\n";
    file.close();

    // Create a LocalVector to read the data into
    LocalVector<T> vec;

    // Call the ReadFileASCII method
    EXPECT_NO_THROW(vec.ReadFileASCII(filename));

    // Validate the vector size
    EXPECT_EQ(vec.GetSize(), 5);

    // Validate the values in the vector
    EXPECT_EQ(vec[0], static_cast<T>(1.0)); // First element
    EXPECT_EQ(vec[1], static_cast<T>(2.0)); // Second element
    EXPECT_EQ(vec[2], static_cast<T>(3.0)); // Third element
    EXPECT_EQ(vec[3], static_cast<T>(4.0)); // Fourth element
    EXPECT_EQ(vec[4], static_cast<T>(5.0)); // Fifth element

    // Clean up the temporary file
    std::remove(filename.c_str());
}

template <typename T>
void testing_copy_from_async()
{
    // Create a source LocalVector and allocate some size
    LocalVector<T> src_vec = getTestVector<T>(5);

    // Create a destination LocalVector
    LocalVector<T> dest_vec;

    // Perform the asynchronous copy
    EXPECT_NO_THROW(dest_vec.CopyFromAsync(src_vec));

    // Synchronize to ensure the copy is complete
    EXPECT_NO_THROW(_rocalution_sync());

    // Validate the copied values
    EXPECT_EQ(dest_vec.GetSize(), src_vec.GetSize());
    EXPECT_EQ(dest_vec[0], static_cast<T>(1.0));
    EXPECT_EQ(dest_vec[1], static_cast<T>(2.0));
    EXPECT_EQ(dest_vec[2], static_cast<T>(3.0));
    EXPECT_EQ(dest_vec[3], static_cast<T>(4.0));
    EXPECT_EQ(dest_vec[4], static_cast<T>(5.0));
}

template <typename T>
void testing_move_to_accelerator_async()
{
    // Create a source LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Perform the asynchronous move to the accelerator
    EXPECT_NO_THROW(vec.MoveToAcceleratorAsync());

    // Synchronize to ensure the move is complete
    EXPECT_NO_THROW(_rocalution_sync());

    // Validate that the vector is still accessible and contains the correct values
    EXPECT_EQ(vec.GetSize(), 5);
    EXPECT_EQ(vec[0], static_cast<T>(1.0));
    EXPECT_EQ(vec[1], static_cast<T>(2.0));
    EXPECT_EQ(vec[2], static_cast<T>(3.0));
    EXPECT_EQ(vec[3], static_cast<T>(4.0));
    EXPECT_EQ(vec[4], static_cast<T>(5.0));
}

template <typename T>
void testing_copy_from_host_data()
{
    // clang-format off
    // Create a host array with test data
    T   host_data[] = {static_cast<T>(1.0),
                       static_cast<T>(2.0),
                       static_cast<T>(3.0),
                       static_cast<T>(4.0),
                       static_cast<T>(5.0)};
    // clang-format on
    int size = 5;

    // Create a LocalVector and allocate the same size
    LocalVector<T> vec;
    vec.Allocate("TestVector", size);

    // Copy data from the host array to the LocalVector
    EXPECT_NO_THROW(vec.CopyFromHostData(host_data));

    // Validate the vector size
    EXPECT_EQ(vec.GetSize(), size);

    // Validate the copied values
    EXPECT_EQ(vec[0], static_cast<T>(1.0));
    EXPECT_EQ(vec[1], static_cast<T>(2.0));
    EXPECT_EQ(vec[2], static_cast<T>(3.0));
    EXPECT_EQ(vec[3], static_cast<T>(4.0));
    EXPECT_EQ(vec[4], static_cast<T>(5.0));
}

template <typename T>
void testing_copy_to_data()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Create a host array to store the copied data
    T host_data[5] = {static_cast<T>(0.0)};

    // Copy data from the LocalVector to the host array
    EXPECT_NO_THROW(vec.CopyToData(host_data));

    // Validate the copied values
    EXPECT_EQ(host_data[0], static_cast<T>(1.0));
    EXPECT_EQ(host_data[1], static_cast<T>(2.0));
    EXPECT_EQ(host_data[2], static_cast<T>(3.0));
    EXPECT_EQ(host_data[3], static_cast<T>(4.0));
    EXPECT_EQ(host_data[4], static_cast<T>(5.0));
}

// Helper function to read the contents of a file into a string
std::string ReadFileContents(const std::string& filename)
{
    std::ifstream     file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

template <typename T>
void testing_write_file_ascii()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Define the filename for the test
    std::string filename = "test_vector.txt";

    // Write the vector to the file
    EXPECT_NO_THROW(vec.WriteFileASCII(filename));

    // Read the contents of the file
    std::string file_contents = ReadFileContents(filename);

    // Define the expected output
    std::string expected_output
        = "1.000000e+00\n2.000000e+00\n3.000000e+00\n4.000000e+00\n5.000000e+00\n";

    // Validate the file contents
    EXPECT_EQ(file_contents, expected_output);

    // Clean up the temporary file
    std::remove(filename.c_str());
}

template <typename T>
void testing_scale_add_scale_with_offsets()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(10);

    // Create another LocalVector for the operation
    LocalVector<T> x;
    x.Allocate("XVector", 10);

    // Fill the second vector with values
    for(int i = 0; i < 10; ++i)
    {
        x[i] = static_cast<T>(10 - i); // Values: 10.0, 9.0, ..., 1.0
    }

    // Define scaling factors and offsets
    T       alpha      = 2.0f;
    T       beta       = 3.0f;
    int64_t src_offset = 2; // Start from index 2 in x
    int64_t dst_offset = 4; // Start from index 4 in vec
    int64_t size       = 3; // Operate on 3 elements

    // Perform the ScaleAddScale operation
    EXPECT_NO_THROW(vec.ScaleAddScale(alpha, x, beta, src_offset, dst_offset, size));

    // Validate the results
    EXPECT_EQ(vec[4], static_cast<T>(alpha * 5.0 + beta * 8.0)); // 2 * 5 + 3 * 8 = 34
    EXPECT_EQ(vec[5], static_cast<T>(alpha * 6.0 + beta * 7.0)); // 2 * 6 + 3 * 7 = 33
    EXPECT_EQ(vec[6], static_cast<T>(alpha * 7.0 + beta * 6.0)); // 2 * 7 + 3 * 6 = 32

    // Ensure other elements remain unchanged
    EXPECT_EQ(vec[0], static_cast<T>(1.0));
    EXPECT_EQ(vec[1], static_cast<T>(2.0));
    EXPECT_EQ(vec[2], static_cast<T>(3.0));
    EXPECT_EQ(vec[3], static_cast<T>(4.0));
    EXPECT_EQ(vec[7], static_cast<T>(8.0));
    EXPECT_EQ(vec[8], static_cast<T>(9.0));
    EXPECT_EQ(vec[9], static_cast<T>(10.0));
}

template <typename T>
void testing_inclusive_sum_void()
{
    LocalVector<T> vec = getTestVector<T>(5);

    EXPECT_EQ(vec.InclusiveSum(), static_cast<T>(15.0)); // 1 + 2 + 3 + 4 + 5 = 15
}

template <typename T>
void testing_inclusive_sum_with_input()
{
    LocalVector<T> input_vec = getTestVector<T>(5);

    LocalVector<T> result_vec;
    result_vec.Allocate("ResultVector", 5);

    EXPECT_NO_THROW(result_vec.InclusiveSum(input_vec));

    EXPECT_EQ(result_vec[0], static_cast<T>(1.0f)); // 1
    EXPECT_EQ(result_vec[1], static_cast<T>(3.0f)); // 1 + 2
    EXPECT_EQ(result_vec[2], static_cast<T>(6.0f)); // 1 + 2 + 3
    EXPECT_EQ(result_vec[3], static_cast<T>(10.0f)); // 1 + 2 + 3 + 4
    EXPECT_EQ(result_vec[4], static_cast<T>(15.0f)); // 1 + 2 + 3 + 4 + 5
}

template <typename T>
void testing_exclusive_sum_void()
{
    LocalVector<T> vec = getTestVector<T>(5);

    EXPECT_EQ(vec.ExclusiveSum(), static_cast<T>(10.0)); // 0 + 1 + 2 + 3 + 4 = 10
}

template <typename T>
void testing_exclusive_sum_with_input()
{
    LocalVector<T> input_vec = getTestVector<T>(5);

    LocalVector<T> result_vec;
    result_vec.Allocate("ResultVector", 5);

    EXPECT_NO_THROW(result_vec.ExclusiveSum(input_vec));

    EXPECT_EQ(result_vec[0], static_cast<T>(0.0)); // 0
    EXPECT_EQ(result_vec[1], static_cast<T>(1.0)); // 0 + 1
    EXPECT_EQ(result_vec[2], static_cast<T>(3.0)); // 0 + 1 + 2
    EXPECT_EQ(result_vec[3], static_cast<T>(6.0)); // 0 + 1 + 2 + 3
    EXPECT_EQ(result_vec[4], static_cast<T>(10.0)); // 0 + 1 + 2 + 3 + 4
}

template <typename T>
void testing_asum()
{
    LocalVector<T> vec = getTestVector<T>(5);

    EXPECT_EQ(vec.Asum(), static_cast<T>(15.0)); // |1| + |2| + |3| + |4| + |5| = 15
}

template <typename T>
void testing_amax()
{
    LocalVector<T> vec = getTestVector<T>(5);

    T   max_value = 0.0;
    int max_index = vec.Amax(max_value);

    EXPECT_EQ(max_index, 4); // Index of the maximum absolute value
    EXPECT_EQ(max_value, static_cast<T>(5.0)); // Maximum absolute value
}

// Helper function to read binary file contents into a vector
template <typename ValueType>
std::vector<ValueType> ReadBinaryFile(const std::string& filename, size_t& size_out)
{
    std::ifstream file(filename, std::ios::binary);

    // Read the size of the vector
    size_t size = 0;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    size_out = size;

    // Read the vector data
    std::vector<ValueType> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(ValueType));
    file.close();

    return data;
}

template <typename T>
void testing_prolongation()
{
    // Create a coarse-level LocalVector and allocate some size
    LocalVector<T> vec_coarse;
    vec_coarse.Allocate("CoarseVector", 3);

    // Fill the coarse-level vector with values
    vec_coarse[0] = static_cast<T>(1.0);
    vec_coarse[1] = static_cast<T>(2.0);
    vec_coarse[2] = static_cast<T>(3.0);

    // Create a mapping vector
    LocalVector<int> map;
    map.Allocate("MappingVector", 6);

    // Define the mapping (fine indices map to coarse indices)
    map[0] = 0;
    map[1] = 0;
    map[2] = 1;
    map[3] = 1;
    map[4] = 2;
    map[5] = 2;

    // Create a fine-level LocalVector to store the result
    LocalVector<T> vec_fine;
    vec_fine.Allocate("FineVector", 6);

    // Perform the prolongation operation
    EXPECT_NO_THROW(vec_fine.Prolongation(vec_coarse, map));

    // Validate the prolonged values
    EXPECT_EQ(vec_fine[0], static_cast<T>(1.0));
    EXPECT_EQ(vec_fine[1], static_cast<T>(1.0));
    EXPECT_EQ(vec_fine[2], static_cast<T>(2.0));
    EXPECT_EQ(vec_fine[3], static_cast<T>(2.0));
    EXPECT_EQ(vec_fine[4], static_cast<T>(3.0));
    EXPECT_EQ(vec_fine[5], static_cast<T>(3.0));
}

template <typename T>
void testing_get_index_values()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Create an index vector
    LocalVector<int> index;
    index.Allocate("IndexVector", 3);
    index[0] = 0; // First element
    index[1] = 2; // Third element
    index[2] = 4; // Fifth element

    // Create a LocalVector to store the values
    LocalVector<T> values;
    values.Allocate("ValuesVector", 3);

    // Call the GetIndexValues method
    EXPECT_NO_THROW(vec.GetIndexValues(index, &values));

    // Validate the retrieved values
    EXPECT_EQ(values[0], static_cast<T>(1.0)); // First element
    EXPECT_EQ(values[1], static_cast<T>(3.0)); // Third element
    EXPECT_EQ(values[2], static_cast<T>(5.0)); // Fifth element
}

template <typename T>
void testing_set_index_values()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec;
    vec.Allocate("TestVector", 5);

    // Create an index vector
    LocalVector<int> index;
    index.Allocate("IndexVector", 3);
    index[0] = 0; // First element
    index[1] = 2; // Third element
    index[2] = 4; // Fifth element

    // Create a LocalVector to store the values
    LocalVector<T> values;
    values.Allocate("ValuesVector", 3);
    values[0] = static_cast<T>(100.0); // First element
    values[1] = static_cast<T>(200.0); // Third element
    values[2] = static_cast<T>(300.0); // Fifth element

    // Call the SetIndexValues method
    EXPECT_NO_THROW(vec.SetIndexValues(index, values));

    // Validate the set values
    EXPECT_EQ(vec[0], static_cast<T>(100.0)); // First element
    EXPECT_EQ(vec[2], static_cast<T>(200.0)); // Third element
    EXPECT_EQ(vec[4], static_cast<T>(300.0)); // Fifth element
}

template <typename T>
void testing_add_index_values()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec;
    vec.Allocate("TestVector", 5);

    // Fill the vector with initial values
    vec[0] = static_cast<T>(10.0);
    vec[2] = static_cast<T>(20.0);
    vec[4] = static_cast<T>(30.0);

    // Create an index vector
    LocalVector<int> index;
    index.Allocate("IndexVector", 3);
    index[0] = 0; // First element
    index[1] = 2; // Third element
    index[2] = 4; // Fifth element

    // Create a LocalVector to store the values to be added
    LocalVector<T> values;
    values.Allocate("ValuesVector", 3);
    values[0] = static_cast<T>(5.0); // First element
    values[1] = static_cast<T>(10.0); // Third element
    values[2] = static_cast<T>(15.0); // Fifth element

    // Call the AddIndexValues method
    EXPECT_NO_THROW(vec.AddIndexValues(index, values));

    // Validate the updated values
    EXPECT_EQ(vec[0], static_cast<T>(15.0)); // First element: 10 + 5
    EXPECT_EQ(vec[2], static_cast<T>(30.0)); // Third element: 20 + 10
    EXPECT_EQ(vec[4], static_cast<T>(45.0)); // Fifth element: 30 + 15
}

template <typename T>
void testing_get_continuous_values()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Create an array to store the continuous values
    T values[3];

    // Call the GetContinuousValues method
    EXPECT_NO_THROW(vec.GetContinuousValues(1, 4, values));

    // Validate the retrieved values
    EXPECT_EQ(values[0], static_cast<T>(2.0)); // Second element
    EXPECT_EQ(values[1], static_cast<T>(3.0)); // Third element
    EXPECT_EQ(values[2], static_cast<T>(4.0)); // Fourth element
}

template <typename T>
void testing_set_continuous_values()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec;
    vec.Allocate("TestVector", 5);

    // Create an array of values to set
    T values[3] = {100.0, 200.0, 300.0};

    // Call the SetContinuousValues method
    EXPECT_NO_THROW(vec.SetContinuousValues(1, 4, values));

    // Validate the set values
    EXPECT_EQ(vec[1], static_cast<T>(100.0)); // Second element
    EXPECT_EQ(vec[2], static_cast<T>(200.0)); // Third element
    EXPECT_EQ(vec[3], static_cast<T>(300.0)); // Fourth element
}

template <typename T>
void testing_extract_coarse_mapping()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec;
    vec.Allocate("TestVector", 5);

    // Create an index vector
    int index[3] = {0, 2, 4};
    int size     = 0;
    int map[3];

    // Call the ExtractCoarseMapping method
    EXPECT_NO_THROW(vec.ExtractCoarseMapping(0, 5, index, 3, &size, map));

    // Validate the extracted mapping
    EXPECT_EQ(size, 5);
    EXPECT_EQ(map[0], 0);
    EXPECT_EQ(map[1], 0);
    EXPECT_EQ(map[2], 0);
}

template <typename T>
void testing_extract_coarse_boundary()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec;
    vec.Allocate("TestVector", 5);

    // Create an index vector
    int index[3] = {0, 2, 4};
    int size     = 0;
    int boundary[3];

    // Call the ExtractCoarseBoundary method
    EXPECT_NO_THROW(vec.ExtractCoarseBoundary(0, 5, index, 3, &size, boundary));

    // Validate the extracted boundary
    EXPECT_EQ(size, 1);
    EXPECT_EQ(boundary[0], 0);
}

template <typename T>
void testing_move_to_host_async()
{
    // Create a LocalVector and allocate some size
    LocalVector<T> vec = getTestVector<T>(5);

    // Move the vector to the accelerator
    EXPECT_NO_THROW(vec.MoveToAccelerator());

    // Perform the asynchronous move to the host
    EXPECT_NO_THROW(vec.MoveToHostAsync());

    // Synchronize to ensure the move is complete
    EXPECT_NO_THROW(_rocalution_sync());

    // Validate that the vector is still accessible and contains the correct values
    EXPECT_EQ(vec.GetSize(), 5);
    for(int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(vec[i], static_cast<T>(i + 1));
    }
}

#endif // TESTING_LOCAL_VECTOR_HPP
