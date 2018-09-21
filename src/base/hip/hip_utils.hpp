/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HIP_HIP_UTILS_HPP_
#define ROCALUTION_HIP_HIP_UTILS_HPP_

#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "backend_hip.hpp"

#include <complex>
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <rocsparse.h>

#define ROCBLAS_HANDLE(handle) *static_cast<rocblas_handle*>(handle)
#define ROCSPARSE_HANDLE(handle) *static_cast<rocsparse_handle*>(handle)

#define CHECK_HIP_ERROR(file, line)                              \
    {                                                            \
        hipError_t err_t;                                        \
        if((err_t = hipGetLastError()) != hipSuccess)            \
        {                                                        \
            LOG_INFO("HIP error: " << hipGetErrorString(err_t)); \
            LOG_INFO("File: " << file << "; line: " << line);    \
            exit(1);                                             \
        }                                                        \
    }

#define CHECK_ROCBLAS_ERROR(stat_t, file, line)               \
    {                                                         \
        if(stat_t != rocblas_status_success)                  \
        {                                                     \
            LOG_INFO("rocBLAS error " << stat_t);             \
            if(stat_t == rocblas_status_invalid_handle)       \
                LOG_INFO("rocblas_status_invalid_handle");    \
            if(stat_t == rocblas_status_not_implemented)      \
                LOG_INFO("rocblas_status_not_implemented");   \
            if(stat_t == rocblas_status_invalid_pointer)      \
                LOG_INFO("rocblas_status_invalid_pointer");   \
            if(stat_t == rocblas_status_invalid_size)         \
                LOG_INFO("rocblas_status_invalid_size");      \
            if(stat_t == rocblas_status_memory_error)         \
                LOG_INFO("rocblas_status_memory_error");      \
            if(stat_t == rocblas_status_internal_error)       \
                LOG_INFO("rocblas_status_internal_error");    \
            LOG_INFO("File: " << file << "; line: " << line); \
            exit(1);                                          \
        }                                                     \
    }

#define CHECK_ROCSPARSE_ERROR(status, file, line)             \
    {                                                         \
        if(status != rocsparse_status_success)                \
        {                                                     \
            LOG_INFO("rocSPARSE error " << status);           \
            if(status == rocsparse_status_invalid_handle)     \
                LOG_INFO("rocsparse_status_invalid_handle");  \
            if(status == rocsparse_status_not_implemented)    \
                LOG_INFO("rocsparse_status_not_implemented"); \
            if(status == rocsparse_status_invalid_pointer)    \
                LOG_INFO("rocsparse_status_invalid_pointer"); \
            if(status == rocsparse_status_invalid_size)       \
                LOG_INFO("rocsparse_status_invalid_size");    \
            if(status == rocsparse_status_memory_error)       \
                LOG_INFO("rocsparse_status_memory_error");    \
            if(status == rocsparse_status_internal_error)     \
                LOG_INFO("rocsparse_status_internal_error");  \
            if(status == rocsparse_status_invalid_value)      \
                LOG_INFO("rocsparse_status_invalid_value");   \
            if(status == rocsparse_status_arch_mismatch)      \
                LOG_INFO("rocsparse_status_arch_mismatch");   \
            LOG_INFO("File: " << file << "; line: " << line); \
            exit(1);                                          \
        }                                                     \
    }

namespace rocalution {

// Type traits to cast STL types to HIP types
template <typename ValueType>
struct HIPType
{
    typedef ValueType Type;
};

#ifdef SUPPORT_COMPLEX
template <>
struct HIPType<std::complex<float>>
{
    typedef hipFloatComplex Type;
};

template <>
struct HIPType<std::complex<double>>
{
    typedef hipDoubleComplex Type;
};
#endif

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_UTILS_HPP_
