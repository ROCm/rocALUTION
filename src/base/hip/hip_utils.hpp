#ifndef ROCALUTION_HIP_HIP_UTILS_HPP_
#define ROCALUTION_HIP_HIP_UTILS_HPP_

#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "backend_hip.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_vector.hpp"
#include "hip_allocate_free.hpp"

#include <stdlib.h>
#include <complex>

#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <rocsparse.h>

#define ROCBLAS_HANDLE(handle) *static_cast<rocblas_handle*>(handle)
#define ROCSPARSE_HANDLE(handle) *static_cast<rocsparse_handle*>(handle)

#define CHECK_HIP_ERROR(file, line) {                                   \
    hipError_t err_t;                                                   \
    if ((err_t = hipGetLastError() ) != hipSuccess) {                   \
      LOG_INFO("HIP error: " << hipGetErrorString(err_t));              \
      LOG_INFO("File: " << file << "; line: " << line);                 \
      exit(1);                                                          \
    }                                                                   \
  }   

#define CHECK_ROCBLAS_ERROR(stat_t, file, line)             \
{                                                           \
    if (stat_t != rocblas_status_success)                   \
    {                                                       \
        LOG_INFO("rocBLAS error " << stat_t);               \
        if (stat_t == rocblas_status_invalid_handle)        \
            LOG_INFO("rocblas_status_invalid_handle");      \
        if (stat_t == rocblas_status_not_implemented)       \
            LOG_INFO("rocblas_status_not_implemented");     \
        if (stat_t == rocblas_status_invalid_pointer)       \
            LOG_INFO("rocblas_status_invalid_pointer");     \
        if (stat_t == rocblas_status_invalid_size)          \
            LOG_INFO("rocblas_status_invalid_size");        \
        if (stat_t == rocblas_status_memory_error)          \
            LOG_INFO("rocblas_status_memory_error");        \
        if (stat_t == rocblas_status_internal_error)        \
            LOG_INFO("rocblas_status_internal_error");      \
        LOG_INFO("File: " << file << "; line: " << line);   \
        exit(1);                                            \
    }                                                       \
}

#define CHECK_ROCSPARSE_ERROR(status, file, line)           \
{                                                           \
    if (status != rocsparse_status_success)                 \
    {                                                       \
        LOG_INFO("rocSPARSE error " << status);             \
        if (status == rocsparse_status_invalid_handle)      \
            LOG_INFO("rocsparse_status_invalid_handle");    \
        if (status == rocsparse_status_not_implemented);    \
            LOG_INFO("rocsparse_status_not_implemented");   \
        if (status == rocsparse_status_invalid_pointer)     \
            LOG_INFO("rocsparse_status_invalid_pointer");   \
        if (status == rocsparse_status_invalid_size)        \
            LOG_INFO("rocsparse_status_invalid_size");      \
        if (status == rocsparse_status_memory_error)        \
            LOG_INFO("rocsparse_status_memory_error");      \
        if (status == rocsparse_status_internal_error)      \
            LOG_INFO("rocsparse_status_internal_error");    \
        if (status == rocsparse_status_invalid_value)       \
            LOG_INFO("rocsparse_status_invalid_value");     \
        if (status == rocsparse_status_arch_mismatch)       \
            LOG_INFO("rocsparse_status_arch_mismatch");     \
        LOG_INFO("File: " << file << "; line: " << line);   \
        exit(1);                                            \
    }                                                       \
}   

namespace rocalution {

// Type traits to cast STL types to HIP types
template <typename ValueType>
struct HIPType {
  typedef ValueType Type;
};

#ifdef SUPPORT_COMPLEX
template <>
struct HIPType<std::complex<float> > {
  typedef cuFloatComplex Type;
};

template <>
struct HIPType<std::complex<double> > {
  typedef cuDoubleComplex Type;
};
#endif

template <typename IndexType, unsigned int BLOCK_SIZE>
bool cum_sum( IndexType*  dst,
              const IndexType*  src,
              const IndexType   numElems) {

  hipMemset(dst, 0, (numElems+1)*sizeof(IndexType));
  CHECK_HIP_ERROR(__FILE__, __LINE__);

  IndexType* d_temp = NULL;
  allocate_hip<IndexType>(numElems+1, &d_temp);

  hipMemset(d_temp, 0, (numElems+1)*sizeof(IndexType));
  CHECK_HIP_ERROR(__FILE__, __LINE__);

  hipLaunchKernelGGL((kernel_red_partial_sum<IndexType, BLOCK_SIZE>),
                     dim3(numElems/BLOCK_SIZE+1), dim3(BLOCK_SIZE), 0, 0,
                     dst+1, src, numElems);
  CHECK_HIP_ERROR(__FILE__,__LINE__);

  hipLaunchKernelGGL((kernel_red_recurse<IndexType>),
                     dim3(numElems/(BLOCK_SIZE*BLOCK_SIZE)+1), dim3(BLOCK_SIZE), 0, 0,
                     d_temp, dst+BLOCK_SIZE, BLOCK_SIZE, (numElems+1));
  CHECK_HIP_ERROR(__FILE__,__LINE__);

  hipLaunchKernelGGL((kernel_red_extrapolate<IndexType>),
                     dim3(numElems/(BLOCK_SIZE*BLOCK_SIZE)+1), dim3(BLOCK_SIZE), 0, 0,
                     dst+1, d_temp, src, numElems);
  CHECK_HIP_ERROR(__FILE__,__LINE__);

  free_hip<int>(&d_temp);

  return true;

}

template <typename IndexType, typename ValueType, unsigned int WARP_SIZE, unsigned int BLOCK_SIZE>
void reduce_hip(const int size, const ValueType *src, ValueType *reduce, ValueType *host_buffer, ValueType *device_buffer) {

  *reduce = (ValueType) 0;

  if (size > 0) {

    dim3 BlockSize(BLOCK_SIZE);
    dim3 GridSize(WARP_SIZE);

    hipLaunchKernelGGL((kernel_reduce<WARP_SIZE, BLOCK_SIZE, ValueType, IndexType>),
                       GridSize, BlockSize, 0, 0,
                       size, src, device_buffer);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    hipMemcpy(host_buffer,                 // dst
              device_buffer,               // src
              WARP_SIZE*sizeof(ValueType), // size
              hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    for (unsigned int i=0; i<WARP_SIZE; ++i) {
      *reduce += host_buffer[i];
    }

  }

}

}

#endif // ROCALUTION_HIP_HIP_UTILS_HPP_
