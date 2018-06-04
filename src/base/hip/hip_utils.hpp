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
#include <hipblas.h>
#include <hipsparse.h>

#define HIPBLAS_HANDLE(handle) *static_cast<hipblasHandle_t*>(handle)
#define HIPSPARSE_HANDLE(handle) *static_cast<hipsparseHandle_t*>(handle)

#define CHECK_HIP_ERROR(file, line) {                                   \
    hipError_t err_t;                                                   \
    if ((err_t = hipGetLastError() ) != hipSuccess) {                   \
      LOG_INFO("HIP error: " << hipGetErrorString(err_t));              \
      LOG_INFO("File: " << file << "; line: " << line);                 \
      exit(1);                                                          \
    }                                                                   \
  }   

#define CHECK_HIPBLAS_ERROR(stat_t, file, line) {                       \
  if (stat_t  != HIPBLAS_STATUS_SUCCESS) {                              \
  LOG_INFO("hipBLAS error " << stat_t);                                 \
  if (stat_t == HIPBLAS_STATUS_NOT_INITIALIZED)                         \
    LOG_INFO("HIPBLAS_STATUS_NOT_INITIALIZED");                         \
  if (stat_t == HIPBLAS_STATUS_ALLOC_FAILED)                            \
    LOG_INFO("HIPBLAS_STATUS_ALLOC_FAILED");                            \
  if (stat_t == HIPBLAS_STATUS_INVALID_VALUE)                           \
    LOG_INFO("HIPBLAS_STATUS_INVALID_VALUE");                           \
  if (stat_t == HIPBLAS_STATUS_MAPPING_ERROR)                           \
    LOG_INFO("HIPBLAS_STATUS_MAPPING_ERROR");                           \
  if (stat_t == HIPBLAS_STATUS_EXECUTION_FAILED)                        \
    LOG_INFO("HIPBLAS_STATUS_EXECUTION_FAILED");                        \
  if (stat_t == HIPBLAS_STATUS_INTERNAL_ERROR)                          \
    LOG_INFO("HIPBLAS_STATUS_INTERNAL_ERROR");                          \
  if (stat_t == HIPBLAS_STATUS_NOT_SUPPORTED)                           \
    LOG_INFO("HIPBLAS_STATUS_NOT_SUPPORTED");                           \
  LOG_INFO("File: " << file << "; line: " << line);                     \
  exit(1);                                                              \
  }                                                                     \
}

#define CHECK_HIPSPARSE_ERROR(stat_t, file, line) {                     \
  if (stat_t  != HIPSPARSE_STATUS_SUCCESS) {                            \
  LOG_INFO("hipSPARSE error " << stat_t);                               \
  if (stat_t == HIPSPARSE_STATUS_NOT_INITIALIZED)                       \
    LOG_INFO("HIPSPARSE_STATUS_NOT_INITIALIZED");                       \
  if (stat_t == HIPSPARSE_STATUS_ALLOC_FAILED)                          \
    LOG_INFO("HIPSPARSE_STATUS_ALLOC_FAILED");                          \
  if (stat_t == HIPSPARSE_STATUS_INVALID_VALUE)                         \
    LOG_INFO("HIPSPARSE_STATUS_INVALID_VALUE");                         \
  if (stat_t == HIPSPARSE_STATUS_ARCH_MISMATCH)                         \
    LOG_INFO("HIPSPARSE_STATUS_ARCH_MISMATCH");                         \
  if (stat_t == HIPSPARSE_STATUS_MAPPING_ERROR)                         \
    LOG_INFO("HIPSPARSE_STATUS_MAPPING_ERROR");                         \
  if (stat_t == HIPSPARSE_STATUS_EXECUTION_FAILED)                      \
    LOG_INFO("HIPSPARSE_STATUS_EXECUTION_FAILED");                      \
  if (stat_t == HIPSPARSE_STATUS_INTERNAL_ERROR)                        \
    LOG_INFO("HIPSPARSE_STATUS_INTERNAL_ERROR");                        \
  if (stat_t == HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)             \
    LOG_INFO("HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED");             \
  LOG_INFO("File: " << file << "; line: " << line);                     \
  exit(1);                                                              \
  }                                                                     \
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
