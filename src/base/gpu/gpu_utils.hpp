#ifndef PARALUTION_GPU_GPU_UTILS_HPP_
#define PARALUTION_GPU_GPU_UTILS_HPP_

#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "backend_gpu.hpp"
#include "cuda_kernels_general.hpp"
#include "cuda_kernels_vector.hpp"
#include "gpu_allocate_free.hpp"

#include <stdlib.h>
#include <complex>

#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#define CUBLAS_HANDLE(handle) *static_cast<cublasHandle_t*>(handle)
#define CUSPARSE_HANDLE(handle) *static_cast<cusparseHandle_t*>(handle)

#define CHECK_CUDA_ERROR(file, line) {                                  \
    cudaError_t err_t;                                                  \
    if ((err_t = cudaGetLastError() ) != cudaSuccess) {                 \
      LOG_INFO("Cuda error: " << cudaGetErrorString(err_t));            \
      LOG_INFO("File: " << file << "; line: " << line);                 \
      exit(1);                                                          \
    }                                                                   \
  }   

#define CHECK_CUBLAS_ERROR(stat_t, file, line) {                        \
  if (stat_t  != CUBLAS_STATUS_SUCCESS) {                               \
  LOG_INFO("Cublas error!");                                            \
  if (stat_t == CUBLAS_STATUS_NOT_INITIALIZED)                          \
    LOG_INFO("CUBLAS_STATUS_NOT_INITIALIZED");                          \
  if (stat_t == CUBLAS_STATUS_ALLOC_FAILED)                             \
    LOG_INFO("CUBLAS_STATUS_ALLOC_FAILED");                             \
  if (stat_t == CUBLAS_STATUS_INVALID_VALUE)                            \
    LOG_INFO("CUBLAS_STATUS_INVALID_VALUE");                            \
  if (stat_t == CUBLAS_STATUS_ARCH_MISMATCH)                            \
    LOG_INFO("CUBLAS_STATUS_ARCH_MISMATCH");                            \
  if (stat_t == CUBLAS_STATUS_MAPPING_ERROR)                            \
    LOG_INFO("CUBLAS_STATUS_MAPPING_ERROR");                            \
  if (stat_t == CUBLAS_STATUS_EXECUTION_FAILED)                         \
    LOG_INFO("CUBLAS_STATUS_EXECUTION_FAILED");                         \
  if (stat_t == CUBLAS_STATUS_INTERNAL_ERROR)                           \
    LOG_INFO("CUBLAS_STATUS_INTERNAL_ERROR");                           \
  LOG_INFO("File: " << file << "; line: " << line);                     \
  exit(1);                                                              \
  }                                                                     \
  }   

#define CHECK_CUSPARSE_ERROR(stat_t, file, line) {                      \
  if (stat_t  != CUSPARSE_STATUS_SUCCESS) {                             \
  LOG_INFO("Cusparse error!");                                          \
  if (stat_t == CUSPARSE_STATUS_NOT_INITIALIZED)                        \
    LOG_INFO("CUSPARSE_STATUS_NOT_INITIALIZED");                        \
  if (stat_t == CUSPARSE_STATUS_ALLOC_FAILED)                           \
    LOG_INFO("CUSPARSE_STATUS_ALLOC_FAILED");                           \
  if (stat_t == CUSPARSE_STATUS_INVALID_VALUE)                          \
    LOG_INFO("CUSPARSE_STATUS_INVALID_VALUE");                          \
  if (stat_t == CUSPARSE_STATUS_ARCH_MISMATCH)                          \
    LOG_INFO("CUSPARSE_STATUS_ARCH_MISMATCH");                          \
  if (stat_t == CUSPARSE_STATUS_MAPPING_ERROR)                          \
    LOG_INFO("CUSPARSE_STATUS_MAPPING_ERROR");                          \
  if (stat_t == CUSPARSE_STATUS_EXECUTION_FAILED)                       \
    LOG_INFO("CUSPARSE_STATUS_EXECUTION_FAILED");                       \
  if (stat_t == CUSPARSE_STATUS_INTERNAL_ERROR)                         \
    LOG_INFO("CUSPARSE_STATUS_INTERNAL_ERROR");                         \
  if (stat_t == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)              \
    LOG_INFO("CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED");              \
  LOG_INFO("File: " << file << "; line: " << line);                     \
  exit(1);                                                              \
  }                                                                     \
  }   

namespace paralution {

// Type traits to cast STL types to CUDA types
template <typename ValueType>
struct CUDAType {
  typedef ValueType Type;
};

template <>
struct CUDAType<std::complex<float> > {
  typedef cuFloatComplex Type;
};

template <>
struct CUDAType<std::complex<double> > {
  typedef cuDoubleComplex Type;
};

// Convert pointers to CUDA types
template <typename ValueType>
inline typename CUDAType<ValueType>::Type *CUDAPtr(ValueType *ptr) {
  return reinterpret_cast<typename CUDAType<ValueType>::Type*>(ptr);
}

template <typename ValueType>
inline const typename CUDAType<ValueType>::Type *CUDAPtr(const ValueType *ptr) {
  return reinterpret_cast<const typename CUDAType<ValueType>::Type*>(ptr);
}

// Convert values to CUDA types
inline float CUDAVal(const float &val) { return val; }
inline double CUDAVal(const double &val) { return val; }
inline cuFloatComplex CUDAVal(const std::complex<float> &val) { return make_cuFloatComplex(val.real(), val.imag()); }
inline cuDoubleComplex CUDAVal(const std::complex<double> &val) { return make_cuDoubleComplex(val.real(), val.imag()); }
inline int CUDAVal(const int &val) { return val; }

template <typename IndexType, unsigned int BLOCK_SIZE>
bool cum_sum( IndexType*  dst,
              const IndexType*  src,
              const IndexType   numElems) {
  
  cudaMemset(dst, 0, (numElems+1)*sizeof(IndexType));
  CHECK_CUDA_ERROR(__FILE__, __LINE__);
  
  IndexType* d_temp = NULL;
  allocate_gpu<IndexType>(numElems+1, &d_temp);
  
  cudaMemset(d_temp, 0, (numElems+1)*sizeof(IndexType));
  CHECK_CUDA_ERROR(__FILE__, __LINE__);
  
  kernel_red_partial_sum <IndexType, BLOCK_SIZE> <<< numElems/BLOCK_SIZE+1, BLOCK_SIZE>>>(dst+1, src, numElems);
  CHECK_CUDA_ERROR(__FILE__,__LINE__);
  
  kernel_red_recurse <IndexType> <<< numElems/(BLOCK_SIZE*BLOCK_SIZE)+1, BLOCK_SIZE>>>(d_temp, dst+BLOCK_SIZE, BLOCK_SIZE, (numElems+1));
  CHECK_CUDA_ERROR(__FILE__,__LINE__);
  
  kernel_red_extrapolate<IndexType> <<< numElems/(BLOCK_SIZE*BLOCK_SIZE)+1, BLOCK_SIZE>>>(dst+1, d_temp, src, numElems);
  CHECK_CUDA_ERROR(__FILE__,__LINE__);
  free_gpu<int>(&d_temp);
  
  return true;
  
}

template <typename IndexType, typename ValueType, unsigned int WARP_SIZE, unsigned int BLOCK_SIZE>
void reduce_cuda(const int size, const ValueType *src, ValueType *reduce, ValueType *host_buffer, ValueType *device_buffer) {

  *reduce = (ValueType) 0;

  if (size > 0) {

    dim3 BlockSize(BLOCK_SIZE);
    dim3 GridSize(WARP_SIZE);

    kernel_reduce<WARP_SIZE, BLOCK_SIZE><<<GridSize, BlockSize>>>(size, CUDAPtr(src), CUDAPtr(device_buffer));
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    cudaMemcpy(host_buffer,                 // dst
               device_buffer,               // src
               WARP_SIZE*sizeof(ValueType), // size
               cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    for (unsigned int i=0; i<WARP_SIZE; ++i) {
      *reduce += host_buffer[i];
    }

  }

}

}

#endif // PARALUTION_GPU_GPU_UTILS_HPP_
