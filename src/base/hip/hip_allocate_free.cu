#include "../../utils/def.hpp"
#include "hip_allocate_free.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "../../utils/allocate_free.hpp"

#include <cmath>
#include <complex>

#include <cuda.h>

namespace paralution {

#ifdef PARALUTION_HIP_PINNED_MEMORY

template <typename DataType>
void allocate_host(const int size, DataType **ptr) {

  LOG_DEBUG(0, "allocate_host()",
            size);

  if (size > 0) {

    assert(*ptr == NULL);

    //    *ptr = new DataType[size];

    cudaMallocHost((void **)ptr, size*sizeof(DataType));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    LOG_DEBUG(0, "allocate_host()",
              *ptr);

    assert(*ptr != NULL);

  }

}

template <typename DataType>
void free_host(DataType **ptr) {

  LOG_DEBUG(0, "free_host()",
            *ptr);

  assert(*ptr != NULL);

  //  delete[] *ptr;
  cudaFreeHost(*ptr);
  CHECK_HIP_ERROR(__FILE__, __LINE__);

  *ptr = NULL;

}

#endif

template <typename DataType>
void allocate_hip(const int size, DataType **ptr) {

  LOG_DEBUG(0, "allocate_hip()",
            size);

  if (size > 0) {

    assert(*ptr == NULL);

    cudaMalloc( (void **)ptr, size*sizeof(DataType));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    assert(*ptr != NULL);

  }

}

template <typename DataType>
void free_hip(DataType **ptr) {

  LOG_DEBUG(0, "free_hip()",
            *ptr);

  assert(*ptr != NULL);

  cudaFree(*ptr);
  CHECK_HIP_ERROR(__FILE__, __LINE__);

  *ptr = NULL;

}

template <typename DataType>
void set_to_zero_hip(const int blocksize,
                     const int max_threads,
                     const int size, DataType *ptr) {

  LOG_DEBUG(0, "set_to_zero_hip()",
            "size =" << size << 
            " ptr=" << ptr);

  if (size > 0) {

    assert(ptr != NULL);

    cudaMemset(ptr, 0, size*sizeof(DataType));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

/*
    int s = size ;
    int k = (size/blocksize)/max_threads + 1;
    if (k > 1) s = size / k;

    dim3 BlockSize(blocksize);
    dim3 GridSize(s / blocksize + 1);

    kernel_set_to_zeros<DataType, int> <<<GridSize, BlockSize>>> (size, ptr);

    CHECK_HIP_ERROR(__FILE__, __LINE__);
*/

/*
    // 1D accessing, no stride
    dim3 BlockSize(blocksize);
    dim3 GridSize(size / blocksize + 1);

    kernel_set_to_zeros<DataType, int> <<<GridSize, BlockSize>>> (size, ptr);

    CHECK_HIP_ERROR(__FILE__, __LINE__);
*/

  }

}

template <typename DataType>
void set_to_one_hip(const int blocksize,
                    const int max_threads,
                    const int size, DataType *ptr) {

  LOG_DEBUG(0, "set_to_zero_hip()",
            "size =" << size << 
            " ptr=" << ptr);

  if (size > 0) {

    assert(ptr != NULL);

/*
    int s = size ;
    int k = (size/blocksize)/max_threads + 1;
    if (k > 1) s = size / k;

    dim3 BlockSize(blocksize);
    dim3 GridSize(s / blocksize + 1);

    kernel_set_to_ones<DataType, int> <<<GridSize, BlockSize>>> (size, ptr);

    CHECK_HIP_ERROR(__FILE__, __LINE__);
*/

    // 1D accessing, no stride
    dim3 BlockSize(blocksize);
    dim3 GridSize(size / blocksize + 1);

    kernel_set_to_ones<DataType, int> <<<GridSize, BlockSize>>> (size, ptr);

  }

}

template <>
void set_to_one_hip(const int blocksize,
                    const int max_threads,
                    const int size, std::complex<double> *ptr) {

  LOG_DEBUG(0, "set_to_zero_hip()",
            "size =" << size << 
            " ptr=" << ptr);

  if (size > 0) {

    assert(ptr != NULL);

/*
    int s = size ;
    int k = (size/blocksize)/max_threads + 1;
    if (k > 1) s = size / k;

    dim3 BlockSize(blocksize);
    dim3 GridSize(s / blocksize + 1);

    kernel_set_to_ones<DataType, int> <<<GridSize, BlockSize>>> (size, ptr);

    CHECK_HIP_ERROR(__FILE__, __LINE__);
*/

    // 1D accessing, no stride
    dim3 BlockSize(blocksize);
    dim3 GridSize(size / blocksize + 1);

    kernel_set_to_ones<cuDoubleComplex, int> <<<GridSize, BlockSize>>> (size, (cuDoubleComplex*)ptr);

  }

}

template <>
void set_to_one_hip(const int blocksize,
                    const int max_threads,
                    const int size, std::complex<float> *ptr) {

  LOG_DEBUG(0, "set_to_zero_hip()",
            "size =" << size << 
            " ptr=" << ptr);

  if (size > 0) {

    assert(ptr != NULL);

/*
    int s = size ;
    int k = (size/blocksize)/max_threads + 1;
    if (k > 1) s = size / k;

    dim3 BlockSize(blocksize);
    dim3 GridSize(s / blocksize + 1);

    kernel_set_to_ones<DataType, int> <<<GridSize, BlockSize>>> (size, ptr);

    CHECK_HIP_ERROR(__FILE__, __LINE__);
*/

    // 1D accessing, no stride
    dim3 BlockSize(blocksize);
    dim3 GridSize(size / blocksize + 1);

    kernel_set_to_ones<cuFloatComplex, int> <<<GridSize, BlockSize>>> (size, (cuFloatComplex*)ptr);

  }

}


#ifdef PARALUTION_HIP_PINNED_MEMORY

template void allocate_host<float                >(const int size, float                **ptr);
template void allocate_host<double               >(const int size, double               **ptr);
#ifdef SUPPORT_COMPLEX
template void allocate_host<std::complex<float>  >(const int size, std::complex<float>  **ptr);
template void allocate_host<std::complex<double> >(const int size, std::complex<double> **ptr);
#endif
template void allocate_host<int                  >(const int size, int                  **ptr);
template void allocate_host<unsigned int         >(const int size, unsigned int         **ptr);
template void allocate_host<char                 >(const int size, char                 **ptr);

template void free_host<float                >(float                **ptr);
template void free_host<double               >(double               **ptr);
#ifdef SUPPORT_COMPLEX
template void free_host<std::complex<float>  >(std::complex<float>  **ptr);
template void free_host<std::complex<double> >(std::complex<double> **ptr);
#endif
template void free_host<int                  >(int                  **ptr);
template void free_host<unsigned int         >(unsigned int         **ptr);
template void free_host<char                 >(char                 **ptr);

#endif

template void allocate_hip<float                >(const int size, float                **ptr);
template void allocate_hip<double               >(const int size, double               **ptr);
#ifdef SUPPORT_COMPLEX
template void allocate_hip<std::complex<float>  >(const int size, std::complex<float>  **ptr);
template void allocate_hip<std::complex<double> >(const int size, std::complex<double> **ptr);
#endif
template void allocate_hip<int                  >(const int size, int                  **ptr);
template void allocate_hip<unsigned int         >(const int size, unsigned int         **ptr);
template void allocate_hip<char                 >(const int size, char                 **ptr);

template void free_hip<float                >(float                **ptr);
template void free_hip<double               >(double               **ptr);
#ifdef SUPPORT_COMPLEX
template void free_hip<std::complex<float>  >(std::complex<float>  **ptr);
template void free_hip<std::complex<double> >(std::complex<double> **ptr);
#endif
template void free_hip<int                  >(int                  **ptr);
template void free_hip<unsigned int         >(unsigned int         **ptr);
template void free_hip<char                 >(char                 **ptr);

template void set_to_zero_hip<float                >(const int blocksize, const int max_threads, const int size, float                *ptr);
template void set_to_zero_hip<double               >(const int blocksize, const int max_threads, const int size, double               *ptr);
#ifdef SUPPORT_COMPLEX
template void set_to_zero_hip<std::complex<float>  >(const int blocksize, const int max_threads, const int size, std::complex<float>  *ptr);
template void set_to_zero_hip<std::complex<double> >(const int blocksize, const int max_threads, const int size, std::complex<double> *ptr);
#endif
template void set_to_zero_hip<int                  >(const int blocksize, const int max_threads, const int size, int                  *ptr);

template void set_to_one_hip<float                >(const int blocksize, const int max_threads, const int size, float                *ptr);
template void set_to_one_hip<double               >(const int blocksize, const int max_threads, const int size, double               *ptr);
#ifdef SUPPORT_COMPLEX
template void set_to_one_hip<std::complex<float>  >(const int blocksize, const int max_threads, const int size, std::complex<float>  *ptr);
template void set_to_one_hip<std::complex<double> >(const int blocksize, const int max_threads, const int size, std::complex<double> *ptr);
#endif
template void set_to_one_hip<int                  >(const int blocksize, const int max_threads, const int size, int                  *ptr);

}
