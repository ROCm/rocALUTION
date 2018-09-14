#include "../../utils/def.hpp"
#include "hip_allocate_free.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "../../utils/allocate_free.hpp"

#include <cmath>
#include <complex>
#include <hip/hip_runtime.h>

namespace rocalution {

#ifdef ROCALUTION_HIP_PINNED_MEMORY
template <typename DataType>
void allocate_host(int size, DataType** ptr)
{
    log_debug(0, "allocate_host()", size, ptr);

    if(size > 0)
    {
        assert(*ptr == NULL);

        //    *ptr = new DataType[size];

        hipMallocHost((void**)ptr, size * sizeof(DataType));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        assert(*ptr != NULL);
    }
}

template <typename DataType>
void free_host(DataType** ptr)
{
    log_debug(0, "free_host()", *ptr);

    assert(*ptr != NULL);

    //  delete[] *ptr;
    hipFreeHost(*ptr);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    *ptr = NULL;
}
#endif

template <typename DataType>
void allocate_hip(int size, DataType** ptr)
{
    log_debug(0, "allocate_hip()", size, ptr);

    if(size > 0)
    {
        assert(*ptr == NULL);

        hipMalloc((void**)ptr, size * sizeof(DataType));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        assert(*ptr != NULL);
    }
}

template <typename DataType>
void free_hip(DataType** ptr)
{
    log_debug(0, "free_hip()", *ptr);

    assert(*ptr != NULL);

    hipFree(*ptr);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    *ptr = NULL;
}

template <typename DataType>
void set_to_zero_hip(int blocksize, int size, DataType* ptr)
{
    log_debug(0, "set_to_zero_hip()", blocksize, size, ptr);

    if(size > 0)
    {
        assert(ptr != NULL);

        hipMemset(ptr, 0, size * sizeof(DataType));
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

template <typename DataType>
void set_to_one_hip(int blocksize, int size, DataType* ptr)
{
    log_debug(0, "set_to_zero_hip()", blocksize, size, ptr);

    if(size > 0)
    {
        assert(ptr != NULL);

        // 1D accessing, no stride
        dim3 BlockSize(blocksize);
        dim3 GridSize(size / blocksize + 1);

        hipLaunchKernelGGL(
            (kernel_set_to_ones<DataType, int>), GridSize, BlockSize, 0, 0, size, ptr);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }
}

#ifdef SUPPORT_COMPLEX
template <>
void set_to_one_hip(int blocksize, int size, std::complex<double>* ptr)
{
    log_debug(0, "set_to_zero_hip()", blocksize, size, ptr);

    if(size > 0)
    {
        assert(ptr != NULL);

        // 1D accessing, no stride
        dim3 BlockSize(blocksize);
        dim3 GridSize(size / blocksize + 1);

        hipLaunchKernelGGL((kernel_set_to_ones<hipDoubleComplex, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           size,
                           (hipDoubleComplex*)ptr);
    }
}

template <>
void set_to_one_hip(int blocksize, int size, std::complex<float>* ptr)
{
    log_debug(0, "set_to_zero_hip()", blocksize, size, ptr);

    if(size > 0)
    {
        assert(ptr != NULL);

        // 1D accessing, no stride
        dim3 BlockSize(blocksize);
        dim3 GridSize(size / blocksize + 1);

        hipLaunchKernelGGL((kernel_set_to_ones<hipFloatComplex, int>),
                           GridSize,
                           BlockSize,
                           0,
                           0,
                           size,
                           (hipFloatComplex*)ptr);
    }
}
#endif

#ifdef ROCALUTION_HIP_PINNED_MEMORY
template void allocate_host<float>(int size, float** ptr);
template void allocate_host<double>(int size, double** ptr);
#ifdef SUPPORT_COMPLEX
template void allocate_host<std::complex<float>>(int size, std::complex<float>** ptr);
template void allocate_host<std::complex<double>>(int size, std::complex<double>** ptr);
#endif
template void allocate_host<int>(int size, int** ptr);
template void allocate_host<unsigned int>(int size, unsigned int** ptr);
template void allocate_host<char>(int size, char** ptr);

template void free_host<float>(float** ptr);
template void free_host<double>(double** ptr);
#ifdef SUPPORT_COMPLEX
template void free_host<std::complex<float>>(std::complex<float>** ptr);
template void free_host<std::complex<double>>(std::complex<double>** ptr);
#endif
template void free_host<int>(int** ptr);
template void free_host<unsigned int>(unsigned int** ptr);
template void free_host<char>(char** ptr);
#endif

template void allocate_hip<float>(int size, float** ptr);
template void allocate_hip<double>(int size, double** ptr);
#ifdef SUPPORT_COMPLEX
template void allocate_hip<std::complex<float>>(int size, std::complex<float>** ptr);
template void allocate_hip<std::complex<double>>(int size, std::complex<double>** ptr);
#endif
template void allocate_hip<int>(int size, int** ptr);
template void allocate_hip<unsigned int>(int size, unsigned int** ptr);
template void allocate_hip<char>(int size, char** ptr);

template void free_hip<float>(float** ptr);
template void free_hip<double>(double** ptr);
#ifdef SUPPORT_COMPLEX
template void free_hip<std::complex<float>>(std::complex<float>** ptr);
template void free_hip<std::complex<double>>(std::complex<double>** ptr);
#endif
template void free_hip<int>(int** ptr);
template void free_hip<unsigned int>(unsigned int** ptr);
template void free_hip<char>(char** ptr);

template void set_to_zero_hip<float>(int blocksize, int size, float* ptr);
template void set_to_zero_hip<double>(int blocksize, int size, double* ptr);
#ifdef SUPPORT_COMPLEX
template void
set_to_zero_hip<std::complex<float>>(int blocksize, int size, std::complex<float>* ptr);
template void
set_to_zero_hip<std::complex<double>>(int blocksize, int size, std::complex<double>* ptr);
#endif
template void set_to_zero_hip<int>(int blocksize, int size, int* ptr);

template void set_to_one_hip<float>(int blocksize, int size, float* ptr);
template void set_to_one_hip<double>(int blocksize, int size, double* ptr);
#ifdef SUPPORT_COMPLEX
template void
set_to_one_hip<std::complex<float>>(int blocksize, int size, std::complex<float>* ptr);
template void
set_to_one_hip<std::complex<double>>(int blocksize, int size, std::complex<double>* ptr);
#endif
template void set_to_one_hip<int>(int blocksize, int size, int* ptr);

} // namespace rocalution
