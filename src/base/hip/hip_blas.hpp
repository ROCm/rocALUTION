#ifndef ROCALUTION_HIP_HIP_BLAS_HPP_
#define ROCALUTION_HIP_HIP_BLAS_HPP_

#include <hipblas.h>

namespace rocalution {

// hipblas axpy
template <typename ValueType>
hipblasStatus_t hipblasTaxpy(hipblasHandle_t handle,
                             int n, const ValueType *alpha,
                             const ValueType *x, int incx,
                             ValueType *y, int incy);

// hipblas dot
template <typename ValueType>
hipblasStatus_t hipblasTdot(hipblasHandle_t handle, int n,
                            const ValueType *x, int incx,
                            const ValueType *y, int incy,
                            ValueType *result);

// hipblas dotconj
template <typename ValueType>
hipblasStatus_t hipblasTdotc(hipblasHandle_t handle, int n,
                             const ValueType *x, int incx,
                             const ValueType *y, int incy,
                             ValueType *result);

// hipblas nrm2
template <typename ValueType>
hipblasStatus_t hipblasTnrm2(hipblasHandle_t handle, int n,
                             const ValueType *x, int incx,
                             ValueType *result);

// hipblas scal
template <typename ValueType>
hipblasStatus_t hipblasTscal(hipblasHandle_t handle, int n,
                             const ValueType *alpha,
                             ValueType *x, int incx);

// hipblas_amax
template <typename ValueType>
hipblasStatus_t hipblasTamax(hipblasHandle_t handle, int n,
                             const ValueType *x, int incx,
                             int *result);

// hipblas_asum
template <typename ValueType>
hipblasStatus_t hipblasTasum(hipblasHandle_t handle, int n,
                             const ValueType *x, int incx,
                             ValueType *result);

// hipblas_gemv
template <typename ValueType>
hipblasStatus_t hipblasTgemv(hipblasHandle_t handle,
                             hipblasOperation_t trans,
                             int m, int n,
                             const ValueType *alpha,
                             const ValueType *A, int lda,
                             const ValueType *x, int incx,
                             const ValueType *beta,
                             ValueType *y, int incy);

// hipblas_gemm
template <typename ValueType>
hipblasStatus_t hipblasTgemm(hipblasHandle_t handle,
                             hipblasOperation_t transa,
                             hipblasOperation_t transb,
                             int m, int n, int k,
                             const ValueType *alpha,
                             const ValueType *A, int lda,
                             const ValueType *B, int ldb,
                             const ValueType *beta,
                             ValueType *C, int ldc);

}

#endif // ROCALUTION_HIP_HIP_BLAS_HPP_
