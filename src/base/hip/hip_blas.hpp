/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HIP_HIP_BLAS_HPP_
#define ROCALUTION_HIP_HIP_BLAS_HPP_

#include <rocblas.h>

namespace rocalution {

// rocblas axpy
template <typename ValueType>
rocblas_status rocblasTaxpy(rocblas_handle handle,
                            int n,
                            const ValueType* alpha,
                            const ValueType* x,
                            int incx,
                            ValueType* y,
                            int incy);

// rocblas dot
template <typename ValueType>
rocblas_status rocblasTdot(rocblas_handle handle,
                           int n,
                           const ValueType* x,
                           int incx,
                           const ValueType* y,
                           int incy,
                           ValueType* result);

// rocblas dotconj
template <typename ValueType>
rocblas_status rocblasTdotc(rocblas_handle handle,
                            int n,
                            const ValueType* x,
                            int incx,
                            const ValueType* y,
                            int incy,
                            ValueType* result);

// rocblas nrm2
template <typename ValueType>
rocblas_status
rocblasTnrm2(rocblas_handle handle, int n, const ValueType* x, int incx, ValueType* result);

// rocblas scal
template <typename ValueType>
rocblas_status
rocblasTscal(rocblas_handle handle, int n, const ValueType* alpha, ValueType* x, int incx);

// rocblas_amax
template <typename ValueType>
rocblas_status
rocblasTamax(rocblas_handle handle, int n, const ValueType* x, int incx, int* result);

// rocblas_asum
template <typename ValueType>
rocblas_status
rocblasTasum(rocblas_handle handle, int n, const ValueType* x, int incx, ValueType* result);

// rocblas_gemv
template <typename ValueType>
rocblas_status rocblasTgemv(rocblas_handle handle,
                            rocblas_operation trans,
                            int m,
                            int n,
                            const ValueType* alpha,
                            const ValueType* A,
                            int lda,
                            const ValueType* x,
                            int incx,
                            const ValueType* beta,
                            ValueType* y,
                            int incy);

// rocblas_gemm
template <typename ValueType>
rocblas_status rocblasTgemm(rocblas_handle handle,
                            rocblas_operation transa,
                            rocblas_operation transb,
                            int m,
                            int n,
                            int k,
                            const ValueType* alpha,
                            const ValueType* A,
                            int lda,
                            const ValueType* B,
                            int ldb,
                            const ValueType* beta,
                            ValueType* C,
                            int ldc);

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_BLAS_HPP_
