#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "hip_blas.hpp"

#include <hipblas.h>
#include <complex>

namespace rocalution {

// hipblas axpy
template <>
hipblasStatus_t hipblasTaxpy(hipblasHandle_t handle,
                             int n, const float *alpha,
                             const float *x, int incx,
                             float *y, int incy)
{
    return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasTaxpy(hipblasHandle_t handle,
                             int n, const double *alpha,
                             const double *x, int incx,
                             double *y, int incy)
{
    return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasTaxpy(hipblasHandle_t handle,
                             int n, const std::complex<float> *alpha,
                             const std::complex<float> *x, int incx,
                             std::complex<float> *y, int incy)
{
//    return hipblas_caxpy(handle, n, alpha, x, incx, y, incy);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
hipblasStatus_t hipblasTaxpy(hipblasHandle_t handle,
                             int n, const std::complex<double> *alpha,
                             const std::complex<double> *x, int incx,
                             std::complex<double> *y, int incy)
{
//    return hipblas_zaxpy(handle, n, alpha, x, incx, y, incy);
    FATAL_ERROR(__FILE__, __LINE__);
}

// hipblas dot
template <>
hipblasStatus_t hipblasTdot(hipblasHandle_t handle, int n,
                            const float *x, int incx,
                            const float *y, int incy,
                            float *result)
{
    return hipblasSdot(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasTdot(hipblasHandle_t handle, int n,
                            const double *x, int incx,
                            const double *y, int incy,
                            double *result)
{
    return hipblasDdot(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasTdot(hipblasHandle_t handle, int n,
                            const std::complex<float> *x, int incx,
                            const std::complex<float> *y, int incy,
                            std::complex<float> *result)
{
//    return hipblas_cdotu(handle, n, x, incx, y, incy, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
hipblasStatus_t hipblasTdot(hipblasHandle_t handle, int n,
                            const std::complex<double> *x, int incx,
                            const std::complex<double> *y, int incy,
                            std::complex<double> *result)
{
//    return hipblas_zdotu(handle, n, x, incx, y, incy, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

// hipblas dotconj
template <>
hipblasStatus_t hipblasTdotc(hipblasHandle_t handle, int n,
                             const float *x, int incx,
                             const float *y, int incy,
                             float *result)
{
    return hipblasSdot(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasTdotc(hipblasHandle_t handle, int n,
                             const double *x, int incx,
                             const double *y, int incy,
                             double *result)
{
    return hipblasDdot(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasTdotc(hipblasHandle_t handle, int n,
                             const std::complex<float> *x, int incx,
                             const std::complex<float> *y, int incy,
                             std::complex<float> *result)
{
//    return hipblas_cdot(handle, n, x, incx, y, incy, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
hipblasStatus_t hipblasTdotc(hipblasHandle_t handle, int n,
                             const std::complex<double> *x, int incx,
                             const std::complex<double> *y, int incy,
                             std::complex<double> *result)
{
//    return hipblas_zdot(handle, n, x, incx, y, incy, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

// hipblas nrm2
template <>
hipblasStatus_t hipblasTnrm2(hipblasHandle_t handle, int n,
                             const float *x, int incx,
                             float *result)
{
    return hipblasSnrm2(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasTnrm2(hipblasHandle_t handle, int n,
                             const double *x, int incx,
                             double *result)
{
    return hipblasDnrm2(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasTnrm2(hipblasHandle_t handle, int n,
                             const std::complex<float> *x, int incx,
                             std::complex<float> *result)
{
//    return hipblas_scnrm2(handle, n, x, incx, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
hipblasStatus_t hipblasTnrm2(hipblasHandle_t handle, int n,
                             const std::complex<double> *x, int incx,
                             std::complex<double> *result)
{
//    return hipblas_dznrm2(handle, n, x, incx, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

// hipblas scal
template <>
hipblasStatus_t hipblasTscal(hipblasHandle_t handle, int n,
                             const float *alpha,
                             float *x, int incx)
{
    return hipblasSscal(handle, n, alpha, x, incx);
}

template <>
hipblasStatus_t hipblasTscal(hipblasHandle_t handle, int n,
                             const double *alpha,
                             double *x, int incx)
{
    return hipblasDscal(handle, n, alpha, x, incx);
}

template <>
hipblasStatus_t hipblasTscal(hipblasHandle_t handle, int n,
                             const std::complex<float> *alpha,
                             std::complex<float> *x, int incx)
{
//    return hipblas_cscal(handle, n, alpha, x, incx);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
hipblasStatus_t hipblasTscal(hipblasHandle_t handle, int n,
                             const std::complex<double> *alpha,
                             std::complex<double> *x, int incx)
{
//    return hipblas_zscal(handle, n, alpha, x, incx);
    FATAL_ERROR(__FILE__, __LINE__);
}

// hipblas_amax
template <>
hipblasStatus_t hipblasTamax(hipblasHandle_t handle, int n,
                             const float *x, int incx,
                             int *result)
{
    return hipblasIsamax(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasTamax(hipblasHandle_t handle, int n,
                             const double *x, int incx,
                             int *result)
{
    return hipblasIdamax(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasTamax(hipblasHandle_t handle, int n,
                             const std::complex<float> *x, int incx,
                             int *result)
{
//    return hipblas_iscamax(handle, n, x, incx, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
hipblasStatus_t hipblasTamax(hipblasHandle_t handle, int n,
                             const std::complex<double> *x, int incx,
                             int *result)
{
//    return hipblas_idzamax(handle, n, x, incx, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

// hipblas_asum
template <>
hipblasStatus_t hipblasTasum(hipblasHandle_t handle, int n,
                             const float *x, int incx,
                             float *result)
{
    return hipblasSasum(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasTasum(hipblasHandle_t handle, int n,
                             const double *x, int incx,
                             double *result)
{
    return hipblasDasum(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasTasum(hipblasHandle_t handle, int n,
                             const std::complex<float> *x, int incx,
                             std::complex<float> *result)
{
//    return hipblas_scasum(handle, n, x, incx, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
hipblasStatus_t hipblasTasum(hipblasHandle_t handle, int n,
                             const std::complex<double> *x, int incx,
                             std::complex<double> *result)
{
//    return hipblas_dzasum(handle, n, x, incx, result);
    FATAL_ERROR(__FILE__, __LINE__);
}

// hipblas_gemv
template <>
hipblasStatus_t hipblasTgemv(hipblasHandle_t handle,
                             hipblasOperation_t trans,
                             int m, int n,
                             const float *alpha,
                             const float *A, int lda,
                             const float *x, int incx,
                             const float *beta,
                             float *y, int incy)
{
    return hipblasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasTgemv(hipblasHandle_t handle,
                             hipblasOperation_t trans,
                             int m, int n,
                             const double *alpha,
                             const double *A, int lda,
                             const double *x, int incx,
                             const double *beta,
                             double *y, int incy)
{
    return hipblasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasTgemv(hipblasHandle_t handle,
                             hipblasOperation_t trans,
                             int m, int n,
                             const std::complex<float> *alpha,
                             const std::complex<float> *A, int lda,
                             const std::complex<float> *x, int incx,
                             const std::complex<float> *beta,
                             std::complex<float> *y, int incy)
{
//    return hipblas_cgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
hipblasStatus_t hipblasTgemv(hipblasHandle_t handle,
                             hipblasOperation_t trans,
                             int m, int n,
                             const std::complex<double> *alpha,
                             const std::complex<double> *A, int lda,
                             const std::complex<double> *x, int incx,
                             const std::complex<double> *beta,
                             std::complex<double> *y, int incy)
{
//    return hipblas_zgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    FATAL_ERROR(__FILE__, __LINE__);
}

// hipblas_gemm
template <>
hipblasStatus_t hipblasTgemm(hipblasHandle_t handle,
                             hipblasOperation_t transa,
                             hipblasOperation_t transb,
                             int m, int n, int k,
                             const float *alpha,
                             const float *A, int lda,
                             const float *B, int ldb,
                             const float *beta,
                             float *C, int ldc)
{
    return hipblasSgemm(handle, transa, transb, m, n, k,
                        alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasTgemm(hipblasHandle_t handle,
                             hipblasOperation_t transa,
                             hipblasOperation_t transb,
                             int m, int n, int k,
                             const double *alpha,
                             const double *A, int lda,
                             const double *B, int ldb,
                             const double *beta,
                             double *C, int ldc)
{
    return hipblasDgemm(handle, transa, transb, m, n, k,
                        alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasTgemm(hipblasHandle_t handle,
                             hipblasOperation_t transa,
                             hipblasOperation_t transb,
                             int m, int n, int k,
                             const std::complex<float> *alpha,
                             const std::complex<float> *A, int lda,
                             const std::complex<float> *B, int ldb,
                             const std::complex<float> *beta,
                             std::complex<float> *C, int ldc)
{
//    return hipblas_cgemm(handle, transa, transb, m, n, k,
//                         alpha, A, lda, B, ldb, beta, C, ldc);
    FATAL_ERROR(__FILE__, __LINE__);
}

template <>
hipblasStatus_t hipblasTgemm(hipblasHandle_t handle,
                             hipblasOperation_t transa,
                             hipblasOperation_t transb,
                             int m, int n, int k,
                             const std::complex<double> *alpha,
                             const std::complex<double> *A, int lda,
                             const std::complex<double> *B, int ldb,
                             const std::complex<double> *beta,
                             std::complex<double> *C, int ldc)
{
//    return hipblas_zgemm(handle, transa, transb, m, n, k,
//                         alpha, A, lda, B, ldb, beta, C, ldc);
    FATAL_ERROR(__FILE__, __LINE__);
}

}
