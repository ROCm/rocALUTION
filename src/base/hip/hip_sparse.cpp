#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "hip_sparse.hpp"

#include <hipsparse.h>
#include <complex>

namespace rocalution {

// hipsparse csrmv
template <>
hipsparseStatus_t hipsparseTcsrmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  int nnz,
                                  const float *alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float *csrSortedValA,
                                  const int *csrSortedRowPtrA,
                                  const int *csrSortedColIndA,
                                  const float *x,
                                  const float *beta,
                                  float *y)
{
    return hipsparseScsrmv(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, x, beta, y);
}

template <>
hipsparseStatus_t hipsparseTcsrmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  int nnz,
                                  const double *alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double *csrSortedValA,
                                  const int *csrSortedRowPtrA,
                                  const int *csrSortedColIndA,
                                  const double *x,
                                  const double *beta,
                                  double *y)
{
    return hipsparseDcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, x, beta, y);
}

// hipsparse coomv
template <>
hipsparseStatus_t hipsparseTcoomv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  int nnz,
                                  const float *alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float *cooSortedValA,
                                  const int *cooSortedRowIndA,
                                  const int *cooSortedColIndA,
                                  const float *x,
                                  const float *beta,
                                  float *y)
{
    return hipsparseScoomv(handle, transA, m, n, nnz, alpha, descrA, cooSortedValA,
                           cooSortedRowIndA, cooSortedColIndA, x, beta, y);
}

template <>
hipsparseStatus_t hipsparseTcoomv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  int nnz,
                                  const double *alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double *cooSortedValA,
                                  const int *cooSortedRowIndA,
                                  const int *cooSortedColIndA,
                                  const double *x,
                                  const double *beta,
                                  double *y)
{
    return hipsparseDcoomv(handle, transA, m, n, nnz, alpha, descrA, cooSortedValA,
                           cooSortedRowIndA, cooSortedColIndA, x, beta, y);
}

// hipsparse ellmv
template <>
hipsparseStatus_t hipsparseTellmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  const float *alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float *ellValA,
                                  const int *ellColIndA,
                                  int ellWidth,
                                  const float *x,
                                  const float *beta,
                                  float *y)
{
    return hipsparseSellmv(handle, transA, m, n, alpha, descrA, ellValA, ellColIndA,
                           ellWidth, x, beta, y);
}

template <>
hipsparseStatus_t hipsparseTellmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  const double *alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double *ellValA,
                                  const int *ellColIndA,
                                  int ellWidth,
                                  const double *x,
                                  const double *beta,
                                  double *y)
{
    return hipsparseDellmv(handle, transA, m, n, alpha, descrA, ellValA, ellColIndA,
                           ellWidth, x, beta, y);
}

// hipsparse csr2ell
template <>
hipsparseStatus_t hipsparseTcsr2ell(hipsparseHandle_t handle,
                                    int m,
                                    const hipsparseMatDescr_t descrA,
                                    const float *csrValA,
                                    const int *csrRowPtrA,
                                    const int *csrColIndA,
                                    const hipsparseMatDescr_t descrC,
                                    int ellWidthC,
                                    float *ellValC,
                                    int *ellColIndC)
{
    return hipsparseScsr2ell(handle, m, descrA, csrValA, csrRowPtrA, csrColIndA,
                             descrC, ellWidthC, ellValC, ellColIndC);
}

template <>
hipsparseStatus_t hipsparseTcsr2ell(hipsparseHandle_t handle,
                                    int m,
                                    const hipsparseMatDescr_t descrA,
                                    const double *csrValA,
                                    const int *csrRowPtrA,
                                    const int *csrColIndA,
                                    const hipsparseMatDescr_t descrC,
                                    int ellWidthC,
                                    double *ellValC,
                                    int *ellColIndC)
{
    return hipsparseDcsr2ell(handle, m, descrA, csrValA, csrRowPtrA, csrColIndA,
                             descrC, ellWidthC, ellValC, ellColIndC);
}

}
