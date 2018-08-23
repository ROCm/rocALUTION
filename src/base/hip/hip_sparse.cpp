#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "hip_sparse.hpp"

#include <hipsparse.h>
#include <complex>

namespace rocalution {

// hipsparse csrmv
template <>
hipsparseStatus_t hipsparseTcsrmv2(hipsparseHandle_t handle,
                                   hipsparseOperation_t transA,
                                   int m,
                                   int n,
                                   int nnz,
                                   const float *alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const float *csrSortedValA,
                                   const int *csrSortedRowPtrA,
                                   const int *csrSortedColIndA,
                                   csrmv2Info_t info,
                                   const float *x,
                                   const float *beta,
                                   float *y)
{
    return hipsparseScsrmv2(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA,
                            csrSortedRowPtrA, csrSortedColIndA, info, x, beta, y);
}

template <>
hipsparseStatus_t hipsparseTcsrmv2(hipsparseHandle_t handle,
                                   hipsparseOperation_t transA,
                                   int m,
                                   int n,
                                   int nnz,
                                   const double *alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const double *csrSortedValA,
                                   const int *csrSortedRowPtrA,
                                   const int *csrSortedColIndA,
                                   csrmv2Info_t info,
                                   const double *x,
                                   const double *beta,
                                   double *y)
{
    return hipsparseDcsrmv2(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA,
                            csrSortedRowPtrA, csrSortedColIndA, info, x, beta, y);
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

// hipsparse csr2csc
template <>
hipsparseStatus_t hipsparseTcsr2csc(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    int nnz,
                                    const float* csrSortedVal,
                                    const int* csrSortedRowPtr,
                                    const int* csrSortedColInd,
                                    float* cscSortedVal,
                                    int* cscSortedRowInd,
                                    int* cscSortedColPtr,
                                    hipsparseAction_t copyValues,
                                    hipsparseIndexBase_t idxBase)
{
    return hipsparseScsr2csc(handle, m, n, nnz,
                             csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                             cscSortedVal, cscSortedRowInd, cscSortedColPtr,
                             copyValues, idxBase);
}

template <>
hipsparseStatus_t hipsparseTcsr2csc(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    int nnz,
                                    const double* csrSortedVal,
                                    const int* csrSortedRowPtr,
                                    const int* csrSortedColInd,
                                    double* cscSortedVal,
                                    int* cscSortedRowInd,
                                    int* cscSortedColPtr,
                                    hipsparseAction_t copyValues,
                                    hipsparseIndexBase_t idxBase)
{
    return hipsparseDcsr2csc(handle, m, n, nnz,
                             csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                             cscSortedVal, cscSortedRowInd, cscSortedColPtr,
                             copyValues, idxBase);
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

// hipsparse ell2csr
template <>
hipsparseStatus_t hipsparseTell2csr(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    const hipsparseMatDescr_t descrA,
                                    int ellWidthA,
                                    const float* ellValA,
                                    const int* ellColIndA,
                                    const hipsparseMatDescr_t descrC,
                                    float* csrValC,
                                    const int* csrRowPtrC,
                                    int* csrColIndC)
{
    return hipsparseSell2csr(handle, m, n, descrA, ellWidthA, ellValA, ellColIndA, descrC, csrValC, csrRowPtrC, csrColIndC);
}

template <>
hipsparseStatus_t hipsparseTell2csr(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    const hipsparseMatDescr_t descrA,
                                    int ellWidthA,
                                    const double* ellValA,
                                    const int* ellColIndA,
                                    const hipsparseMatDescr_t descrC,
                                    double* csrValC,
                                    const int* csrRowPtrC,
                                    int* csrColIndC)
{
    return hipsparseDell2csr(handle, m, n, descrA, ellWidthA, ellValA, ellColIndA, descrC, csrValC, csrRowPtrC, csrColIndC);
}

} // namespace rocalution
