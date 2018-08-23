#ifndef ROCALUTION_HIP_HIP_SPARSE_HPP_
#define ROCALUTION_HIP_HIP_SPARSE_HPP_

#include <hipsparse.h>

namespace rocalution {

// hipsparse csrmv
template <typename ValueType>
hipsparseStatus_t hipsparseTcsrmv2(hipsparseHandle_t handle,
                                   hipsparseOperation_t transA,
                                   int m,
                                   int n,
                                   int nnz,
                                   const ValueType *alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const ValueType *csrSortedValA,
                                   const int *csrSortedRowPtrA,
                                   const int *csrSortedColIndA,
                                   csrmv2Info_t info,
                                   const ValueType *x,
                                   const ValueType *beta,
                                   ValueType *y);

// hipsparse coomv
template <typename ValueType>
hipsparseStatus_t hipsparseTcoomv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  int nnz,
                                  const ValueType *alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const ValueType *cooSortedValA,
                                  const int *cooSortedRowIndA,
                                  const int *cooSortedColIndA,
                                  const ValueType *x,
                                  const ValueType *beta,
                                  ValueType *y);

// hipsparse ellmv
template <typename ValueType>
hipsparseStatus_t hipsparseTellmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  const ValueType *alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const ValueType *ellValA,
                                  const int *ellColIndA,
                                  int ellWidth,
                                  const ValueType *x,
                                  const ValueType *beta,
                                  ValueType *y);

// hipsparse csr2csc
template <typename ValueType>
hipsparseStatus_t hipsparseTcsr2csc(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    int nnz,
                                    const ValueType* csrSortedVal,
                                    const int* csrSortedRowPtr,
                                    const int* csrSortedColInd,
                                    ValueType* cscSortedVal,
                                    int* cscSortedRowInd,
                                    int* cscSortedColPtr,
                                    hipsparseAction_t copyValues,
                                    hipsparseIndexBase_t idxBase);

// hipsparse csr2ell
template <typename ValueType>
hipsparseStatus_t hipsparseTcsr2ell(hipsparseHandle_t handle,
                                    int m,
                                    const hipsparseMatDescr_t descrA,
                                    const ValueType *csrValA,
                                    const int *csrRowPtrA,
                                    const int *csrColIndA,
                                    const hipsparseMatDescr_t descrC,
                                    int ellWidthC,
                                    ValueType *ellValC,
                                    int *ellColIndC);

// hipsparse ell2csr
template <typename ValueType>
hipsparseStatus_t hipsparseTell2csr(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    const hipsparseMatDescr_t descrA,
                                    int ellWidthA,
                                    const ValueType* ellValA,
                                    const int* ellColIndA,
                                    const hipsparseMatDescr_t descrC,
                                    ValueType* csrValC,
                                    const int* csrRowPtrC,
                                    int* csrColIndC);

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_SPARSE_HPP_
