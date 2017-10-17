#ifndef PARALUTION_GPU_CUSPARSE_CSR_HPP_
#define PARALUTION_GPU_CUSPARSE_CSR_HPP_

namespace paralution {

cusparseStatus_t __cusparseXcsrgeam__(cusparseHandle_t handle, int m, int n,
                                      const double *alpha,
                                      const cusparseMatDescr_t descrA, int nnzA,
                                      const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                      const double *beta,
                                      const cusparseMatDescr_t descrB, int nnzB,
                                      const double *csrValB, const int *csrRowPtrB, const int *csrColIndB,
                                      const cusparseMatDescr_t descrC,
                                      double *csrValC, int *csrRowPtrC, int *csrColIndC) {

  return cusparseDcsrgeam(handle, 
                          m, n,
                          alpha,
                          descrA, nnzA,
                          csrValA, csrRowPtrA, csrColIndA,
                          beta,
                          descrB, nnzB,
                          csrValB, csrRowPtrB, csrColIndB,
                          descrC,
                          csrValC, csrRowPtrC, csrColIndC);
}

cusparseStatus_t __cusparseXcsrgeam__(cusparseHandle_t handle, int m, int n,
                                      const float *alpha,
                                      const cusparseMatDescr_t descrA, int nnzA,
                                      const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                      const float *beta,
                                      const cusparseMatDescr_t descrB, int nnzB,
                                      const float *csrValB, const int *csrRowPtrB, const int *csrColIndB,
                                      const cusparseMatDescr_t descrC,
                                      float *csrValC, int *csrRowPtrC, int *csrColIndC) {
  return cusparseScsrgeam(handle, 
                          m, n,
                          alpha,
                          descrA, nnzA,
                          csrValA, csrRowPtrA, csrColIndA,
                          beta,
                          descrB, nnzB,
                          csrValB, csrRowPtrB, csrColIndB,
                          descrC,
                          csrValC, csrRowPtrC, csrColIndC);

}

cusparseStatus_t __cusparseXcsrgeam__(cusparseHandle_t handle, int m, int n,
                                      const std::complex<double> *alpha,
                                      const cusparseMatDescr_t descrA, int nnzA,
                                      const std::complex<double> *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                      const std::complex<double> *beta,
                                      const cusparseMatDescr_t descrB, int nnzB,
                                      const std::complex<double> *csrValB, const int *csrRowPtrB, const int *csrColIndB,
                                      const cusparseMatDescr_t descrC,
                                      std::complex<double> *csrValC, int *csrRowPtrC, int *csrColIndC) {
  return cusparseZcsrgeam(handle, 
                          m, n,
                          (cuDoubleComplex*)alpha,
                          descrA, nnzA,
                          (cuDoubleComplex*)csrValA, csrRowPtrA, csrColIndA,
                          (cuDoubleComplex*)beta,
                          descrB, nnzB,
                          (cuDoubleComplex*)csrValB, csrRowPtrB, csrColIndB,
                          descrC,
                          (cuDoubleComplex*)csrValC, csrRowPtrC, csrColIndC);

}

cusparseStatus_t __cusparseXcsrgeam__(cusparseHandle_t handle, int m, int n,
                                      const std::complex<float> *alpha,
                                      const cusparseMatDescr_t descrA, int nnzA,
                                      const std::complex<float> *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                      const std::complex<float> *beta,
                                      const cusparseMatDescr_t descrB, int nnzB,
                                      const std::complex<float> *csrValB, const int *csrRowPtrB, const int *csrColIndB,
                                      const cusparseMatDescr_t descrC,
                                      std::complex<float> *csrValC, int *csrRowPtrC, int *csrColIndC) {
  return cusparseCcsrgeam(handle, 
                          m, n,
                          (cuFloatComplex*)alpha,
                          descrA, nnzA,
                          (cuFloatComplex*)csrValA, csrRowPtrA, csrColIndA,
                          (cuFloatComplex*)beta,
                          descrB, nnzB,
                          (cuFloatComplex*)csrValB, csrRowPtrB, csrColIndB,
                          descrC,
                          (cuFloatComplex*)csrValC, csrRowPtrC, csrColIndC);

}

cusparseStatus_t  __cusparseXcsrgemm__(cusparseHandle_t handle,
                                       cusparseOperation_t transA, cusparseOperation_t transB,
                                       int m, int n, int k,
                                       const cusparseMatDescr_t descrA, const int nnzA,
                                       const double *csrValA,
                                       const int *csrRowPtrA, const int *csrColIndA,
                                       const cusparseMatDescr_t descrB, const int nnzB,                            
                                       const double *csrValB, 
                                       const int *csrRowPtrB, const int *csrColIndB,
                                       const cusparseMatDescr_t descrC,
                                       double *csrValC,
                                       const int *csrRowPtrC, int *csrColIndC ) {
  
  return cusparseDcsrgemm(handle,
                          transA, transB,
                          m, n, k,
                          descrA, nnzA,
                          csrValA,
                          csrRowPtrA, csrColIndA,
                          descrB, nnzB,                            
                          csrValB, 
                          csrRowPtrB, csrColIndB,
                          descrC,
                          csrValC,
                          csrRowPtrC, csrColIndC );

}

cusparseStatus_t  __cusparseXcsrgemm__(cusparseHandle_t handle,
                                       cusparseOperation_t transA, cusparseOperation_t transB,
                                       int m, int n, int k,
                                       const cusparseMatDescr_t descrA, const int nnzA,
                                       const float *csrValA,
                                       const int *csrRowPtrA, const int *csrColIndA,
                                       const cusparseMatDescr_t descrB, const int nnzB,                            
                                       const float *csrValB, 
                                       const int *csrRowPtrB, const int *csrColIndB,
                                       const cusparseMatDescr_t descrC,
                                       float *csrValC,
                                       const int *csrRowPtrC, int *csrColIndC ) {

  return cusparseScsrgemm(handle,
                          transA, transB,
                          m, n, k,
                          descrA, nnzA,
                          csrValA,
                          csrRowPtrA, csrColIndA,
                          descrB, nnzB,                            
                          csrValB, 
                          csrRowPtrB, csrColIndB,
                          descrC,
                          csrValC,
                          csrRowPtrC, csrColIndC );

}

cusparseStatus_t  __cusparseXcsrgemm__(cusparseHandle_t handle,
                                       cusparseOperation_t transA, cusparseOperation_t transB,
                                       int m, int n, int k,
                                       const cusparseMatDescr_t descrA, const int nnzA,
                                       const std::complex<double> *csrValA,
                                       const int *csrRowPtrA, const int *csrColIndA,
                                       const cusparseMatDescr_t descrB, const int nnzB,                            
                                       const std::complex<double> *csrValB, 
                                       const int *csrRowPtrB, const int *csrColIndB,
                                       const cusparseMatDescr_t descrC,
                                       std::complex<double> *csrValC,
                                       const int *csrRowPtrC, int *csrColIndC ) {

  return cusparseZcsrgemm(handle,
                          transA, transB,
                          m, n, k,
                          descrA, nnzA,
                          (cuDoubleComplex*)csrValA,
                          csrRowPtrA, csrColIndA,
                          descrB, nnzB,                            
                          (cuDoubleComplex*)csrValB, 
                          csrRowPtrB, csrColIndB,
                          descrC,
                          (cuDoubleComplex*)csrValC,
                          csrRowPtrC, csrColIndC );

}

cusparseStatus_t  __cusparseXcsrgemm__(cusparseHandle_t handle,
                                       cusparseOperation_t transA, cusparseOperation_t transB,
                                       int m, int n, int k,
                                       const cusparseMatDescr_t descrA, const int nnzA,
                                       const std::complex<float> *csrValA,
                                       const int *csrRowPtrA, const int *csrColIndA,
                                       const cusparseMatDescr_t descrB, const int nnzB,                            
                                       const std::complex<float> *csrValB, 
                                       const int *csrRowPtrB, const int *csrColIndB,
                                       const cusparseMatDescr_t descrC,
                                       std::complex<float> *csrValC,
                                       const int *csrRowPtrC, int *csrColIndC ) {

  return cusparseCcsrgemm(handle,
                          transA, transB,
                          m, n, k,
                          descrA, nnzA,
                          (cuFloatComplex*)csrValA,
                          csrRowPtrA, csrColIndA,
                          descrB, nnzB,                            
                          (cuFloatComplex*)csrValB, 
                          csrRowPtrB, csrColIndB,
                          descrC,
                          (cuFloatComplex*)csrValC,
                          csrRowPtrC, csrColIndC );

}


}

#endif
