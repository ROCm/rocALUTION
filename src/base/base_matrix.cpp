#include "../utils/def.hpp"
#include "base_matrix.hpp"
#include "base_vector.hpp"
#include "backend_manager.hpp"
#include "../utils/log.hpp"

#include <complex>

namespace paralution {

template <typename ValueType>
BaseMatrix<ValueType>::BaseMatrix() {

  LOG_DEBUG(this, "BaseMatrix::BaseMatrix()",
            "default constructor");

  this->nrow_ = 0;
  this->ncol_ = 0;
  this->nnz_  = 0;

}

template <typename ValueType>
BaseMatrix<ValueType>::~BaseMatrix() {

  LOG_DEBUG(this, "BaseMatrix::~BaseMatrix()",
            "default destructor");

}

template <typename ValueType>
inline int BaseMatrix<ValueType>::get_nrow(void) const {

  return this->nrow_;

}

template <typename ValueType>
inline int BaseMatrix<ValueType>::get_ncol(void) const {

  return this->ncol_;

}

template <typename ValueType>
inline int BaseMatrix<ValueType>::get_nnz(void) const {

  return this->nnz_;

}

template <typename ValueType>
void BaseMatrix<ValueType>::set_backend(const Paralution_Backend_Descriptor local_backend) {

  this->local_backend_ = local_backend;

}

template <typename ValueType>
bool BaseMatrix<ValueType>::Check(void) const {

  LOG_INFO("BaseMatrix<ValueType>::Check()");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)!");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyFromCSR(const int *row_offsets, const int *col, const ValueType *val) {

  LOG_INFO("CopyFromCSR(const int *row_offsets, const int *col, const ValueType *val)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("This function is not available for this backend");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyToCSR(int *row_offsets, int *col, ValueType *val) const {

  LOG_INFO("CopyToCSR(int *row_offsets, int *col, ValueType *val) const");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("This function is not available for this backend");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyFromCOO(const int *row, const int *col, const ValueType *val) {

  LOG_INFO("CopyFromCOO(const int *row, const int *col, const ValueType *val)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("This function is not available for this backend");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyToCOO(int *row, int *col, ValueType *val) const {

  LOG_INFO("CopyToCOO(const int *row, const int *col, const ValueType *val) const");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("This function is not available for this backend");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyFromHostCSR(const int *row_offset, const int *col, const ValueType *val,
                                            const int nnz, const int nrow, const int ncol) {

  LOG_INFO("CopyFromHostCSR(const int *row_offsets, const int *col, const ValueType *val, const int nnz, const int nrow, const int ncol)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("This function is not available for this backend");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateCSR(const int nnz, const int nrow, const int ncol) {

  LOG_INFO("AllocateCSR(const int nnz, const int nrow, const int ncol)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("This is NOT a CSR matrix");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateCOO(const int nnz, const int nrow, const int ncol) {

  LOG_INFO("AllocateCOO(const int nnz, const int nrow, const int ncol)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("This is NOT a COO matrix");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateDIA(const int nnz, const int nrow, const int ncol, const int ndiag) {

  LOG_INFO("AllocateDIA(const int nnz, const int nrow, const int ncol, const int ndiag)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("This is NOT a DIA matrix");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateELL(const int nnz, const int nrow, const int ncol, const int max_row) {

  LOG_INFO("AllocateELL(const int nnz, const int nrow, const int ncol, const int max_row)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("This is NOT a ELL matrix");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateHYB(const int ell_nnz, const int coo_nnz, const int ell_max_row,
                                        const int nrow, const int ncol) {

  LOG_INFO("AllocateHYB(const int ell_nnz, const int coo_nnz, const int ell_max_row, const int nrow, const int ncol)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("This is NOT a HYB matrix");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateDENSE(const int nrow, const int ncol) {

  LOG_INFO("AllocateDENSE(const int nrow, const int ncol)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("This is NOT a DENSE matrix");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::AllocateMCSR(const int nnz, const int nrow, const int ncol) {

  LOG_INFO("AllocateMCSR(const int nnz, const int nrow, const int ncol)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("This is NOT a MCSR matrix");
  FATAL_ERROR(__FILE__, __LINE__);

}

// The conversion CSR->COO (or X->CSR->COO)
template <typename ValueType>
bool BaseMatrix<ValueType>::ReadFileMTX(const std::string filename) {
  return false;
}

// The conversion CSR->COO (or X->CSR->COO)
template <typename ValueType>
bool BaseMatrix<ValueType>::WriteFileMTX(const std::string filename) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ReadFileCSR(const std::string filename) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::WriteFileCSR(const std::string filename) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractDiagonal(BaseVector<ValueType> *vec_diag) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractInverseDiagonal(BaseVector<ValueType> *vec_inv_diag) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractSubMatrix(const int row_offset,
                                             const int col_offset,
                                             const int row_size,
                                             const int col_size,
                                             BaseMatrix<ValueType> *mat) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractL(BaseMatrix<ValueType> *L) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractLDiagonal(BaseMatrix<ValueType> *L) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractU(BaseMatrix<ValueType> *U) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractUDiagonal(BaseMatrix<ValueType> *U) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::LUSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::LLSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::LLSolve(const BaseVector<ValueType> &in, const BaseVector<ValueType> &inv_diag,
                                    BaseVector<ValueType> *out) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ILU0Factorize(void) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ILUTFactorize(const double t, const int maxrow) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ICFactorize(BaseVector<ValueType> *inv_diag) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Permute(const BaseVector<int> &permutation) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::PermuteBackward(const BaseVector<int> &permutation) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::CMK(BaseVector<int> *permutation) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::RCMK(BaseVector<int> *permutation) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ConnectivityOrder(BaseVector<int> *permutation) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::MultiColoring(int &num_colors, int **size_colors, BaseVector<int> *permutation) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::MaximalIndependentSet(int &size, BaseVector<int> *permutation) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ZeroBlockPermutation(int &size, BaseVector<int> *permutation) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::SymbolicPower(const int p) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::SymbolicMatMatMult(const BaseMatrix<ValueType> &src) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ILUpFactorizeNumeric(const int p, const BaseMatrix<ValueType> &mat) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::MatrixAdd(const BaseMatrix<ValueType> &mat, const ValueType alpha, 
                                      const ValueType beta, const bool structure) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Gershgorin(ValueType &lambda_min,
                                       ValueType &lambda_max) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Scale(const ValueType alpha) {
  return false;
}


template <typename ValueType>
bool BaseMatrix<ValueType>::ScaleDiagonal(const ValueType alpha) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ScaleOffDiagonal(const ValueType alpha) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AddScalar(const ValueType alpha) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AddScalarDiagonal(const ValueType alpha) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AddScalarOffDiagonal(const ValueType alpha) {
  return false;
}

template <typename ValueType>
void BaseMatrix<ValueType>::LUAnalyse(void) {

  LOG_INFO("BaseMatrix<ValueType>::LUAnalyse(void)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)!");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::LUAnalyseClear(void) {

  LOG_INFO("BaseMatrix<ValueType>::LUAnalyseClear(void)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)!");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::LLAnalyse(void) {

  LOG_INFO("BaseMatrix<ValueType>::LLAnalyse(void)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)!");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::LLAnalyseClear(void) {

  LOG_INFO("BaseMatrix<ValueType>::LLAnalyseClear(void)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)!");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::LAnalyse(const bool diag_unit) {

  LOG_INFO("BaseMatrix<ValueType>::LAnalyse(const bool diag_unit=false)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)!");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::LAnalyseClear(void) {

  LOG_INFO("BaseMatrix<ValueType>::LAnalyseClear(void)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)!");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
bool BaseMatrix<ValueType>::LSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {
  return false;
}

template <typename ValueType>
void BaseMatrix<ValueType>::UAnalyse(const bool diag_unit) {

  LOG_INFO("BaseMatrix<ValueType>::UAnalyse(const bool diag_unit=false)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)!");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::UAnalyseClear(void) {

  LOG_INFO("BaseMatrix<ValueType>::UAnalyseClear(void)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)!");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
bool BaseMatrix<ValueType>::USolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::NumericMatMatMult(const BaseMatrix<ValueType> &A, const BaseMatrix<ValueType> &B) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::SymbolicMatMatMult(const BaseMatrix<ValueType> &A, const BaseMatrix<ValueType> &B) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AMGConnect(const ValueType eps, BaseVector<int> *connections) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AMGAggregate(const BaseVector<int> &connections, BaseVector<int> *aggregates) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AMGSmoothedAggregation(const ValueType relax,
                                                   const BaseVector<int> &aggregates,
                                                   const BaseVector<int> &connections,
                                                         BaseMatrix<ValueType> *prolong,
                                                         BaseMatrix<ValueType> *restrict) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::AMGAggregation(const BaseVector<int> &aggregates,
                                                 BaseMatrix<ValueType> *prolong,
                                                 BaseMatrix<ValueType> *restrict) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::RugeStueben(const ValueType eps, BaseMatrix<ValueType> *prolong,
                                                             BaseMatrix<ValueType> *restrict) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::InitialPairwiseAggregation(const ValueType beta, int &nc, BaseVector<int> *G, int &Gsize,
                                                       int **rG, int &rGsize, const int ordering) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::InitialPairwiseAggregation(const BaseMatrix<ValueType> &mat, const ValueType beta, int &nc,
                                                       BaseVector<int> *G, int &Gsize, int **rG, int &rGsize,
                                                       const int ordering) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::FurtherPairwiseAggregation(const ValueType beta, int &nc, BaseVector<int> *G, int &Gsize,
                                                       int **rG, int &rGsize, const int ordering) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::FurtherPairwiseAggregation(const BaseMatrix<ValueType> &mat, const ValueType beta, int &nc,
                                                       BaseVector<int> *G, int &Gsize, int **rG, int &rGsize,
                                                       const int ordering) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::CoarsenOperator(BaseMatrix<ValueType> *Ac, const int nrow, const int ncol, const BaseVector<int> &G,
                                            const int Gsize, const int *rG, const int rGsize) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::LUFactorize(void) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Householder(const int idx, ValueType &beta, BaseVector<ValueType> *vec) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::QRDecompose(void) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::QRSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Invert(void) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::FSAI(const int power, const BaseMatrix<ValueType> *pattern) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::SPAI(void) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::DiagonalMatrixMultR(const BaseVector<ValueType> &diag) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::DiagonalMatrixMultL(const BaseVector<ValueType> &diag) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::MatMatMult(const BaseMatrix<ValueType> &A, const BaseMatrix<ValueType> &B) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Zeros(void) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Compress(const double drop_off) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Transpose(void) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Sort(void) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::Key(long int &row_key,
                                long int &col_key,
                                long int &val_key) const {
  return false;
}

template <typename ValueType>
void BaseMatrix<ValueType>::SetDataPtrCOO(int **row, int **col, ValueType **val,
                                          const int nnz, const int nrow, const int ncol) {

  LOG_INFO("BaseMatrix<ValueType>::SetDataPtrCOO(...)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)! Check the backend?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::LeaveDataPtrCOO(int **row, int **col, ValueType **val) {

  LOG_INFO("BaseMatrix<ValueType>::LeaveDataPtrCOO(...)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)! Check the backend?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::SetDataPtrCSR(int **row_offset, int **col, ValueType **val,
                                          const int nnz, const int nrow, const int ncol) {

  LOG_INFO("BaseMatrix<ValueType>::SetDataPtrCSR(...)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)! Check the backend?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::LeaveDataPtrCSR(int **row_offset, int **col, ValueType **val) {

  LOG_INFO("BaseMatrix<ValueType>::LeaveDataPtrCSR(...)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)! Check the backend?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::SetDataPtrMCSR(int **row_offset, int **col, ValueType **val,
                                           const int nnz, const int nrow, const int ncol) {

  LOG_INFO("BaseMatrix<ValueType>::SetDataPtrMCSR(...)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)! Check the backend?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::LeaveDataPtrMCSR(int **row_offset, int **col, ValueType **val) {

  LOG_INFO("BaseMatrix<ValueType>::LeaveDataPtrMCSR(...)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)! Check the backend?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::SetDataPtrDENSE(ValueType **val, const int nrow, const int ncol) {

  LOG_INFO("BaseMatrix<ValueType>::SetDataPtrDENSE(...)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)! Check the backend?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::LeaveDataPtrDENSE(ValueType **val) {

  LOG_INFO("BaseMatrix<ValueType>::LeaveDataPtrDENSE(...)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)! Check the backend?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::SetDataPtrELL(int **col, ValueType **val,
                                          const int nnz, const int nrow, const int ncol, const int max_row) {

  LOG_INFO("BaseMatrix<ValueType>::SetDataPtrELL(...)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)! Check the backend?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::LeaveDataPtrELL(int **col, ValueType **val, int &max_row) {

  LOG_INFO("BaseMatrix<ValueType>::LeaveDataPtrELL(...)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)! Check the backend?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::SetDataPtrDIA(int **offset, ValueType **val,
                                          const int nnz, const int nrow, const int ncol, const int num_diag) {

  LOG_INFO("BaseMatrix<ValueType>::SetDataPtrDIA(...)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)! Check the backend?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void BaseMatrix<ValueType>::LeaveDataPtrDIA(int **offset, ValueType **val, int &num_diag) {

  LOG_INFO("BaseMatrix<ValueType>::LeaveDataPtrDIA(...)");
  LOG_INFO("Matrix format=" << _matrix_format_names[this->get_mat_format()]);
  this->info();
  LOG_INFO("The function is not implemented (yet)! Check the backend?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
bool BaseMatrix<ValueType>::CreateFromMap(const BaseVector<int> &map, const int n, const int m) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::CreateFromMap(const BaseVector<int> &map, const int n, const int m, BaseMatrix<ValueType> *pro) {
  return false;
}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &mat) {

  // default is no async
  LOG_VERBOSE_INFO(4, "*** info: BaseMatrix::CopyFromAsync() no async available)");        

  this->CopyFrom(mat);

}

template <typename ValueType>
void BaseMatrix<ValueType>::CopyToAsync(BaseMatrix<ValueType> *mat) const {

  // default is no async
  LOG_VERBOSE_INFO(4, "*** info: BaseMatrix::CopyToAsync() no async available)");        

  this->CopyTo(mat);

}

template <typename ValueType>
bool BaseMatrix<ValueType>::ReplaceColumnVector(const int idx, const BaseVector<ValueType> &vec) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractColumnVector(const int idx, BaseVector<ValueType> *vec) const {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ReplaceRowVector(const int idx, const BaseVector<ValueType> &vec) {
  return false;
}

template <typename ValueType>
bool BaseMatrix<ValueType>::ExtractRowVector(const int idx, BaseVector<ValueType> *vec) const {
  return false;
}





  //TODO print also parameters info?

template <typename ValueType>
HostMatrix<ValueType>::HostMatrix() {
}

template <typename ValueType>
HostMatrix<ValueType>::~HostMatrix() {
}







template <typename ValueType>
AcceleratorMatrix<ValueType>::AcceleratorMatrix() {
}

template <typename ValueType>
AcceleratorMatrix<ValueType>::~AcceleratorMatrix() {
}


template <typename ValueType>
void AcceleratorMatrix<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  // default is no async
  this->CopyFromHostAsync(src);

}


template <typename ValueType>
void AcceleratorMatrix<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  // default is no async
  this->CopyToHostAsync(dst);

}




template <typename ValueType>
HIPAcceleratorMatrix<ValueType>::HIPAcceleratorMatrix() {
}

template <typename ValueType>
HIPAcceleratorMatrix<ValueType>::~HIPAcceleratorMatrix() {
}


template class BaseMatrix<double>;
template class BaseMatrix<float>;
#ifdef SUPPORT_COMPLEX
template class BaseMatrix<std::complex<double> >;
template class BaseMatrix<std::complex<float> >;
#endif
template class BaseMatrix<int>;

template class HostMatrix<double>;
template class HostMatrix<float>;
#ifdef SUPPORT_COMPLEX
template class HostMatrix<std::complex<double> >;
template class HostMatrix<std::complex<float> >;
#endif
template class HostMatrix<int>;

template class AcceleratorMatrix<double>;
template class AcceleratorMatrix<float>;
#ifdef SUPPORT_COMPLEX
template class AcceleratorMatrix<std::complex<double> >;
template class AcceleratorMatrix<std::complex<float> >;
#endif
template class AcceleratorMatrix<int>;

template class HIPAcceleratorMatrix<double>;
template class HIPAcceleratorMatrix<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorMatrix<std::complex<double> >;
template class HIPAcceleratorMatrix<std::complex<float> >;
#endif
template class HIPAcceleratorMatrix<int>;

}
