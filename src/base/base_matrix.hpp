#ifndef PARALUTION_BASE_MATRIX_HPP_
#define PARALUTION_BASE_MATRIX_HPP_

#include "matrix_formats.hpp"
#include "backend_manager.hpp"

namespace paralution {

template <typename ValueType>
class BaseVector;
template <typename ValueType>
class HostVector;
template <typename ValueType>
class GPUAcceleratorVector;

template <typename ValueType>
class HostMatrixCSR;
template <typename ValueType>
class HostMatrixCOO;
template <typename ValueType>
class HostMatrixDIA;
template <typename ValueType>
class HostMatrixELL;
template <typename ValueType>
class HostMatrixHYB;
template <typename ValueType>
class HostMatrixDENSE;
template <typename ValueType>
class HostMatrixMCSR;
template <typename ValueType>
class HostMatrixBCSR;

template <typename ValueType>
class GPUAcceleratorMatrixCSR;
template <typename ValueType>
class GPUAcceleratorMatrixMCSR;
template <typename ValueType>
class GPUAcceleratorMatrixBCSR;
template <typename ValueType>
class GPUAcceleratorMatrixCOO;
template <typename ValueType>
class GPUAcceleratorMatrixDIA;
template <typename ValueType>
class GPUAcceleratorMatrixELL;
template <typename ValueType>
class GPUAcceleratorMatrixHYB;
template <typename ValueType>
class GPUAcceleratorMatrixDENSE;

/// Base class for all host/accelerator matrices
template <typename ValueType>
class BaseMatrix {

public:

  BaseMatrix();
  virtual ~BaseMatrix();

  /// Return the number of rows in the matrix
  int get_nrow(void) const;
  /// Return the number of columns in the matrix
  int get_ncol(void) const;
  /// Return the non-zeros of the matrix
  int get_nnz(void) const;
  /// Shows simple info about the object
  virtual void info(void) const = 0;
  /// Return the matrix format id (see matrix_formats.hpp)
  virtual unsigned int get_mat_format(void) const = 0;
  /// Copy the backend descriptor information
  virtual void set_backend(const Paralution_Backend_Descriptor local_backend);

  virtual bool Check(void) const;

  /// Allocate CSR Matrix
  virtual void AllocateCSR(const int nnz, const int nrow, const int ncol);
  /// Allocate MCSR Matrix
  virtual void AllocateMCSR(const int nnz, const int nrow, const int ncol);
  /// Allocate COO Matrix
  virtual void AllocateCOO(const int nnz, const int nrow, const int ncol);
  /// Allocate DIA Matrix
  virtual void AllocateDIA(const int nnz, const int nrow, const int ncol, const int ndiag);
  /// Allocate ELL Matrix
  virtual void AllocateELL(const int nnz, const int nrow, const int ncol, const int max_row);
  /// Allocate HYB Matrix
  virtual void AllocateHYB(const int ell_nnz, const int coo_nnz, const int ell_max_row, 
                           const int nrow, const int ncol);
  /// Allocate DENSE Matrix
  virtual void AllocateDENSE(const int nrow, const int ncol);

  /// Initialize a COO matrix on the Host with externally allocated data
  virtual void SetDataPtrCOO(int **row, int **col, ValueType **val,
                             const int nnz, const int nrow, const int ncol);
  /// Leave a COO matrix to Host pointers
  virtual void LeaveDataPtrCOO(int **row, int **col, ValueType **val);

  /// Initialize a CSR matrix on the Host with externally allocated data
  virtual void SetDataPtrCSR(int **row_offset, int **col, ValueType **val,
                             const int nnz, const int nrow, const int ncol);
  /// Leave a CSR matrix to Host pointers
  virtual void LeaveDataPtrCSR(int **row_offset, int **col, ValueType **val);

  /// Initialize a MCSR matrix on the Host with externally allocated data
  virtual void SetDataPtrMCSR(int **row_offset, int **col, ValueType **val,
                              const int nnz, const int nrow, const int ncol);
  /// Leave a MCSR matrix to Host pointers
  virtual void LeaveDataPtrMCSR(int **row_offset, int **col, ValueType **val);

  /// Initialize an ELL matrix on the Host with externally allocated data
  virtual void SetDataPtrELL(int **col, ValueType **val,
                             const int nnz, const int nrow, const int ncol, const int max_row);
  /// Leave an ELL matrix to Host pointers
  virtual void LeaveDataPtrELL(int **col, ValueType **val, int &max_row);

  /// Initialize a DIA matrix on the Host with externally allocated data
  virtual void SetDataPtrDIA(int **offset, ValueType **val,
                             const int nnz, const int nrow, const int ncol, const int num_diag);
  /// Leave a DIA matrix to Host pointers
  virtual void LeaveDataPtrDIA(int **offset, ValueType **val, int &num_diag);

  /// Initialize a DENSE matrix on the Host with externally allocated data
  virtual void SetDataPtrDENSE(ValueType **val, const int nrow, const int ncol);
  /// Leave a DENSE matrix to Host pointers
  virtual void LeaveDataPtrDENSE(ValueType **val);

  /// Clear (free) the matrix
  virtual void Clear(void) = 0;

  /// Set all the values to zero
  virtual bool Zeros(void);

  /// Scale all values
  virtual bool Scale(const ValueType alpha);
  /// Scale the diagonal entries of the matrix with alpha
  virtual bool ScaleDiagonal(const ValueType alpha);
  /// Scale the off-diagonal entries of the matrix with alpha
  virtual bool ScaleOffDiagonal(const ValueType alpha);
  /// Add alpha to all values
  virtual bool AddScalar(const ValueType alpha);
  /// Add alpha to the diagonal entries of the matrix
  virtual bool AddScalarDiagonal(const ValueType alpha);
  /// Add alpha to the off-diagonal entries of the matrix
  virtual bool AddScalarOffDiagonal(const ValueType alpha);

  /// Extrat a sub-matrix with row/col_offset and row/col_size
  virtual bool ExtractSubMatrix(const int row_offset,
                                const int col_offset,
                                const int row_size,
                                const int col_size,
                                BaseMatrix<ValueType> *mat) const;

  /// Extract the diagonal values of the matrix into a LocalVector
  virtual bool ExtractDiagonal(BaseVector<ValueType> *vec_diag) const;
  /// Extract the inverse (reciprocal) diagonal values of the matrix into a LocalVector
  virtual bool ExtractInverseDiagonal(BaseVector<ValueType> *vec_inv_diag) const;
  /// Extract the upper triangular matrix
  virtual bool ExtractU(BaseMatrix<ValueType> *U) const;
  /// Extract the upper triangular matrix including diagonal
  virtual bool ExtractUDiagonal(BaseMatrix<ValueType> *U) const;
  /// Extract the lower triangular matrix
  virtual bool ExtractL(BaseMatrix<ValueType> *L) const;
  /// Extract the lower triangular matrix including diagonal
  virtual bool ExtractLDiagonal(BaseMatrix<ValueType> *L) const;

  /// Perform (forward) permutation of the matrix
  virtual bool Permute(const BaseVector<int> &permutation);

  /// Perform (backward) permutation of the matrix
  virtual bool PermuteBackward(const BaseVector<int> &permutation);

  /// Create permutation vector for CMK reordering of the matrix
  virtual bool CMK(BaseVector<int> *permutation) const;
  /// Create permutation vector for reverse CMK reordering of the matrix
  virtual bool RCMK(BaseVector<int> *permutation) const;
  /// Create permutation vector for connectivity reordering of the matrix (increasing nnz per row)
  virtual bool ConnectivityOrder(BaseVector<int> *permutation) const;

  /// Perform multi-coloring decomposition of the matrix; Returns number of
  /// colors, the corresponding sizes (the array is allocated in the function)
  /// and the permutation
  virtual bool MultiColoring(int &num_colors, int **size_colors, BaseVector<int> *permutation) const;

  /// Perform maximal independent set decomposition of the matrix; Returns the 
  /// size of the maximal independent set and the corresponding permutation
  virtual bool MaximalIndependentSet(int &size, BaseVector<int> *permutation) const;

  /// Return a permutation for saddle-point problems (zero diagonal entries),
  /// where all zero diagonal elements are mapped to the last block;
  /// the return size is the size of the first block
  virtual bool ZeroBlockPermutation(int &size, BaseVector<int> *permutation) const;

  /// Convert the matrix from another matrix (with different structure)
  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat) = 0;

  /// Copy from another matrix
  virtual void CopyFrom(const BaseMatrix<ValueType> &mat) = 0;

  /// Copy to another matrix
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const = 0;

  /// Async copy from another matrix
  virtual void CopyFromAsync(const BaseMatrix<ValueType> &mat);

  /// Copy to another matrix
  virtual void CopyToAsync(BaseMatrix<ValueType> *mat) const;

  /// Copy from CSR array (the matrix has to be allocated)
  virtual void CopyFromCSR(const int *row_offsets, const int *col, const ValueType *val);

  /// Copy to CSR array (the arrays have to be allocated)
  virtual void CopyToCSR(int *row_offsets, int *col, ValueType *val) const;

  /// Copy from COO array (the matrix has to be allocated)
  virtual void CopyFromCOO(const int *row, const int *col, const ValueType *val);

  /// Copy to COO array (the arrays have to be allocated)
  virtual void CopyToCOO(int *row, int *col, ValueType *val) const;

  /// Allocates and copies a host CSR matrix
  virtual void CopyFromHostCSR(const int *row_offset, const int *col, const ValueType *val,
                               const int nnz, const int nrow, const int ncol);

  /// Create a restriction matrix operator based on an int vector map
  virtual bool CreateFromMap(const BaseVector<int> &map, const int n, const int m);
  /// Create a restriction and prolongation matrix operator based on an int vector map
  virtual bool CreateFromMap(const BaseVector<int> &map, const int n, const int m, BaseMatrix<ValueType> *pro);

  /// Read matrix from MTX (Matrix Market Format) file
  virtual bool ReadFileMTX(const std::string filename);
  /// Write matrix to MTX (Matrix Market Format) file
  virtual bool WriteFileMTX(const std::string filename) const;

  /// Read matrix from CSR (PARALUTION binary format) file
  virtual bool ReadFileCSR(const std::string filename);
  /// Write matrix to CSR (PARALUTION binary format) file
  virtual bool WriteFileCSR(const std::string filename) const;

  /// Perform symbolic computation (structure only) of |this|^p
  virtual bool SymbolicPower(const int p);

  /// Perform symbolic matrix-matrix multiplication (i.e. determine the structure),
  /// this = this*src
  virtual bool SymbolicMatMatMult(const BaseMatrix<ValueType> &src);
  /// Multiply two matrices, this = A * B
  virtual bool MatMatMult(const BaseMatrix<ValueType> &A, const BaseMatrix<ValueType> &B);
  /// Perform symbolic matrix-matrix multiplication (i.e. determine the structure),
  /// this = A*B
  virtual bool SymbolicMatMatMult(const BaseMatrix<ValueType> &A, const BaseMatrix<ValueType> &B);
  /// Perform numerical matrix-matrix multiplication (i.e. value computation),
  /// this = A*B
  virtual bool NumericMatMatMult(const BaseMatrix<ValueType> &A, const BaseMatrix<ValueType> &B);
  /// Multiply the matrix with diagonal matrix (stored in LocalVector),
  /// this=this*diag (right multiplication)
  virtual bool DiagonalMatrixMultR(const BaseVector<ValueType> &diag);
  /// Multiply the matrix with diagonal matrix (stored in LocalVector),
  /// this=diag*this (left multiplication)
  virtual bool DiagonalMatrixMultL(const BaseVector<ValueType> &diag);
  /// Perform matrix addition, this = alpha*this + beta*mat;
  /// if structure==false the structure of the matrix is not changed,
  /// if structure==true new data structure is computed
  virtual bool MatrixAdd(const BaseMatrix<ValueType> &mat, const ValueType alpha,
                         const ValueType beta, const bool structure);

  /// Perform ILU(0) factorization
  virtual bool ILU0Factorize(void);
  /// Perform LU factorization
  virtual bool LUFactorize(void);
  /// Perform ILU(t,m) factorization based on threshold and maximum
  /// number of elements per row
  virtual bool ILUTFactorize(const double t, const int maxrow);
  /// Perform ILU(p) factorization based on power (see power(q)-pattern method, D. Lukarski
  ///  "Parallel Sparse Linear Algebra for Multi-core and Many-core Platforms - Parallel Solvers and
  /// Preconditioners", PhD Thesis, 2012, KIT)
  virtual bool ILUpFactorizeNumeric(const int p, const BaseMatrix<ValueType> &mat);

  /// Perform IC(0) factorization
  virtual bool ICFactorize(BaseVector<ValueType> *inv_diag);

  /// Analyse the structure (level-scheduling)
  virtual void LUAnalyse(void);
  /// Delete the analysed data (see LUAnalyse)
  virtual void LUAnalyseClear(void);
  /// Solve LU out = in; if level-scheduling algorithm is provided then the graph
  /// traversing is performed in parallel
  virtual bool LUSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;

  /// Analyse the structure (level-scheduling)
  virtual void LLAnalyse(void);
  /// Delete the analysed data (see LLAnalyse)
  virtual void LLAnalyseClear(void);
  /// Solve LL^T out = in; if level-scheduling algorithm is provided then the graph
  // traversing is performed in parallel
  virtual bool LLSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;
  virtual bool LLSolve(const BaseVector<ValueType> &in, const BaseVector<ValueType> &inv_diag,
                       BaseVector<ValueType> *out) const;

  /// Analyse the structure (level-scheduling) L-part
  /// diag_unit == true the diag is 1;
  /// diag_unit == false the diag is 0;
  virtual void LAnalyse(const bool diag_unit=false);
  /// Delete the analysed data (see LAnalyse) L-party
  virtual void LAnalyseClear(void);
  /// Solve L out = in; if level-scheduling algorithm is provided then the
  /// graph traversing is performed in parallel
  virtual bool LSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;

  /// Analyse the structure (level-scheduling) U-part;
  /// diag_unit == true the diag is 1;
  /// diag_unit == false the diag is 0;
  virtual void UAnalyse(const bool diag_unit=false);
  /// Delete the analysed data (see UAnalyse) U-party
  virtual void UAnalyseClear(void);
  /// Solve U out = in; if level-scheduling algorithm is provided then the
  /// graph traversing is performed in parallel
  virtual bool USolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;

  /// Compute Householder vector
  virtual bool Householder(const int idx, ValueType &beta, BaseVector<ValueType> *vec) const;
  /// QR Decomposition
  virtual bool QRDecompose(void);
  /// Solve QR out = in
  virtual bool QRSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;

  /// Invert this
  virtual bool Invert(void);

  /// Compute the spectrum approximation with Gershgorin circles theorem
  virtual bool Gershgorin(ValueType &lambda_min, ValueType &lambda_max) const;

  /// Apply the matrix to vector, out = this*in;
  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const = 0;
  /// Apply and add the matrix to vector, out = out + scalar*this*in;
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const = 0;

  /// Delete all entries abs(a_ij) <= drop_off;
  /// the diagonal elements are never deleted
  virtual bool Compress(const double drop_off);

  /// Transpose the matrix
  virtual bool Transpose(void);

  /// Sort the matrix indices
  virtual bool Sort(void);

  // Return key for row, col and val
  virtual bool Key(long int &row_key,
                   long int &col_key,
                   long int &val_key) const;

  /// Replace a column vector of a matrix 
  virtual bool ReplaceColumnVector(const int idx, const BaseVector<ValueType> &vec);

  /// Replace a column vector of a matrix
  virtual bool ReplaceRowVector(const int idx, const BaseVector<ValueType> &vec);

  /// Extract values from a column of a matrix to a vector
  virtual bool ExtractColumnVector(const int idx, BaseVector<ValueType> *vec) const;

  /// Extract values from a row of a matrix to a vector
  virtual bool ExtractRowVector(const int idx, BaseVector<ValueType> *vec) const;

  virtual bool AMGConnect(const ValueType eps, BaseVector<int> *connections) const;
  virtual bool AMGAggregate(const BaseVector<int> &connections, BaseVector<int> *aggregates) const;
  virtual bool AMGSmoothedAggregation(const ValueType relax,
                                      const BaseVector<int> &aggregates,
                                      const BaseVector<int> &connections,
                                            BaseMatrix<ValueType> *prolong,
                                            BaseMatrix<ValueType> *restrict) const;
  virtual bool AMGAggregation(const BaseVector<int> &aggregates,
                                    BaseMatrix<ValueType> *prolong,
                                    BaseMatrix<ValueType> *restrict) const;

  /// Ruge Stüben coarsening
  virtual bool RugeStueben(const ValueType eps, BaseMatrix<ValueType> *prolong,
                                                BaseMatrix<ValueType> *restrict) const;

  /// Factorized Sparse Approximate Inverse assembly for given system
  /// matrix power pattern or external sparsity pattern
  virtual bool FSAI(const int power, const BaseMatrix<ValueType> *pattern);

  /// SParse Approximate Inverse assembly for given system matrix pattern
  virtual bool SPAI(void);

  /// Initial Pairwise Aggregation scheme
  virtual bool InitialPairwiseAggregation(const ValueType beta, int &nc, BaseVector<int> *G, int &Gsize,
                                          int **rG, int &rGsize, const int ordering) const;
  /// Initial Pairwise Aggregation scheme for split matrices
  virtual bool InitialPairwiseAggregation(const BaseMatrix<ValueType> &mat, const ValueType beta, int &nc,
                                          BaseVector<int> *G, int &Gsize, int **rG, int &rGsize,
                                          const int ordering) const;
  /// Further Pairwise Aggregation scheme
  virtual bool FurtherPairwiseAggregation(const ValueType beta, int &nc, BaseVector<int> *G, int &Gsize,
                                          int **rG, int &rGsize, const int ordering) const;
  /// Further Pairwise Aggregation scheme for split matrices
  virtual bool FurtherPairwiseAggregation(const BaseMatrix<ValueType> &mat, const ValueType beta, int &nc,
                                          BaseVector<int> *G, int &Gsize, int **rG, int &rGsize,
                                          const int ordering) const;
  /// Build coarse operator for pairwise aggregation scheme
  virtual bool CoarsenOperator(BaseMatrix<ValueType> *Ac, const int nrow, const int ncol, const BaseVector<int> &G,
                               const int Gsize, const int *rG, const int rGsize) const;

protected:

  /// Number of rows
  int nrow_;
  /// Number of columns
  int ncol_;
  /// Number of non-zero elements
  int nnz_;

  /// Backend descriptor (local copy)
  Paralution_Backend_Descriptor local_backend_;

  friend class BaseVector<ValueType>;
  friend class HostVector<ValueType>;
  friend class AcceleratorVector<ValueType>;
  friend class GPUAcceleratorVector<ValueType>;

};

template <typename ValueType>
class HostMatrix : public BaseMatrix<ValueType> {

public:

  HostMatrix();
  virtual ~HostMatrix();

};

template <typename ValueType>
class AcceleratorMatrix : public BaseMatrix<ValueType> {

public:

  AcceleratorMatrix();
  virtual ~AcceleratorMatrix();

  /// Copy (accelerator matrix) from host matrix
  virtual void CopyFromHost(const HostMatrix<ValueType> &src) = 0;

  /// Async copy (accelerator matrix) from host matrix
  virtual void CopyFromHostAsync(const HostMatrix<ValueType> &src);

  /// Copy (accelerator matrix) to host matrix
  virtual void CopyToHost(HostMatrix<ValueType> *dst) const = 0;

  /// Async opy (accelerator matrix) to host matrix
  virtual void CopyToHostAsync(HostMatrix<ValueType> *dst) const;

};

template <typename ValueType>
class GPUAcceleratorMatrix : public AcceleratorMatrix<ValueType> {

public:

  GPUAcceleratorMatrix();
  virtual ~GPUAcceleratorMatrix();

};

}

#endif // PARALUTION_BASE_MATRIX_HPP_
