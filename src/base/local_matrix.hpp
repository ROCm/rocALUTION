/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_LOCAL_MATRIX_HPP_
#define ROCALUTION_LOCAL_MATRIX_HPP_

#include "../utils/types.hpp"
#include "operator.hpp"
#include "backend_manager.hpp"
#include "matrix_formats.hpp"

namespace rocalution {

template <typename ValueType>
class BaseMatrix;

template <typename ValueType>
class LocalVector;
template <typename ValueType>
class GlobalVector;

template <typename ValueType>
class GlobalMatrix;

// Local Matrix
template <typename ValueType>
class LocalMatrix : public Operator<ValueType>
{
    public:
    LocalMatrix();
    virtual ~LocalMatrix();

    virtual void Info(void) const;

    /// Return the matrix format id (see matrix_formats.hpp)
    unsigned int GetFormat(void) const;

    virtual IndexType2 GetM(void) const;
    virtual IndexType2 GetN(void) const;
    virtual IndexType2 GetNnz(void) const;

    /// Return true if the matrix is ok (empty matrix is also ok)
    //  and false if there is something wrong with the strcture or some of values are NaN
    bool Check(void) const;

    /// Allocate CSR Matrix
    void AllocateCSR(const std::string name, int nnz, int nrow, int ncol);
    /// Allocate BCSR Matrix - not implemented
    void AllocateBCSR(void){};
    /// Allocate MCSR Matrix
    void AllocateMCSR(const std::string name, int nnz, int nrow, int ncol);
    /// Allocate COO Matrix
    void AllocateCOO(const std::string name, int nnz, int nrow, int ncol);
    /// Allocate DIA Matrix
    void AllocateDIA(const std::string name, int nnz, int nrow, int ncol, int ndiag);
    /// Allocate ELL Matrix
    void AllocateELL(const std::string name, int nnz, int nrow, int ncol, int max_row);
    /// Allocate HYB Matrix
    void AllocateHYB(
        const std::string name, int ell_nnz, int coo_nnz, int ell_max_row, int nrow, int ncol);
    /// Allocate DENSE Matrix
    void AllocateDENSE(const std::string name, int nrow, int ncol);

    /// Initialize a COO matrix on the Host with externally allocated data
    void SetDataPtrCOO(
        int** row, int** col, ValueType** val, std::string name, int nnz, int nrow, int ncol);
    /// Leave a COO matrix to Host pointers
    void LeaveDataPtrCOO(int** row, int** col, ValueType** val);

    /// Initialize a CSR matrix on the Host with externally allocated data
    void SetDataPtrCSR(int** row_offset,
                       int** col,
                       ValueType** val,
                       std::string name,
                       int nnz,
                       int nrow,
                       int ncol);
    /// Leave a CSR matrix to Host pointers
    void LeaveDataPtrCSR(int** row_offset, int** col, ValueType** val);

    /// Initialize a MCSR matrix on the Host with externally allocated data
    void SetDataPtrMCSR(int** row_offset,
                        int** col,
                        ValueType** val,
                        std::string name,
                        int nnz,
                        int nrow,
                        int ncol);
    /// Leave a MCSR matrix to Host pointers
    void LeaveDataPtrMCSR(int** row_offset, int** col, ValueType** val);

    /// Initialize an ELL matrix on the Host with externally allocated data
    void SetDataPtrELL(
        int** col, ValueType** val, std::string name, int nnz, int nrow, int ncol, int max_row);
    /// Leave an ELL matrix to Host pointers
    void LeaveDataPtrELL(int** col, ValueType** val, int& max_row);

    /// Initialize a DIA matrix on the Host with externally allocated data
    void SetDataPtrDIA(
        int** offset, ValueType** val, std::string name, int nnz, int nrow, int ncol, int num_diag);
    /// Leave a DIA matrix to Host pointers
    void LeaveDataPtrDIA(int** offset, ValueType** val, int& num_diag);

    /// Initialize a DENSE matrix on the Host with externally allocated data
    void SetDataPtrDENSE(ValueType** val, std::string name, int nrow, int ncol);
    /// Leave a DENSE matrix to Host pointers
    void LeaveDataPtrDENSE(ValueType** val);

    void Clear(void);

    /// Set all the values to zero
    void Zeros(void);

    /// Scale all the values in the matrix
    void Scale(ValueType alpha);
    /// Scale the diagonal entries of the matrix with alpha,
    /// the diagonal elements must exist
    void ScaleDiagonal(ValueType alpha);
    /// Scale the off-diagonal entries of the matrix with alpha,
    /// the diagonal elements must exist
    void ScaleOffDiagonal(ValueType alpha);

    /// Add a scalar to all values
    void AddScalar(ValueType alpha);
    /// Add alpha to the diagonal entries of the matrix,
    /// the diagonal elements must exist
    void AddScalarDiagonal(ValueType alpha);
    /// Add alpha to the off-diagonal entries of the matrix,
    /// the diagonal elements must exist
    void AddScalarOffDiagonal(ValueType alpha);

    /// Extract a sub-matrix with row/col_offset and row/col_size
    void ExtractSubMatrix(int row_offset,
                          int col_offset,
                          int row_size,
                          int col_size,
                          LocalMatrix<ValueType>* mat) const;

    /// Extract array of non-overlapping sub-matrices (row/col_num_blocks define the
    /// blocks for rows/columns; row/col_offset have sizes col/row_num_blocks+1, where [i+1]-[i]
    /// defines the i-th size of the sub-matrix)
    void ExtractSubMatrices(int row_num_blocks,
                            int col_num_blocks,
                            const int* row_offset,
                            const int* col_offset,
                            LocalMatrix<ValueType>*** mat) const;

    /// Extract the diagonal values of the matrix into a LocalVector
    void ExtractDiagonal(LocalVector<ValueType>* vec_diag) const;

    /// Extract the inverse (reciprocal) diagonal values of the matrix into a LocalVector
    void ExtractInverseDiagonal(LocalVector<ValueType>* vec_inv_diag) const;

    /// Extract the upper triangular matrix
    void ExtractU(LocalMatrix<ValueType>* U, bool diag) const;
    /// Extract the lower triangular matrix
    void ExtractL(LocalMatrix<ValueType>* L, bool diag) const;

    /// Perform (forward) permutation of the matrix
    void Permute(const LocalVector<int>& permutation);

    /// Perform (backward) permutation of the matrix
    void PermuteBackward(const LocalVector<int>& permutation);

    /// Create permutation vector for CMK reordering of the matrix
    void CMK(LocalVector<int>* permutation) const;
    /// Create permutation vector for reverse CMK reordering of the matrix
    void RCMK(LocalVector<int>* permutation) const;
    /// Create permutation vector for connectivity reordering of the matrix (increasing nnz per row)
    void ConnectivityOrder(LocalVector<int>* permutation) const;

    /// Perform multi-coloring decomposition of the matrix; Returns
    /// number of colors, the corresponding sizes (the array is allocated in
    /// the function) and the permutation
    void MultiColoring(int& num_colors, int** size_colors, LocalVector<int>* permutation) const;

    /// Perform maximal independent set decomposition of the matrix;
    /// Returns the size of the maximal independent set and the
    /// corresponding permutation
    void MaximalIndependentSet(int& size, LocalVector<int>* permutation) const;

    /// Return a permutation for saddle-point problems (zero diagonal entries),
    /// where all zero diagonal elements are mapped to the last block;
    /// the return size is the size of the first block
    void ZeroBlockPermutation(int& size, LocalVector<int>* permutation) const;

    /// Perform ILU(0) factorization
    void ILU0Factorize(void);
    /// Perform LU factorization
    void LUFactorize(void);

    /// Perform ILU(t,m) factorization based on threshold and maximum
    /// number of elements per row
    void ILUTFactorize(double t, int maxrow);

    /// Perform ILU(p) factorization based on power (see power(q)-pattern method,
    /// D. Lukarski "Parallel Sparse Linear Algebra for Multi-core and Many-core
    /// Platforms - Parallel Solvers and Preconditioners", PhD Thesis, 2012, KIT);
    /// level==true build the structure based on levels; level==false build the
    /// structure only based on the power(p+1)
    void ILUpFactorize(int p, bool level = true);
    /// Analyse the structure (level-scheduling)
    void LUAnalyse(void);
    /// Delete the analysed data (see LUAnalyse)
    void LUAnalyseClear(void);
    /// Solve LU out = in; if level-scheduling algorithm is provided then the
    /// graph traversing is performed in parallel
    void LUSolve(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;

    /// Perform IC(0) factorization
    void ICFactorize(LocalVector<ValueType>* inv_diag);

    /// Analyse the structure (level-scheduling)
    void LLAnalyse(void);
    /// Delete the analysed data (see LLAnalyse)
    void LLAnalyseClear(void);
    /// Solve LL^T out = in; if level-scheduling algorithm is provided then the
    /// graph traversing is performed in parallel
    void LLSolve(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;
    void LLSolve(const LocalVector<ValueType>& in,
                 const LocalVector<ValueType>& inv_diag,
                 LocalVector<ValueType>* out) const;

    /// Analyse the structure (level-scheduling) L-part
    /// diag_unit == true the diag is 1;
    /// diag_unit == false the diag is 0;
    void LAnalyse(bool diag_unit = false);
    /// Delete the analysed data (see LAnalyse) L-party
    void LAnalyseClear(void);
    /// Solve L out = in; if level-scheduling algorithm is provided then the
    /// graph traversing is performed in parallel
    void LSolve(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;

    /// Analyse the structure (level-scheduling) U-part;
    /// diag_unit == true the diag is 1;
    /// diag_unit == false the diag is 0;
    void UAnalyse(bool diag_unit = false);
    /// Delete the analysed data (see UAnalyse) U-party
    void UAnalyseClear(void);
    /// Solve U out = in; if level-scheduling algorithm is provided then the
    /// graph traversing is performed in parallel
    void USolve(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;

    /// Compute Householder vector
    void Householder(int idx, ValueType& beta, LocalVector<ValueType>* vec) const;
    /// QR Decomposition
    void QRDecompose(void);
    /// Solve QR out = in
    void QRSolve(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;

    /// Matrix inversion using QR decomposition
    void Invert(void);

    /// Read matrix from MTX (Matrix Market Format) file
    void ReadFileMTX(const std::string filename);
    /// Write matrix to MTX (Matrix Market Format) file
    void WriteFileMTX(const std::string filename) const;

    /// Read matrix from CSR (ROCALUTION binary format) file
    void ReadFileCSR(const std::string filename);
    /// Write matrix to CSR (ROCALUTION binary format) file
    void WriteFileCSR(const std::string filename) const;

    virtual void MoveToAccelerator(void);
    virtual void MoveToAcceleratorAsync(void);
    virtual void MoveToHost(void);
    virtual void MoveToHostAsync(void);
    virtual void Sync(void);

    /// Copy matrix (values and structure) from another LocalMatrix
    void CopyFrom(const LocalMatrix<ValueType>& src);

    /// Async copy matrix (values and structure) from another LocalMatrix
    void CopyFromAsync(const LocalMatrix<ValueType>& src);

    /// Clone the entire matrix (values,structure+backend descr)
    /// from another LocalMatrix
    void CloneFrom(const LocalMatrix<ValueType>& src);

    /// Update CSR matrix entries only, structure will remain the same
    void UpdateValuesCSR(ValueType* val);

    /// Copy (import) CSR matrix described in three arrays (offsets, columns, values).
    /// The object data has to be allocated (call AllocateCSR first)
    void CopyFromCSR(const int* row_offsets, const int* col, const ValueType* val);

    /// Copy (export) CSR matrix described in three arrays
    /// (offsets, columns, values). The output arrays have to be allocated
    void CopyToCSR(int* row_offsets, int* col, ValueType* val) const;

    /// Copy (import) COO matrix described in three arrays (rows, columns, values).
    /// The object data has to be allocated (call AllocateCOO first)
    void CopyFromCOO(const int* row, const int* col, const ValueType* val);

    /// Copy (export) COO matrix described in three arrays
    /// (rows, columns, values). The output arrays have to be allocated
    void CopyToCOO(int* row, int* col, ValueType* val) const;

    /// Allocates and copies (import) a host CSR matrix
    void CopyFromHostCSR(const int* row_offset,
                         const int* col,
                         const ValueType* val,
                         const std::string name,
                         int nnz,
                         int nrow,
                         int ncol);

    /// Create a restriction matrix operator based on an int vector map
    void CreateFromMap(const LocalVector<int>& map, int n, int m);
    /// Create a restriction and prolongation matrix operator based on an int vector map
    void CreateFromMap(const LocalVector<int>& map, int n, int m, LocalMatrix<ValueType>* pro);

    /// Convert the matrix to CSR structure
    void ConvertToCSR(void);
    /// Convert the matrix to MCSR structure
    void ConvertToMCSR(void);
    /// Convert the matrix to BCSR structure
    void ConvertToBCSR(void);
    /// Convert the matrix to COO structure
    void ConvertToCOO(void);
    /// Convert the matrix to ELL structure
    void ConvertToELL(void);
    /// Convert the matrix to DIA structure
    void ConvertToDIA(void);
    /// Convert the matrix to HYB structure
    void ConvertToHYB(void);
    /// Convert the matrix to DENSE structure
    void ConvertToDENSE(void);
    /// Convert the matrix to specified matrix ID format
    void ConvertTo(unsigned int matrix_format);

    virtual void Apply(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;
    virtual void
    ApplyAdd(const LocalVector<ValueType>& in, ValueType scalar, LocalVector<ValueType>* out) const;

    /// Perform symbolic computation (structure only) of |this|^p
    void SymbolicPower(int p);

    /// Perform matrix addition, this = alpha*this + beta*mat;
    /// if structure==false the sparsity pattern of the matrix is not changed;
    /// if structure==true a new sparsity pattern is computed
    void MatrixAdd(const LocalMatrix<ValueType>& mat,
                   ValueType alpha = static_cast<ValueType>(1),
                   ValueType beta  = static_cast<ValueType>(1),
                   bool structure  = false);

    /// Multiply two matrices, this = A * B
    void MatrixMult(const LocalMatrix<ValueType>& A, const LocalMatrix<ValueType>& B);

    /// Multiply the matrix with diagonal matrix (stored in LocalVector),
    /// as DiagonalMatrixMultR()
    void DiagonalMatrixMult(const LocalVector<ValueType>& diag);

    /// Multiply the matrix with diagonal matrix (stored in LocalVector),
    /// this=diag*this
    void DiagonalMatrixMultL(const LocalVector<ValueType>& diag);

    /// Multiply the matrix with diagonal matrix (stored in LocalVector),
    /// this=this*diag
    void DiagonalMatrixMultR(const LocalVector<ValueType>& diag);

    /// Compute the spectrum approximation with Gershgorin circles theorem
    void Gershgorin(ValueType& lambda_min, ValueType& lambda_max) const;

    /// Delete all entries in the matrix which abs(a_ij) <= drop_off;
    /// the diagonal elements are never deleted;
    void Compress(double drop_off);

    /// Transpose the matrix
    void Transpose(void);

    /// Sort the matrix indices
    void Sort(void);

    void Key(long int& row_key, long int& col_key, long int& val_key) const;

    /// Replace a column vector of a matrix
    void ReplaceColumnVector(int idx, const LocalVector<ValueType>& vec);

    /// Replace a row vector of a matrix
    void ReplaceRowVector(int idx, const LocalVector<ValueType>& vec);

    /// Extract values from a column of a matrix to a vector
    void ExtractColumnVector(int idx, LocalVector<ValueType>* vec) const;

    /// Extract values from a row of a matrix to a vector
    void ExtractRowVector(int idx, LocalVector<ValueType>* vec) const;

    /// Strong couplings for aggregation-based AMG
    void AMGConnect(ValueType eps, LocalVector<int>* connections) const;
    /// Plain aggregation - Modification of a greedy aggregation scheme from Vanek (1996)
    void AMGAggregate(const LocalVector<int>& connections, LocalVector<int>* aggregates) const;
    /// Interpolation scheme based on smoothed aggregation from Vanek (1996)
    void AMGSmoothedAggregation(ValueType relax,
                                const LocalVector<int>& aggregates,
                                const LocalVector<int>& connections,
                                LocalMatrix<ValueType>* prolong,
                                LocalMatrix<ValueType>* restrict) const;
    /// Aggregation-based interpolation scheme
    void AMGAggregation(const LocalVector<int>& aggregates,
                        LocalMatrix<ValueType>* prolong,
                        LocalMatrix<ValueType>* restrict) const;

    /// Ruge Stüben coarsening
    void RugeStueben(ValueType eps,
                     LocalMatrix<ValueType>* prolong,
                     LocalMatrix<ValueType>* restrict) const;

    /// Factorized Sparse Approximate Inverse assembly for given system
    /// matrix power pattern or external sparsity pattern
    void FSAI(int power, const LocalMatrix<ValueType>* pattern);

    /// SParse Approximate Inverse assembly for given system matrix pattern
    void SPAI(void);

    /// Initial Pairwise Aggregation scheme
    void InitialPairwiseAggregation(ValueType beta,
                                    int& nc,
                                    LocalVector<int>* G,
                                    int& Gsize,
                                    int** rG,
                                    int& rGsize,
                                    int ordering) const;
    /// Initial Pairwise Aggregation scheme for split matrices
    void InitialPairwiseAggregation(const LocalMatrix<ValueType>& mat,
                                    ValueType beta,
                                    int& nc,
                                    LocalVector<int>* G,
                                    int& Gsize,
                                    int** rG,
                                    int& rGsize,
                                    int ordering) const;
    /// Further Pairwise Aggregation scheme
    void FurtherPairwiseAggregation(ValueType beta,
                                    int& nc,
                                    LocalVector<int>* G,
                                    int& Gsize,
                                    int** rG,
                                    int& rGsize,
                                    int ordering) const;
    /// Further Pairwise Aggregation scheme for split matrices
    void FurtherPairwiseAggregation(const LocalMatrix<ValueType>& mat,
                                    ValueType beta,
                                    int& nc,
                                    LocalVector<int>* G,
                                    int& Gsize,
                                    int** rG,
                                    int& rGsize,
                                    int ordering) const;
    /// Build coarse operator for pairwise aggregation scheme
    void CoarsenOperator(LocalMatrix<ValueType>* Ac,
                         int nrow,
                         int ncol,
                         const LocalVector<int>& G,
                         int Gsize,
                         const int* rG,
                         int rGsize) const;

    protected:
    virtual bool is_host(void) const;
    virtual bool is_accel(void) const;

    private:
    /// Pointer from the base matrix class to the current
    /// allocated matrix (host_ or accel_)
    BaseMatrix<ValueType>* matrix_;

    /// Host Matrix
    HostMatrix<ValueType>* matrix_host_;

    /// Accelerator Matrix
    AcceleratorMatrix<ValueType>* matrix_accel_;

    int* assembly_rank;
    int* assembly_irank;
    int* assembly_loop_start;
    int* assembly_loop_end;
    int assembly_threads;

    void free_assembly_data(void);

    friend class LocalVector<ValueType>;
    friend class GlobalVector<ValueType>;
    friend class GlobalMatrix<ValueType>;
};

} // namespace rocalution

#endif // ROCALUTION_LOCAL_MATRIX_HPP_
