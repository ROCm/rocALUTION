/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "backend_manager.hpp"
#include "matrix_formats.hpp"
#include "operator.hpp"
#include "rocalution/export.hpp"

namespace rocalution
{

    template <typename ValueType>
    class BaseMatrix;

    template <typename ValueType>
    class LocalVector;
    template <typename ValueType>
    class GlobalVector;

    template <typename ValueType>
    class GlobalMatrix;

    /** \ingroup op_vec_module
  * \class LocalMatrix
  * \brief LocalMatrix class
  * \details
  * A LocalMatrix is called local, because it will always stay on a single system. The
  * system can contain several CPUs via UMA or NUMA memory system or it can contain an
  * accelerator.
  *
  * \tparam ValueType - can be int, float, double, std::complex<float> and
  *                     std::complex<double>
  *
  * A number of matrix formats are supported. These are CSR, BCSR, MCSR, COO, DIA, ELL, HYB, and DENSE.
  * \note For CSR type matrices, the column indices must be sorted in increasing order. For COO matrices, the row
  * indices must be sorted in increasing order. The function \p Check can be used to check whether a matrix
  * contains valid data. For CSR and COO matrices, the function \p Sort can be used to sort the row or column
  * indices respectively.
  */
    template <typename ValueType>
    class LocalMatrix : public Operator<ValueType>
    {
    public:
        ROCALUTION_EXPORT
        LocalMatrix();
        ROCALUTION_EXPORT
        virtual ~LocalMatrix();

        ROCALUTION_EXPORT
        virtual void Info(void) const;

        /** \brief Return the matrix format id (see matrix_formats.hpp) */
        ROCALUTION_EXPORT
        unsigned int GetFormat(void) const;
        /** \brief Return the matrix block dimension */
        ROCALUTION_EXPORT
        int GetBlockDimension(void) const;

        ROCALUTION_EXPORT
        virtual IndexType2 GetM(void) const;
        ROCALUTION_EXPORT
        virtual IndexType2 GetN(void) const;
        ROCALUTION_EXPORT
        virtual IndexType2 GetNnz(void) const;

        /** \brief Perform a sanity check of the matrix
      * \details
      * Checks, if the matrix contains valid data, i.e. if the values are not infinity
      * and not NaN (not a number) and if the structure of the matrix is correct (e.g.
      * indices cannot be negative, CSR and COO matrices have to be sorted, etc.).
      *
      * \retval true if the matrix is ok (empty matrix is also ok).
      * \retval false if there is something wrong with the structure or values.
      */
        ROCALUTION_EXPORT
        bool Check(void) const;

        /** \brief Allocate a local matrix with name and sizes
      * \details
      * The local matrix allocation functions require a name of the object (this is only
      * for information purposes) and corresponding number of non-zero elements, number
      * of rows and number of columns. Furthermore, depending on the matrix format,
      * additional parameters are required.
      *
      * \par Example
      * \code{.cpp}
      *   LocalMatrix<ValueType> mat;
      *
      *   mat.AllocateCSR("my CSR matrix", 456, 100, 100);
      *   mat.Clear();
      *
      *   mat.AllocateCOO("my COO matrix", 200, 100, 100);
      *   mat.Clear();
      * \endcode
      */
        /**@{*/
        ROCALUTION_EXPORT
        void AllocateCSR(const std::string& name, int nnz, int nrow, int ncol);
        ROCALUTION_EXPORT
        void AllocateBCSR(const std::string& name, int nnzb, int nrowb, int ncolb, int blockdim);
        ROCALUTION_EXPORT
        void AllocateMCSR(const std::string& name, int nnz, int nrow, int ncol);
        ROCALUTION_EXPORT
        void AllocateCOO(const std::string& name, int nnz, int nrow, int ncol);
        ROCALUTION_EXPORT
        void AllocateDIA(const std::string& name, int nnz, int nrow, int ncol, int ndiag);
        ROCALUTION_EXPORT
        void AllocateELL(const std::string& name, int nnz, int nrow, int ncol, int max_row);
        ROCALUTION_EXPORT
        void AllocateHYB(
            const std::string& name, int ell_nnz, int coo_nnz, int ell_max_row, int nrow, int ncol);
        ROCALUTION_EXPORT
        void AllocateDENSE(const std::string& name, int nrow, int ncol);
        /**@}*/

        /** \brief Initialize a LocalMatrix on the host with externally allocated data
      * \details
      * \p SetDataPtr functions have direct access to the raw data via pointers. Already
      * allocated data can be set by passing their pointers.
      *
      * \note
      * Setting data pointers will leave the original pointers empty (set to \p NULL).
      *
      * \par Example
      * \code{.cpp}
      *   // Allocate a CSR matrix
      *   int* csr_row_ptr   = new int[100 + 1];
      *   int* csr_col_ind   = new int[345];
      *   ValueType* csr_val = new ValueType[345];
      *
      *   // Fill the CSR matrix
      *   // ...
      *
      *   // rocALUTION local matrix object
      *   LocalMatrix<ValueType> mat;
      *
      *   // Set the CSR matrix data, csr_row_ptr, csr_col and csr_val pointers become
      *   // invalid
      *   mat.SetDataPtrCSR(&csr_row_ptr, &csr_col, &csr_val, "my_matrix", 345, 100, 100);
      * \endcode
      */
        /**@{*/
        ROCALUTION_EXPORT
        void SetDataPtrCOO(
            int** row, int** col, ValueType** val, std::string name, int nnz, int nrow, int ncol);
        ROCALUTION_EXPORT
        void SetDataPtrCSR(int**       row_offset,
                           int**       col,
                           ValueType** val,
                           std::string name,
                           int         nnz,
                           int         nrow,
                           int         ncol);
        ROCALUTION_EXPORT
        void SetDataPtrBCSR(int**       row_offset,
                            int**       col,
                            ValueType** val,
                            std::string name,
                            int         nnzb,
                            int         nrowb,
                            int         ncolb,
                            int         blockdim);
        ROCALUTION_EXPORT
        void SetDataPtrMCSR(int**       row_offset,
                            int**       col,
                            ValueType** val,
                            std::string name,
                            int         nnz,
                            int         nrow,
                            int         ncol);
        ROCALUTION_EXPORT
        void SetDataPtrELL(
            int** col, ValueType** val, std::string name, int nnz, int nrow, int ncol, int max_row);
        ROCALUTION_EXPORT
        void SetDataPtrDIA(int**       offset,
                           ValueType** val,
                           std::string name,
                           int         nnz,
                           int         nrow,
                           int         ncol,
                           int         num_diag);
        ROCALUTION_EXPORT
        void SetDataPtrDENSE(ValueType** val, std::string name, int nrow, int ncol);
        /**@}*/

        /** \brief Leave a LocalMatrix to host pointers
      * \details
      * \p LeaveDataPtr functions have direct access to the raw data via pointers. A
      * LocalMatrix object can leave its raw data to host pointers. This will leave the
      * LocalMatrix empty.
      *
      * \par Example
      * \code{.cpp}
      *   // rocALUTION CSR matrix object
      *   LocalMatrix<ValueType> mat;
      *
      *   // Allocate the CSR matrix
      *   mat.AllocateCSR("my_matrix", 345, 100, 100);
      *
      *   // Fill CSR matrix
      *   // ...
      *
      *   int* csr_row_ptr   = NULL;
      *   int* csr_col_ind   = NULL;
      *   ValueType* csr_val = NULL;
      *
      *   // Get (steal) the data from the matrix, this will leave the local matrix
      *   // object empty
      *   mat.LeaveDataPtrCSR(&csr_row_ptr, &csr_col_ind, &csr_val);
      * \endcode
      */
        /**@{*/
        ROCALUTION_EXPORT
        void LeaveDataPtrCOO(int** row, int** col, ValueType** val);
        ROCALUTION_EXPORT
        void LeaveDataPtrCSR(int** row_offset, int** col, ValueType** val);
        ROCALUTION_EXPORT
        void LeaveDataPtrBCSR(int** row_offset, int** col, ValueType** val, int& blockdim);
        ROCALUTION_EXPORT
        void LeaveDataPtrMCSR(int** row_offset, int** col, ValueType** val);
        ROCALUTION_EXPORT
        void LeaveDataPtrELL(int** col, ValueType** val, int& max_row);
        ROCALUTION_EXPORT
        void LeaveDataPtrDIA(int** offset, ValueType** val, int& num_diag);
        ROCALUTION_EXPORT
        void LeaveDataPtrDENSE(ValueType** val);
        /**@}*/

        ROCALUTION_EXPORT
        void Clear(void);

        /** \brief Set all matrix values to zero */
        ROCALUTION_EXPORT
        void Zeros(void);

        /** \brief Scale all values in the matrix */
        ROCALUTION_EXPORT
        void Scale(ValueType alpha);
        /** \brief Scale the diagonal entries of the matrix with alpha, all diagonal elements
      * must exist
      */
        ROCALUTION_EXPORT
        void ScaleDiagonal(ValueType alpha);
        /** \brief Scale the off-diagonal entries of the matrix with alpha, all diagonal
      * elements must exist */
        ROCALUTION_EXPORT
        void ScaleOffDiagonal(ValueType alpha);

        /** \brief Add a scalar to all matrix values */
        ROCALUTION_EXPORT
        void AddScalar(ValueType alpha);
        /** \brief Add alpha to the diagonal entries of the matrix, all diagonal elements
      * must exist
      */
        ROCALUTION_EXPORT
        void AddScalarDiagonal(ValueType alpha);
        /** \brief Add alpha to the off-diagonal entries of the matrix, all diagonal elements
      * must exist
      */
        ROCALUTION_EXPORT
        void AddScalarOffDiagonal(ValueType alpha);

        /** \brief Extract a sub-matrix with row/col_offset and row/col_size */
        ROCALUTION_EXPORT
        void ExtractSubMatrix(int                     row_offset,
                              int                     col_offset,
                              int                     row_size,
                              int                     col_size,
                              LocalMatrix<ValueType>* mat) const;

        /** \brief Extract array of non-overlapping sub-matrices (row/col_num_blocks define
      * the blocks for rows/columns; row/col_offset have sizes col/row_num_blocks+1,
      * where [i+1]-[i] defines the i-th size of the sub-matrix)
      */
        ROCALUTION_EXPORT
        void ExtractSubMatrices(int                       row_num_blocks,
                                int                       col_num_blocks,
                                const int*                row_offset,
                                const int*                col_offset,
                                LocalMatrix<ValueType>*** mat) const;

        /** \brief Extract the diagonal values of the matrix into a LocalVector */
        ROCALUTION_EXPORT
        void ExtractDiagonal(LocalVector<ValueType>* vec_diag) const;

        /** \brief Extract the inverse (reciprocal) diagonal values of the matrix into a
      * LocalVector
      */
        ROCALUTION_EXPORT
        void ExtractInverseDiagonal(LocalVector<ValueType>* vec_inv_diag) const;

        /** \brief Extract the upper triangular matrix */
        ROCALUTION_EXPORT
        void ExtractU(LocalMatrix<ValueType>* U, bool diag) const;
        /** \brief Extract the lower triangular matrix */
        ROCALUTION_EXPORT
        void ExtractL(LocalMatrix<ValueType>* L, bool diag) const;

        /** \brief Perform (forward) permutation of the matrix */
        ROCALUTION_EXPORT
        void Permute(const LocalVector<int>& permutation);

        /** \brief Perform (backward) permutation of the matrix */
        ROCALUTION_EXPORT
        void PermuteBackward(const LocalVector<int>& permutation);

        /** \brief Create permutation vector for CMK reordering of the matrix
      * \details
      * The Cuthill-McKee ordering minimize the bandwidth of a given sparse matrix.
      *
      * @param[out]
      * permutation permutation vector for CMK reordering
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<int> cmk;
      *
      *   mat.CMK(&cmk);
      *   mat.Permute(cmk);
      * \endcode
      */
        ROCALUTION_EXPORT
        void CMK(LocalVector<int>* permutation) const;

        /** \brief Create permutation vector for reverse CMK reordering of the matrix
      * \details
      * The Reverse Cuthill-McKee ordering minimize the bandwidth of a given sparse
      * matrix.
      *
      * @param[out]
      * permutation permutation vector for reverse CMK reordering
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<int> rcmk;
      *
      *   mat.RCMK(&rcmk);
      *   mat.Permute(rcmk);
      * \endcode
      */
        ROCALUTION_EXPORT
        void RCMK(LocalVector<int>* permutation) const;

        /** \brief Create permutation vector for connectivity reordering of the matrix
      * \details
      * Connectivity ordering returns a permutation, that sorts the matrix by non-zero
      * entries per row.
      *
      * @param[out]
      * permutation permutation vector for connectivity reordering
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<int> conn;
      *
      *   mat.ConnectivityOrder(&conn);
      *   mat.Permute(conn);
      * \endcode
      */
        ROCALUTION_EXPORT
        void ConnectivityOrder(LocalVector<int>* permutation) const;

        /** \brief Perform multi-coloring decomposition of the matrix
      * \details
      * The Multi-Coloring algorithm builds a permutation (coloring of the matrix) in a
      * way such that no two adjacent nodes in the sparse matrix have the same color.
      *
      * @param[out]
      * num_colors  number of colors
      * @param[out]
      * size_colors pointer to array that holds the number of nodes for each color
      * @param[out]
      * permutation permutation vector for multi-coloring reordering
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<int> mc;
      *   int num_colors;
      *   int* block_colors = NULL;
      *
      *   mat.MultiColoring(num_colors, &block_colors, &mc);
      *   mat.Permute(mc);
      * \endcode
      */
        ROCALUTION_EXPORT
        void MultiColoring(int& num_colors, int** size_colors, LocalVector<int>* permutation) const;

        /** \brief Perform maximal independent set decomposition of the matrix
      * \details
      * The Maximal Independent Set algorithm finds a set with maximal size, that
      * contains elements that do not depend on other elements in this set.
      *
      * @param[out]
      * size        number of independent sets
      * @param[out]
      * permutation permutation vector for maximal independent set reordering
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<int> mis;
      *   int size;
      *
      *   mat.MaximalIndependentSet(size, &mis);
      *   mat.Permute(mis);
      * \endcode
      */
        ROCALUTION_EXPORT
        void MaximalIndependentSet(int& size, LocalVector<int>* permutation) const;

        /** \brief Return a permutation for saddle-point problems (zero diagonal entries)
      * \details
      * For Saddle-Point problems, (i.e. matrices with zero diagonal entries), the Zero
      * Block Permutation maps all zero-diagonal elements to the last block of the
      * matrix.
      *
      * @param[out]
      * size
      * @param[out]
      * permutation permutation vector for zero block permutation
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<int> zbp;
      *   int size;
      *
      *   mat.ZeroBlockPermutation(size, &zbp);
      *   mat.Permute(zbp);
      * \endcode

      */
        ROCALUTION_EXPORT
        void ZeroBlockPermutation(int& size, LocalVector<int>* permutation) const;

        /** \brief Perform ILU(0) factorization */
        ROCALUTION_EXPORT
        void ILU0Factorize(void);
        /** \brief Perform LU factorization */
        ROCALUTION_EXPORT
        void LUFactorize(void);

        /** \brief Perform ILU(t,m) factorization based on threshold and maximum number of
      * elements per row
      */
        ROCALUTION_EXPORT
        void ILUTFactorize(double t, int maxrow);

        /** \brief Perform ILU(p) factorization based on power */
        ROCALUTION_EXPORT
        void ILUpFactorize(int p, bool level = true);
        /** \brief Analyse the structure (level-scheduling) */
        ROCALUTION_EXPORT
        void LUAnalyse(void);
        /** \brief Delete the analysed data (see LUAnalyse) */
        ROCALUTION_EXPORT
        void LUAnalyseClear(void);
        /** \brief Solve LU out = in; if level-scheduling algorithm is provided then the
      * graph traversing is performed in parallel
      */
        ROCALUTION_EXPORT
        void LUSolve(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;

        /** \brief Perform IC(0) factorization */
        ROCALUTION_EXPORT
        void ICFactorize(LocalVector<ValueType>* inv_diag);

        /** \brief Analyse the structure (level-scheduling) */
        ROCALUTION_EXPORT
        void LLAnalyse(void);
        /** \brief Delete the analysed data (see LLAnalyse) */
        ROCALUTION_EXPORT
        void LLAnalyseClear(void);
        /** \brief Solve LL^T out = in; if level-scheduling algorithm is provided then the
      * graph traversing is performed in parallel
      */
        ROCALUTION_EXPORT
        void LLSolve(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;
        /** \brief Solve LL^T out = in; if level-scheduling algorithm is provided then the
      * graph traversing is performed in parallel
      */
        ROCALUTION_EXPORT
        void LLSolve(const LocalVector<ValueType>& in,
                     const LocalVector<ValueType>& inv_diag,
                     LocalVector<ValueType>*       out) const;

        /** \brief Analyse the structure (level-scheduling) L-part
      * - diag_unit == true the diag is 1;
      * - diag_unit == false the diag is 0;
      */
        ROCALUTION_EXPORT
        void LAnalyse(bool diag_unit = false);
        /** \brief Delete the analysed data (see LAnalyse) L-part */
        ROCALUTION_EXPORT
        void LAnalyseClear(void);
        /** \brief Solve L out = in; if level-scheduling algorithm is provided then the
      * graph traversing is performed in parallel
      */
        ROCALUTION_EXPORT
        void LSolve(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;

        /** \brief Analyse the structure (level-scheduling) U-part;
      * - diag_unit == true the diag is 1;
      * - diag_unit == false the diag is 0;
      */
        ROCALUTION_EXPORT
        void UAnalyse(bool diag_unit = false);
        /** \brief Delete the analysed data (see UAnalyse) U-part */
        ROCALUTION_EXPORT
        void UAnalyseClear(void);
        /** \brief Solve U out = in; if level-scheduling algorithm is provided then the
      * graph traversing is performed in parallel
      */
        ROCALUTION_EXPORT
        void USolve(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;

        /** \brief Compute Householder vector */
        ROCALUTION_EXPORT
        void Householder(int idx, ValueType& beta, LocalVector<ValueType>* vec) const;
        /** \brief QR Decomposition */
        ROCALUTION_EXPORT
        void QRDecompose(void);
        /** \brief Solve QR out = in */
        ROCALUTION_EXPORT
        void QRSolve(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;

        /** \brief Matrix inversion using QR decomposition */
        ROCALUTION_EXPORT
        void Invert(void);

        /** \brief Read matrix from MTX (Matrix Market Format) file
      * \details
      * Read a matrix from Matrix Market Format file.
      *
      * @param[in]
      * filename    name of the file containing the MTX data.
      *
      * \par Example
      * \code{.cpp}
      *   LocalMatrix<ValueType> mat;
      *   mat.ReadFileMTX("my_matrix.mtx");
      * \endcode
      */
        ROCALUTION_EXPORT
        void ReadFileMTX(const std::string& filename);

        /** \brief Write matrix to MTX (Matrix Market Format) file
      * \details
      * Write a matrix to Matrix Market Format file.
      *
      * @param[in]
      * filename    name of the file to write the MTX data to.
      *
      * \par Example
      * \code{.cpp}
      *   LocalMatrix<ValueType> mat;
      *
      *   // Allocate and fill mat
      *   // ...
      *
      *   mat.WriteFileMTX("my_matrix.mtx");
      * \endcode
      */
        ROCALUTION_EXPORT
        void WriteFileMTX(const std::string& filename) const;

        /** \brief Read matrix from CSR (rocALUTION binary format) file
      * \details
      * Read a CSR matrix from binary file. For details on the format, see
      * WriteFileCSR().
      *
      * @param[in]
      * filename    name of the file containing the data.
      *
      * \par Example
      * \code{.cpp}
      *   LocalMatrix<ValueType> mat;
      *   mat.ReadFileCSR("my_matrix.csr");
      * \endcode
      */
        ROCALUTION_EXPORT
        void ReadFileCSR(const std::string& filename);

        /** \brief Write CSR matrix to binary file
      * \details
      * Write a CSR matrix to binary file.
      *
      * The binary format contains a header, the rocALUTION version and the matrix data
      * as follows
      * \code{.cpp}
      *   // Header
      *   out << "#rocALUTION binary csr file" << std::endl;
      *
      *   // rocALUTION version
      *   out.write((char*)&version, sizeof(int));
      *
      *   // CSR matrix data
      *   out.write((char*)&m, sizeof(int));
      *   out.write((char*)&n, sizeof(int));
      *   out.write((char*)&nnz, sizeof(int));
      *   out.write((char*)csr_row_ptr, (m + 1) * sizeof(int));
      *   out.write((char*)csr_col_ind, nnz * sizeof(int));
      *   out.write((char*)csr_val, nnz * sizeof(double));
      * \endcode
      *
      * \note
      * Vector values array is always stored in double precision (e.g. double or
      * std::complex<double>).
      *
      * @param[in]
      * filename    name of the file to write the data to.
      *
      * \par Example
      * \code{.cpp}
      *   LocalMatrix<ValueType> mat;
      *
      *   // Allocate and fill mat
      *   // ...
      *
      *   mat.WriteFileCSR("my_matrix.csr");
      * \endcode
      */
        ROCALUTION_EXPORT
        void WriteFileCSR(const std::string& filename) const;

        ROCALUTION_EXPORT
        virtual void MoveToAccelerator(void);
        ROCALUTION_EXPORT
        virtual void MoveToAcceleratorAsync(void);
        ROCALUTION_EXPORT
        virtual void MoveToHost(void);
        ROCALUTION_EXPORT
        virtual void MoveToHostAsync(void);
        ROCALUTION_EXPORT
        virtual void Sync(void);

        /** \brief Copy matrix from another LocalMatrix
      * \details
      * \p CopyFrom copies values and structure from another local matrix. Source and
      * destination matrix should be in the same format.
      *
      * \note
      * This function allows cross platform copying. One of the objects could be
      * allocated on the accelerator backend.
      *
      * @param[in]
      * src Local matrix where values and structure should be copied from.
      *
      * \par Example
      * \code{.cpp}
      *   LocalMatrix<ValueType> mat1, mat2;
      *
      *   // Allocate and initialize mat1 and mat2
      *   // ...
      *
      *   // Move mat1 to accelerator
      *   // mat1.MoveToAccelerator();
      *
      *   // Now, mat1 is on the accelerator (if available)
      *   // and mat2 is on the host
      *
      *   // Copy mat1 to mat2 (or vice versa) will move data between host and
      *   // accelerator backend
      *   mat1.CopyFrom(mat2);
      * \endcode
      */
        ROCALUTION_EXPORT
        void CopyFrom(const LocalMatrix<ValueType>& src);

        /** \brief Async copy matrix (values and structure) from another LocalMatrix */
        ROCALUTION_EXPORT
        void CopyFromAsync(const LocalMatrix<ValueType>& src);

        /** \brief Clone the matrix
      * \details
      * \p CloneFrom clones the entire matrix, including values, structure and backend
      * descriptor from another LocalMatrix.
      *
      * @param[in]
      * src LocalMatrix to clone from.
      *
      * \par Example
      * \code{.cpp}
      *   LocalMatrix<ValueType> mat;
      *
      *   // Allocate and initialize mat (host or accelerator)
      *   // ...
      *
      *   LocalMatrix<ValueType> tmp;
      *
      *   // By cloning mat, tmp will have identical values and structure and will be on
      *   // the same backend as mat
      *   tmp.CloneFrom(mat);
      * \endcode
      */
        ROCALUTION_EXPORT
        void CloneFrom(const LocalMatrix<ValueType>& src);

        /** \brief Update CSR matrix entries only, structure will remain the same */
        ROCALUTION_EXPORT
        void UpdateValuesCSR(ValueType* val);

        /** \brief Copy (import) CSR matrix described in three arrays (offsets, columns,
      * values). The object data has to be allocated (call AllocateCSR first)
      */
        ROCALUTION_EXPORT
        void CopyFromCSR(const int* row_offsets, const int* col, const ValueType* val);

        /** \brief Copy (export) CSR matrix described in three arrays (offsets, columns,
      * values). The output arrays have to be allocated
      */
        ROCALUTION_EXPORT
        void CopyToCSR(int* row_offsets, int* col, ValueType* val) const;

        /** \brief Copy (import) COO matrix described in three arrays (rows, columns,
      * values). The object data has to be allocated (call AllocateCOO first)
      */
        ROCALUTION_EXPORT
        void CopyFromCOO(const int* row, const int* col, const ValueType* val);

        /** \brief Copy (export) COO matrix described in three arrays (rows, columns,
      * values). The output arrays have to be allocated
      */
        ROCALUTION_EXPORT
        void CopyToCOO(int* row, int* col, ValueType* val) const;

        /** \brief Allocates and copies (imports) a host CSR matrix
      * \details
      * If the CSR matrix data pointers are only accessible as constant, the user can
      * create a LocalMatrix object and pass const CSR host pointers. The LocalMatrix
      * will then be allocated and the data will be copied to the corresponding backend,
      * where the original object was located at.
      *
      * @param[in]
      * row_offset  CSR matrix row offset pointers.
      * @param[in]
      * col         CSR matrix column indices.
      * @param[in]
      * val         CSR matrix values array.
      * @param[in]
      * name        Matrix object name.
      * @param[in]
      * nnz         Number of non-zero elements.
      * @param[in]
      * nrow        Number of rows.
      * @param[in]
      * ncol        Number of columns.
      */
        ROCALUTION_EXPORT
        void CopyFromHostCSR(const int*         row_offset,
                             const int*         col,
                             const ValueType*   val,
                             const std::string& name,
                             int                nnz,
                             int                nrow,
                             int                ncol);

        /** \brief Create a restriction matrix operator based on an int vector map */
        ROCALUTION_EXPORT
        void CreateFromMap(const LocalVector<int>& map, int n, int m);
        /** \brief Create a restriction and prolongation matrix operator based on an int
      * vector map
      */
        ROCALUTION_EXPORT
        void CreateFromMap(const LocalVector<int>& map, int n, int m, LocalMatrix<ValueType>* pro);

        /** \brief Convert the matrix to CSR structure */
        ROCALUTION_EXPORT
        void ConvertToCSR(void);
        /** \brief Convert the matrix to MCSR structure */
        ROCALUTION_EXPORT
        void ConvertToMCSR(void);
        /** \brief Convert the matrix to BCSR structure */
        ROCALUTION_EXPORT
        void ConvertToBCSR(int blockdim);
        /** \brief Convert the matrix to COO structure */
        ROCALUTION_EXPORT
        void ConvertToCOO(void);
        /** \brief Convert the matrix to ELL structure */
        ROCALUTION_EXPORT
        void ConvertToELL(void);
        /** \brief Convert the matrix to DIA structure */
        ROCALUTION_EXPORT
        void ConvertToDIA(void);
        /** \brief Convert the matrix to HYB structure */
        ROCALUTION_EXPORT
        void ConvertToHYB(void);
        /** \brief Convert the matrix to DENSE structure */
        ROCALUTION_EXPORT
        void ConvertToDENSE(void);
        /** \brief Convert the matrix to specified matrix ID format */
        ROCALUTION_EXPORT
        void ConvertTo(unsigned int matrix_format, int blockdim = 1);

        ROCALUTION_EXPORT
        virtual void Apply(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;
        ROCALUTION_EXPORT
        virtual void ApplyAdd(const LocalVector<ValueType>& in,
                              ValueType                     scalar,
                              LocalVector<ValueType>*       out) const;

        /** \brief Perform symbolic computation (structure only) of \f$|this|^p\f$ */
        ROCALUTION_EXPORT
        void SymbolicPower(int p);

        /** \brief Perform matrix addition, this = alpha*this + beta*mat;
      * - if structure==false the sparsity pattern of the matrix is not changed;
      * - if structure==true a new sparsity pattern is computed
      */
        ROCALUTION_EXPORT
        void MatrixAdd(const LocalMatrix<ValueType>& mat,
                       ValueType                     alpha     = static_cast<ValueType>(1),
                       ValueType                     beta      = static_cast<ValueType>(1),
                       bool                          structure = false);

        /** \brief Multiply two matrices, this = A * B */
        ROCALUTION_EXPORT
        void MatrixMult(const LocalMatrix<ValueType>& A, const LocalMatrix<ValueType>& B);

        /** \brief Multiply the matrix with diagonal matrix (stored in LocalVector), as
      * DiagonalMatrixMultR()
      */
        ROCALUTION_EXPORT
        void DiagonalMatrixMult(const LocalVector<ValueType>& diag);

        /** \brief Multiply the matrix with diagonal matrix (stored in LocalVector),
      * this=diag*this
      */
        ROCALUTION_EXPORT
        void DiagonalMatrixMultL(const LocalVector<ValueType>& diag);

        /** \brief Multiply the matrix with diagonal matrix (stored in LocalVector),
      * this=this*diag
      */
        ROCALUTION_EXPORT
        void DiagonalMatrixMultR(const LocalVector<ValueType>& diag);

        /** \brief Compute the spectrum approximation with Gershgorin circles theorem */
        ROCALUTION_EXPORT
        void Gershgorin(ValueType& lambda_min, ValueType& lambda_max) const;

        /** \brief Delete all entries in the matrix which abs(a_ij) <= drop_off;
      * the diagonal elements are never deleted
      */
        ROCALUTION_EXPORT
        void Compress(double drop_off);

        /** \brief Transpose the matrix */
        ROCALUTION_EXPORT
        void Transpose(void);

        /** \brief Transpose the matrix */
        ROCALUTION_EXPORT
        void Transpose(LocalMatrix<ValueType>* T) const;

        /** \brief Sort the matrix indices
      * \details
      * Sorts the matrix by indices.
      * - For CSR matrices, column values are sorted.
      * - For COO matrices, row indices are sorted.
      */
        ROCALUTION_EXPORT
        void Sort(void);

        /** \brief Compute a unique hash key for the matrix arrays
      * \details
      * Typically, it is hard to compare if two matrices have the same structure (and
      * values). To do so, rocALUTION provides a keying function, that generates three
      * keys, for the row index, column index and values array.
      *
      * @param[out]
      * row_key row index array key
      * @param[out]
      * col_key column index array key
      * @param[out]
      * val_key values array key
      */
        ROCALUTION_EXPORT
        void Key(long int& row_key, long int& col_key, long int& val_key) const;

        /** \brief Replace a column vector of a matrix */
        ROCALUTION_EXPORT
        void ReplaceColumnVector(int idx, const LocalVector<ValueType>& vec);

        /** \brief Replace a row vector of a matrix */
        ROCALUTION_EXPORT
        void ReplaceRowVector(int idx, const LocalVector<ValueType>& vec);

        /** \brief Extract values from a column of a matrix to a vector */
        ROCALUTION_EXPORT
        void ExtractColumnVector(int idx, LocalVector<ValueType>* vec) const;

        /** \brief Extract values from a row of a matrix to a vector */
        ROCALUTION_EXPORT
        void ExtractRowVector(int idx, LocalVector<ValueType>* vec) const;

        /** \brief Strong couplings for aggregation-based AMG */
        ROCALUTION_EXPORT
        void AMGConnect(ValueType eps, LocalVector<int>* connections) const;
        /** \brief Plain aggregation - Modification of a greedy aggregation scheme from
      * Vanek (1996)
      */
        ROCALUTION_EXPORT
        void AMGAggregate(const LocalVector<int>& connections, LocalVector<int>* aggregates) const;

        /** \brief Parallel aggregation - Parallel maximal independent set aggregation scheme from
      * Bell, Dalton, & Olsen (2012)
      */
        ROCALUTION_EXPORT
        void AMGPMISAggregate(const LocalVector<int>& connections,
                              LocalVector<int>*       aggregates) const;

        /** \brief Interpolation scheme based on smoothed aggregation from Vanek (1996) */
        ROCALUTION_EXPORT
        void AMGSmoothedAggregation(ValueType               relax,
                                    const LocalVector<int>& aggregates,
                                    const LocalVector<int>& connections,
                                    LocalMatrix<ValueType>* prolong,
                                    LocalMatrix<ValueType>* restrict,
                                    int lumping_strat = 0) const;
        /** \brief Aggregation-based interpolation scheme */
        ROCALUTION_EXPORT
        void AMGAggregation(const LocalVector<int>& aggregates,
                            LocalMatrix<ValueType>* prolong,
                            LocalMatrix<ValueType>* restrict) const;

        /** \brief Ruge Stueben coarsening */
        ROCALUTION_EXPORT
        void RSCoarsening(float eps, LocalVector<int>* CFmap, LocalVector<bool>* S) const;
        /** \brief Parallel maximal independent set coarsening for RS AMG*/
        ROCALUTION_EXPORT
        void RSPMISCoarsening(float eps, LocalVector<int>* CFmap, LocalVector<bool>* S) const;

        /** \brief Ruge Stueben Direct Interpolation */
        ROCALUTION_EXPORT
        void RSDirectInterpolation(const LocalVector<int>&  CFmap,
                                   const LocalVector<bool>& S,
                                   LocalMatrix<ValueType>*  prolong,
                                   LocalMatrix<ValueType>* restrict) const;

        /** \brief Ruge Stueben Ext+i Interpolation */
        ROCALUTION_EXPORT
        void RSExtPIInterpolation(const LocalVector<int>&  CFmap,
                                  const LocalVector<bool>& S,
                                  bool                     FF1,
                                  float                    trunc,
                                  LocalMatrix<ValueType>*  prolong,
                                  LocalMatrix<ValueType>* restrict) const;

        /** \brief Generate Ruge Stueben operators */
        ROCALUTION_EXPORT
        void RugeStueben(float                   eps,
                         LocalMatrix<ValueType>* prolong,
                         LocalMatrix<ValueType>* restrict) const;

        /** \brief Factorized Sparse Approximate Inverse assembly for given system matrix
      * power pattern or external sparsity pattern
      */
        ROCALUTION_EXPORT
        void FSAI(int power, const LocalMatrix<ValueType>* pattern);

        /** \brief SParse Approximate Inverse assembly for given system matrix pattern */
        ROCALUTION_EXPORT
        void SPAI(void);

        /** \brief Initial Pairwise Aggregation scheme */
        ROCALUTION_EXPORT
        void InitialPairwiseAggregation(ValueType         beta,
                                        int&              nc,
                                        LocalVector<int>* G,
                                        int&              Gsize,
                                        int**             rG,
                                        int&              rGsize,
                                        int               ordering) const;
        /** \brief Initial Pairwise Aggregation scheme for split matrices */
        ROCALUTION_EXPORT
        void InitialPairwiseAggregation(const LocalMatrix<ValueType>& mat,
                                        ValueType                     beta,
                                        int&                          nc,
                                        LocalVector<int>*             G,
                                        int&                          Gsize,
                                        int**                         rG,
                                        int&                          rGsize,
                                        int                           ordering) const;
        /** \brief Further Pairwise Aggregation scheme */
        ROCALUTION_EXPORT
        void FurtherPairwiseAggregation(ValueType         beta,
                                        int&              nc,
                                        LocalVector<int>* G,
                                        int&              Gsize,
                                        int**             rG,
                                        int&              rGsize,
                                        int               ordering) const;
        /** \brief Further Pairwise Aggregation scheme for split matrices */
        ROCALUTION_EXPORT
        void FurtherPairwiseAggregation(const LocalMatrix<ValueType>& mat,
                                        ValueType                     beta,
                                        int&                          nc,
                                        LocalVector<int>*             G,
                                        int&                          Gsize,
                                        int**                         rG,
                                        int&                          rGsize,
                                        int                           ordering) const;
        /** \brief Build coarse operator for pairwise aggregation scheme */
        ROCALUTION_EXPORT
        void CoarsenOperator(LocalMatrix<ValueType>* Ac,
                             ParallelManager*        pm,
                             int                     nrow,
                             int                     ncol,
                             const LocalVector<int>& G,
                             int                     Gsize,
                             const int*              rG,
                             int                     rGsize) const;

    protected:
        virtual bool is_host_(void) const;
        virtual bool is_accel_(void) const;

    private:
        // Pointer from the base matrix class to the current
        // allocated matrix (host_ or accel_)
        BaseMatrix<ValueType>* matrix_;

        // Host Matrix
        HostMatrix<ValueType>* matrix_host_;

        // Accelerator Matrix
        AcceleratorMatrix<ValueType>* matrix_accel_;

        friend class LocalVector<ValueType>;
        friend class GlobalVector<ValueType>;
        friend class GlobalMatrix<ValueType>;
    };

} // namespace rocalution

#endif // ROCALUTION_LOCAL_MATRIX_HPP_
