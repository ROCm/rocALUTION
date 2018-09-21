/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HIP_MATRIX_CSR_HPP_
#define ROCALUTION_HIP_MATRIX_CSR_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

#include <rocsparse.h>

namespace rocalution {

template <typename ValueType>
class HIPAcceleratorMatrixCSR : public HIPAcceleratorMatrix<ValueType>
{
    public:
    HIPAcceleratorMatrixCSR();
    HIPAcceleratorMatrixCSR(const Rocalution_Backend_Descriptor local_backend);
    virtual ~HIPAcceleratorMatrixCSR();

    virtual void Info(void) const;
    virtual unsigned int GetMatFormat(void) const { return CSR; }

    virtual void Clear(void);
    virtual bool Zeros(void);

    virtual void AllocateCSR(const int nnz, const int nrow, const int ncol);
    virtual void SetDataPtrCSR(int** row_offset,
                               int** col,
                               ValueType** val,
                               const int nnz,
                               const int nrow,
                               const int ncol);
    virtual void LeaveDataPtrCSR(int** row_offset, int** col, ValueType** val);

    virtual bool ConvertFrom(const BaseMatrix<ValueType>& mat);

    virtual void CopyFrom(const BaseMatrix<ValueType>& mat);
    virtual void CopyFromAsync(const BaseMatrix<ValueType>& mat);
    virtual void CopyTo(BaseMatrix<ValueType>* mat) const;
    virtual void CopyToAsync(BaseMatrix<ValueType>* mat) const;

    virtual void CopyFromHost(const HostMatrix<ValueType>& src);
    virtual void CopyFromHostAsync(const HostMatrix<ValueType>& src);
    virtual void CopyToHost(HostMatrix<ValueType>* dst) const;
    virtual void CopyToHostAsync(HostMatrix<ValueType>* dst) const;

    virtual void CopyFromCSR(const int* row_offsets, const int* col, const ValueType* val);
    virtual void CopyToCSR(int* row_offsets, int* col, ValueType* val) const;

    virtual void CopyFromHostCSR(const int* row_offset,
                                 const int* col,
                                 const ValueType* val,
                                 const int nnz,
                                 const int nrow,
                                 const int ncol);

    virtual bool Permute(const BaseVector<int>& permutation);

    virtual bool Scale(const ValueType alpha);
    virtual bool ScaleDiagonal(const ValueType alpha);
    virtual bool ScaleOffDiagonal(const ValueType alpha);
    virtual bool AddScalar(const ValueType alpha);
    virtual bool AddScalarDiagonal(const ValueType alpha);
    virtual bool AddScalarOffDiagonal(const ValueType alpha);

    virtual bool ExtractSubMatrix(const int row_offset,
                                  const int col_offset,
                                  const int row_size,
                                  const int col_size,
                                  BaseMatrix<ValueType>* mat) const;

    virtual bool ExtractDiagonal(BaseVector<ValueType>* vec_diag) const;
    virtual bool ExtractInverseDiagonal(BaseVector<ValueType>* vec_inv_diag) const;
    virtual bool ExtractL(BaseMatrix<ValueType>* L) const;
    virtual bool ExtractLDiagonal(BaseMatrix<ValueType>* L) const;

    virtual bool ExtractU(BaseMatrix<ValueType>* U) const;
    virtual bool ExtractUDiagonal(BaseMatrix<ValueType>* U) const;

    virtual bool MaximalIndependentSet(int& size, BaseVector<int>* permutation) const;
    virtual bool
    MultiColoring(int& num_colors, int** size_colors, BaseVector<int>* permutation) const;

    virtual bool DiagonalMatrixMultR(const BaseVector<ValueType>& diag);
    virtual bool DiagonalMatrixMultL(const BaseVector<ValueType>& diag);

    virtual bool MatMatMult(const BaseMatrix<ValueType>& A, const BaseMatrix<ValueType>& B);

    virtual bool MatrixAdd(const BaseMatrix<ValueType>& mat,
                           const ValueType alpha,
                           const ValueType beta,
                           const bool structure);

    virtual bool ILU0Factorize(void);

    virtual bool ICFactorize(BaseVector<ValueType>* inv_diag = NULL);

    virtual void LUAnalyse(void);
    virtual void LUAnalyseClear(void);
    virtual bool LUSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

    virtual void LLAnalyse(void);
    virtual void LLAnalyseClear(void);
    virtual bool LLSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
    virtual bool LLSolve(const BaseVector<ValueType>& in,
                         const BaseVector<ValueType>& inv_diag,
                         BaseVector<ValueType>* out) const;

    virtual void LAnalyse(const bool diag_unit = false);
    virtual void LAnalyseClear(void);
    virtual bool LSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

    virtual void UAnalyse(const bool diag_unit = false);
    virtual void UAnalyseClear(void);
    virtual bool USolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

    virtual bool Gershgorin(ValueType& lambda_min, ValueType& lambda_max) const;

    void ApplyAnalysis(void);
    virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
    virtual void ApplyAdd(const BaseVector<ValueType>& in,
                          const ValueType scalar,
                          BaseVector<ValueType>* out) const;

    virtual bool Compress(const double drop_off);

    virtual bool ReplaceColumnVector(const int idx, const BaseVector<ValueType>& vec);
    virtual bool ExtractColumnVector(const int idx, BaseVector<ValueType>* vec) const;
    virtual bool ExtractRowVector(const int idx, BaseVector<ValueType>* vec) const;

    virtual bool Transpose(void);

    private:
    MatrixCSR<ValueType, int> mat_;

    rocsparse_mat_descr L_mat_descr_;
    rocsparse_mat_descr U_mat_descr_;
    rocsparse_mat_descr mat_descr_;

    rocsparse_mat_info mat_info_;

    // Matrix buffer (csrilu0, csric0, csrsv)
    size_t mat_buffer_size_;
    void* mat_buffer_;

    HIPAcceleratorVector<ValueType>* tmp_vec_;

    friend class HIPAcceleratorMatrixCOO<ValueType>;
    friend class HIPAcceleratorMatrixDIA<ValueType>;
    friend class HIPAcceleratorMatrixELL<ValueType>;
    friend class HIPAcceleratorMatrixHYB<ValueType>;

    friend class BaseVector<ValueType>;
    friend class AcceleratorVector<ValueType>;
    friend class HIPAcceleratorVector<ValueType>;
};

} // namespace rocalution

#endif // ROCALUTION_HIP_MATRIX_CSR_HPP_
