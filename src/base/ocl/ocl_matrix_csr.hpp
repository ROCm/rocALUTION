#ifndef PARALUTION_OCL_MATRIX_CSR_HPP_
#define PARALUTION_OCL_MATRIX_CSR_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace paralution {

template <class ValueType>
class OCLAcceleratorMatrixCSR : public OCLAcceleratorMatrix<ValueType> {

public:

  OCLAcceleratorMatrixCSR();
  OCLAcceleratorMatrixCSR(const Paralution_Backend_Descriptor local_backend);
  virtual ~OCLAcceleratorMatrixCSR();

  virtual void info(void) const;
  virtual unsigned int get_mat_format(void) const { return CSR; }

  virtual void Clear(void);
  virtual bool Zeros(void);

  virtual void AllocateCSR(const int nnz, const int nrow, const int ncol);
  virtual void SetDataPtrCSR(int **row_offset, int **col, ValueType **val,
                             const int nnz, const int nrow, const int ncol);
  virtual void LeaveDataPtrCSR(int **row_offset, int **col, ValueType **val);

  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat);

  virtual void CopyFrom(const BaseMatrix<ValueType> &mat);
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const;

  virtual void CopyFromHost(const HostMatrix<ValueType> &src);
  virtual void CopyToHost(HostMatrix<ValueType> *dst) const;

  virtual void CopyFromHostCSR(const int *row_offset, const int *col, const ValueType *val,
                               const int nnz, const int nrow, const int ncol);

  virtual bool Permute(const BaseVector<int> &permutation);

  virtual bool Scale(const ValueType alpha);
  virtual bool ScaleDiagonal(const ValueType alpha);
  virtual bool ScaleOffDiagonal(const ValueType alpha);
  virtual bool AddScalar(const ValueType alpha);
  virtual bool AddScalarDiagonal(const ValueType alpha);
  virtual bool AddScalarOffDiagonal(const ValueType alpha);

  virtual bool ExtractSubMatrix(const int row_offset, const int col_offset,
                                const int row_size, const int col_size,
                                BaseMatrix<ValueType> *mat) const;

  virtual bool ExtractDiagonal(BaseVector<ValueType> *vec_diag) const;
  virtual bool ExtractInverseDiagonal(BaseVector<ValueType> *vec_inv_diag) const;
  virtual bool ExtractL(BaseMatrix<ValueType> *L) const;
  virtual bool ExtractLDiagonal(BaseMatrix<ValueType> *L) const;

  virtual bool ExtractU(BaseMatrix<ValueType> *U) const;
  virtual bool ExtractUDiagonal(BaseMatrix<ValueType> *U) const;

  virtual bool MaximalIndependentSet(int &size, BaseVector<int> *permutation) const;
  virtual bool MultiColoring(int &num_colors, int **size_colors, BaseVector<int> *permutation) const;

  virtual bool DiagonalMatrixMultR(const BaseVector<ValueType> &diag);
  virtual bool DiagonalMatrixMultL(const BaseVector<ValueType> &diag);

  virtual bool MatrixAdd(const BaseMatrix<ValueType> &mat, const ValueType alpha,
                         const ValueType beta, const bool structure);

  virtual void LUAnalyse(void);
  virtual void LUAnalyseClear(void);

  virtual void LLAnalyse(void);
  virtual void LLAnalyseClear(void);

  virtual void LAnalyse(const bool diag_unit = false);
  virtual void LAnalyseClear(void);

  virtual void UAnalyse(const bool diag_unit = false);
  virtual void UAnalyseClear(void);

  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar, BaseVector<ValueType> *out) const;

  virtual bool Compress(const double drop_off);

  virtual bool ReplaceColumnVector(const int idx, const BaseVector<ValueType> &vec);
  virtual bool ExtractColumnVector(const int idx, BaseVector<ValueType> *vec) const;

private:

  MatrixCSR<ValueType, int> mat_;

  friend class OCLAcceleratorMatrixCOO<ValueType>;
  friend class OCLAcceleratorMatrixDIA<ValueType>;
  friend class OCLAcceleratorMatrixELL<ValueType>;
  friend class OCLAcceleratorMatrixHYB<ValueType>;

  friend class BaseVector<ValueType>;
  friend class AcceleratorVector<ValueType>;
  friend class OCLAcceleratorVector<ValueType>;

  OCLAcceleratorVector<ValueType> *tmp_vec_;

};


}

#endif // PARALUTION_OCL_MATRIX_CSR_HPP_
