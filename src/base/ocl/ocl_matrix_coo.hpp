#ifndef PARALUTION_OCL_MATRIX_COO_HPP_
#define PARALUTION_OCL_MATRIX_COO_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace paralution {

template <class ValueType>
class OCLAcceleratorMatrixCOO : public OCLAcceleratorMatrix<ValueType> {

public:

  OCLAcceleratorMatrixCOO();
  OCLAcceleratorMatrixCOO(const Paralution_Backend_Descriptor local_backend);
  virtual ~OCLAcceleratorMatrixCOO();

  virtual void info(void) const;
  virtual unsigned int get_mat_format(void) const { return COO; }

  virtual void Clear(void);
  virtual void AllocateCOO(const int nnz, const int nrow, const int ncol);
  virtual void SetDataPtrCOO(int **row, int **col, ValueType **val,
                             const int nnz, const int nrow, const int ncol);
  virtual void LeaveDataPtrCOO(int **row, int **col, ValueType **val);

  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat);

  virtual bool Permute(const BaseVector<int> &permutation);
  virtual bool PermuteBackward(const BaseVector<int> &permutation);

  virtual void CopyFrom(const BaseMatrix<ValueType> &mat);
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const;

  virtual void CopyFromHost(const HostMatrix<ValueType> &src);
  virtual void CopyToHost(HostMatrix<ValueType> *dst) const;

  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar, BaseVector<ValueType> *out) const;

private:

  MatrixCOO<ValueType, int> mat_;

  friend class BaseVector<ValueType>;
  friend class AcceleratorVector<ValueType>;
  friend class OCLAcceleratorVector<ValueType>;

};


}

#endif // PARALUTION_OCL_MATRIX_COO_HPP_
