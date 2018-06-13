#ifndef ROCALUTION_HIP_MATRIX_COO_HPP_
#define ROCALUTION_HIP_MATRIX_COO_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

#include <hipsparse.h>

namespace rocalution {

template <typename ValueType>
class HIPAcceleratorMatrixCOO : public HIPAcceleratorMatrix<ValueType> {

public:

  HIPAcceleratorMatrixCOO();
  HIPAcceleratorMatrixCOO(const Rocalution_Backend_Descriptor local_backend);
  virtual ~HIPAcceleratorMatrixCOO();

  virtual void Info(void) const;
  virtual unsigned int GetMatFormat(void) const { return COO; }

  virtual void Clear(void);
  virtual void AllocateCOO(const int nnz, const int nrow, const int ncol);

  virtual void SetDataPtrCOO(int **row, int **col, ValueType **val,
                             const int nnz, const int nrow, const int ncol);
  virtual void LeaveDataPtrCOO(int **row, int **col, ValueType **val);

  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat);

  virtual bool Permute(const BaseVector<int> &permutation);
  virtual bool PermuteBackward(const BaseVector<int> &permutation);

  virtual void CopyFrom(const BaseMatrix<ValueType> &mat);
  virtual void CopyFromAsync(const BaseMatrix<ValueType> &mat);
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const;
  virtual void CopyToAsync(BaseMatrix<ValueType> *mat) const;

  virtual void CopyFromHost(const HostMatrix<ValueType> &src);
  virtual void CopyFromHostAsync(const HostMatrix<ValueType> &src);
  virtual void CopyToHost(HostMatrix<ValueType> *dst) const;
  virtual void CopyToHostAsync(HostMatrix<ValueType> *dst) const;

  virtual void CopyFromCOO(const int *row, const int *col, const ValueType *val);
  virtual void CopyToCOO(int *row, int *col, ValueType *val) const;

  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const;

private:

  MatrixCOO<ValueType, int> mat_;

  hipsparseMatDescr_t mat_descr_;

  friend class BaseVector<ValueType>;
  friend class AcceleratorVector<ValueType>;
  friend class HIPAcceleratorVector<ValueType>;

};


}

#endif // ROCALUTION_HIP_MATRIX_COO_HPP_
