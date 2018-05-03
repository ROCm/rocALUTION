#ifndef ROCALUTION_HIP_MATRIX_DIA_HPP_
#define ROCALUTION_HIP_MATRIX_DIA_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace rocalution {

template <typename ValueType>
class HIPAcceleratorMatrixDIA : public HIPAcceleratorMatrix<ValueType> {

public:

  HIPAcceleratorMatrixDIA();
  HIPAcceleratorMatrixDIA(const Rocalution_Backend_Descriptor local_backend);
  virtual ~HIPAcceleratorMatrixDIA();

  inline int get_ndiag(void) const { return mat_.num_diag; }

  virtual void info(void) const;
  virtual unsigned int get_mat_format(void) const { return DIA; }

  virtual void Clear(void);
  virtual void AllocateDIA(const int nnz, const int nrow, const int ncol, const int ndiag);
  virtual void SetDataPtrDIA(int **offset, ValueType **val,
                     const int nnz, const int nrow, const int ncol, const int num_diag);
  virtual void LeaveDataPtrDIA(int **offset, ValueType **val, int &num_diag);

  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat);

  virtual void CopyFrom(const BaseMatrix<ValueType> &mat);
  virtual void CopyFromAsync(const BaseMatrix<ValueType> &mat);
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const;
  virtual void CopyToAsync(BaseMatrix<ValueType> *mat) const;

  virtual void CopyFromHost(const HostMatrix<ValueType> &src);
  virtual void CopyFromHostAsync(const HostMatrix<ValueType> &src);
  virtual void CopyToHost(HostMatrix<ValueType> *dst) const;
  virtual void CopyToHostAsync(HostMatrix<ValueType> *dst) const;

  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const;

private:

  MatrixDIA<ValueType, int> mat_;

  friend class BaseVector<ValueType>;
  friend class AcceleratorVector<ValueType>;
  friend class HIPAcceleratorVector<ValueType>;

};


}

#endif // ROCALUTION_HIP_MATRIX_DIA_HPP_
