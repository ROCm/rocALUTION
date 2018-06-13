#ifndef ROCALUTION_HOST_MATRIX_DIA_HPP_
#define ROCALUTION_HOST_MATRIX_DIA_HPP_

#include "../base_vector.hpp"
#include "../base_matrix.hpp"
#include "../matrix_formats.hpp"

namespace rocalution {

template <typename ValueType>
class HostMatrixDIA : public HostMatrix<ValueType> {

public:

  HostMatrixDIA();
  HostMatrixDIA(const Rocalution_Backend_Descriptor local_backend);
  virtual ~HostMatrixDIA();

  inline int get_ndiag(void) const { return mat_.num_diag; }

  virtual void Info(void) const;
  virtual unsigned int get_mat_format(void) const { return  DIA; }

  virtual void Clear(void);
  virtual void AllocateDIA(const int nnz, const int nrow, const int ncol, const int ndiag);
  virtual void SetDataPtrDIA(int **offset, ValueType **val,
                     const int nnz, const int nrow, const int ncol, const int num_diag);
  virtual void LeaveDataPtrDIA(int **offset, ValueType **val, int &num_diag);

  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat);

  virtual void CopyFrom(const BaseMatrix<ValueType> &mat);
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const;

  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const;

private:

  MatrixDIA<ValueType, int> mat_;

  friend class BaseVector<ValueType>;
  friend class HostVector<ValueType>;
  friend class HostMatrixCSR<ValueType>;
  friend class HostMatrixCOO<ValueType>;
  friend class HostMatrixELL<ValueType>;
  friend class HostMatrixHYB<ValueType>;
  friend class HostMatrixDENSE<ValueType>;

  friend class HIPAcceleratorMatrixDIA<ValueType>;

};


}

#endif // ROCALUTION_HOST_MATRIX_DIA_HPP_
