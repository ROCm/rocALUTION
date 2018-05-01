#ifndef ROCALUTION_HOST_MATRIX_BCSR_HPP_
#define ROCALUTION_HOST_MATRIX_BCSR_HPP_

#include "../base_vector.hpp"
#include "../base_matrix.hpp"
#include "../matrix_formats.hpp"

namespace rocalution {

template <typename ValueType>
class HostMatrixBCSR : public HostMatrix<ValueType> {

public:

  HostMatrixBCSR();
  HostMatrixBCSR(const Rocalution_Backend_Descriptor local_backend);
  virtual ~HostMatrixBCSR();

  virtual void info(void) const;
  virtual unsigned int get_mat_format(void) const { return  BCSR; }

  virtual void Clear(void);
  virtual void AllocateBCSR(const int nnz, const int nrow, const int ncol);

  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat);

  virtual void CopyFrom(const BaseMatrix<ValueType> &mat);
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const;

  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const;

private:

  MatrixBCSR<ValueType, int> mat_;

  friend class BaseVector<ValueType>;
  friend class HostVector<ValueType>;
  friend class HostMatrixCSR<ValueType>;
  friend class HostMatrixCOO<ValueType>;
  friend class HostMatrixHYB<ValueType>;
  friend class HostMatrixDENSE<ValueType>;

  friend class HIPAcceleratorMatrixBCSR<ValueType>;

};


}

#endif // ROCALUTION_HOST_MATRIX_BCSR_HPP_
