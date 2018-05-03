#ifndef ROCALUTION_HIP_MATRIX_BCSR_HPP_
#define ROCALUTION_HIP_MATRIX_BCSR_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace rocalution {

template <typename ValueType>
class HIPAcceleratorMatrixBCSR : public HIPAcceleratorMatrix<ValueType> {

public:

  HIPAcceleratorMatrixBCSR();
  HIPAcceleratorMatrixBCSR(const Rocalution_Backend_Descriptor local_backend);
  virtual ~HIPAcceleratorMatrixBCSR();

  virtual void info(void) const;
  virtual unsigned int get_mat_format(void) const { return BCSR; }

  virtual void Clear(void);
  virtual void AllocateBCSR(const int nnz, const int nrow, const int ncol);

  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat);

  virtual void CopyFrom(const BaseMatrix<ValueType> &mat);
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const;

  virtual void CopyFromHost(const HostMatrix<ValueType> &src);
  virtual void CopyToHost(HostMatrix<ValueType> *dst) const;

  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const;

private:

  MatrixBCSR<ValueType, int> mat_;

  friend class BaseVector<ValueType>;
  friend class AcceleratorVector<ValueType>;
  friend class HIPAcceleratorVector<ValueType>;

};


}

#endif // ROCALUTION_HIP_MATRIX_BCSR_HPP_
