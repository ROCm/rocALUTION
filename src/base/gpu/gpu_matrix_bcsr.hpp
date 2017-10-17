#ifndef PARALUTION_GPU_MATRIX_BCSR_HPP_
#define PARALUTION_GPU_MATRIX_BCSR_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace paralution {

template <typename ValueType>
class GPUAcceleratorMatrixBCSR : public GPUAcceleratorMatrix<ValueType> {

public:

  GPUAcceleratorMatrixBCSR();
  GPUAcceleratorMatrixBCSR(const Paralution_Backend_Descriptor local_backend);
  virtual ~GPUAcceleratorMatrixBCSR();

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
  friend class GPUAcceleratorVector<ValueType>;

};


}

#endif // PARALUTION_GPU_MATRIX_BCSR_HPP_
