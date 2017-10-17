#ifndef PARALUTION_GPU_MATRIX_DENSE_HPP_
#define PARALUTION_GPU_MATRIX_DENSE_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace paralution {

template <typename ValueType>
class GPUAcceleratorMatrixDENSE : public GPUAcceleratorMatrix<ValueType> {

public:

  GPUAcceleratorMatrixDENSE();
  GPUAcceleratorMatrixDENSE(const Paralution_Backend_Descriptor local_backend);
  virtual ~GPUAcceleratorMatrixDENSE();

  virtual void info(void) const;
  virtual unsigned int get_mat_format(void) const { return DENSE; }

  virtual void Clear(void);
  virtual void AllocateDENSE(const int nrow, const int ncol);
  virtual void SetDataPtrDENSE(ValueType **val, const int nrow, const int ncol);
  virtual void LeaveDataPtrDENSE(ValueType **val);

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

  virtual bool MatMatMult(const BaseMatrix<ValueType> &A, const BaseMatrix<ValueType> &B);

  virtual bool ReplaceColumnVector(const int idx, const BaseVector<ValueType> &vec);
  virtual bool ReplaceRowVector(const int idx, const BaseVector<ValueType> &vec);
  virtual bool ExtractColumnVector(const int idx, BaseVector<ValueType> *vec) const;
  virtual bool ExtractRowVector(const int idx, BaseVector<ValueType> *vec) const;

private:

  MatrixDENSE<ValueType> mat_;

  friend class BaseVector<ValueType>;
  friend class AcceleratorVector<ValueType>;
  friend class GPUAcceleratorVector<ValueType>;

};


}

#endif // PARALUTION_GPU_MATRIX_DENSE_HPP_
