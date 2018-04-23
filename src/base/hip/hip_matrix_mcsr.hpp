#ifndef PARALUTION_HIP_MATRIX_MCSR_HPP_
#define PARALUTION_HIP_MATRIX_MCSR_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace paralution {

template <typename ValueType>
class HIPAcceleratorMatrixMCSR : public HIPAcceleratorMatrix<ValueType> {

public:

  HIPAcceleratorMatrixMCSR();
  HIPAcceleratorMatrixMCSR(const Paralution_Backend_Descriptor local_backend);
  virtual ~HIPAcceleratorMatrixMCSR();

  virtual void info(void) const;
  virtual unsigned int get_mat_format(void) const { return MCSR; }

  virtual void Clear(void);
  virtual void AllocateMCSR(const int nnz, const int nrow, const int ncol);
  virtual void SetDataPtrMCSR(int **row_offset, int **col, ValueType **val,
                              const int nnz, const int nrow, const int ncol);
  virtual void LeaveDataPtrMCSR(int **row_offset, int **col, ValueType **val);

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

  MatrixMCSR<ValueType, int> mat_;

  friend class BaseVector<ValueType>;
  friend class AcceleratorVector<ValueType>;
  friend class HIPAcceleratorVector<ValueType>;

};


}

#endif // PARALUTION_HIP_MATRIX_MCSR_HPP_
