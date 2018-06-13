#ifndef ROCALUTION_HOST_MATRIX_MCSR_HPP_
#define ROCALUTION_HOST_MATRIX_MCSR_HPP_

#include "../base_vector.hpp"
#include "../base_matrix.hpp"
#include "../matrix_formats.hpp"

namespace rocalution {

template <typename ValueType>
class HostMatrixMCSR : public HostMatrix<ValueType> {

public:

  HostMatrixMCSR();
  HostMatrixMCSR(const Rocalution_Backend_Descriptor local_backend);
  virtual ~HostMatrixMCSR();

  virtual void Info(void) const;
  virtual unsigned int get_mat_format(void) const { return  MCSR; }

  virtual void Clear(void);
  virtual void AllocateMCSR(const int nnz, const int nrow, const int ncol);
  virtual void SetDataPtrMCSR(int **row_offset, int **col, ValueType **val,
                              const int nnz, const int nrow, const int ncol);
  virtual void LeaveDataPtrMCSR(int **row_offset, int **col, ValueType **val);

  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat);

  virtual void CopyFrom(const BaseMatrix<ValueType> &mat);
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const;

  virtual bool ILU0Factorize(void);

  virtual void LUAnalyse(void);
  virtual void LUAnalyseClear(void);
  virtual bool LUSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;

  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const;
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const;

private:

  MatrixMCSR<ValueType, int> mat_;

  friend class BaseVector<ValueType>;
  friend class HostVector<ValueType>;
  friend class HostMatrixCSR<ValueType>;
  friend class HostMatrixCOO<ValueType>;
  friend class HostMatrixHYB<ValueType>;
  friend class HostMatrixDENSE<ValueType>;

  friend class HIPAcceleratorMatrixMCSR<ValueType>;

};


}

#endif // ROCALUTION_HOST_MATRIX_MCSR_HPP_
