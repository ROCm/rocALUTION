#ifndef PARALUTION_MIC_MATRIX_DIA_HPP_
#define PARALUTION_MIC_MATRIX_DIA_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace paralution {

template <typename ValueType>
class MICAcceleratorMatrixDIA : public MICAcceleratorMatrix<ValueType> {
  
public:

  MICAcceleratorMatrixDIA();
  MICAcceleratorMatrixDIA(const Paralution_Backend_Descriptor local_backend);
  virtual ~MICAcceleratorMatrixDIA();

  inline int get_ndiag(void) const { return mat_.num_diag; }

  virtual void info(void) const;
  virtual unsigned int get_mat_format(void) const{ return DIA; }

  virtual void Clear(void);
  virtual void AllocateDIA(const int nnz, const int nrow, const int ncol, const int ndiag);

  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat);

  virtual void CopyFrom(const BaseMatrix<ValueType> &mat);
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const;

  virtual void CopyFromHost(const HostMatrix<ValueType> &src);
  virtual void CopyToHost(HostMatrix<ValueType> *dst) const;

  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const; 
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const; 
  
private:
  
  MatrixDIA<ValueType, int> mat_;

  friend class BaseVector<ValueType>;  
  friend class AcceleratorVector<ValueType>;  
  friend class MICAcceleratorVector<ValueType>;  

};

};

#endif // PARALUTION_MIC_MATRIX_DIA_HPP_
