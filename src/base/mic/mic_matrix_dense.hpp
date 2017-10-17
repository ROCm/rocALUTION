#ifndef PARALUTION_MIC_MATRIX_DENSE_HPP_
#define PARALUTION_MIC_MATRIX_DENSE_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace paralution {

template <typename ValueType>
class MICAcceleratorMatrixDENSE : public MICAcceleratorMatrix<ValueType> {
  
public:

  MICAcceleratorMatrixDENSE();
  MICAcceleratorMatrixDENSE(const Paralution_Backend_Descriptor local_backend);
  virtual ~MICAcceleratorMatrixDENSE();

  virtual void info(void) const;
  virtual unsigned int get_mat_format(void) const{ return DENSE; }

  virtual void Clear(void);
  virtual void AllocateDENSE(const int nrow, const int ncol);

  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat);

  virtual void CopyFrom(const BaseMatrix<ValueType> &mat);
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const;

  virtual void CopyFromHost(const HostMatrix<ValueType> &src);
  virtual void CopyToHost(HostMatrix<ValueType> *dst) const;

  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const; 
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const; 
  
private:
  
  MatrixDENSE<ValueType> mat_;

  friend class BaseVector<ValueType>;  
  friend class AcceleratorVector<ValueType>;  
  friend class MICAcceleratorVector<ValueType>;  

};

};

#endif // PARALUTION_MIC_MATRIX_DENSE_HPP_
