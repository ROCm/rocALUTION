#ifndef PARALUTION_MIC_MATRIX_COO_HPP_
#define PARALUTION_MIC_MATRIX_COO_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace paralution {

template <typename ValueType>
class MICAcceleratorMatrixCOO : public MICAcceleratorMatrix<ValueType> {
  
public:

  MICAcceleratorMatrixCOO();
  MICAcceleratorMatrixCOO(const Paralution_Backend_Descriptor local_backend);
  virtual ~MICAcceleratorMatrixCOO();

  virtual void info(void) const;
  virtual unsigned int get_mat_format(void) const{ return COO; }

  virtual void Clear(void);
  virtual void AllocateCOO(const int nnz, const int nrow, const int ncol);

  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat);

  virtual bool Permute(const BaseVector<int> &permutation);
  virtual bool PermuteBackward(const BaseVector<int> &permutation);

  virtual void CopyFrom(const BaseMatrix<ValueType> &mat);
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const;

  virtual void CopyFromHost(const HostMatrix<ValueType> &src);
  virtual void CopyToHost(HostMatrix<ValueType> *dst) const;

  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const; 
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const; 
  
private:
  
  MatrixCOO<ValueType, int> mat_;

  friend class BaseVector<ValueType>;  
  friend class AcceleratorVector<ValueType>;  
  friend class MICAcceleratorVector<ValueType>;  

};

};

#endif // PARALUTION_MIC_MATRIX_COO_HPP_
