#ifndef PARALUTION_MIC_MATRIX_CSR_HPP_
#define PARALUTION_MIC_MATRIX_CSR_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace paralution {

template <typename ValueType>
class MICAcceleratorMatrixCSR : public MICAcceleratorMatrix<ValueType> {
  
public:

  MICAcceleratorMatrixCSR();
  MICAcceleratorMatrixCSR(const Paralution_Backend_Descriptor local_backend);
  virtual ~MICAcceleratorMatrixCSR();

  virtual void info(void) const;
  virtual unsigned int get_mat_format(void) const { return CSR; }

  virtual void Clear(void);
  virtual bool Zeros(void);

  virtual void AllocateCSR(const int nnz, const int nrow, const int ncol);

  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat);

  virtual void CopyFrom(const BaseMatrix<ValueType> &mat);
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const;

  virtual void CopyFromHost(const HostMatrix<ValueType> &src);
  virtual void CopyToHost(HostMatrix<ValueType> *dst) const;

  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const; 
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const; 

private:
  
  MatrixCSR<ValueType, int> mat_;

  friend class MICAcceleratorMatrixDIA<ValueType>;
  friend class MICAcceleratorMatrixELL<ValueType>;
  friend class MICAcceleratorMatrixHYB<ValueType>;

  friend class BaseVector<ValueType>;  
  friend class AcceleratorVector<ValueType>;  
  friend class MICAcceleratorVector<ValueType>;  

  MICAcceleratorVector<ValueType> *tmp_vec_;

};

};

#endif // PARALUTION_MIC_MATRIX_CSR_HPP_
