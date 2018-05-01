#ifndef ROCALUTION_HIP_VECTOR_HPP_
#define ROCALUTION_HIP_VECTOR_HPP_

#include "../base_vector.hpp"
#include "../base_matrix.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"

#include <complex>

namespace rocalution {

template <typename ValueType>
class HIPAcceleratorVector : public AcceleratorVector<ValueType> {

public:

  HIPAcceleratorVector();
  HIPAcceleratorVector(const Rocalution_Backend_Descriptor local_backend);
  virtual ~HIPAcceleratorVector();

  virtual void info(void) const;

  virtual void Allocate(const int n);
  virtual void SetDataPtr(ValueType **ptr, const int size);
  virtual void LeaveDataPtr(ValueType **ptr);
  virtual void Clear(void);
  virtual void Zeros(void);
  virtual void Ones(void);
  virtual void SetValues(const ValueType val);

  virtual void CopyFrom(const BaseVector<ValueType> &src);
  virtual void CopyFromAsync(const BaseVector<ValueType> &src);
  virtual void CopyFrom(const BaseVector<ValueType> &src,
                        const int src_offset,
                        const int dst_offset,
                        const int size);

  virtual void CopyTo(BaseVector<ValueType> *dst) const;
  virtual void CopyToAsync(BaseVector<ValueType> *dst) const;
  virtual void CopyFromFloat(const BaseVector<float> &src);
  virtual void CopyFromDouble(const BaseVector<double> &src);

  virtual void CopyFromHostAsync(const HostVector<ValueType> &src);
  virtual void CopyFromHost(const HostVector<ValueType> &src);
  virtual void CopyToHostAsync(HostVector<ValueType> *dst) const;
  virtual void CopyToHost(HostVector<ValueType> *dst) const;

  virtual void CopyFromData(const ValueType *data);
  virtual void CopyToData(ValueType *data) const;

  virtual void CopyFromPermute(const BaseVector<ValueType> &src,
                               const BaseVector<int> &permutation);
  virtual void CopyFromPermuteBackward(const BaseVector<ValueType> &src,
                                       const BaseVector<int> &permutation);

  virtual void Permute(const BaseVector<int> &permutation);
  virtual void PermuteBackward(const BaseVector<int> &permutation);

  // this = this + alpha*x
  virtual void AddScale(const BaseVector<ValueType> &x, const ValueType alpha);
  // this = alpha*this + x
  virtual void ScaleAdd(const ValueType alpha, const BaseVector<ValueType> &x);
  // this = alpha*this + x*beta
  virtual void ScaleAddScale(const ValueType alpha, const BaseVector<ValueType> &x, const ValueType beta);
  virtual void ScaleAddScale(const ValueType alpha, const BaseVector<ValueType> &x, const ValueType beta,
                             const int src_offset, const int dst_offset,const int size);
  // this = alpha*this + x*beta + y*gamma
  virtual void ScaleAdd2(const ValueType alpha, const BaseVector<ValueType> &x,
                         const ValueType beta, const BaseVector<ValueType> &y,
                         const ValueType gamma);
  // this = alpha*this
  virtual void Scale(const ValueType alpha);

  virtual void ExclusiveScan(const BaseVector<ValueType> &x);
  // this^T x
  virtual ValueType Dot(const BaseVector<ValueType> &x) const;
  // this^T x
  virtual ValueType DotNonConj(const BaseVector<ValueType> &x) const;
  // srqt(this^T this)
  virtual ValueType Norm(void) const;
  // reduce
  virtual ValueType Reduce(void) const;
  // Compute sum of absolute values of this
  virtual ValueType Asum(void) const;
  // Compute absolute value of this
  virtual int Amax(ValueType &value) const;
  // point-wise multiplication
  virtual void PointWiseMult(const BaseVector<ValueType> &x);
  virtual void PointWiseMult(const BaseVector<ValueType> &x, const BaseVector<ValueType> &y);
  virtual void Power(const double power);

  // set index array
  virtual void SetIndexArray(const int size, const int *index);
  // get index values
  virtual void GetIndexValues(ValueType *values) const;
  // set index values
  virtual void SetIndexValues(const ValueType *values);
  // get continuous index values
  virtual void GetContinuousValues(const int start, const int end, ValueType *values) const;
  // set continuous index values
  virtual void SetContinuousValues(const int start, const int end, const ValueType *values);

  // get coarse boundary mapping
  virtual void ExtractCoarseMapping(const int start, const int end, const int *index,
                                    const int nc, int *size, int *map) const;
  // get coarse boundary index
  virtual void ExtractCoarseBoundary(const int start, const int end, const int *index,
                                     const int nc, int *size, int *boundary) const;

private:

  ValueType *vec_;

  int *index_array_;
  ValueType *index_buffer_;

  ValueType *host_buffer_;
  ValueType *device_buffer_;

  friend class HIPAcceleratorVector<float>;
  friend class HIPAcceleratorVector<double>;
  friend class HIPAcceleratorVector<std::complex<float> >;
  friend class HIPAcceleratorVector<std::complex<double> >;
  friend class HIPAcceleratorVector<int>;

  friend class HostVector<ValueType>;
  friend class AcceleratorMatrix<ValueType>;

  friend class HIPAcceleratorMatrixCSR<ValueType>;
  friend class HIPAcceleratorMatrixMCSR<ValueType>;
  friend class HIPAcceleratorMatrixBCSR<ValueType>;
  friend class HIPAcceleratorMatrixCOO<ValueType>;
  friend class HIPAcceleratorMatrixDIA<ValueType>;
  friend class HIPAcceleratorMatrixELL<ValueType>;
  friend class HIPAcceleratorMatrixDENSE<ValueType>;
  friend class HIPAcceleratorMatrixHYB<ValueType>;

  friend class HIPAcceleratorMatrixCOO<double>;
  friend class HIPAcceleratorMatrixCOO<float>;
  friend class HIPAcceleratorMatrixCOO<std::complex<double> >;
  friend class HIPAcceleratorMatrixCOO<std::complex<float> >;

  friend class HIPAcceleratorMatrixCSR<double>;
  friend class HIPAcceleratorMatrixCSR<float>;
  friend class HIPAcceleratorMatrixCSR<std::complex<double> >;
  friend class HIPAcceleratorMatrixCSR<std::complex<float> >;

};


}

#endif // ROCALUTION_BASE_VECTOR_HPP_
