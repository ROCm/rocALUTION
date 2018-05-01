#ifndef ROCALUTION_HOST_VECTOR_HPP_
#define ROCALUTION_HOST_VECTOR_HPP_

#include "../base_vector.hpp"
#include "../base_matrix.hpp"
#include "../base_stencil.hpp"

#include <complex>

namespace rocalution {

template <typename ValueType>
class LocalVector;

template <typename ValueType>
class HostVector : public BaseVector<ValueType> {

public:

  HostVector();
  HostVector(const Rocalution_Backend_Descriptor local_backend);
  virtual ~HostVector();

  virtual void info(void) const;

  virtual bool Check(void) const;
  virtual void Allocate(const int n);
  virtual void SetDataPtr(ValueType **ptr, const int size);
  virtual void LeaveDataPtr(ValueType **ptr);
  virtual void Clear(void);
  virtual void Zeros(void);
  virtual void Ones(void);
  virtual void SetValues(const ValueType val);
  virtual void SetRandom(const ValueType a, const ValueType b, const int seed);

  virtual void CopyFrom(const BaseVector<ValueType> &vec);
  virtual void CopyFromFloat(const BaseVector<float> &vec);
  virtual void CopyFromDouble(const BaseVector<double> &vec);
  virtual void CopyTo(BaseVector<ValueType> *vec) const;
  virtual void CopyFrom(const BaseVector<ValueType> &src,
                        const int src_offset,
                        const int dst_offset,
                        const int size);
  virtual void CopyFromPermute(const BaseVector<ValueType> &src,
                               const BaseVector<int> &permutation);
  virtual void CopyFromPermuteBackward(const BaseVector<ValueType> &src,
                                       const BaseVector<int> &permutation);

  virtual void CopyFromData(const ValueType *data);
  virtual void CopyToData(ValueType *data) const;

  virtual void Permute(const BaseVector<int> &permutation);
  virtual void PermuteBackward(const BaseVector<int> &permutation);

  virtual bool Restriction(const BaseVector<ValueType> &vec_fine, const BaseVector<int> &map);
  virtual bool Prolongation(const BaseVector<ValueType> &vec_coarse, const BaseVector<int> &map);

  /// Read vector from ASCII file
  void ReadFileASCII(const std::string filename);
  /// Write vector to ASCII file
  void WriteFileASCII(const std::string filename) const;
  /// Read vector from binary file
  void ReadFileBinary(const std::string filename);
  /// Write vector to binary file
  void WriteFileBinary(const std::string filename) const;

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
  // reduce vector
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

  // for [] operator in LocalVector
  friend class LocalVector<ValueType>;

  friend class HostVector<double>;
  friend class HostVector<float>;
  friend class HostVector<std::complex<double> >;
  friend class HostVector<std::complex<float> >;
  friend class HostVector<int>;

  friend class HostMatrix<ValueType>;
  friend class HostMatrixCSR<ValueType>;
  friend class HostMatrixCOO<ValueType>;
  friend class HostMatrixDIA<ValueType>;
  friend class HostMatrixELL<ValueType>;
  friend class HostMatrixHYB<ValueType>;
  friend class HostMatrixDENSE<ValueType>;
  friend class HostMatrixMCSR<ValueType>;
  friend class HostMatrixBCSR<ValueType>;

  friend class HostMatrixCOO<float>;
  friend class HostMatrixCOO<double>;
  friend class HostMatrixCOO<std::complex<float> >;
  friend class HostMatrixCOO<std::complex<double> >;

  friend class HostMatrixCSR<double>;
  friend class HostMatrixCSR<float>;
  friend class HostMatrixCSR<std::complex<float> >;
  friend class HostMatrixCSR<std::complex<double> >;

  friend class HostMatrixDENSE<double>;
  friend class HostMatrixDENSE<float>;

  friend class HIPAcceleratorVector<ValueType>;

  friend class HostStencil<ValueType>;
  friend class HostStencilLaplace2D<ValueType>;

};


}

#endif // ROCALUTION_HOST_VECTOR_HPP_
