#ifndef ROCALUTION_GLOBAL_VECTOR_HPP_
#define ROCALUTION_GLOBAL_VECTOR_HPP_

#include "../utils/types.hpp"
#include "vector.hpp"
#include "parallel_manager.hpp"

namespace rocalution {

template <typename ValueType>
class LocalVector;
template <typename ValueType>
class LocalMatrix;
template <typename ValueType>
class GlobalMatrix;
struct MRequest;

// Global vector
template <typename ValueType>
class GlobalVector : public Vector<ValueType> {

public:

  GlobalVector();
  GlobalVector(const ParallelManager &pm);
  virtual ~GlobalVector();

  virtual void MoveToAccelerator(void);
  virtual void MoveToHost(void);

  virtual void Info(void) const;
  virtual bool Check(void) const;

  virtual IndexType2 get_size(void) const;
  virtual int get_local_size(void) const;
  virtual int get_ghost_size(void) const;

  const LocalVector<ValueType>& GetInterior() const;
  LocalVector<ValueType>& GetInterior();
  const LocalVector<ValueType>& GetGhost() const;

  virtual void Allocate(std::string name, const IndexType2 size);
  virtual void Clear(void);

  void SetParallelManager(const ParallelManager &pm);

  virtual void Zeros(void);
  virtual void Ones(void);
  virtual void SetValues(const ValueType val);
  virtual void SetRandom(const ValueType a = -1.0, const ValueType b = 1.0, const int seed = 0);
  void CloneFrom(const GlobalVector<ValueType> &src);

  // Accessing operator - only for host data
  ValueType& operator[](const int i);
  const ValueType& operator[](const int i) const;

  void SetDataPtr(ValueType **ptr, std::string name, const IndexType2 size);
  void LeaveDataPtr(ValueType **ptr);

  virtual void CopyFrom(const GlobalVector<ValueType> &src);
  virtual void ReadFileASCII(const std::string filename);
  virtual void WriteFileASCII(const std::string filename) const;
  virtual void ReadFileBinary(const std::string filename);
  virtual void WriteFileBinary(const std::string filename) const;

  // this = this + alpha*x
  virtual void AddScale(const GlobalVector<ValueType> &x, const ValueType alpha);
  // this = alpha*this + x
  virtual void ScaleAdd(const ValueType alpha, const GlobalVector<ValueType> &x);
  // this = alpha*this + x*beta + y*gamma
  virtual void ScaleAdd2(const ValueType alpha, const GlobalVector<ValueType> &x,
                         const ValueType beta, const GlobalVector<ValueType> &y,
                         const ValueType gamma);
  // this = alpha*this
  virtual void Scale(const ValueType alpha);
  // this^T x
  virtual ValueType Dot(const GlobalVector<ValueType> &x) const;
  // this^T x
  virtual ValueType DotNonConj(const GlobalVector<ValueType> &x) const;
  // sqrt(this^T this)
  virtual ValueType Norm(void) const;
  // reduce
  virtual ValueType Reduce(void) const;
  // L1 norm, sum(|this|)
  virtual ValueType Asum(void) const;
  // Amax, max(|this|)
  virtual int Amax(ValueType &value) const;
  // point-wise multiplication
  virtual void PointWiseMult(const GlobalVector<ValueType> &x);
  virtual void PointWiseMult(const GlobalVector<ValueType> &x, const GlobalVector<ValueType> &y);

  virtual void Power(const double power);

  // Restriction operator based on restriction mapping vector
  void Restriction(const GlobalVector<ValueType> &vec_fine, const LocalVector<int> &map);

  // Prolongation operator based on restriction(!) mapping vector
  void Prolongation(const GlobalVector<ValueType> &vec_coarse, const LocalVector<int> &map);

protected:

  virtual bool is_host(void) const;
  virtual bool is_accel(void) const;

  void UpdateGhostValuesAsync(const GlobalVector<ValueType> &in);
  void UpdateGhostValuesSync(void);

private:

  MRequest *recv_event_;
  MRequest *send_event_;

  ValueType *recv_boundary_;
  ValueType *send_boundary_;

  LocalVector<ValueType> vector_interior_;
  LocalVector<ValueType> vector_ghost_;

  friend class LocalMatrix<ValueType>;
  friend class GlobalMatrix<ValueType>;

  friend class BaseRocalution<ValueType>;

};


}

#endif // ROCALUTION_GLOBAL_VECTOR_HPP_
