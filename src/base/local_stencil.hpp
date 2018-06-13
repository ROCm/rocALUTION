#ifndef ROCALUTION_LOCAL_STENCIL_HPP_
#define ROCALUTION_LOCAL_STENCIL_HPP_

#include "../utils/types.hpp"
#include "operator.hpp"
#include "local_vector.hpp"
#include "stencil_types.hpp"

namespace rocalution {

template <typename ValueType>
class BaseStencil;
template <typename ValueType>
class HostStencil;
template <typename ValueType>
class AcceleratorStencil;

template <typename ValueType>
class LocalVector;
template <typename ValueType>
class GlobalVector;

// Local Stencil
template <typename ValueType>
class LocalStencil : public Operator<ValueType> {

public:

  LocalStencil();
  LocalStencil(unsigned int type);
  virtual ~LocalStencil();

  virtual void Info() const;

  virtual int get_ndim(void) const;
  virtual IndexType2 get_nrow(void) const;
  virtual IndexType2 get_ncol(void) const;
  virtual IndexType2 get_nnz(void) const;

  virtual void SetGrid(const int size);

  virtual void Clear();

  virtual void Apply(const LocalVector<ValueType> &in, LocalVector<ValueType> *out) const; 
  virtual void ApplyAdd(const LocalVector<ValueType> &in, const ValueType scalar, 
                        LocalVector<ValueType> *out) const; 

  virtual void MoveToAccelerator(void);
  virtual void MoveToHost(void);

protected:

  virtual bool is_host(void) const {return true;};
  virtual bool is_accel(void) const {return false;};

private:
  
  std::string object_name_ ;

  BaseStencil<ValueType> *stencil_;

  HostStencil<ValueType> *stencil_host_;
  AcceleratorStencil<ValueType> *stencil_accel_;

  
  friend class LocalVector<ValueType>;
  friend class GlobalVector<ValueType>;

};


}

#endif // ROCALUTION_LOCAL_STENCIL_HPP_
