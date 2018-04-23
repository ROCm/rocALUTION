#ifndef PARALUTION_BASE_STENCIL_HPP_
#define PARALUTION_BASE_STENCIL_HPP_

#include "base_paralution.hpp"

namespace paralution {

template <typename ValueType>
class BaseVector;
template <typename ValueType>
class HostVector;
template <typename ValueType>
class HIPAcceleratorVector;

template <typename ValueType>
class HostStencilLaplace2D;
template <typename ValueType>
class HIPAcceleratorStencil;
template <typename ValueType>
class HIPAcceleratorStencilLaplace2D;

/// Base class for all host/accelerator stencils
template <typename ValueType>
class BaseStencil {

public:

  BaseStencil();
  virtual ~BaseStencil();

  /// Return the number of rows in the stencil
  int get_nrow(void) const;
  /// Return the number of columns in the stencil
  int get_ncol(void) const;
  /// Return the dimension of the stencil
  int get_ndim(void) const;
  /// Return the nnz per row
  virtual int get_nnz(void) const = 0;

  /// Shows simple info about the object
  virtual void info(void) const = 0;
  /// Return the stencil format id (see stencil_formats.hpp)
  virtual unsigned int get_stencil_id(void) const = 0 ;
  /// Copy the backend descriptor information
  virtual void set_backend(const Paralution_Backend_Descriptor local_backend);
  // Set the grid size
  virtual void SetGrid(const int size);

  /// Apply the stencil to vector, out = this*in;
  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const = 0; 
  /// Apply and add the stencil to vector, out = out + scalar*this*in;
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const = 0; 

protected:

  /// Number of rows
  int ndim_;
  /// Number of columns
  int size_;


  /// Backend descriptor (local copy)
  Paralution_Backend_Descriptor local_backend_;

  friend class BaseVector<ValueType>;
  friend class HostVector<ValueType>;
  friend class AcceleratorVector<ValueType>;
  friend class HIPAcceleratorVector<ValueType>;

};

template <typename ValueType>
class HostStencil : public BaseStencil<ValueType> {

public:

  HostStencil();
  virtual ~HostStencil();

};

template <typename ValueType>
class AcceleratorStencil : public BaseStencil<ValueType> {

public:

  AcceleratorStencil();
  virtual ~AcceleratorStencil();

  /// Copy (accelerator stencil) from host stencil
  virtual void CopyFromHost(const HostStencil<ValueType> &src) = 0;

  /// Copy (accelerator stencil) to host stencil
  virtual void CopyToHost(HostStencil<ValueType> *dst) const = 0;

};

template <typename ValueType>
class HIPAcceleratorStencil : public AcceleratorStencil<ValueType> {

public:

  HIPAcceleratorStencil();
  virtual ~HIPAcceleratorStencil();

};

}

#endif // PARALUTION_BASE_STENCIL_HPP_
