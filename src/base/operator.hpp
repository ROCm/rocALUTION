#ifndef ROCALUTION_OPERATOR_HPP_
#define ROCALUTION_OPERATOR_HPP_

#include "../utils/types.hpp"
#include "base_rocalution.hpp"

#include <iostream>
#include <string>
#include <cstdlib>

namespace rocalution {

template <typename ValueType>
class GlobalVector;
template <typename ValueType>
class LocalVector;

/// Operator class defines the generic interface
/// for applying an operator (e.g. matrix, stencil)
/// from/to global and local vectors
template <typename ValueType>
class Operator : public BaseRocalution<ValueType> {

public:

  Operator();
  virtual ~Operator();

  /// Return the number of rows in the matrix/stencil
  virtual IndexType2 get_nrow(void) const = 0;
  /// Return the number of columns in the matrix/stencil
  virtual IndexType2 get_ncol(void) const = 0;
  /// Return the number of non-zeros in the matrix/stencil
  virtual IndexType2 get_nnz(void) const = 0;

  /// Return the number of rows in the local matrix/stencil
  virtual int get_local_nrow(void) const;
  /// Return the number of columns in the local matrix/stencil
  virtual int get_local_ncol(void) const;
  /// Return the number of non-zeros in the local matrix/stencil
  virtual int get_local_nnz(void) const;

  /// Return the number of rows in the ghost matrix/stencil
  virtual int get_ghost_nrow(void) const;
  /// Return the number of columns in the ghost matrix/stencil
  virtual int get_ghost_ncol(void) const;
  /// Return the number of non-zeros in the ghost matrix/stencil
  virtual int get_ghost_nnz(void) const;

  /// Apply the operator, out = Operator(in), where in, out are local vectors
  virtual void Apply(const LocalVector<ValueType> &in, LocalVector<ValueType> *out) const;

  /// Apply and add the operator, out = out + scalar*Operator(in), where in, out are local vectors
  virtual void ApplyAdd(const LocalVector<ValueType> &in, const ValueType scalar,
                        LocalVector<ValueType> *out) const;

  /// Apply the operator, out = Operator(in), where in, out are global vectors
  virtual void Apply(const GlobalVector<ValueType> &in, GlobalVector<ValueType> *out) const; 

  /// Apply and add the operator, out = out + scalar*Operator(in), where in, out are global vectors
  virtual void ApplyAdd(const GlobalVector<ValueType> &in, const ValueType scalar, 
                        GlobalVector<ValueType> *out) const; 

};


}

#endif // ROCALUTION_OPERTOR_HPP_
