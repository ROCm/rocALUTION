/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_BASE_HPP_
#define ROCALUTION_BASE_HPP_

#include "backend_manager.hpp"

#include <complex>
#include <vector>

namespace rocalution {

template <typename ValueType>
class GlobalVector;
template <typename ValueType>
class GlobalMatrix;
class ParallelManager;

class RocalutionObj {

public:

  RocalutionObj();
  virtual ~RocalutionObj();

  /// Clear (free all data) the object
  virtual void Clear() = 0;

protected:
  size_t global_obj_id;

};

/// Global data for all ROCALUTION objects
struct Rocalution_Object_Data {
  
  std::vector<class RocalutionObj*> all_obj;

};

/// Global obj tracking structure
extern struct Rocalution_Object_Data Rocalution_Object_Data_Tracking;

/// Base class for operator and vector 
/// (i.e. global/local matrix/stencil/vector) classes,
/// all the backend-related interface and data 
/// are defined here
template <typename ValueType>
class BaseRocalution : public RocalutionObj {

public:

  BaseRocalution();
  BaseRocalution(const BaseRocalution<ValueType> &src);
  virtual ~BaseRocalution();

  BaseRocalution<ValueType>& operator=(const BaseRocalution<ValueType> &src);

  /// Move the object to the Accelerator backend
  virtual void MoveToAccelerator(void) = 0;

  /// Move the object to the Host backend
  virtual void MoveToHost(void) = 0;

  /// Move the object to the Accelerator backend with async move
  virtual void MoveToAcceleratorAsync(void);

  /// Move the object to the Host backend with async move
  virtual void MoveToHostAsync(void);

  // Sync (the async move)
  virtual void Sync(void);

  /// Clone the Backend descriptor from another object
  virtual void CloneBackend(const BaseRocalution<ValueType> &src);

  /// Clone the Backend descriptor from another object with different template ValueType
  template <typename ValueType2>
  void CloneBackend(const BaseRocalution<ValueType2> &src);

  /// Print the object information (properties, backends)
  virtual void Info() const = 0;

  /// Clear (free all data) the object
  virtual void Clear() = 0;

protected:

  /// Name of the object
  std::string object_name_;

  /// Parallel Manager
  const ParallelManager *pm_;

  /// Backend descriptor 
  Rocalution_Backend_Descriptor local_backend_;

  /// Return true if the object is on the host
  virtual bool is_host(void) const = 0;

  /// Return true if the object is on the accelerator
  virtual bool is_accel(void) const = 0;

  // active async transfer
  bool asyncf;

  friend class BaseRocalution<double>;
  friend class BaseRocalution<float>;
  friend class BaseRocalution<std::complex<double> >;
  friend class BaseRocalution<std::complex<float> >;

  friend class BaseRocalution<int>;

  friend class GlobalVector<int>;
  friend class GlobalVector<float>;
  friend class GlobalVector<double>;

  friend class GlobalMatrix<int>;
  friend class GlobalMatrix<float>;
  friend class GlobalMatrix<double>;

};

} // namespace rocalution

#endif // ROCALUTION_LOCAL_BASE_HPP_
