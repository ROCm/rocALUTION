#ifndef PARALUTION_BASE_HPP_
#define PARALUTION_BASE_HPP_

#include "backend_manager.hpp"

#include <complex>
#include <vector>

namespace paralution {

template <typename ValueType>
class GlobalVector;
template <typename ValueType>
class GlobalMatrix;
class ParallelManager;

class ParalutionObj {

public:

  ParalutionObj();
  virtual ~ParalutionObj();

  /// Clear (free all data) the object
  virtual void Clear() = 0;

protected:
  size_t global_obj_id;

};

/// Global data for all PARALUTION objects
struct Paralution_Object_Data {
  
  std::vector<class ParalutionObj*> all_obj;

};

/// Global obj tracking structure
extern struct Paralution_Object_Data Paralution_Object_Data_Tracking;

/// Base class for operator and vector 
/// (i.e. global/local matrix/stencil/vector) classes,
/// all the backend-related interface and data 
/// are defined here
template <typename ValueType>
class BaseParalution : public ParalutionObj {

public:

  BaseParalution();
  BaseParalution(const BaseParalution<ValueType> &src);
  virtual ~BaseParalution();

  BaseParalution<ValueType>& operator=(const BaseParalution<ValueType> &src);

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
  virtual void CloneBackend(const BaseParalution<ValueType> &src);

  /// Clone the Backend descriptor from another object with different template ValueType
  template <typename ValueType2>
  void CloneBackend(const BaseParalution<ValueType2> &src);

  /// Print the object information (properties, backends)
  virtual void info() const = 0;

  /// Clear (free all data) the object
  virtual void Clear() = 0;

protected:

  /// Name of the object
  std::string object_name_;

  /// Parallel Manager
  const ParallelManager *pm_;

  /// Backend descriptor 
  Paralution_Backend_Descriptor local_backend_;

  /// Return true if the object is on the host
  virtual bool is_host(void) const = 0;

  /// Return true if the object is on the accelerator
  virtual bool is_accel(void) const = 0;

  // active async transfer
  bool asyncf;

  friend class BaseParalution<double>;
  friend class BaseParalution<float>;
  friend class BaseParalution<std::complex<double> >;
  friend class BaseParalution<std::complex<float> >;

  friend class BaseParalution<int>;

  friend class GlobalVector<int>;
  friend class GlobalVector<float>;
  friend class GlobalVector<double>;

  friend class GlobalMatrix<int>;
  friend class GlobalMatrix<float>;
  friend class GlobalMatrix<double>;

};


}

#endif // PARALUTION_LOCAL_BASE_HPP_
