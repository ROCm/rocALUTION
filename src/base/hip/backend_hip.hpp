#ifndef ROCALUTION_BACKEND_HIP_HPP_
#define ROCALUTION_BACKEND_HIP_HPP_

#include "../backend_manager.hpp"

#include <iostream>

namespace rocalution {

template <typename ValueType>
class AcceleratorVector;
template <typename ValueType>
class AcceleratorMatrix;
template <typename ValueType>
class HostMatrix;

/// Initialize HIP (rocBLAS, rocSPARSE)
bool rocalution_init_hip();
/// Release HIP resources (rocBLAS, rocSPARSE)
void rocalution_stop_hip();

/// Print information about the HIPs in the systems
void rocalution_info_hip(const struct Rocalution_Backend_Descriptor);

/// Sync the device (for async transfers)
void rocalution_hip_sync(void);

/// Build (and return) a vector on HIP
template <typename ValueType>
AcceleratorVector<ValueType>* _rocalution_init_base_hip_vector(const struct Rocalution_Backend_Descriptor backend_descriptor);

/// Build (and return) a matrix on HIP
template <typename ValueType>
AcceleratorMatrix<ValueType>* _rocalution_init_base_hip_matrix(const struct Rocalution_Backend_Descriptor backend_descriptor,
                                                               const unsigned int matrix_format);


}

#endif // ROCALUTION_BACKEND_HIP_HPP_
