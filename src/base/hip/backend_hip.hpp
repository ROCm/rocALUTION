#ifndef PARALUTION_BACKEND_HIP_HPP_
#define PARALUTION_BACKEND_HIP_HPP_

#include "../backend_manager.hpp"

#include <iostream>

namespace paralution {

template <typename ValueType>
class AcceleratorVector;
template <typename ValueType>
class AcceleratorMatrix;
template <typename ValueType>
class HostMatrix;

/// Initialize HIP (rocBLAS, rocSPARSE)
bool paralution_init_hip();
/// Release HIP resources (rocBLAS, rocSPARSE)
void paralution_stop_hip();

/// Print information about the HIPs in the systems
void paralution_info_hip(const struct Paralution_Backend_Descriptor);

/// Sync the device (for async transfers)
void paralution_hip_sync(void);

/// Build (and return) a vector on HIP
template <typename ValueType>
AcceleratorVector<ValueType>* _paralution_init_base_hip_vector(const struct Paralution_Backend_Descriptor backend_descriptor);

/// Build (and return) a matrix on HIP
template <typename ValueType>
AcceleratorMatrix<ValueType>* _paralution_init_base_hip_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                               const unsigned int matrix_format);


}

#endif // PARALUTION_BACKEND_HIP_HPP_
