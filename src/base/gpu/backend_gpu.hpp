#ifndef PARALUTION_BACKEND_GPU_HPP_
#define PARALUTION_BACKEND_GPU_HPP_

#include "../backend_manager.hpp"

#include <iostream>

namespace paralution {

template <typename ValueType>
class AcceleratorVector;
template <typename ValueType>
class AcceleratorMatrix;
template <typename ValueType>
class HostMatrix;

/// Initialize a GPU (CUDA, CUBLAS, CUSPARSE)
bool paralution_init_gpu();
/// Release the GPU resources (CUDA, CUBLAS, CUSPARSE)
void paralution_stop_gpu();

/// Print information about the GPUs in the systems
void paralution_info_gpu(const struct Paralution_Backend_Descriptor);

/// Sync the device (for async transfers)
void paralution_gpu_sync(void);

/// Build (and return) a vector on GPU
template <typename ValueType>
AcceleratorVector<ValueType>* _paralution_init_base_gpu_vector(const struct Paralution_Backend_Descriptor backend_descriptor);

/// Build (and return) a matrix on GPU
template <typename ValueType>
AcceleratorMatrix<ValueType>* _paralution_init_base_gpu_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                               const unsigned int matrix_format);


}

#endif // PARALUTION_BACKEND_GPU_HPP_
