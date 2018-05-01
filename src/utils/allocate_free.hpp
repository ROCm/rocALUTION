#ifndef ROCALUTION_UTILS_ALLOCATE_FREE_HPP_
#define ROCALUTION_UTILS_ALLOCATE_FREE_HPP_

#include <iostream>

// When CUDA backend is available the host memory allocation
// can use cudaMallocHost() function for pinned memory
// thus the memory transfers to the GPU are faster
// and this also enables the async transfers.
// Uncomment to use pinned memory
//
// To force standard CPU allocation comment the following line 
// #define ROCALUTION_CUDA_PINNED_MEMORY

namespace rocalution {

/// Allocate buffer on the host
template <typename DataType>
void allocate_host(const int size, DataType **ptr);

/// Free buffer on the host
template <typename DataType>
void free_host(DataType **ptr);

/// set a buffer to zero on the host
template <typename DataType>
void set_to_zero_host(const int size, DataType *ptr);

}

#endif // ROCALUTION_UTILS_ALLOCATE_FREE_HPP_
