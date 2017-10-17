#ifndef PARALUTION_OCL_ALLOCATE_FREE_HPP_
#define PARALUTION_OCL_ALLOCATE_FREE_HPP_

namespace paralution {

/// Allocate device memory
template <typename DataType>
void allocate_ocl(const int size, void *context, DataType **ptr);

/// Free device memory
template <typename DataType>
void free_ocl(DataType **ptr);

/// Set device object to specific values
template <typename DataType>
void ocl_set_to(const int size, const DataType val, DataType *ptr, void *command_queue);

/// Copy object from host to device memory
template <typename DataType>
void ocl_host2dev(const int size, const DataType *src, DataType *dst, void *command_queue);

/// Copy object from device to host memory
template <typename DataType>
void ocl_dev2host(const int size, const DataType *src, DataType *dst, void *command_queue);

/// Copy object from device to device (intra) memory
template <typename DataType>
void ocl_dev2dev(const int size, const DataType *src, DataType *dst, void *command_queue);


}

#endif // PARALUTION_OCL_ALLOCATE_FREE_HPP_
