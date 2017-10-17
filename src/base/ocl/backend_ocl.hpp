#ifndef PARALUTION_BACKEND_OCL_HPP_
#define PARALUTION_BACKEND_OCL_HPP_

#include "../backend_manager.hpp"

#if defined(__APPLE__) && defined(__MACH__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace paralution {

template <typename ValueType>
class AcceleratorVector;
template <typename ValueType>
class AcceleratorMatrix;
template <typename ValueType>
class HostMatrix;

/// Initialize OpenCL
bool paralution_init_ocl();
/// Release the OpenCL resources
void paralution_stop_ocl();

/// Print information about the GPUs in the systems
void paralution_info_ocl(const struct Paralution_Backend_Descriptor);

/// Sync the device
void paralution_ocl_sync(void);

/// Get OpenCL kernel
template <typename ValueType>
cl_int _paralution_get_opencl_kernel(const char *name, cl_kernel *kernel);

/// Build (and return) an OpenCL vector
template <typename ValueType>
AcceleratorVector<ValueType>* _paralution_init_base_ocl_vector(const struct Paralution_Backend_Descriptor backend_descriptor);

/// Build (and return) an OpenCL matrix
template <typename ValueType>
AcceleratorMatrix<ValueType>* _paralution_init_base_ocl_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                               const unsigned int matrix_format);

}

#endif // PARALUTION_BACKEND_OCL_HPP_
