#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "ocl_allocate_free.hpp"
#include "ocl_utils.hpp"

#include <complex>

#if defined(__APPLE__) && defined(__MACH__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace paralution {

// Allocate memory on device
template <typename DataType>
void allocate_ocl(const int size, void *context, DataType **ptr) {

  LOG_DEBUG(0, "allocate_ocl()",
            size);

  if (size > 0) {

    assert (*ptr == NULL);

    cl_int err;

    // Allocate memory on device
    cl_mem data = clCreateBuffer((cl_context) context, CL_MEM_READ_WRITE, sizeof(DataType)*size, NULL, &err);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    *ptr = (DataType*) data;

    assert (*ptr != NULL);

  }

}

// Free memory on device
template <typename DataType>
void free_ocl(DataType **ptr) {

  LOG_DEBUG(0, "free_ocl()",
            "");

  // Free memory on device
  cl_int err = clReleaseMemObject((cl_mem) *ptr);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  *ptr = NULL;

}

// Set object on device to specific value (sync)
template<typename DataType>
void ocl_set_to(const int size, const DataType val, DataType *ptr, void *command_queue) {

  LOG_DEBUG(0, "ocl_set_to()",
            "size=" << size << " value=" << val);

  assert (ptr != NULL);

  if (size > 0) {

    cl_event event;
    cl_int err = clEnqueueFillBuffer((cl_command_queue) command_queue, (cl_mem) ptr, &val, sizeof(DataType), 0,
                                     size*sizeof(DataType), 0, NULL, &event);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    err = clWaitForEvents(1, &event);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    err = clReleaseEvent(event);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

// Copy object from host to device memory (sync)
template <typename DataType>
void ocl_host2dev(const int size, const DataType *src, DataType *dst, void *command_queue) {

  LOG_DEBUG(0, "ocl_host2dev()",
            size);

  if (size > 0) {

    assert (src != NULL);
    assert (dst != NULL);

    // Copy object from host to device memory
    cl_int err = clEnqueueWriteBuffer((cl_command_queue) command_queue, (cl_mem) dst, CL_TRUE, 0, size*sizeof(DataType), src, 0, NULL, NULL);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

// Copy object from device to host memory (sync)
template<typename DataType>
void ocl_dev2host(const int size, const DataType *src, DataType *dst, void *command_queue) {

  LOG_DEBUG(0, "ocl_dev2host()",
            size);

  if (size > 0) {

    assert (src != NULL);
    assert (dst != NULL);

    // Copy object from device to host memory
    cl_int err = clEnqueueReadBuffer((cl_command_queue) command_queue, (cl_mem) src, CL_TRUE, 0, size*sizeof(DataType), dst, 0, NULL, NULL);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

// Copy object from device to device memory (internal copy, sync)
template<typename DataType>
void ocl_dev2dev(const int size, const DataType *src, DataType *dst, void *command_queue) {

  LOG_DEBUG(0, "ocl_dev2dev()",
            size);

  if (size > 0) {

    assert (src != NULL);
    assert (dst != NULL);

    // Copy object from device to device memory (internal copy)
    cl_event event;
    cl_int err = clEnqueueCopyBuffer((cl_command_queue) command_queue, (cl_mem) src, (cl_mem) dst, 0, 0, size*sizeof(DataType), 0, NULL, &event);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    err = clWaitForEvents(1, &event);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    err = clReleaseEvent(event);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}


template void allocate_ocl<double      >(const int size, void *context, double       **ptr);
template void allocate_ocl<float       >(const int size, void *context, float        **ptr);
#ifdef SUPPORT_COMPLEX
template void allocate_ocl<std::complex<double> >(const int size, void *context, std::complex<double> **ptr);
template void allocate_ocl<std::complex<float>  >(const int size, void *context, std::complex<float>  **ptr);
#endif
template void allocate_ocl<int         >(const int size, void *context, int          **ptr);
template void allocate_ocl<unsigned int>(const int size, void *context, unsigned int **ptr);
template void allocate_ocl<char        >(const int size, void *context, char         **ptr);

template void free_ocl<double      >(double       **ptr);
template void free_ocl<float       >(float        **ptr);
#ifdef SUPPORT_COMPLEX
template void free_ocl<std::complex<double> >(std::complex<double> **ptr);
template void free_ocl<std::complex<float > >(std::complex<float > **ptr);
#endif
template void free_ocl<int         >(int          **ptr);
template void free_ocl<unsigned int>(unsigned int **ptr);
template void free_ocl<char        >(char         **ptr);

template void ocl_set_to<double      >(const int size, const double       val, double       *ptr, void *command_queue);
template void ocl_set_to<float       >(const int size, const float        val, float        *ptr, void *command_queue);
#ifdef SUPPORT_COMPLEX
template void ocl_set_to<std::complex<double> >(const int size, const std::complex<double> val, std::complex<double> *ptr, void *command_queue);
template void ocl_set_to<std::complex<float>  >(const int size, const std::complex<float>  val, std::complex<float>  *ptr, void *command_queue);
#endif
template void ocl_set_to<int         >(const int size, const int          val, int          *ptr, void *command_queue);
template void ocl_set_to<unsigned int>(const int size, const unsigned int val, unsigned int *ptr, void *command_queue);
template void ocl_set_to<char        >(const int size, const char         val, char         *ptr, void *command_queue);

template void ocl_host2dev<double      >(const int size, const double       *src, double       *dst, void *command_queue);
template void ocl_host2dev<float       >(const int size, const float        *src, float        *dst, void *command_queue);
#ifdef SUPPORT_COMPLEX
template void ocl_host2dev<std::complex<double> >(const int size, const std::complex<double> *src, std::complex<double> *dst, void *command_queue);
template void ocl_host2dev<std::complex<float>  >(const int size, const std::complex<float>  *src, std::complex<float>  *dst, void *command_queue);
#endif
template void ocl_host2dev<int         >(const int size, const int          *src, int          *dst, void *command_queue);
template void ocl_host2dev<unsigned int>(const int size, const unsigned int *src, unsigned int *dst, void *command_queue);
template void ocl_host2dev<char        >(const int size, const char         *src, char         *dst, void *command_queue);

template void ocl_dev2host<double      >(const int size, const double       *src, double       *dst, void *command_queue);
template void ocl_dev2host<float       >(const int size, const float        *src, float        *dst, void *command_queue);
#ifdef SUPPORT_COMPLEX
template void ocl_dev2host<std::complex<double> >(const int size, const std::complex<double> *src, std::complex<double> *dst, void *command_queue);
template void ocl_dev2host<std::complex<float>  >(const int size, const std::complex<float>  *src, std::complex<float>  *dst, void *command_queue);
#endif
template void ocl_dev2host<int         >(const int size, const int          *src, int          *dst, void *command_queue);
template void ocl_dev2host<unsigned int>(const int size, const unsigned int *src, unsigned int *dst, void *command_queue);
template void ocl_dev2host<char        >(const int size, const char         *src, char         *dst, void *command_queue);

template void ocl_dev2dev<double      >(const int size, const double       *src, double       *dst, void *command_queue);
template void ocl_dev2dev<float       >(const int size, const float        *src, float        *dst, void *command_queue);
#ifdef SUPPORT_COMPLEX
template void ocl_dev2dev<std::complex<double> >(const int size, const std::complex<double> *src, std::complex<double> *dst, void *command_queue);
template void ocl_dev2dev<std::complex<float>  >(const int size, const std::complex<float>  *src, std::complex<float>  *dst, void *command_queue);
#endif
template void ocl_dev2dev<int         >(const int size, const int          *src, int          *dst, void *command_queue);
template void ocl_dev2dev<unsigned int>(const int size, const unsigned int *src, unsigned int *dst, void *command_queue);
template void ocl_dev2dev<char        >(const int size, const char         *src, char         *dst, void *command_queue);

}
