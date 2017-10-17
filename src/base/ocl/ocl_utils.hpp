#ifndef PARALUTION_OCL_OCL_UTILS_HPP_
#define PARALUTION_OCL_OCL_UTILS_HPP_

#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "backend_ocl.hpp"

#if defined(__APPLE__) && defined(__MACH__)
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#define CL_KERNEL(...) #__VA_ARGS__
#define CL_DEVICE_REGISTERS_PER_BLOCK_NV      0x4002
#define CL_DEVICE_WARP_SIZE_NV                0x4003

static const char *OCL_ERROR[] = {
  "CL_SUCCESS",
  "CL_DEVICE_NOT_FOUND",
  "CL_DEVICE_NOT_AVAILABLE",
  "CL_COMPILER_NOT_AVAILABLE",
  "CL_MEM_OBJECT_ALLOCATION_FAILURE",
  "CL_OUT_OF_RESOURCES",
  "CL_OUT_OF_HOST_MEMORY",
  "CL_PROFILING_INFO_NOT_AVAILABLE",
  "CL_MEM_COPY_OVERLAP",
  "CL_IMAGE_FORMAT_MISMATCH",
  "CL_IMAGE_FORMAT_NOT_SUPPORTED",
  "CL_BUILD_PROGRAM_FAILURE",
  "CL_MAP_FAILURE",
  "CL_MISALIGNED_SUB_BUFFER_OFFSET",
  "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
  "CL_COMPILE_PROGRAM_FAILURE",
  "CL_LINKER_NOT_AVAILABLE",
  "CL_LINK_PROGRAM_FAILURE",
  "CL_DEVICE_PARTITION_FAILED",
  "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
  "",
  "",
  "",
  "",
  "",
  "",
  "",
  "",
  "",
  "",
  "CL_INVALID_VALUE",
  "CL_INVALID_DEVICE_TYPE",
  "CL_INVALID_PLATFORM",
  "CL_INVALID_DEVICE",
  "CL_INVALID_CONTEXT",
  "CL_INVALID_QUEUE_PROPERTIES",
  "CL_INVALID_COMMAND_QUEUE",
  "CL_INVALID_HOST_PTR",
  "CL_INVALID_MEM_OBJECT",
  "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
  "CL_INVALID_IMAGE_SIZE",
  "CL_INVALID_SAMPLER",
  "CL_INVALID_BINARY",
  "CL_INVALID_BUILD_OPTIONS",
  "CL_INVALID_PROGRAM",
  "CL_INVALID_PROGRAM_EXECUTABLE",
  "CL_INVALID_KERNEL_NAME",
  "CL_INVALID_KERNEL_DEFINITION",
  "CL_INVALID_KERNEL",
  "CL_INVALID_ARG_INDEX",
  "CL_INVALID_ARG_VALUE",
  "CL_INVALID_ARG_SIZE",
  "CL_INVALID_KERNEL_ARGS",
  "CL_INVALID_WORK_DIMENSION",
  "CL_INVALID_WORK_GROUP_SIZE",
  "CL_INVALID_WORK_ITEM_SIZE",
  "CL_INVALID_GLOBAL_OFFSET",
  "CL_INVALID_EVENT_WAIT_LIST",
  "CL_INVALID_EVENT",
  "CL_INVALID_OPERATION",
  "CL_INVALID_GL_OBJECT",
  "CL_INVALID_BUFFER_SIZE",
  "CL_INVALID_MIP_LEVEL",
  "CL_INVALID_GLOBAL_WORK_SIZE",
  "CL_INVALID_PROPERTY",
  "CL_INVALID_IMAGE_DESCRIPTOR",
  "CL_INVALID_COMPILER_OPTIONS",
  "CL_INVALID_LINKER_OPTIONS",
  "CL_INVALID_DEVICE_PARTITION_COUNT",
  "CL_INVALID_PIPE_SIZE",
  "CL_INVALID_DEVICE_QUEUE"
};

#define CHECK_OCL_ERROR(err_t, file, line) {                                                    \
  if (err_t != CL_SUCCESS) {                                                                    \
    LOG_INFO("OPENCL ERROR: " << OCL_ERROR[-err_t]);                                            \
    LOG_INFO("File: " << file << "; line: " << line);                                           \
    exit(1);                                                                                    \
  }                                                                                             \
}

namespace paralution {

template <typename ValueType, typename A0, typename A1>
cl_int ocl_kernel(const char *name, void *command_queue, const size_t LocalSize, const size_t GlobalSize, A0 a0, A1 a1) {

  cl_kernel kernel;
  cl_int err = _paralution_get_opencl_kernel<ValueType>(name, &kernel);
  if (err != CL_SUCCESS) return err;

  err = clSetKernelArg(kernel, 0, sizeof(A0), (void*) &a0); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 1, sizeof(A1), (void*) &a1); if (err != CL_SUCCESS) return err;

  return clEnqueueNDRangeKernel((cl_command_queue) command_queue, kernel, 1, NULL, &GlobalSize, &LocalSize, 0, NULL, NULL);

}

template <typename ValueType, typename A0, typename A1, typename A2>
cl_int ocl_kernel(const char *name, void *command_queue, const size_t LocalSize, const size_t GlobalSize, A0 a0, A1 a1, A2 a2) {

  cl_kernel kernel;
  cl_int err = _paralution_get_opencl_kernel<ValueType>(name, &kernel);
  if (err != CL_SUCCESS) return err;

  err = clSetKernelArg(kernel, 0, sizeof(A0), (void*) &a0); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 1, sizeof(A1), (void*) &a1); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 2, sizeof(A2), (void*) &a2); if (err != CL_SUCCESS) return err;

  return clEnqueueNDRangeKernel((cl_command_queue) command_queue, kernel, 1, NULL, &GlobalSize, &LocalSize, 0, NULL, NULL);

}

template <typename ValueType, typename A0, typename A1, typename A2, typename A3>
cl_int ocl_kernel(const char *name, void *command_queue, const size_t LocalSize, const size_t GlobalSize, A0 a0, A1 a1, A2 a2, A3 a3) {

  cl_kernel kernel;
  cl_int err = _paralution_get_opencl_kernel<ValueType>(name, &kernel);
  if (err != CL_SUCCESS) return err;

  err = clSetKernelArg(kernel, 0, sizeof(A0), (void*) &a0); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 1, sizeof(A1), (void*) &a1); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 2, sizeof(A2), (void*) &a2); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 3, sizeof(A3), (void*) &a3); if (err != CL_SUCCESS) return err;

  return clEnqueueNDRangeKernel((cl_command_queue) command_queue, kernel, 1, NULL, &GlobalSize, &LocalSize, 0, NULL, NULL);

}

template <typename ValueType, typename A0, typename A1, typename A2, typename A3, typename A4>
cl_int ocl_kernel(const char *name, void *command_queue, const size_t LocalSize, const size_t GlobalSize, A0 a0, A1 a1, A2 a2, A3 a3, A4 a4) {

  cl_kernel kernel;
  cl_int err = _paralution_get_opencl_kernel<ValueType>(name, &kernel);
  if (err != CL_SUCCESS) return err;

  err = clSetKernelArg(kernel, 0, sizeof(A0), (void*) &a0); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 1, sizeof(A1), (void*) &a1); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 2, sizeof(A2), (void*) &a2); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 3, sizeof(A3), (void*) &a3); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 4, sizeof(A4), (void*) &a4); if (err != CL_SUCCESS) return err;

  return clEnqueueNDRangeKernel((cl_command_queue) command_queue, kernel, 1, NULL, &GlobalSize, &LocalSize, 0, NULL, NULL);

}

template <typename ValueType, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5>
cl_int ocl_kernel(const char *name, void *command_queue, const size_t LocalSize, const size_t GlobalSize, A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5) {

  cl_kernel kernel;
  cl_int err = _paralution_get_opencl_kernel<ValueType>(name, &kernel);
  if (err != CL_SUCCESS) return err;

  err = clSetKernelArg(kernel, 0, sizeof(A0), (void*) &a0); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 1, sizeof(A1), (void*) &a1); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 2, sizeof(A2), (void*) &a2); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 3, sizeof(A3), (void*) &a3); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 4, sizeof(A4), (void*) &a4); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 5, sizeof(A5), (void*) &a5); if (err != CL_SUCCESS) return err;

  return clEnqueueNDRangeKernel((cl_command_queue) command_queue, kernel, 1, NULL, &GlobalSize, &LocalSize, 0, NULL, NULL);

}

template <typename ValueType, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
cl_int ocl_kernel(const char *name, void *command_queue, const size_t LocalSize, const size_t GlobalSize, A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6) {

  cl_kernel kernel;
  cl_int err = _paralution_get_opencl_kernel<ValueType>(name, &kernel);
  if (err != CL_SUCCESS) return err;

  err = clSetKernelArg(kernel, 0, sizeof(A0), (void*) &a0); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 1, sizeof(A1), (void*) &a1); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 2, sizeof(A2), (void*) &a2); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 3, sizeof(A3), (void*) &a3); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 4, sizeof(A4), (void*) &a4); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 5, sizeof(A5), (void*) &a5); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 6, sizeof(A6), (void*) &a6); if (err != CL_SUCCESS) return err;

  return clEnqueueNDRangeKernel((cl_command_queue) command_queue, kernel, 1, NULL, &GlobalSize, &LocalSize, 0, NULL, NULL);

}

template <typename ValueType, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7>
cl_int ocl_kernel(const char *name, void *command_queue, const size_t LocalSize, const size_t GlobalSize, A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7) {

  cl_kernel kernel;
  cl_int err = _paralution_get_opencl_kernel<ValueType>(name, &kernel);
  if (err != CL_SUCCESS) return err;

  err = clSetKernelArg(kernel, 0, sizeof(A0), (void*) &a0); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 1, sizeof(A1), (void*) &a1); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 2, sizeof(A2), (void*) &a2); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 3, sizeof(A3), (void*) &a3); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 4, sizeof(A4), (void*) &a4); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 5, sizeof(A5), (void*) &a5); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 6, sizeof(A6), (void*) &a6); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 7, sizeof(A7), (void*) &a7); if (err != CL_SUCCESS) return err;

  return clEnqueueNDRangeKernel((cl_command_queue) command_queue, kernel, 1, NULL, &GlobalSize, &LocalSize, 0, NULL, NULL);

}

template <typename ValueType, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8>
cl_int ocl_kernel(const char *name, void *command_queue, const size_t LocalSize, const size_t GlobalSize, A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8) {

  cl_kernel kernel;
  cl_int err = _paralution_get_opencl_kernel<ValueType>(name, &kernel);
  if (err != CL_SUCCESS) return err;

  err = clSetKernelArg(kernel, 0, sizeof(A0), (void*) &a0); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 1, sizeof(A1), (void*) &a1); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 2, sizeof(A2), (void*) &a2); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 3, sizeof(A3), (void*) &a3); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 4, sizeof(A4), (void*) &a4); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 5, sizeof(A5), (void*) &a5); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 6, sizeof(A6), (void*) &a6); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 7, sizeof(A7), (void*) &a7); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 8, sizeof(A8), (void*) &a8); if (err != CL_SUCCESS) return err;

  return clEnqueueNDRangeKernel((cl_command_queue) command_queue, kernel, 1, NULL, &GlobalSize, &LocalSize, 0, NULL, NULL);

}

template <typename ValueType, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9>
cl_int ocl_kernel(const char *name, void *command_queue, const size_t LocalSize, const size_t GlobalSize, A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9) {

  cl_kernel kernel;
  cl_int err = _paralution_get_opencl_kernel<ValueType>(name, &kernel);
  if (err != CL_SUCCESS) return err;

  err = clSetKernelArg(kernel, 0, sizeof(A0), (void*) &a0); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 1, sizeof(A1), (void*) &a1); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 2, sizeof(A2), (void*) &a2); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 3, sizeof(A3), (void*) &a3); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 4, sizeof(A4), (void*) &a4); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 5, sizeof(A5), (void*) &a5); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 6, sizeof(A6), (void*) &a6); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 7, sizeof(A7), (void*) &a7); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 8, sizeof(A8), (void*) &a8); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 9, sizeof(A9), (void*) &a9); if (err != CL_SUCCESS) return err;

  return clEnqueueNDRangeKernel((cl_command_queue) command_queue, kernel, 1, NULL, &GlobalSize, &LocalSize, 0, NULL, NULL);

}

template <typename ValueType, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10>
cl_int ocl_kernel(const char *name, void *command_queue, const size_t LocalSize, const size_t GlobalSize, A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10) {

  cl_kernel kernel;
  cl_int err = _paralution_get_opencl_kernel<ValueType>(name, &kernel);
  if (err != CL_SUCCESS) return err;

  err = clSetKernelArg(kernel,  0, sizeof( A0), (void*)  &a0); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel,  1, sizeof( A1), (void*)  &a1); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel,  2, sizeof( A2), (void*)  &a2); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel,  3, sizeof( A3), (void*)  &a3); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel,  4, sizeof( A4), (void*)  &a4); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel,  5, sizeof( A5), (void*)  &a5); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel,  6, sizeof( A6), (void*)  &a6); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel,  7, sizeof( A7), (void*)  &a7); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel,  8, sizeof( A8), (void*)  &a8); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel,  9, sizeof( A9), (void*)  &a9); if (err != CL_SUCCESS) return err;
  err = clSetKernelArg(kernel, 10, sizeof(A10), (void*) &a10); if (err != CL_SUCCESS) return err;

  return clEnqueueNDRangeKernel((cl_command_queue) command_queue, kernel, 1, NULL, &GlobalSize, &LocalSize, 0, NULL, NULL);

}

}

#endif // PARALUTION_OCL_OCL_UTILS_HPP_
