#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../backend_manager.hpp"
#include "../base_vector.hpp"
#include "../base_matrix.hpp"
#include "ocl_utils.hpp"
#include "backend_ocl.hpp" 

#include "ocl_vector.hpp"
#include "ocl_matrix_csr.hpp"
#include "ocl_matrix_coo.hpp"
#include "ocl_matrix_mcsr.hpp"
#include "ocl_matrix_bcsr.hpp"
#include "ocl_matrix_hyb.hpp"
#include "ocl_matrix_dia.hpp"
#include "ocl_matrix_ell.hpp"
#include "ocl_matrix_dense.hpp"

#include "kernels/ocl_kernels_general.hpp"
#include "kernels/ocl_kernels_vector.hpp"
#include "kernels/ocl_kernels_csr.hpp"
#include "kernels/ocl_kernels_bcsr.hpp"
#include "kernels/ocl_kernels_mcsr.hpp"
#include "kernels/ocl_kernels_dense.hpp"
#include "kernels/ocl_kernels_ell.hpp"
#include "kernels/ocl_kernels_dia.hpp"
#include "kernels/ocl_kernels_coo.hpp"
#include "kernels/ocl_kernels_hyb.hpp"
#include "kernels/ocl_kernels_math_int.hpp"
#include "kernels/ocl_kernels_math_real.hpp"
#include "kernels/ocl_kernels_math_complex.hpp"

#include <sstream>
#include <complex>
#include <typeinfo>

namespace paralution {

// Initalizes the OpenCL backend
bool paralution_init_ocl(void) {

  LOG_DEBUG(0, "paralution_init_ocl()",
            "* begin");

  cl_int err;

  // Get number of OpenCL platforms available
  cl_uint num_platforms;

  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (num_platforms < 1 || err != CL_SUCCESS) {
    LOG_INFO("No OpenCL platform available");
    LOG_INFO("OpenCL has NOT been initialized!");
    paralution_stop_ocl();
    return false;
  }

  // Get all OpenCL platforms
  cl_platform_id *platforms = new cl_platform_id[num_platforms];

  err = clGetPlatformIDs(num_platforms, platforms, NULL);

  if (err != CL_SUCCESS) {
    LOG_INFO("Cannot retrieve OpenCL platforms");
    paralution_stop_ocl();
    return false;
  }

  LOG_INFO("Number of OpenCL capable platforms: " << num_platforms);

  // Get all OpenCL devices for each platform and look up the first GPU to make it default device
  cl_uint total_devices = 0;

  bool GPU_found = false;
  int GPU_platform = 0;
  int GPU_device = 0;

  cl_device_id **devices = new cl_device_id*[num_platforms];

  for (cl_uint i=0; i<num_platforms; ++i) {

    // Get number of OpenCL devices for this platform
    cl_uint num_devices;

    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

    if (err != CL_SUCCESS) {
      LOG_INFO("Cannot query for the number of OpenCL devices for platform " << i);
      paralution_stop_ocl();
      return false;
    }

    total_devices += num_devices;

    devices[i] = new cl_device_id[num_devices];

    // Get all OpenCL devices for this platform
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices[i], NULL);

    if (err != CL_SUCCESS) {
      LOG_INFO("Cannot retrieve OpenCL devices for platform " << i);
      paralution_stop_ocl();
      return false;
    }

    // Check if there is a GPU
    if (GPU_found == false) {

      cl_device_type device_type;

      for (cl_uint j=0; j<num_devices; ++j) {

        err = clGetDeviceInfo(devices[i][j], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);

        if (err != CL_SUCCESS) {
          LOG_INFO("Cannot query for CL_DEVICE_TYPE");
          paralution_stop_ocl();
          return false;
        }

        if (device_type & CL_DEVICE_TYPE_GPU) {
          GPU_found = true;
          GPU_platform = i;
          GPU_device = j;
          break;
        }

      }

    }

  }

  if (total_devices < 1) {
    LOG_INFO("No OpenCL capable device found");
    paralution_stop_ocl();
    return false;
  }

  LOG_INFO("Number of OpenCL capable devices: " << total_devices);

  // Set the default compute device, if no user specification is done
  if (_get_backend_descriptor()->OCL_plat == -1 && _get_backend_descriptor()->OCL_dev == -1) {

    _get_backend_descriptor()->OCL_plat = GPU_platform;
    _get_backend_descriptor()->OCL_dev = GPU_device;

  }

  int current_platform = _get_backend_descriptor()->OCL_plat;
  int current_device   = _get_backend_descriptor()->OCL_dev;

  _get_backend_descriptor()->OCL_platform_id = platforms[current_platform];
  _get_backend_descriptor()->OCL_device_id   = devices[current_platform][current_device];

  for (cl_uint i=0; i<num_platforms; ++i) {
    delete[] devices[i];
  }

  delete[] platforms;
  delete[] devices;

  // Query device vendor
  int vendor;
  err = clGetDeviceInfo((cl_device_id) _get_backend_descriptor()->OCL_device_id, CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, NULL);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  // Query device type
  cl_device_type devicetype;
  err = clGetDeviceInfo((cl_device_id) _get_backend_descriptor()->OCL_device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &devicetype, NULL);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  if (devicetype & CL_DEVICE_TYPE_CPU) {
    LOG_VERBOSE_INFO(2, "*** warning: It is highly recommended to switch to OpenMP backend for CPU devices");
  }

  // Query for maximum threads per block / max work group size
  size_t workgroupsize;
  err = clGetDeviceInfo((cl_device_id) _get_backend_descriptor()->OCL_device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroupsize), &workgroupsize, NULL);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);
  _get_backend_descriptor()->OCL_threads_per_proc = static_cast<int> (workgroupsize);

  if (vendor == 4318) {

    // NVIDIA
    // Query for warp size
    int warpsize;
    err = clGetDeviceInfo((cl_device_id) _get_backend_descriptor()->OCL_device_id, CL_DEVICE_WARP_SIZE_NV, sizeof(warpsize), &warpsize, NULL);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);
    _get_backend_descriptor()->OCL_warp_size = warpsize;

    // Query for registers per block
    unsigned int registers;
    err = clGetDeviceInfo((cl_device_id) _get_backend_descriptor()->OCL_device_id, CL_DEVICE_REGISTERS_PER_BLOCK_NV, sizeof(registers), &registers, NULL);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);
    _get_backend_descriptor()->OCL_regs_per_block = (size_t) registers;

  } else if (vendor == 4098) {

    // AMD
    _get_backend_descriptor()->OCL_warp_size = 64;
    _get_backend_descriptor()->OCL_regs_per_block = 65536;
    _get_backend_descriptor()->OCL_threads_per_proc = 1024;

  } else if (vendor == 32902) {

    // Intel
    _get_backend_descriptor()->OCL_warp_size = 128;
    _get_backend_descriptor()->OCL_regs_per_block = 65536;

  } else {

    LOG_INFO("Unknown OpenCL vendor");
    _get_backend_descriptor()->OCL_warp_size = 32;
    _get_backend_descriptor()->OCL_regs_per_block = 65536;

  }

  // Query for number of multiprocessors
  int computeunits;
  err = clGetDeviceInfo((cl_device_id) _get_backend_descriptor()->OCL_device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeunits), &computeunits, NULL);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);
  _get_backend_descriptor()->OCL_num_procs = computeunits;

  // Create OpenCL context
  _get_backend_descriptor()->OCL_context = clCreateContext(NULL, 1, (cl_device_id*) &_get_backend_descriptor()->OCL_device_id, NULL, NULL, &err);

  if (err != CL_SUCCESS) {
    LOG_INFO("Cannot create OpenCL context");
    paralution_stop_ocl();
    return false;
  }

  // Create OpenCL command queue
  if (vendor != 4318) {
#ifdef CL_VERSION_2_0
    _get_backend_descriptor()->OCL_command_queue = clCreateCommandQueueWithProperties((cl_context) _get_backend_descriptor()->OCL_context,
                                                                                      (cl_device_id) _get_backend_descriptor()->OCL_device_id,
                                                                                      NULL, &err);
#else
    _get_backend_descriptor()->OCL_command_queue = clCreateCommandQueue((cl_context) _get_backend_descriptor()->OCL_context,
                                                                        (cl_device_id) _get_backend_descriptor()->OCL_device_id,
                                                                        0, &err);
#endif
  } else {

    _get_backend_descriptor()->OCL_command_queue = clCreateCommandQueue((cl_context) _get_backend_descriptor()->OCL_context,
                                                                        (cl_device_id) _get_backend_descriptor()->OCL_device_id,
                                                                        0, &err);

  }

  if (err != CL_SUCCESS) {
    LOG_INFO("Cannot create OpenCL command queue for current device");
    paralution_stop_ocl();
    return false;
  }

  // Possible precision
  const int nprec = 5;
  std::string precision[nprec];
  _get_backend_descriptor()->OCL_program.resize(nprec, NULL);

  precision[0] = "int";
  precision[1] = "float";
  precision[2] = "double";
  precision[3] = "float2";
  precision[4] = "double2";

  // Create programs
  for (int i=0; i<nprec; ++i) {

    std::stringstream sizes;
    sizes << "#define BLOCK_SIZE " << _get_backend_descriptor()->OCL_block_size << "\n";
    sizes << "#define WARP_SIZE "  << _get_backend_descriptor()->OCL_warp_size << "\n";

//    std::string source_str = ("#include \"matrix_formats_ind.hpp\"\n");
    std::string source_str = ("#define IndexType int\n");
    source_str.append("#define ValueType " + precision[i] + "\n");
    source_str.append(sizes.str());

    if (precision[i] == "float") {
      source_str.append("#define AtomicType uint\n");
      source_str.append("#pragma OPENCL EXTENSION cl_khr_fp64 : disable\n");
      source_str.append("#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n");
      source_str.append(ocl_kernels_math_real);
    }

    if (precision[i] == "double") {
      source_str.append("#define AtomicType ulong\n");
      source_str.append("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
      source_str.append("#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable\n");
      source_str.append(ocl_kernels_math_real);
    }

    if (precision[i] == "float2") {
      source_str.append("#define RealType float\n");
      source_str.append("#pragma OPENCL EXTENSION cl_khr_fp64 : disable\n");
      source_str.append(ocl_kernels_math_complex);
    }

    if (precision[i] == "double2") {
      source_str.append("#define RealType double\n");
      source_str.append("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
      source_str.append(ocl_kernels_math_complex);
    }

    if (precision[i] == "int") {
      source_str.append("#define AtomicType uint\n");
      source_str.append("#pragma OPENCL EXTENSION cl_khr_fp64 : disable\n");
      source_str.append("#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n");
      source_str.append(ocl_kernels_math_int);
    }

    // Add all programs
    source_str.append(ocl_kernels_general);
//      source_str.append(ocl_kernel_conversion);
    source_str.append(ocl_kernels_vector);
    source_str.append(ocl_kernels_csr);
    source_str.append(ocl_kernels_bcsr);
    source_str.append(ocl_kernels_mcsr);
    source_str.append(ocl_kernels_dense);
    source_str.append(ocl_kernels_ell);
    source_str.append(ocl_kernels_dia);
    source_str.append(ocl_kernels_coo);
    source_str.append(ocl_kernels_hyb);

    const char *source[] = { source_str.c_str() };

    // Create OpenCL program
    _get_backend_descriptor()->OCL_program[i] = (void*) clCreateProgramWithSource((cl_context) _get_backend_descriptor()->OCL_context, 1, source, NULL, &err);

    if (err != CL_SUCCESS) {
      LOG_INFO("Cannot create OpenCL program");
      paralution_stop_ocl();
      return false;
    }

    // Compile OpenCL program
    err = clBuildProgram((cl_program) _get_backend_descriptor()->OCL_program[i], 0, NULL, "-w -cl-mad-enable -cl-no-signed-zeros", NULL, NULL);

    // If build fails, get compiler log
    if (err == CL_BUILD_PROGRAM_FAILURE) {

      size_t logsize;
      err = clGetProgramBuildInfo((cl_program) _get_backend_descriptor()->OCL_program[i], (cl_device_id) _get_backend_descriptor()->OCL_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);

      if (err != CL_SUCCESS) {
        LOG_INFO("Cannot retrieve OpenCL program build log size");
        paralution_stop_ocl();
        return false;
      }

      char *log = new char[logsize];
      err = clGetProgramBuildInfo((cl_program) _get_backend_descriptor()->OCL_program[i], (cl_device_id) _get_backend_descriptor()->OCL_device_id, CL_PROGRAM_BUILD_LOG, logsize, log, NULL);

      if (err != CL_SUCCESS) {
        LOG_INFO("Cannot retrieve OpenCL program build log");
        paralution_stop_ocl();
        return false;
      }

      LOG_INFO("OpenCL error in program " << precision[i]);
      LOG_INFO("OpenCL build log:\n" << log);

      delete[] log;

      return false;

    }

  }

  LOG_DEBUG(0, "paralution_init_ocl()",
            "* end");

  return true;

}

// Stop OpenCL backend
void paralution_stop_ocl(void) {

  LOG_DEBUG(0, "paralution_stop_ocl()",
            "* begin");

  if (_get_backend_descriptor()->accelerator == true) {

    cl_int err;

    // Release OpenCL command queue
    if (_get_backend_descriptor()->OCL_command_queue != NULL) {
      err = clReleaseCommandQueue((cl_command_queue) _get_backend_descriptor()->OCL_command_queue);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);
      _get_backend_descriptor()->OCL_command_queue = NULL;
    }

    // Release OpenCL kernels
    if (_get_backend_descriptor()->OCL_kernels.empty() == false) {
      for (std::map<std::string, void*>::iterator it=_get_backend_descriptor()->OCL_kernels.begin(); it!=_get_backend_descriptor()->OCL_kernels.end(); ++it) {
        err = clReleaseKernel((cl_kernel) it->second);
        CHECK_OCL_ERROR(err, __FILE__, __LINE__);
      }
    }

    // Release OpenCL program
    if (_get_backend_descriptor()->OCL_program.size() > 0) {
      for (unsigned int i=0; i<_get_backend_descriptor()->OCL_program.size(); ++i) {
        err = clReleaseProgram((cl_program) _get_backend_descriptor()->OCL_program[i]);
        CHECK_OCL_ERROR(err, __FILE__, __LINE__);
      }
    }

    // Release OpenCL context
    if (_get_backend_descriptor()->OCL_context != NULL) {
      err = clReleaseContext((cl_context) _get_backend_descriptor()->OCL_context);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);
      _get_backend_descriptor()->OCL_context = NULL;
    }

  }

  _get_backend_descriptor()->OCL_plat = -1;
  _get_backend_descriptor()->OCL_dev  = -1;

  LOG_DEBUG(0, "paralution_stop_ocl()",
            "* end");

}

// Print OpenCL computation device information to screen
void paralution_info_ocl(const struct Paralution_Backend_Descriptor backend_descriptor) {

  LOG_DEBUG(0, "paralution_info_ocl()",
            "* begin");

  if (_get_backend_descriptor()->OCL_plat != -1 && _get_backend_descriptor()->OCL_dev != -1) {
    LOG_INFO("Selected OpenCL platform: " << _get_backend_descriptor()->OCL_plat);
    LOG_INFO("Selected OpenCL device: " << _get_backend_descriptor()->OCL_dev);
  } else {
    LOG_INFO("No OpenCL device is selected!");
  }

  cl_int err;

  // Get number of OpenCL platforms available
  cl_uint num_platforms;

  err = clGetPlatformIDs(0, NULL, &num_platforms);

  if (num_platforms < 1 || err != CL_SUCCESS) {
    LOG_INFO("No OpenCL capable platform found");
    return;
  }

  // Get all OpenCL platforms
  cl_platform_id *platforms = new cl_platform_id[num_platforms];

  err = clGetPlatformIDs(num_platforms, platforms, NULL);

  if (err != CL_SUCCESS) {
    LOG_INFO("Cannot retrieve OpenCL platforms");
    return;
  }

  // Get all OpenCL devices for each platform and print info
  for (cl_uint i=0; i<num_platforms; ++i) {

    // Get number of OpenCL devices for this platform
    cl_uint num_devices;

    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

    if (err != CL_SUCCESS) {
      LOG_INFO("Cannot query for the number of OpenCL devices for platform " << i);
      return;
    }

    cl_device_id *devices = new cl_device_id[num_devices];

    // Get all OpenCL devices for this platform
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

    if (err != CL_SUCCESS) {
      LOG_INFO("Cannot retrieve OpenCL devices for platform " << i);
      return;
    }

    // Print info for each device
    for (cl_uint j=0; j<num_devices; ++j) {

      // Platform name
      size_t length;
      err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &length);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      char *platform_name = new char[length];
      err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, length, platform_name, NULL);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      // Device name
      err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &length);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      char *device_name = new char[length];
      err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, length, device_name, NULL);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      // Global memory
      cl_ulong global_memory;
      err = clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_memory), &global_memory, NULL);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      // Clock frequency
      cl_uint clock_frequency;
      err = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      // OpenCL version
      err = clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &length);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      char *opencl_version = new char[length];
      err = clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, length, opencl_version, NULL);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      // Device type
      cl_device_type device_type;
      err = clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      std::string type;
      if (device_type & CL_DEVICE_TYPE_CPU)         type = "CPU";
      if (device_type & CL_DEVICE_TYPE_GPU)         type = "GPU";
      if (device_type & CL_DEVICE_TYPE_ACCELERATOR) type = "ACCELERATOR";
      if (device_type & CL_DEVICE_TYPE_DEFAULT)     type = "DEFAULT";

      LOG_INFO("------------------------------------------------");
      LOG_INFO("Platform number: " << i);
      LOG_INFO("Platform name: "   << platform_name);
      LOG_INFO("Device number: "   << j);
      LOG_INFO("Device name: "     << device_name);
      LOG_INFO("Device type: "     << type);
      LOG_INFO("totalGlobalMem: "  << (global_memory >> 20) <<" MByte");
      LOG_INFO("clockRate: "       << clock_frequency);
      LOG_INFO("OpenCL version: "  << opencl_version);
      LOG_INFO("------------------------------------------------");

      delete[] platform_name;
      delete[] device_name;
      delete[] opencl_version;

    }

    delete[] devices;

  }

  delete[] platforms;

  LOG_DEBUG(0, "paralution_info_ocl()",
            "* end");

}

void paralution_ocl_sync(void) {

  LOG_DEBUG(0, "paralution_ocl_sync()",
            "");

  assert (_get_backend_descriptor()->OCL_command_queue != NULL);

  cl_int err = clFinish((cl_command_queue) _get_backend_descriptor()->OCL_command_queue);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

}

template <typename ValueType>
cl_int _paralution_get_opencl_kernel(const char *name, cl_kernel *kernel) {

  cl_int err = CL_SUCCESS;

  // Check precision to return appropriate kernel
  std::string precision;
  int index;

  if (typeid(ValueType) == typeid(float)) {
    precision = "float";
    index = 1;
  } else if (typeid(ValueType) == typeid(double)) {
    precision = "double";
    index = 2;
  } else if (typeid(ValueType) == typeid(std::complex<float>)) {
    precision = "float2";
    index = 3;
  } else if (typeid(ValueType) == typeid(std::complex<double>)) {
    precision = "double2";
    index = 4;
  } else if (typeid(ValueType) == typeid(int)) {
    precision = "int";
    index = 0;
  } else {
    return CL_BUILD_PROGRAM_FAILURE;
  }

  std::string kernel_name = name + precision;

  if (_get_backend_descriptor()->OCL_kernels.find(kernel_name) == _get_backend_descriptor()->OCL_kernels.end()) {

    _get_backend_descriptor()->OCL_kernels[kernel_name] = (void*) clCreateKernel((cl_program) _get_backend_descriptor()->OCL_program[index], name, &err);

    if (err != CL_SUCCESS) {
      return err;
    }

  }

  *kernel = (cl_kernel) _get_backend_descriptor()->OCL_kernels[kernel_name];

  return err;

}

template <typename ValueType>
AcceleratorMatrix<ValueType>* _paralution_init_base_ocl_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                               const unsigned int matrix_format) {
  assert (backend_descriptor.backend == OCL);

  switch (matrix_format) {

  case CSR:
    return new OCLAcceleratorMatrixCSR<ValueType>(backend_descriptor);
    
  case COO:
    return new OCLAcceleratorMatrixCOO<ValueType>(backend_descriptor);

  case MCSR:
    return new OCLAcceleratorMatrixMCSR<ValueType>(backend_descriptor);

  case DIA:
    return new OCLAcceleratorMatrixDIA<ValueType>(backend_descriptor);
    
  case ELL:
    return new OCLAcceleratorMatrixELL<ValueType>(backend_descriptor);

  case DENSE:
    return new OCLAcceleratorMatrixDENSE<ValueType>(backend_descriptor);

  case HYB:
    return new OCLAcceleratorMatrixHYB<ValueType>(backend_descriptor);

  case BCSR:
    return new OCLAcceleratorMatrixBCSR<ValueType>(backend_descriptor);

  default:
    LOG_INFO("This backend is not supported for Matrix types");
    FATAL_ERROR(__FILE__, __LINE__);   
    return NULL;

  }

}

template <typename ValueType>
AcceleratorVector<ValueType>* _paralution_init_base_ocl_vector(const struct Paralution_Backend_Descriptor backend_descriptor) {

  assert (backend_descriptor.backend == OCL);

  return new OCLAcceleratorVector<ValueType>(backend_descriptor);

}


template cl_int _paralution_get_opencl_kernel<float>(const char *name, cl_kernel *kernel);
template cl_int _paralution_get_opencl_kernel<double>(const char *name, cl_kernel *kernel);
template cl_int _paralution_get_opencl_kernel<std::complex<float> >(const char *name, cl_kernel *kernel);
template cl_int _paralution_get_opencl_kernel<std::complex<double> >(const char *name, cl_kernel *kernel);
template cl_int _paralution_get_opencl_kernel<int>(const char *name, cl_kernel *kernel);

template AcceleratorVector<double>* _paralution_init_base_ocl_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
template AcceleratorVector<float>*  _paralution_init_base_ocl_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
#ifdef SUPPORT_COMPLEX
template AcceleratorVector<std::complex<double> >* _paralution_init_base_ocl_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
template AcceleratorVector<std::complex<float> >*  _paralution_init_base_ocl_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
#endif
template AcceleratorVector<int>*    _paralution_init_base_ocl_vector(const struct Paralution_Backend_Descriptor backend_descriptor);

template AcceleratorMatrix<double>* _paralution_init_base_ocl_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                     const unsigned int matrix_format);
template AcceleratorMatrix<float>*  _paralution_init_base_ocl_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                     const unsigned int matrix_format);
#ifdef SUPPORT_COMPLEX
template AcceleratorMatrix<std::complex<double> >* _paralution_init_base_ocl_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                                    const unsigned int matrix_format);
template AcceleratorMatrix<std::complex<float> >*  _paralution_init_base_ocl_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                                    const unsigned int matrix_format);
#endif

}
