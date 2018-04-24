#include "../../utils/def.hpp"
#include "../backend_manager.hpp"
#include "backend_hip.hpp" 
#include "../../utils/log.hpp"
#include "hip_utils.hpp"
#include "../base_vector.hpp"
#include "../base_matrix.hpp"

#include "hip_vector.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_matrix_coo.hpp"
#include "hip_matrix_mcsr.hpp"
#include "hip_matrix_bcsr.hpp"
#include "hip_matrix_hyb.hpp"
#include "hip_matrix_dia.hpp"
#include "hip_matrix_ell.hpp"
#include "hip_matrix_dense.hpp"

#include <hip/hip_runtime_api.h>
#include <hipblas.h>
//#include <rocsparse.h>
#include <complex>

namespace paralution {

bool paralution_init_hip(void) {

  LOG_DEBUG(0, "paralution_init_hip()",
            "* begin");

  assert(_get_backend_descriptor()->HIP_blas_handle == NULL);
  assert(_get_backend_descriptor()->HIP_sparse_handle == NULL);
  //  assert(_get_backend_descriptor()->HIP_dev == -1);

  // create a handle
  _get_backend_descriptor()->HIP_blas_handle = new hipblasHandle_t;
//  _get_backend_descriptor()->HIP_sparse_handle = new rocsparse_handle;

  // get last error (if any)
//TODO port to hip  hipGetLastError();

  hipError_t hip_status_t;
  int num_dev;
  hipGetDeviceCount(&num_dev);
  hip_status_t = hipGetLastError();

  // if querying for device count fails, fall back to host backend
  if (hip_status_t != hipSuccess) {
    LOG_INFO("Querying for HIP devices failed - falling back to host backend");
    return false;
  }

  LOG_INFO("Number of HIP devices in the system: " << num_dev);

  if (num_dev < 1) {

    LOG_INFO("No HIP device found");

  } else {

    if (_get_backend_descriptor()->HIP_dev != -1)
      num_dev = 1;

    for (int idev=0; idev<num_dev; idev++) {

      int dev = idev;

      if (_get_backend_descriptor()->HIP_dev != -1) {
        dev = _get_backend_descriptor()->HIP_dev;
      }

      hipSetDevice(dev);
      hip_status_t = hipGetLastError();

      if (hip_status_t == hipSuccess) {

        if ((hipblasCreate(static_cast<hipblasHandle_t*>(_get_backend_descriptor()->HIP_blas_handle)) == HIPBLAS_STATUS_SUCCESS)) {// &&
//            (rocsparse_create_handle(static_cast<rocsparse_handle*>(_get_backend_descriptor()->HIP_sparse_handle)) == rocsparse_status_success)) {
          _get_backend_descriptor()->HIP_dev = dev;
          break;
        } else
          LOG_INFO("HIP device " << dev << " cannot create rocBLAS/rocSPARSE context");
      }

      if (hip_status_t == hipErrorContextAlreadyInUse)
        LOG_INFO("HIP context of device " << dev << " is already in use");

      if (hip_status_t == hipErrorInvalidDevice)
        LOG_INFO("HIP device " << dev << " is invalid");

    }

  }

  if (_get_backend_descriptor()->HIP_dev == -1) {
    LOG_INFO("HIP and rocBLAS/rocSPARSE have NOT been initialized!");
    return false;
  }

  struct hipDeviceProp_t dev_prop;      
  hipGetDeviceProperties(&dev_prop, _get_backend_descriptor()->HIP_dev);

  // TODO
  if (dev_prop.major < 2) {
    LOG_INFO("HIP device " << _get_backend_descriptor()->HIP_dev << " has low compute capability (min 2.0 is needed)");
    return false;
  }

  // Get some properties from the device TODO
  _get_backend_descriptor()->HIP_warp = dev_prop.warpSize;
  _get_backend_descriptor()->HIP_num_procs = dev_prop.multiProcessorCount;
  _get_backend_descriptor()->HIP_threads_per_proc = dev_prop.maxThreadsPerMultiProcessor;
  _get_backend_descriptor()->HIP_max_threads = 65536; //dev_prop.regsPerBlock;

  LOG_DEBUG(0, "paralution_init_hip()",
            "* end");

  return true;

}


void paralution_stop_hip(void) {

  LOG_DEBUG(0, "paralution_stop_hip()",
            "* begin");

  if (_get_backend_descriptor()->accelerator) {

    if (hipblasDestroy(*(static_cast<hipblasHandle_t*>(_get_backend_descriptor()->HIP_blas_handle))) != HIPBLAS_STATUS_SUCCESS) {
      LOG_INFO("Error in rocblas_destroy_handle");
    }
/*
    if (rocsparse_destroy_handle(*(static_cast<rocsparse_handle*>(_get_backend_descriptor()->HIP_sparse_handle))) != rocsparse_status_success) {
      LOG_INFO("Error in rocsparse_destroy_handle");
    }
*/
  }

  delete (static_cast<hipblasHandle_t*>(_get_backend_descriptor()->HIP_blas_handle));
//  delete (static_cast<rocsparse_handle*>(_get_backend_descriptor()->HIP_sparse_handle));

  _get_backend_descriptor()->HIP_blas_handle = NULL; 
  _get_backend_descriptor()->HIP_sparse_handle = NULL;

  _get_backend_descriptor()->HIP_dev = -1;

  LOG_DEBUG(0, "paralution_stop_hip()",
            "* end");

}

void paralution_info_hip(const struct Paralution_Backend_Descriptor backend_descriptor) {

    int num_dev;

    hipGetDeviceCount(&num_dev);
    hipGetLastError(); 
    // TODO CHECK_HIP_ERROR(__FILE__, __LINE__);

    //    LOG_INFO("Number of HIP devices in the sytem: " << num_dev);    

    if (_get_backend_descriptor()->HIP_dev >= 0) {
      LOG_INFO("Selected HIP device: " << backend_descriptor.HIP_dev);

    } else {

      LOG_INFO("No HIP device is selected!");      

    }

    for (int dev = 0; dev < num_dev; dev++) {

      struct hipDeviceProp_t dev_prop;      
      hipGetDeviceProperties(&dev_prop, dev);
     
      LOG_INFO("------------------------------------------------");        
      LOG_INFO("Device number: "               << dev);
      LOG_INFO("Device name: "                 << dev_prop.name);                        //    char name[256];
      LOG_INFO("totalGlobalMem: "              << (dev_prop.totalGlobalMem >> 20) <<" MByte");//    size_t totalGlobalMem;
      /*
      LOG_INFO("sharedMemPerBlock: "           << dev_prop.sharedMemPerBlock);           //    size_t sharedMemPerBlock;
      LOG_INFO("regsPerBlock: "                << dev_prop.regsPerBlock);                //    int regsPerBlock;
      LOG_INFO("warpSize: "                    << dev_prop.warpSize);                    //    int warpSize;
      LOG_INFO("memPitch: "                    << dev_prop.memPitch);                    //    size_t memPitch;
      LOG_INFO("maxThreadsPerBlock: "          << dev_prop.maxThreadsPerBlock);          //    int maxThreadsPerBlock;
      LOG_INFO("maxThreadsDim[0]: "            << dev_prop.maxThreadsDim[0]);            //    int maxThreadsDim[0];
      LOG_INFO("maxThreadsDim[1]: "            << dev_prop.maxThreadsDim[1]);            //    int maxThreadsDim[1];
      LOG_INFO("maxThreadsDim[2]: "            << dev_prop.maxThreadsDim[2]);            //    int maxThreadsDim[2];
      LOG_INFO("maxGridSize[0]: "              << dev_prop.maxGridSize[0]);              //    int maxGridSize[0];
      LOG_INFO("maxGridSize[1]: "              << dev_prop.maxGridSize[1]);              //    int maxGridSize[1];
      LOG_INFO("maxGridSize[2]: "              << dev_prop.maxGridSize[2]);              //    int maxGridSize[2];
      */
      LOG_INFO("clockRate: "                   << dev_prop.clockRate);                   //    int clockRate;
      /*
      LOG_INFO("totalConstMem: "               << dev_prop.totalConstMem);               //    size_t totalConstMem;
      */
      /*
      LOG_INFO("major: "                       << dev_prop.major);                       //    int major;
      LOG_INFO("minor: "                       << dev_prop.minor);                       //    int minor;
      */
      LOG_INFO("compute capability: "           << dev_prop.major << "." << dev_prop.minor);
      /*
      LOG_INFO("textureAlignment: "            << dev_prop.textureAlignment);            //    size_t textureAlignment;
      LOG_INFO("deviceOverlap: "               << dev_prop.deviceOverlap);               //    int deviceOverlap;
      LOG_INFO("multiProcessorCount: "         << dev_prop.multiProcessorCount);         //    int multiProcessorCount;
      LOG_INFO("kernelExecTimeoutEnabled: "    << dev_prop.kernelExecTimeoutEnabled);    //    int kernelExecTimeoutEnabled;
      LOG_INFO("integrated: "                  << dev_prop.integrated);                  //    int integrated;
      LOG_INFO("canMapHostMemory: "            << dev_prop.canMapHostMemory);            //    int canMapHostMemory;
      LOG_INFO("computeMode: "                 << dev_prop.computeMode);                 //    int computeMode;
      LOG_INFO("maxTexture1D: "                << dev_prop.maxTexture1D);                //    int maxTexture1D;
      LOG_INFO("maxTexture2D[0]: "             << dev_prop.maxTexture2D[0]);             //    int maxTexture2D[0];
      LOG_INFO("maxTexture2D[1]: "             << dev_prop.maxTexture2D[1]);             //    int maxTexture2D[1];
      LOG_INFO("maxTexture3D[0]: "             << dev_prop.maxTexture3D[0]);             //    int maxTexture3D[0];
      LOG_INFO("maxTexture3D[1]: "             << dev_prop.maxTexture3D[1]);             //    int maxTexture3D[1];
      LOG_INFO("maxTexture3D[2]: "             << dev_prop.maxTexture3D[2]);             //    int maxTexture3D[2];
      LOG_INFO("maxTexture1DLayered[0]: "      << dev_prop.maxTexture1DLayered[0]);      //    int maxTexture1DLayered[0];
      LOG_INFO("maxTexture1DLayered[1]: "      << dev_prop.maxTexture1DLayered[1]);      //    int maxTexture1DLayered[1];
      LOG_INFO("maxTexture2DLayered[0]: "      << dev_prop.maxTexture2DLayered[0]);      //    int maxTexture2DLayered[0];
      LOG_INFO("maxTexture2DLayered[1]: "      << dev_prop.maxTexture2DLayered[1]);      //    int maxTexture2DLayered[1];
      LOG_INFO("maxTexture2DLayered[2]: "      << dev_prop.maxTexture2DLayered[2]);      //    int maxTexture2DLayered[2];
      LOG_INFO("surfaceAlignment: "            << dev_prop.surfaceAlignment);            //    size_t surfaceAlignment;
      LOG_INFO("concurrentKernels: "           << dev_prop.concurrentKernels);           //    int concurrentKernels;
      LOG_INFO("ECCEnabled: "                  << dev_prop.ECCEnabled);                  //    int ECCEnabled;
      LOG_INFO("pciBusID: "                    << dev_prop.pciBusID);                    //    int pciBusID;
      LOG_INFO("pciDeviceID: "                 << dev_prop.pciDeviceID);                 //    int pciDeviceID;
      LOG_INFO("pciDomainID: "                 << dev_prop.pciDomainID);                 //    int pciDomainID;
      LOG_INFO("tccDriver: "                   << dev_prop.tccDriver);                   //    int tccDriver;
      LOG_INFO("asyncEngineCount: "            << dev_prop.asyncEngineCount);            //    int asyncEngineCount;
      LOG_INFO("unifiedAddressing: "           << dev_prop.unifiedAddressing);           //    int unifiedAddressing;
      LOG_INFO("memoryClockRate: "             << dev_prop.memoryClockRate);             //    int memoryClockRate;
      LOG_INFO("memoryBusWidth: "              << dev_prop.memoryBusWidth);              //    int memoryBusWidth;
      LOG_INFO("l2CacheSize: "                 << dev_prop.l2CacheSize);                 //    int l2CacheSize;
      LOG_INFO("maxThreadsPerMultiProcessor: " << dev_prop.maxThreadsPerMultiProcessor); //    int maxThreadsPerMultiProcessor;
      */
      LOG_INFO("------------------------------------------------");  

      
    }
    
}

template <typename ValueType>
AcceleratorMatrix<ValueType>* _paralution_init_base_hip_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                               const unsigned int matrix_format) {
  assert(backend_descriptor.backend == HIP);

  switch (matrix_format) {

  case CSR:
    return new HIPAcceleratorMatrixCSR<ValueType>(backend_descriptor);

  case COO:
    return new HIPAcceleratorMatrixCOO<ValueType>(backend_descriptor);

  case MCSR:
    return new HIPAcceleratorMatrixMCSR<ValueType>(backend_descriptor);

  case DIA:
    return new HIPAcceleratorMatrixDIA<ValueType>(backend_descriptor);

  case ELL:
    return new HIPAcceleratorMatrixELL<ValueType>(backend_descriptor);

  case DENSE:
    return new HIPAcceleratorMatrixDENSE<ValueType>(backend_descriptor);

  case HYB:
    return new HIPAcceleratorMatrixHYB<ValueType>(backend_descriptor);

  case BCSR:
    return new HIPAcceleratorMatrixBCSR<ValueType>(backend_descriptor);

  default:
    LOG_INFO("This backed is not supported for Matrix types");
    FATAL_ERROR(__FILE__, __LINE__);   
    return NULL;
  }

}

template <typename ValueType>
AcceleratorVector<ValueType>* _paralution_init_base_hip_vector(const struct Paralution_Backend_Descriptor backend_descriptor) {

  assert(backend_descriptor.backend == HIP);

  return new HIPAcceleratorVector<ValueType>(backend_descriptor);

}

void paralution_hip_sync(void) {

  hipDeviceSynchronize();
  CHECK_HIP_ERROR(__FILE__, __LINE__);

}

template AcceleratorVector<float>* _paralution_init_base_hip_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
template AcceleratorVector<double>* _paralution_init_base_hip_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
#ifdef SUPPORT_COMPLEX
template AcceleratorVector<std::complex<float> >* _paralution_init_base_hip_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
template AcceleratorVector<std::complex<double> >* _paralution_init_base_hip_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
#endif
template AcceleratorVector<int>* _paralution_init_base_hip_vector(const struct Paralution_Backend_Descriptor backend_descriptor);

template AcceleratorMatrix<float>* _paralution_init_base_hip_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                    const unsigned int matrix_format);
template AcceleratorMatrix<double>* _paralution_init_base_hip_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                     const unsigned int matrix_format);
#ifdef SUPPORT_COMPLEX
template AcceleratorMatrix<std::complex<float> >* _paralution_init_base_hip_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                                   const unsigned int matrix_format);
template AcceleratorMatrix<std::complex<double> >* _paralution_init_base_hip_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                                    const unsigned int matrix_format);
#endif

}
