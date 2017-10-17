#include "../../utils/def.hpp"
#include "../backend_manager.hpp"
#include "backend_gpu.hpp" 
#include "../../utils/log.hpp"
#include "gpu_utils.hpp"
#include "../base_vector.hpp"
#include "../base_matrix.hpp"

#include "gpu_vector.hpp"
#include "gpu_matrix_csr.hpp"
#include "gpu_matrix_coo.hpp"
#include "gpu_matrix_mcsr.hpp"
#include "gpu_matrix_bcsr.hpp"
#include "gpu_matrix_hyb.hpp"
#include "gpu_matrix_dia.hpp"
#include "gpu_matrix_ell.hpp"
#include "gpu_matrix_dense.hpp"

#include <complex>

#include <cuda.h>
#include <cublas_v2.h>

namespace paralution {

bool paralution_init_gpu(void) {

  LOG_DEBUG(0, "paralution_init_gpu()",
            "* begin");

  assert(_get_backend_descriptor()->GPU_cublas_handle == NULL);
  assert(_get_backend_descriptor()->GPU_cusparse_handle == NULL);
  //  assert(_get_backend_descriptor()->GPU_dev == -1);

  // create a handle
  _get_backend_descriptor()->GPU_cublas_handle = new cublasHandle_t;
  _get_backend_descriptor()->GPU_cusparse_handle = new cusparseHandle_t;

  // get last cuda error (if any)
  cudaGetLastError();

  cudaError_t cuda_status_t;
  int num_dev;
  cudaGetDeviceCount(&num_dev);
  cuda_status_t = cudaGetLastError();

  // if querying for device count fails, fall back to host backend
  if (cuda_status_t != cudaSuccess) {
    LOG_INFO("Querying for GPU devices failed - falling back to host backend");
    return false;
  }

  LOG_INFO("Number of GPU devices in the system: " << num_dev);

  if (num_dev < 1) {

    LOG_INFO("No GPU device found");

  } else {

    if (_get_backend_descriptor()->GPU_dev != -1)
      num_dev = 1;

    for (int idev=0; idev<num_dev; idev++) {

      int dev = idev;

      if (_get_backend_descriptor()->GPU_dev != -1) {
        dev = _get_backend_descriptor()->GPU_dev;
      }

      cudaSetDevice(dev);
      cuda_status_t = cudaGetLastError();

      if (cuda_status_t == cudaSuccess) {

        if ((cublasCreate(static_cast<cublasHandle_t*>(_get_backend_descriptor()->GPU_cublas_handle)) == CUBLAS_STATUS_SUCCESS) &&
            (cusparseCreate(static_cast<cusparseHandle_t*>(_get_backend_descriptor()->GPU_cusparse_handle)) == CUSPARSE_STATUS_SUCCESS)) {

          _get_backend_descriptor()->GPU_dev = dev;
          break;

        } else
          LOG_INFO("GPU device " << dev << " cannot create CUBLAS/CUSPARSE context");

      }

      if (cuda_status_t == cudaErrorDeviceAlreadyInUse)
        LOG_INFO("GPU device " << dev << " is already in use");

      if (cuda_status_t == cudaErrorInvalidDevice)
        LOG_INFO("GPU device " << dev << " is invalid NVIDIA GPU device");

    }

  }

  if (_get_backend_descriptor()->GPU_dev == -1) {
    LOG_INFO("CUDA and CUBLAS/CUSPARSE have NOT been initialized!");
    return false;
  }


  struct cudaDeviceProp dev_prop;      
  cudaGetDeviceProperties(&dev_prop, _get_backend_descriptor()->GPU_dev);

  if (dev_prop.major < 2) {
    LOG_INFO("GPU device " << _get_backend_descriptor()->GPU_dev << " has low compute capability (min 2.0 is needed)");    
    return false;
  }

  // Get some properties from the device
  _get_backend_descriptor()->GPU_warp = dev_prop.warpSize;
  _get_backend_descriptor()->GPU_num_procs = dev_prop.multiProcessorCount;
  _get_backend_descriptor()->GPU_threads_per_proc = dev_prop.maxThreadsPerMultiProcessor;
  _get_backend_descriptor()->GPU_max_threads = dev_prop.regsPerBlock;

  LOG_DEBUG(0, "paralution_init_gpu()",
            "* end");

  return true;

}


void paralution_stop_gpu(void) {

  LOG_DEBUG(0, "paralution_stop_gpu()",
            "* begin");


  if (_get_backend_descriptor()->accelerator) {

    if (cublasDestroy(*(static_cast<cublasHandle_t*>(_get_backend_descriptor()->GPU_cublas_handle))) != CUBLAS_STATUS_SUCCESS) {
      LOG_INFO("Error in cublasDestroy");
    }

    if (cusparseDestroy(*(static_cast<cusparseHandle_t*>(_get_backend_descriptor()->GPU_cusparse_handle))) != CUSPARSE_STATUS_SUCCESS) {
      LOG_INFO("Error in cusparseDestroy");
    }

  }

    delete (static_cast<cublasHandle_t*>(_get_backend_descriptor()->GPU_cublas_handle));
    delete (static_cast<cusparseHandle_t*>(_get_backend_descriptor()->GPU_cusparse_handle));

    _get_backend_descriptor()->GPU_cublas_handle = NULL; 
    _get_backend_descriptor()->GPU_cusparse_handle = NULL;

    _get_backend_descriptor()->GPU_dev = -1;

  LOG_DEBUG(0, "paralution_stop_gpu()",
            "* end");

}

void paralution_info_gpu(const struct Paralution_Backend_Descriptor backend_descriptor) {

    int num_dev;

    cudaGetDeviceCount(&num_dev);
    cudaGetLastError(); 
    //  CHECK_CUDA_ERROR(__FILE__, __LINE__);

    //    LOG_INFO("Number of GPU devices in the sytem: " << num_dev);    

    if (_get_backend_descriptor()->GPU_dev >= 0) {
      LOG_INFO("Selected GPU device: " << backend_descriptor.GPU_dev);

    } else {

      LOG_INFO("No GPU device is selected!");      

    }

    for (int dev = 0; dev < num_dev; dev++) {

      struct cudaDeviceProp dev_prop;      
      cudaGetDeviceProperties(&dev_prop, dev);
     
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
      */
      LOG_INFO("ECCEnabled: "                  << dev_prop.ECCEnabled);                  //    int ECCEnabled;
      /*
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
AcceleratorMatrix<ValueType>* _paralution_init_base_gpu_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                               const unsigned int matrix_format) {
  assert(backend_descriptor.backend == GPU);

  switch (matrix_format) {

  case CSR:
    return new GPUAcceleratorMatrixCSR<ValueType>(backend_descriptor);
    
  case COO:
    return new GPUAcceleratorMatrixCOO<ValueType>(backend_descriptor);

  case MCSR:
    return new GPUAcceleratorMatrixMCSR<ValueType>(backend_descriptor);

  case DIA:
    return new GPUAcceleratorMatrixDIA<ValueType>(backend_descriptor);
    
  case ELL:
    return new GPUAcceleratorMatrixELL<ValueType>(backend_descriptor);

  case DENSE:
    return new GPUAcceleratorMatrixDENSE<ValueType>(backend_descriptor);

  case HYB:
    return new GPUAcceleratorMatrixHYB<ValueType>(backend_descriptor);

  case BCSR:
    return new GPUAcceleratorMatrixBCSR<ValueType>(backend_descriptor);

  default:
    LOG_INFO("This backed is not supported for Matrix types");
    FATAL_ERROR(__FILE__, __LINE__);   
    return NULL;
  } 

}

template <typename ValueType>
AcceleratorVector<ValueType>* _paralution_init_base_gpu_vector(const struct Paralution_Backend_Descriptor backend_descriptor) {

  assert(backend_descriptor.backend == GPU);

  return new GPUAcceleratorVector<ValueType>(backend_descriptor);

}

void paralution_gpu_sync(void) {

  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR(__FILE__, __LINE__);

}

template AcceleratorVector<float>* _paralution_init_base_gpu_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
template AcceleratorVector<double>* _paralution_init_base_gpu_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
#ifdef SUPPORT_COMPLEX
template AcceleratorVector<std::complex<float> >* _paralution_init_base_gpu_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
template AcceleratorVector<std::complex<double> >* _paralution_init_base_gpu_vector(const struct Paralution_Backend_Descriptor backend_descriptor);
#endif
template AcceleratorVector<int>* _paralution_init_base_gpu_vector(const struct Paralution_Backend_Descriptor backend_descriptor);

template AcceleratorMatrix<float>* _paralution_init_base_gpu_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                    const unsigned int matrix_format);
template AcceleratorMatrix<double>* _paralution_init_base_gpu_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                     const unsigned int matrix_format);
#ifdef SUPPORT_COMPLEX
template AcceleratorMatrix<std::complex<float> >* _paralution_init_base_gpu_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                                   const unsigned int matrix_format);
template AcceleratorMatrix<std::complex<double> >* _paralution_init_base_gpu_matrix(const struct Paralution_Backend_Descriptor backend_descriptor,
                                                                                    const unsigned int matrix_format);
#endif

}
