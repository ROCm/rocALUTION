#include "../../utils/def.hpp"
#include "hip_vector.hpp"
#include "../base_vector.hpp"
#include "../host/host_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/math_functions.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_vector.hpp"
#include "hip_allocate_free.hpp"

#include <cuda.h>
#include <cuComplex.h>
#include <cublas_v2.h>

namespace paralution {

template <typename ValueType>
HIPAcceleratorVector<ValueType>::HIPAcceleratorVector() {

  // no default constructors
    LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HIPAcceleratorVector<ValueType>::HIPAcceleratorVector(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "HIPAcceleratorVector::HIPAcceleratorVector()",
            "constructor with local_backend");

  this->vec_ = NULL;
  this->set_backend(local_backend);

  this->index_array_  = NULL;
  this->index_buffer_ = NULL;

  this->host_buffer_ = NULL;
  this->device_buffer_ = NULL;

  CHECK_HIP_ERROR(__FILE__, __LINE__);

}


template <typename ValueType>
HIPAcceleratorVector<ValueType>::~HIPAcceleratorVector() {

  LOG_DEBUG(this, "HIPAcceleratorVector::~HIPAcceleratorVector()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::info(void) const {

  LOG_INFO("HIPAcceleratorVector<ValueType>");

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::Allocate(const int n) {

  assert(n >= 0);

  if (this->get_size() >0)
    this->Clear();

  if (n > 0) {

    allocate_hip(n, &this->vec_);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    this->local_backend_.HIP_max_threads,
                    n, this->vec_);

    allocate_host(this->local_backend_.HIP_warp, &this->host_buffer_);
    allocate_hip(this->local_backend_.HIP_warp, &this->device_buffer_);

    this->size_ = n;
  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::SetDataPtr(ValueType **ptr, const int size) {

  assert(*ptr != NULL);
  assert(size > 0);

  cudaDeviceSynchronize();

  this->vec_ = *ptr;
  this->size_ = size;

  allocate_host(this->local_backend_.HIP_warp, &this->host_buffer_);
  allocate_hip(this->local_backend_.HIP_warp, &this->device_buffer_);

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::LeaveDataPtr(ValueType **ptr) {

  assert(this->get_size() > 0);

  cudaDeviceSynchronize();
  *ptr = this->vec_;
  this->vec_ = NULL;

  free_host(&this->host_buffer_);
  free_hip(&this->device_buffer_);

  this->size_ = 0;

}


template <typename ValueType>
void HIPAcceleratorVector<ValueType>::Clear(void) {
  
  if (this->get_size() > 0) {

    free_hip(&this->vec_);
    this->size_ = 0;

  }

  if (this->index_size_ > 0) {

    free_hip(&this->index_buffer_);
    free_hip(&this->index_array_);
    this->index_size_ = 0;

    free_host(&this->host_buffer_);
    free_hip(&this->device_buffer_);

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromHost(const HostVector<ValueType> &src) {

  // CPU to HIP copy
  const HostVector<ValueType> *cast_vec;
  if ((cast_vec = dynamic_cast<const HostVector<ValueType>*> (&src)) != NULL) {

    if (this->get_size() == 0) {

      // Allocate local structure
      this->Allocate(cast_vec->get_size());

      // Check for boundary
      assert(this->index_size_ == 0);
      if (cast_vec->index_size_ > 0) {

        this->index_size_ = cast_vec->index_size_;
        allocate_hip<int>(this->index_size_, &this->index_array_);
        allocate_hip<ValueType>(this->index_size_, &this->index_buffer_);

      }

    }

    assert(cast_vec->get_size() == this->get_size());
    assert(cast_vec->index_size_ == this->index_size_);

    if (this->get_size() > 0) {      

      cublasStatus_t stat_t;
      stat_t = cublasSetVector(this->get_size(), sizeof(ValueType),
                               cast_vec->vec_, // src
                               1,
                               this->vec_, // dst
                               1);
      CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

      stat_t = cublasSetVector(this->index_size_, sizeof(int),
                               cast_vec->index_array_,
                               1,
                               this->index_array_,
                               1);
      CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    }

  } else {

    LOG_INFO("Error unsupported HIP vector type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyToHost(HostVector<ValueType> *dst) const {

  // HIP to CPU copy
  HostVector<ValueType> *cast_vec;
  if ((cast_vec = dynamic_cast<HostVector<ValueType>*> (dst)) != NULL) {

    if (cast_vec->get_size() == 0) {

      // Allocate local vector
      cast_vec->Allocate(this->get_size());

      // Check for boundary
      assert(cast_vec->index_size_ == 0);
      if (this->index_size_ > 0) {

        cast_vec->index_size_ = this->index_size_;
        allocate_host(this->index_size_, &cast_vec->index_array_);

      }

    }
      
    assert(cast_vec->get_size() == this->get_size());
    assert(cast_vec->index_size_ == this->index_size_);

    if (this->get_size() > 0) {

      cublasStatus_t stat_t;
      stat_t = cublasGetVector(this->get_size(), sizeof(ValueType),
                               this->vec_, // src
                               1,
                               cast_vec->vec_, // dst
                               1);
      CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

      stat_t = cublasGetVector(this->index_size_, sizeof(int),
                               this->index_array_,
                               1,
                               cast_vec->index_array_,
                               1);
      CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    }

  } else {
    
    LOG_INFO("Error unsupported HIP vector type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

  
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromHostAsync(const HostVector<ValueType> &src) {

  // CPU to HIP copy
  const HostVector<ValueType> *cast_vec;
  if ((cast_vec = dynamic_cast<const HostVector<ValueType>*> (&src)) != NULL) {

    if (this->get_size() == 0) {

      // Allocate local vector
      this->Allocate(cast_vec->get_size());

      // Check for boundary
      assert(this->index_size_ == 0);
      if (cast_vec->index_size_ > 0) {

        this->index_size_ = cast_vec->index_size_;
        allocate_hip<int>(this->index_size_, &this->index_array_);
        allocate_hip<ValueType>(this->index_size_, &this->index_buffer_);

      }

    }

    assert(cast_vec->get_size() == this->get_size());
    assert(cast_vec->index_size_ == this->index_size_);

    if (this->get_size() > 0) {

      cudaMemcpyAsync(this->vec_,     // dst
                      cast_vec->vec_, // src
                      this->get_size()*sizeof(ValueType), // size
                      cudaMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);

      cudaMemcpyAsync(this->index_array_,     // dst
                      cast_vec->index_array_, // src
                      this->index_size_*sizeof(int), // size
                      cudaMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);

    }

  } else {

    LOG_INFO("Error unsupported HIP vector type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyToHostAsync(HostVector<ValueType> *dst) const {

  // HIP to CPU copy
  HostVector<ValueType> *cast_vec;
  if ((cast_vec = dynamic_cast<HostVector<ValueType>*> (dst)) != NULL) {

    if (cast_vec->get_size() == 0) {

      // Allocate local vector
      cast_vec->Allocate(this->get_size());

      // Check for boundary
      assert(cast_vec->index_size_ == 0);
      if (this->index_size_ > 0) {

        cast_vec->index_size_ = this->index_size_;
        allocate_host(this->index_size_, &cast_vec->index_array_);

      }

    }

    assert(cast_vec->get_size() == this->get_size());
    assert(cast_vec->index_size_ == this->index_size_);

    if (this->get_size() > 0) {

      cudaMemcpyAsync(cast_vec->vec_,  // dst
                      this->vec_,      // src
                      this->get_size()*sizeof(ValueType), // size
                      cudaMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);

      cudaMemcpyAsync(cast_vec->index_array_,  // dst
                      this->index_array_,      // src
                      this->index_size_*sizeof(int), // size
                      cudaMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);

    }

  } else {

    LOG_INFO("Error unsupported HIP vector type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFrom(const BaseVector<ValueType> &src) {

  const HIPAcceleratorVector<ValueType> *hip_cast_vec;
  const HostVector<ValueType> *host_cast_vec;

  // HIP to HIP copy
  if ((hip_cast_vec = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&src)) != NULL) {

    if (this->get_size() == 0) {

      // Allocate local vector
      this->Allocate(hip_cast_vec->get_size());

      // Check for boundary
      assert(this->index_size_ == 0);
      if (hip_cast_vec->index_size_ > 0) {

        this->index_size_ = hip_cast_vec->index_size_;
        allocate_hip<int>(this->index_size_, &this->index_array_);
        allocate_hip<ValueType>(this->index_size_, &this->index_buffer_);

      }

    }

    assert(hip_cast_vec->get_size() == this->get_size());
    assert(hip_cast_vec->index_size_ == this->index_size_);

    if (this != hip_cast_vec)  {  

      if (this->get_size() > 0) {

        cudaMemcpy(this->vec_,         // dst
                   hip_cast_vec->vec_, // src
                   this->get_size()*sizeof(ValueType), // size
                   cudaMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        cudaMemcpy(this->index_array_,            // dst
                   hip_cast_vec->index_array_,    // src
                   this->index_size_*sizeof(int), // size
                   cudaMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

      }

    }

  } else {
    
    //HIP to CPU copy
    if ((host_cast_vec = dynamic_cast<const HostVector<ValueType>*> (&src)) != NULL) {
      

      this->CopyFromHost(*host_cast_vec);
      
    
    } else {

      LOG_INFO("Error unsupported HIP vector type");
      this->info();
      src.info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }
    
  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromAsync(const BaseVector<ValueType> &src) {

  const HIPAcceleratorVector<ValueType> *hip_cast_vec;
  const HostVector<ValueType> *host_cast_vec;

  // HIP to HIP copy
  if ((hip_cast_vec = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&src)) != NULL) {

    if (this->get_size() == 0) {

      // Allocate local vector
      this->Allocate(hip_cast_vec->get_size());

      // Check for boundary
      assert(this->index_size_ == 0);
      if (hip_cast_vec->index_size_ > 0) {

        this->index_size_ = hip_cast_vec->index_size_;
        allocate_hip<int>(this->index_size_, &this->index_array_);
        allocate_hip<ValueType>(this->index_size_, &this->index_buffer_);

      }

    }

    assert(hip_cast_vec->get_size() == this->get_size());
    assert(hip_cast_vec->index_size_ == this->index_size_);

    if (this != hip_cast_vec) {

      if (this->get_size() > 0) {

        cudaMemcpy(this->vec_,         // dst
                   hip_cast_vec->vec_, // src
                   this->get_size()*sizeof(ValueType), // size
                   cudaMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        cudaMemcpy(this->index_array_,         // dst
                   hip_cast_vec->index_array_, // src
                   this->index_size_*sizeof(int), // size
                   cudaMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

      }

    }

  } else {

    //HIP to CPU copy
    if ((host_cast_vec = dynamic_cast<const HostVector<ValueType>*> (&src)) != NULL) {

      this->CopyFromHostAsync(*host_cast_vec);

    } else {

      LOG_INFO("Error unsupported HIP vector type");
      this->info();
      src.info();
      FATAL_ERROR(__FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFrom(const BaseVector<ValueType> &src,
                                               const int src_offset,
                                               const int dst_offset,
                                               const int size) {

  assert(&src != this);
  assert(this->get_size() > 0);
  assert(src.  get_size() > 0);
  assert(size > 0);

  assert(src_offset + size <= src.get_size());
  assert(dst_offset + size <= this->get_size());

  const HIPAcceleratorVector<ValueType> *cast_src = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&src);
  assert(cast_src != NULL);

  dim3 BlockSize(this->local_backend_.HIP_block_size);
  dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

  kernel_copy_offset_from<ValueType, int> <<<GridSize, BlockSize>>> (size, src_offset, dst_offset,
                                                                     cast_src->vec_, this->vec_);

  CHECK_HIP_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyTo(BaseVector<ValueType> *dst) const{

  HIPAcceleratorVector<ValueType> *hip_cast_vec;
  HostVector<ValueType> *host_cast_vec;

    // HIP to HIP copy
    if ((hip_cast_vec = dynamic_cast<HIPAcceleratorVector<ValueType>*> (dst)) != NULL) {

      if (hip_cast_vec->get_size() == 0) {

        // Allocate local vector
        hip_cast_vec->Allocate(this->get_size());

        // Check for boundary
        assert(hip_cast_vec->index_size_ == 0);
        if (this->index_size_ > 0) {

          hip_cast_vec->index_size_ = this->index_size_;
          allocate_hip<int>(this->index_size_, &hip_cast_vec->index_array_);
          allocate_hip<ValueType>(this->index_size_, &hip_cast_vec->index_buffer_);

        }

      }

      assert(hip_cast_vec->get_size() == this->get_size());
      assert(hip_cast_vec->index_size_ == this->index_size_);

      if (this != hip_cast_vec)  {

        if (this->get_size() >0) {

          cudaMemcpy(hip_cast_vec->vec_, // dst
                     this->vec_,         // src
                     this->get_size()*sizeof(ValueType), // size
                     cudaMemcpyDeviceToDevice);
          CHECK_HIP_ERROR(__FILE__, __LINE__);

          cudaMemcpy(hip_cast_vec->index_array_,    // dst
                     this->index_array_,            // src
                     this->index_size_*sizeof(int), // size
                     cudaMemcpyDeviceToDevice);
          CHECK_HIP_ERROR(__FILE__, __LINE__);

        }
      }

    } else {
      
      //HIP to CPU copy
      if ((host_cast_vec = dynamic_cast<HostVector<ValueType>*> (dst)) != NULL) {
        

        this->CopyToHost(host_cast_vec);
        
      
      } else {

        LOG_INFO("Error unsupported HIP vector type");
        this->info();
        dst->info();
        FATAL_ERROR(__FILE__, __LINE__);
        
      }
      
    }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyToAsync(BaseVector<ValueType> *dst) const {

  HIPAcceleratorVector<ValueType> *hip_cast_vec;
  HostVector<ValueType> *host_cast_vec;

  // HIP to HIP copy
  if ((hip_cast_vec = dynamic_cast<HIPAcceleratorVector<ValueType>*> (dst)) != NULL) {

    if (hip_cast_vec->get_size() == 0) {

      // Allocate local vector
      hip_cast_vec->Allocate(this->get_size());

      // Check for boundary
      assert(hip_cast_vec->index_size_ == 0);
      if (this->index_size_ > 0) {

        hip_cast_vec->index_size_ = this->index_size_;
        allocate_hip<int>(this->index_size_, &hip_cast_vec->index_array_);
        allocate_hip<ValueType>(this->index_size_, &hip_cast_vec->index_buffer_);

      }

    }

    assert(hip_cast_vec->get_size() == this->get_size());
    assert(hip_cast_vec->index_size_ == this->index_size_);

    if (this != hip_cast_vec) {

      if (this->get_size() > 0) {

        cudaMemcpy(hip_cast_vec->vec_, // dst
                   this->vec_,         // src
                   this->get_size()*sizeof(ValueType), // size
                   cudaMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        cudaMemcpy(hip_cast_vec->index_array_, // dst
                   this->index_array_,         // src
                   this->index_size_*sizeof(int), // size
                   cudaMemcpyDeviceToDevice);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

      }

    }

  } else {

    //HIP to CPU copy
    if ((host_cast_vec = dynamic_cast<HostVector<ValueType>*> (dst)) != NULL) {

      this->CopyToHostAsync(host_cast_vec);

    } else {

      LOG_INFO("Error unsupported HIP vector type");
      this->info();
      dst->info();
      FATAL_ERROR(__FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromFloat(const BaseVector<float> &src) {

  LOG_INFO("Mixed precision for non-complex to complex casting is not allowed");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <>
void HIPAcceleratorVector<double>::CopyFromFloat(const BaseVector<float> &src) {

  const HIPAcceleratorVector<float> *hip_cast_vec;

  // HIP to HIP copy
  if ((hip_cast_vec = dynamic_cast<const HIPAcceleratorVector<float>*> (&src)) != NULL) {

    if (this->get_size() == 0)
      this->Allocate(hip_cast_vec->get_size());

    assert(hip_cast_vec->get_size() == this->get_size());

    if (this->get_size() > 0) {

      dim3 BlockSize(this->local_backend_.HIP_block_size);
      dim3 GridSize(this->get_size() / this->local_backend_.HIP_block_size + 1);

      kernel_copy_from_float<double, int> <<<GridSize, BlockSize>>>(this->get_size(), hip_cast_vec->vec_, this->vec_);

      CHECK_HIP_ERROR(__FILE__, __LINE__);

    }

  } else {

    LOG_INFO("Error unsupported HIP vector type");
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromDouble(const BaseVector<double> &src) {

  LOG_INFO("Mixed precision for non-complex to complex casting is not allowed");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <>
void HIPAcceleratorVector<float>::CopyFromDouble(const BaseVector<double> &src) {

  const HIPAcceleratorVector<double> *hip_cast_vec;

  // HIP to HIP copy
  if ((hip_cast_vec = dynamic_cast<const HIPAcceleratorVector<double>*> (&src)) != NULL) {

    if (this->get_size() == 0)
      this->Allocate(hip_cast_vec->get_size());

    assert(hip_cast_vec->get_size() == this->get_size());

    if (this->get_size()  >0) {

      dim3 BlockSize(this->local_backend_.HIP_block_size);
      dim3 GridSize(this->get_size() / this->local_backend_.HIP_block_size + 1);

      kernel_copy_from_double<float, int> <<<GridSize, BlockSize>>>(this->get_size(), hip_cast_vec->vec_, this->vec_);

      CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

  } else {
    LOG_INFO("Error unsupported HIP vector type");
    FATAL_ERROR(__FILE__, __LINE__);

  }
  
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromData(const ValueType *data) {

  if (this->get_size() > 0) {

    cudaMemcpy(this->vec_,                         // dst
               data,                               // src
               this->get_size()*sizeof(ValueType), // size
               cudaMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyToData(ValueType *data) const {

  if (this->get_size() > 0) {

    cudaMemcpy(data,                               // dst
               this->vec_,                         // src
               this->get_size()*sizeof(ValueType), // size
               cudaMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::Zeros(void) {

  if (this->get_size() > 0) {

    set_to_zero_hip(this->local_backend_.HIP_block_size,
                    this->local_backend_.HIP_max_threads,
                    this->get_size(), this->vec_);

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::Ones(void) {

  if (this->get_size() > 0)
    set_to_one_hip(this->local_backend_.HIP_block_size, 
                   this->local_backend_.HIP_max_threads,
                   this->get_size(), this->vec_);

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::SetValues(const ValueType val) {

  LOG_INFO("HIPAcceleratorVector::SetValues NYI");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <>
void HIPAcceleratorVector<double>::AddScale(const BaseVector<double> &x, const double alpha) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());
    
    const HIPAcceleratorVector<double> *cast_x = dynamic_cast<const HIPAcceleratorVector<double>*> (&x);
    assert(cast_x != NULL);
    
    cublasStatus_t stat_t;
    
    stat_t = cublasDaxpy(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), 
                         this->get_size(), 
                         &alpha, 
                         cast_x->vec_, 1,
                         this->vec_, 1);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorVector<float>::AddScale(const BaseVector<float> &x, const float alpha) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());
    
    const HIPAcceleratorVector<float> *cast_x = dynamic_cast<const HIPAcceleratorVector<float>*> (&x);
    assert(cast_x != NULL);
    
    cublasStatus_t stat_t;
    
    stat_t = cublasSaxpy(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), 
                         this->get_size(), 
                         &alpha, 
                         cast_x->vec_, 1,
                         this->vec_, 1);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorVector<std::complex<double> >::AddScale(const BaseVector<std::complex<double> > &x,
                                                           const std::complex<double> alpha) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());

    const HIPAcceleratorVector<std::complex<double> > *cast_x = dynamic_cast<const HIPAcceleratorVector<std::complex<double> >*> (&x);
    assert(cast_x != NULL);

    cublasStatus_t stat_t;

    stat_t = cublasZaxpy(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), 
                         this->get_size(),
                         (cuDoubleComplex*)&alpha,
                         (cuDoubleComplex*)cast_x->vec_, 1,
                         (cuDoubleComplex*)this->vec_, 1);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorVector<std::complex<float> >::AddScale(const BaseVector<std::complex<float> > &x,
                                                          const std::complex<float> alpha) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());

    const HIPAcceleratorVector<std::complex<float> > *cast_x = dynamic_cast<const HIPAcceleratorVector<std::complex<float> >*> (&x);
    assert(cast_x != NULL);

    cublasStatus_t stat_t;

    stat_t = cublasCaxpy(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), 
                         this->get_size(),
                         (cuFloatComplex*)&alpha,
                         (cuFloatComplex*)cast_x->vec_, 1,
                         (cuFloatComplex*)this->vec_, 1);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorVector<int>::AddScale(const BaseVector<int> &x, const int alpha) {

  LOG_INFO("No int CUBLAS axpy function");
  FATAL_ERROR(__FILE__, __LINE__);
 
}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::ScaleAdd(const ValueType alpha, const BaseVector<ValueType> &x) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());

    const HIPAcceleratorVector<ValueType> *cast_x = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&x);
    assert(cast_x != NULL);

    int size = this->get_size();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

    kernel_scaleadd<<<GridSize, BlockSize>>> (size, HIPVal(alpha), HIPPtr(cast_x->vec_), HIPPtr(this->vec_));

    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::ScaleAddScale(const ValueType alpha, const BaseVector<ValueType> &x, const ValueType beta) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());

    const HIPAcceleratorVector<ValueType> *cast_x = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&x);
    assert(cast_x != NULL);

    int size = this->get_size();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

    kernel_scaleaddscale<<<GridSize, BlockSize>>> (size, HIPVal(alpha), HIPVal(beta), HIPPtr(cast_x->vec_), HIPPtr(this->vec_));

    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::ScaleAddScale(const ValueType alpha, const BaseVector<ValueType> &x, const ValueType beta,
                                                    const int src_offset, const int dst_offset,const int size) {

  if (this->get_size() > 0) {

    assert(this->get_size() > 0);
    assert(x.get_size() > 0);
    assert(size > 0);
    assert(src_offset + size <= x.get_size());
    assert(dst_offset + size <= this->get_size());

    const HIPAcceleratorVector<ValueType> *cast_x = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&x);
    assert(cast_x != NULL);

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

    kernel_scaleaddscale_offset<<<GridSize, BlockSize>>> (size, src_offset, dst_offset,
                                                          HIPVal(alpha), HIPVal(beta),
                                                          HIPPtr(cast_x->vec_), HIPPtr(this->vec_));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::ScaleAdd2(const ValueType alpha, const BaseVector<ValueType> &x,
                                                const ValueType beta, const BaseVector<ValueType> &y,
                                                const ValueType gamma) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());
    assert(this->get_size() == y.get_size());

    const HIPAcceleratorVector<ValueType> *cast_x = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&x);
    const HIPAcceleratorVector<ValueType> *cast_y = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&y);
    assert(cast_x != NULL);
    assert(cast_y != NULL);

    int size = this->get_size();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

    kernel_scaleadd2<<<GridSize, BlockSize>>> (size, HIPVal(alpha), HIPVal(beta), HIPVal(gamma),
                                               HIPPtr(cast_x->vec_), HIPPtr(cast_y->vec_), HIPPtr(this->vec_));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorVector<double>::Scale(const double alpha) {

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasDscal(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(), &alpha,
                         this->vec_, 1);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorVector<float>::Scale(const float alpha) {

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasSscal(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(), &alpha,
                         this->vec_, 1);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorVector<std::complex<double> >::Scale(const std::complex<double> alpha) {

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasZscal(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(), (cuDoubleComplex*)&alpha,
                         (cuDoubleComplex*)this->vec_, 1);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorVector<std::complex<float> >::Scale(const std::complex<float> alpha) {

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasCscal(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(), (cuFloatComplex*)&alpha,
                         (cuFloatComplex*)this->vec_, 1);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorVector<int>::Scale(const int alpha) {

  LOG_INFO("No int CUBLAS scale function");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::ExclusiveScan(const BaseVector<ValueType> &x) {

  LOG_INFO("HIPAcceleratorVector::ExclusiveScan() NYI");
  FATAL_ERROR(__FILE__, __LINE__); 

}

template <>
double HIPAcceleratorVector<double>::Dot(const BaseVector<double> &x) const {

  assert(this->get_size() == x.get_size());

  const HIPAcceleratorVector<double> *cast_x = dynamic_cast<const HIPAcceleratorVector<double>*> (&x);
  assert(cast_x != NULL);

  double res = 0;

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasDdot(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                        this->get_size(),
                        this->vec_, 1,
                        cast_x->vec_, 1, &res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return res;

}

template <>
float HIPAcceleratorVector<float>::Dot(const BaseVector<float> &x) const {

  assert(this->get_size() == x.get_size());

  const HIPAcceleratorVector<float> *cast_x = dynamic_cast<const HIPAcceleratorVector<float>*> (&x);
  assert(cast_x != NULL);

  float res = 0;

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasSdot(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                        this->get_size(),
                        this->vec_, 1,
                        cast_x->vec_, 1, &res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return res;

}

template <>
std::complex<double> HIPAcceleratorVector<std::complex<double> >::Dot(const BaseVector<std::complex<double> > &x) const {

  assert(this->get_size() == x.get_size());

  const HIPAcceleratorVector<std::complex<double> > *cast_x = dynamic_cast<const HIPAcceleratorVector<std::complex<double> >*> (&x);
  assert(cast_x != NULL);

  std::complex<double> res = std::complex<double>(double(0.0), double(0.0));

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasZdotc(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(),
                         (cuDoubleComplex*)this->vec_, 1,
                         (cuDoubleComplex*)cast_x->vec_, 1, (cuDoubleComplex*)&res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return res;

}

template <>
std::complex<float> HIPAcceleratorVector<std::complex<float> >::Dot(const BaseVector<std::complex<float> > &x) const {

  assert(this->get_size() == x.get_size());

  const HIPAcceleratorVector<std::complex<float> > *cast_x = dynamic_cast<const HIPAcceleratorVector<std::complex<float> >*> (&x);
  assert(cast_x != NULL);

  std::complex<float> res = std::complex<float>(float(0.0), float(0.0));

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasCdotc(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(),
                         (cuFloatComplex*)this->vec_, 1,
                         (cuFloatComplex*)cast_x->vec_, 1, (cuFloatComplex*)&res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return res;

}

template <>
int HIPAcceleratorVector<int>::Dot(const BaseVector<int> &x) const {

  LOG_INFO("No int CUBLAS dot function");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
ValueType HIPAcceleratorVector<ValueType>::DotNonConj(const BaseVector<ValueType> &x) const {

  return this->Dot(x);

}

template <>
std::complex<double> HIPAcceleratorVector<std::complex<double> >::DotNonConj(const BaseVector<std::complex<double> > &x) const {

  assert(this->get_size() == x.get_size());

  const HIPAcceleratorVector<std::complex<double> > *cast_x = dynamic_cast<const HIPAcceleratorVector<std::complex<double> >*> (&x);
  assert(cast_x != NULL);

  std::complex<double> res = std::complex<double>(double(0.0), double(0.0));

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasZdotu(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(),
                         (cuDoubleComplex*)this->vec_, 1,
                         (cuDoubleComplex*)cast_x->vec_, 1, (cuDoubleComplex*)&res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return res;

}

template <>
std::complex<float> HIPAcceleratorVector<std::complex<float> >::DotNonConj(const BaseVector<std::complex<float> > &x) const {

  assert(this->get_size() == x.get_size());

  const HIPAcceleratorVector<std::complex<float> > *cast_x = dynamic_cast<const HIPAcceleratorVector<std::complex<float> >*> (&x);
  assert(cast_x != NULL);

  std::complex<float> res = std::complex<float>(float(0.0), float(0.0));

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasCdotu(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(),
                         (cuFloatComplex*)this->vec_, 1,
                         (cuFloatComplex*)cast_x->vec_, 1, (cuFloatComplex*)&res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return res;

}

template <>
int HIPAcceleratorVector<int>::DotNonConj(const BaseVector<int> &x) const {

  LOG_INFO("No int CUBLAS dot function");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <>
double HIPAcceleratorVector<double>::Norm(void) const {

  double res = 0;

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasDnrm2(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(),
                         this->vec_, 1, &res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return res;

}

template <>
float HIPAcceleratorVector<float>::Norm(void) const {

  float res = 0;

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasSnrm2(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(),
                         this->vec_, 1, &res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return res;

}

template <>
std::complex<double> HIPAcceleratorVector<std::complex<double> >::Norm(void) const {

  double res = double(0.0);

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasDznrm2(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                          this->get_size(),
                          (cuDoubleComplex*)this->vec_, 1, &res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  std::complex<double> c_res = (std::complex<double>) res;
  return c_res;

}

template <>
std::complex<float> HIPAcceleratorVector<std::complex<float> >::Norm(void) const {

  float res = float(0.0);

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasScnrm2(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                          this->get_size(),
                          (cuFloatComplex*)this->vec_, 1, &res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  std::complex<float> c_res = (std::complex<float>) res;
  return c_res;

}

template <>
int HIPAcceleratorVector<int>::Norm(void) const {

  LOG_INFO("What is int HIPAcceleratorVector<ValueType>::Norm(void) const?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
ValueType HIPAcceleratorVector<ValueType>::Reduce(void) const {

  ValueType res = (ValueType) 0;

  if (this->get_size() > 0) {

    reduce_hip<int, ValueType, 32, 256>(this->get_size(), this->vec_, &res, this->host_buffer_, this->device_buffer_);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  return res;

}

template <>
int HIPAcceleratorVector<int>::Reduce(void) const {

  LOG_INFO("Reduce<int> not implemented");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <>
double HIPAcceleratorVector<double>::Asum(void) const {

  double res = 0.0;

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasDasum(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(),
                         this->vec_, 1,
                         &res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return res;

}

template <>
float HIPAcceleratorVector<float>::Asum(void) const {

  float res = 0.0;

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasSasum(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(),
                         this->vec_, 1,
                         &res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return res;

}

template <>
std::complex<double> HIPAcceleratorVector<std::complex<double> >::Asum(void) const {

  double res = double(0.0);

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasDzasum(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(),
                         (cuDoubleComplex*)this->vec_, 1,
                         &res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return res;

}

template <>
std::complex<float> HIPAcceleratorVector<std::complex<float> >::Asum(void) const {

  float res = float(0.0);

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasScasum(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                         this->get_size(),
                         (cuFloatComplex*)this->vec_, 1,
                         &res);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return res;

}

template <>
int HIPAcceleratorVector<int>::Asum(void) const {

  LOG_INFO("Asum<int> not implemented");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <>
int HIPAcceleratorVector<double>::Amax(double &value) const {

  int index = 0;
  value = 0.0;

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasIdamax(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                          this->get_size(),
                          this->vec_, 1, &index);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    // cublas returns 1-based indexing
    --index;

    cudaMemcpy(&value,
               this->vec_+index,
               sizeof(double),
               cudaMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  value = paralution_abs(value);
  return index;

}

template <>
int HIPAcceleratorVector<float>::Amax(float &value) const {

  int index = 0;
  value = 0.0;

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasIsamax(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                          this->get_size(),
                          this->vec_, 1, &index);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    // cublas returns 1-based indexing
    --index;

    cudaMemcpy(&value,
               this->vec_+index,
               sizeof(float),
               cudaMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  value = paralution_abs(value);
  return index;

}

template <>
int HIPAcceleratorVector<std::complex<double> >::Amax(std::complex<double> &value) const {

  int index = 0;
  value = std::complex<double>(double(0.0), double(0.0));

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasIzamax(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                          this->get_size(),
                          (cuDoubleComplex*)this->vec_, 1, &index);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    // cublas returns 1-based indexing
    --index;

    cudaMemcpy(&value,
               (cuDoubleComplex*)this->vec_+index,
               sizeof(std::complex<double>),
               cudaMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  value = paralution_abs(value);
  return index;

}

template <>
int HIPAcceleratorVector<std::complex<float> >::Amax(std::complex<float> &value) const {

  int index = 0;
  value = std::complex<float>(float(0.0), float(0.0));

  if (this->get_size() > 0) {

    cublasStatus_t stat_t;

    stat_t = cublasIcamax(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle),
                          this->get_size(),
                          (cuFloatComplex*)this->vec_, 1, &index);
    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    // cublas returns 1-based indexing
    --index;

    cudaMemcpy(&value,
               (cuFloatComplex*)this->vec_+index,
               sizeof(std::complex<float>),
               cudaMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  value = paralution_abs(value);
  return index;

}

template <>
int HIPAcceleratorVector<int>::Amax(int &value) const {

  LOG_INFO("Amax<int> not implemented");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::PointWiseMult(const BaseVector<ValueType> &x) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());

    const HIPAcceleratorVector<ValueType> *cast_x = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&x);
    assert(cast_x != NULL);

    int size = this->get_size();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

    kernel_pointwisemult<<<GridSize, BlockSize>>> (size, HIPPtr(cast_x->vec_), HIPPtr(this->vec_));

    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::PointWiseMult(const BaseVector<ValueType> &x, const BaseVector<ValueType> &y) {

  if (this->get_size() > 0) {

    assert(this->get_size() == x.get_size());
    assert(this->get_size() == y.get_size());

    const HIPAcceleratorVector<ValueType> *cast_x = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&x);
    const HIPAcceleratorVector<ValueType> *cast_y = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&y);
    assert(cast_x != NULL);
    assert(cast_y != NULL);

    int size = this->get_size();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

    kernel_pointwisemult2<<<GridSize, BlockSize>>> (size, HIPPtr(cast_x->vec_), HIPPtr(cast_y->vec_), HIPPtr(this->vec_));

    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::Permute(const BaseVector<int> &permutation) {

  if (this->get_size() > 0) {

    assert(this->get_size() == permutation.get_size());
    
    const HIPAcceleratorVector<int> *cast_perm = dynamic_cast<const HIPAcceleratorVector<int>*> (&permutation);
    assert(cast_perm != NULL);
    
    HIPAcceleratorVector<ValueType> vec_tmp(this->local_backend_);     
    vec_tmp.Allocate(this->get_size());
    vec_tmp.CopyFrom(*this);
    
    int size = this->get_size();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);
    
    //    this->vec_[ cast_perm->vec_[i] ] = vec_tmp.vec_[i];  
    kernel_permute<ValueType, int> <<<GridSize, BlockSize>>> (size, cast_perm->vec_, vec_tmp.vec_, this->vec_);
    
    CHECK_HIP_ERROR(__FILE__, __LINE__);      
  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::PermuteBackward(const BaseVector<int> &permutation) {

  if (this->get_size() > 0) {

    assert(this->get_size() == permutation.get_size());
    
    const HIPAcceleratorVector<int> *cast_perm = dynamic_cast<const HIPAcceleratorVector<int>*> (&permutation);
    assert(cast_perm != NULL);
    
    HIPAcceleratorVector<ValueType> vec_tmp(this->local_backend_);   
    vec_tmp.Allocate(this->get_size());
    vec_tmp.CopyFrom(*this);
    
    int size = this->get_size();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);
    
    //    this->vec_[i] = vec_tmp.vec_[ cast_perm->vec_[i] ];
    kernel_permute_backward<ValueType, int> <<<GridSize, BlockSize>>> (size, cast_perm->vec_, vec_tmp.vec_, this->vec_);
    
    CHECK_HIP_ERROR(__FILE__, __LINE__);      
  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromPermute(const BaseVector<ValueType> &src,
                                                      const BaseVector<int> &permutation) { 

  if (this->get_size() > 0) {

    assert(this != &src);
    
    const HIPAcceleratorVector<ValueType> *cast_vec = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&src);
    const HIPAcceleratorVector<int> *cast_perm      = dynamic_cast<const HIPAcceleratorVector<int>*> (&permutation) ; 
    assert(cast_perm != NULL);
    assert(cast_vec  != NULL);
    
    assert(cast_vec ->get_size() == this->get_size());
    assert(cast_perm->get_size() == this->get_size());
    
    int size = this->get_size();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);
    
    //    this->vec_[ cast_perm->vec_[i] ] = cast_vec->vec_[i];
    kernel_permute<ValueType, int> <<<GridSize, BlockSize>>> (size, cast_perm->vec_, cast_vec->vec_, this->vec_);
    
    CHECK_HIP_ERROR(__FILE__, __LINE__);      
  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::CopyFromPermuteBackward(const BaseVector<ValueType> &src,
                                                              const BaseVector<int> &permutation) {

  if (this->get_size() > 0) {

    assert(this != &src);
    
    const HIPAcceleratorVector<ValueType> *cast_vec = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&src);
    const HIPAcceleratorVector<int> *cast_perm      = dynamic_cast<const HIPAcceleratorVector<int>*> (&permutation) ; 
    assert(cast_perm != NULL);
    assert(cast_vec  != NULL);
    
    assert(cast_vec ->get_size() == this->get_size());
    assert(cast_perm->get_size() == this->get_size());
    
    
    int size = this->get_size();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);
    
    //    this->vec_[i] = cast_vec->vec_[ cast_perm->vec_[i] ];
    kernel_permute_backward<ValueType, int> <<<GridSize, BlockSize>>> (size, cast_perm->vec_, cast_vec->vec_, this->vec_);
    
    CHECK_HIP_ERROR(__FILE__, __LINE__);      
  }

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::SetIndexArray(const int size, const int *index) {

  assert(size > 0);
  assert(this->get_size() >= size);

  this->index_size_ = size;

  allocate_hip<int>(this->index_size_, &this->index_array_);
  allocate_hip<ValueType>(this->index_size_, &this->index_buffer_);

  cudaMemcpy(this->index_array_,            // dst
             index,                         // src
             this->index_size_*sizeof(int), // size
             cudaMemcpyHostToDevice);

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::GetIndexValues(ValueType *values) const {

  assert(values != NULL);

  dim3 BlockSize(this->local_backend_.HIP_block_size);
  dim3 GridSize(this->index_size_ / this->local_backend_.HIP_block_size + 1);

  kernel_get_index_values<ValueType, int> <<<GridSize, BlockSize>>> (this->index_size_, this->index_array_,
                                                                     this->vec_, this->index_buffer_);
  CHECK_HIP_ERROR(__FILE__, __LINE__);

  cudaMemcpy(values,                              // dst
             this->index_buffer_,                 // src
             this->index_size_*sizeof(ValueType), // size
             cudaMemcpyDeviceToHost);
  CHECK_HIP_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::SetIndexValues(const ValueType *values) {

  assert(values != NULL);

  cudaMemcpy(this->index_buffer_,
             values,
             this->index_size_*sizeof(ValueType),
             cudaMemcpyHostToDevice);
  CHECK_HIP_ERROR(__FILE__, __LINE__);

  dim3 BlockSize(this->local_backend_.HIP_block_size);
  dim3 GridSize(this->index_size_ / this->local_backend_.HIP_block_size + 1);

  kernel_set_index_values<ValueType, int> <<<GridSize, BlockSize>>> (this->index_size_, this->index_array_,
                                                                     this->index_buffer_, this->vec_);
  CHECK_HIP_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::GetContinuousValues(const int start, const int end, ValueType *values) const {

  assert(start >= 0);
  assert(end >= start);
  assert(end <= this->get_size());
  assert(values != NULL);

  cudaMemcpy(values,                        // dst
             this->vec_+start,              // src
             (end-start)*sizeof(ValueType), // size
             cudaMemcpyDeviceToHost);
  CHECK_HIP_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::SetContinuousValues(const int start, const int end, const ValueType *values) {

  assert(start >= 0);
  assert(end >= start);
  assert(end <= this->get_size());
  assert(values != NULL);

  cudaMemcpy(this->vec_+start,              // dst
             values,                        // src
             (end-start)*sizeof(ValueType), // size
             cudaMemcpyHostToDevice);
  CHECK_HIP_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::ExtractCoarseMapping(const int start, const int end, const int *index,
                                                           const int nc, int *size, int *map) const {

  LOG_INFO("ExtractCoarseMapping() NYI for HIP");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void HIPAcceleratorVector<ValueType>::ExtractCoarseBoundary(const int start, const int end, const int *index,
                                                            const int nc, int *size, int *boundary) const {

  LOG_INFO("ExtractCoarseBoundary() NYI for HIP");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <>
void HIPAcceleratorVector<double>::Power(const double power) {

  if (this->get_size() > 0) {

    int size = this->get_size();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

    kernel_powerd<int> <<<GridSize, BlockSize>>> (size, power, this->vec_);

    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorVector<float>::Power(const double power) {

  if (this->get_size() > 0) {

    int size = this->get_size();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(size / this->local_backend_.HIP_block_size + 1);

    kernel_powerf<int> <<<GridSize, BlockSize>>> (size, power, this->vec_);

    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorVector<std::complex<double> >::Power(const double power) {

  if (this->get_size() > 0) {

    LOG_INFO("HIPAcceleratorVector::Power(), no pow() for std::complex<double> in HIP");
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorVector<std::complex<float> >::Power(const double power) {

  if (this->get_size() > 0) {

    LOG_INFO("HIPAcceleratorVector::Power(), no pow() for std::complex<float> in HIP");
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorVector<int>::Power(const double power) {

  if (this->get_size() > 0) {

    LOG_INFO("HIPAcceleratorVector::Power(), no pow() for int in HIP");
    FATAL_ERROR(__FILE__, __LINE__);


  }

}


template class HIPAcceleratorVector<double>;
template class HIPAcceleratorVector<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorVector<std::complex<double> >;
template class HIPAcceleratorVector<std::complex<float> >;
#endif
template class HIPAcceleratorVector<int>;

}
