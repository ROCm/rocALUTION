#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/math_functions.hpp"
#include "ocl_allocate_free.hpp"
#include "ocl_utils.hpp"
#include "ocl_vector.hpp"
#include "../host/host_vector.hpp"
#include "../backend_manager.hpp"

#include <math.h>
#include <complex>

namespace paralution {

template <typename ValueType>
OCLAcceleratorVector<ValueType>::OCLAcceleratorVector() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
OCLAcceleratorVector<ValueType>::OCLAcceleratorVector(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "OCLAcceleratorVector::OCLAcceleratorVector()",
            "constructor with local_backend");

  this->vec_ = NULL;

  this->set_backend(local_backend);

  this->index_array_  = NULL;
  this->index_buffer_ = NULL;

  this->host_buffer_ = NULL;
  this->device_buffer_ = NULL;

}

template <typename ValueType>
OCLAcceleratorVector<ValueType>::~OCLAcceleratorVector() {

  LOG_DEBUG(this, "OCLAcceleratorVector::~OCLAcceleratorVector()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::info(void) const {

  LOG_INFO("OCLAcceleratorVector<ValueType>");

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::Allocate(const int n) {

  assert (n >= 0);

  if (this->size_ > 0)
    this->Clear();

  if (n > 0) {

    allocate_ocl(n, this->local_backend_.OCL_context, &this->vec_);

    allocate_host(this->local_backend_.OCL_warp_size, &this->host_buffer_);
    allocate_ocl(this->local_backend_.OCL_warp_size, this->local_backend_.OCL_context, &this->device_buffer_);

    ocl_set_to(n, (ValueType) 0, this->vec_, this->local_backend_.OCL_command_queue);

    this->size_ = n;

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::SetDataPtr(ValueType **ptr, const int size) {

  assert (*ptr != NULL);
  assert (size > 0);

  this->Clear();

  cl_int err = clFinish((cl_command_queue) this->local_backend_.OCL_command_queue);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  this->vec_ = *ptr;
  *ptr = NULL;

  allocate_host(this->local_backend_.OCL_warp_size, &this->host_buffer_);
  allocate_ocl(this->local_backend_.OCL_warp_size, this->local_backend_.OCL_context, &this->device_buffer_);

  this->size_ = size;

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::LeaveDataPtr(ValueType **ptr) {

  assert (this->size_ > 0);

  cl_int err = clFinish((cl_command_queue) this->local_backend_.OCL_command_queue);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  *ptr = this->vec_;
  this->vec_ = NULL;

  free_host(&this->host_buffer_);
  free_ocl(&this->device_buffer_);

  this->size_ = 0;

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::Clear(void) {

  if (this->size_ > 0) {

    free_ocl(&this->vec_);

    free_host(&this->host_buffer_);
    free_ocl(&this->device_buffer_);

    this->size_ = 0;

  }

  if (this->index_size_ > 0) {

    free_ocl(&this->index_buffer_);
    free_ocl(&this->index_array_);
    this->index_size_ = 0;

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::CopyFromHost(const HostVector<ValueType> &src) {

  // CPU to OCL copy
  const HostVector<ValueType> *cast_vec;
  if ((cast_vec = dynamic_cast<const HostVector<ValueType>*> (&src)) != NULL) {

    if (this->size_ == 0) {

      // Allocate local vector
      this->Allocate(cast_vec->size_);

      // Check for boundary
      assert (this->index_size_ == 0);
      if (cast_vec->index_size_ > 0) {

        this->index_size_ = cast_vec->index_size_;
        allocate_ocl(this->index_size_, this->local_backend_.OCL_context, &this->index_array_);
        allocate_ocl(this->index_size_, this->local_backend_.OCL_context, &this->index_buffer_);

      }

    }
    
    assert (cast_vec->size_ == this->size_);
    assert (cast_vec->index_size_ == this->index_size_);

    if (this->size_ > 0) {

      ocl_host2dev(this->size_,    // size
                   cast_vec->vec_, // src
                   this->vec_,     // dst
                   this->local_backend_.OCL_command_queue);

      ocl_host2dev(this->index_size_,      // size
                   cast_vec->index_array_, // src
                   this->index_array_,     // dst
                   this->local_backend_.OCL_command_queue);

    }

  } else {

    LOG_INFO("Error unsupported OpenCL vector type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::CopyToHost(HostVector<ValueType> *dst) const {

  assert (dst != NULL);

  // OCL to CPU copy
  HostVector<ValueType> *cast_vec;
  if ((cast_vec = dynamic_cast<HostVector<ValueType>*> (dst)) != NULL) {

    if (cast_vec->size_ == 0) {

      // Allocate local vector
      cast_vec->Allocate(this->size_);

      // Check for boundary
      assert (cast_vec->index_size_ == 0);
      if (this->index_size_ > 0) {

        cast_vec->index_size_ = this->index_size_;
        allocate_host(this->index_size_, &cast_vec->index_array_);

      }

    }

    assert (cast_vec->size_ == this->size_);
    assert (cast_vec->index_size_ == this->index_size_);

    if (this->size_ > 0) {

      ocl_dev2host(this->size_,    // size
                   this->vec_,     // src
                   cast_vec->vec_, // dst
                   this->local_backend_.OCL_command_queue);

      ocl_dev2host(this->index_size_,      // size
                   this->index_array_,     // src
                   cast_vec->index_array_, // dst
                   this->local_backend_.OCL_command_queue);

    }

  } else {

    LOG_INFO("Error unsupported OCL vector type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::CopyFrom(const BaseVector<ValueType> &src) {

  const OCLAcceleratorVector<ValueType> *ocl_cast_vec;
  const HostVector<ValueType> *host_cast_vec;

  // OCL to OCL copy
  if ((ocl_cast_vec = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&src)) != NULL) {

    if (this->size_ == 0) {

      // Allocate local vector
      this->Allocate(ocl_cast_vec->size_);

      // Check for boundary
      assert (this->index_size_ == 0);
      if (ocl_cast_vec->index_size_ > 0) {

        this->index_size_ = ocl_cast_vec->index_size_;
        allocate_ocl(this->index_size_, this->local_backend_.OCL_context, &this->index_array_);
        allocate_ocl(this->index_size_, this->local_backend_.OCL_context, &this->index_buffer_);

      }

    }

    assert (ocl_cast_vec->size_ == this->size_);
    assert (ocl_cast_vec->index_size_ == this->index_size_);

    if (this != ocl_cast_vec) {

      if (this->size_ > 0) {

        // must be within same opencl context
        ocl_dev2dev(this->size_,        // size
                    ocl_cast_vec->vec_, // src
                    this->vec_,         // dst
                    this->local_backend_.OCL_command_queue);

        // must be within same opencl context
        ocl_dev2dev(this->index_size_,          // size
                    ocl_cast_vec->index_array_, // src
                    this->index_array_,         // dst
                    this->local_backend_.OCL_command_queue);

      }

    }

  } else {

    //OCL to CPU copy
    if ((host_cast_vec = dynamic_cast<const HostVector<ValueType>*> (&src)) != NULL) {

      this->CopyFromHost(*host_cast_vec);

    } else {

      LOG_INFO("Error unsupported OpenCL vector type");
      this->info();
      src.info();
      FATAL_ERROR(__FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::CopyFrom(const BaseVector<ValueType> &src,
                                               const int src_offset,
                                               const int dst_offset,
                                               const int size) {

  assert (&src != this);
  assert (this->size_ > 0);
  assert (src.get_size() > 0);
  assert (size > 0);

  assert (src_offset + size <= src.get_size());
  assert (dst_offset + size <= this->size_);

  const OCLAcceleratorVector<ValueType> *cast_src = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&src);

  assert (cast_src != NULL);

  size_t LocalSize  = this->local_backend_.OCL_block_size;
  size_t GlobalSize = (size / LocalSize + 1) * LocalSize;

  cl_int err = ocl_kernel<ValueType>("kernel_copy_offset_from",
                                     this->local_backend_.OCL_command_queue,
                                     LocalSize, GlobalSize,
                                     size, src_offset, dst_offset, cast_src->vec_, this->vec_);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::CopyTo(BaseVector<ValueType> *dst) const{

  assert (dst != NULL);

  OCLAcceleratorVector<ValueType> *ocl_cast_vec;
  HostVector<ValueType> *host_cast_vec;

  // OCL to OCL copy
  if ((ocl_cast_vec = dynamic_cast<OCLAcceleratorVector<ValueType>*> (dst)) != NULL) {

    if (this != ocl_cast_vec) {

      if (ocl_cast_vec->size_ == 0) {

        // Allocate local vector
        ocl_cast_vec->Allocate(this->size_);

        // Check for boundary
        assert (ocl_cast_vec->index_size_ == 0);
        if (this->index_size_ > 0) {

          ocl_cast_vec->index_size_ = this->index_size_;
          allocate_ocl(this->index_size_, this->local_backend_.OCL_context, &ocl_cast_vec->index_array_);
          allocate_ocl(this->index_size_, this->local_backend_.OCL_context, &ocl_cast_vec->index_buffer_);

        }

      }

      assert (ocl_cast_vec->size_ == this->size_);
      assert (ocl_cast_vec->index_size_ == this->index_size_);

      if (this->size_ > 0) {

        // must be within same opencl context
        ocl_dev2dev(this->size_,        // size
                    this->vec_,         // src
                    ocl_cast_vec->vec_, // dst
                    this->local_backend_.OCL_command_queue);

        // must be within same opencl context
        ocl_dev2dev(this->index_size_,          // size
                    this->index_array_,         // src
                    ocl_cast_vec->index_array_, // dst
                    this->local_backend_.OCL_command_queue);

      }

    }

  } else {

    //OCL to CPU copy
    if ((host_cast_vec = dynamic_cast<HostVector<ValueType>*> (dst)) != NULL) {

      this->CopyToHost(host_cast_vec);

    } else {

      LOG_INFO("Error unsupported OpenCL vector type");
      this->info();
      dst->info();
      FATAL_ERROR(__FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::CopyFromFloat(const BaseVector<float> &src) {

  LOG_INFO("Mixed precision for non-complex to complex casting is not allowed");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <>
void OCLAcceleratorVector<double>::CopyFromFloat(const BaseVector<float> &src) {

  const OCLAcceleratorVector<float> *ocl_cast_vec;

  // OCL to OCL copy
  if ((ocl_cast_vec = dynamic_cast<const OCLAcceleratorVector<float>*> (&src)) != NULL) {

    if (this->size_ == 0)
      this->Allocate(ocl_cast_vec->size_);

    assert (ocl_cast_vec->size_ == this->size_);

    if (this->size_ > 0) {

      size_t LocalSize  = this->local_backend_.OCL_block_size;
      size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

      cl_int err = ocl_kernel<double>("kernel_copy_from_float",
                                      this->local_backend_.OCL_command_queue,
                                      LocalSize, GlobalSize,
                                      this->size_, ocl_cast_vec->vec_, this->vec_);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    }

  } else {

    LOG_INFO("Error unsupported OCL vector type");
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::CopyFromDouble(const BaseVector<double> &src) {

  LOG_INFO("Mixed precision for non-complex to complex casting is not allowed");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <>
void OCLAcceleratorVector<float>::CopyFromDouble(const BaseVector<double> &src) {

  const OCLAcceleratorVector<double> *ocl_cast_vec;

  // GPU to GPU copy
  if ((ocl_cast_vec = dynamic_cast<const OCLAcceleratorVector<double>*> (&src)) != NULL) {

    if (this->size_ == 0)
      this->Allocate(ocl_cast_vec->size_);

    assert (ocl_cast_vec->size_ == this->size_);

    if (this->size_ > 0) {

      size_t LocalSize  = this->local_backend_.OCL_block_size;
      size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

      cl_int err = ocl_kernel<float>("kernel_copy_from_double",
                                     this->local_backend_.OCL_command_queue,
                                     LocalSize, GlobalSize,
                                     this->size_, ocl_cast_vec->vec_, this->vec_);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    }

  } else {
    LOG_INFO("Error unsupported OCL vector type");
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::Zeros(void) {

  if (this->size_ > 0) {

    ocl_set_to(this->size_, (ValueType) 0, this->vec_, this->local_backend_.OCL_command_queue);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::Ones(void) {

  if (this->size_ > 0) {

    ocl_set_to(this->size_, (ValueType) 1, this->vec_, this->local_backend_.OCL_command_queue);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::SetValues(const ValueType val) {

  if (this->size_ > 0) {

    ocl_set_to(this->size_, val, this->vec_, this->local_backend_.OCL_command_queue);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::AddScale(const BaseVector<ValueType> &x, const ValueType alpha) {

  if (this->size_ > 0) {

    assert (this->size_ == x.get_size());

    const OCLAcceleratorVector<ValueType> *cast_x = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&x);

    assert (cast_x != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_axpy",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, alpha, cast_x->vec_, this->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::ScaleAdd(const ValueType alpha, const BaseVector<ValueType> &x) {

  if (this->size_ > 0) {

    assert (this->size_ == x.get_size());

    const OCLAcceleratorVector<ValueType> *cast_x = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&x);

    assert (cast_x != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_scaleadd",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, alpha, cast_x->vec_, this->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::ScaleAddScale(const ValueType alpha,
                                                    const BaseVector<ValueType> &x,
                                                    const ValueType beta) {

  if (this->size_ > 0) {

    assert (this->size_ == x.get_size());

    const OCLAcceleratorVector<ValueType> *cast_x = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&x);

    assert (cast_x != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_scaleaddscale",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, alpha, beta, cast_x->vec_, this->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::ScaleAddScale(const ValueType alpha, const BaseVector<ValueType> &x, const ValueType beta,
                                                    const int src_offset, const int dst_offset,const int size) {

  if (this->size_ > 0) {

    assert (this->size_ > 0);
    assert (x.get_size() > 0);
    assert (size > 0);
    assert (src_offset + size <= x.get_size());
    assert (dst_offset + size <= this->size_);

    const OCLAcceleratorVector<ValueType> *cast_x = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&x);

    assert (cast_x != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (size / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_scaleaddscale_offset",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       size, src_offset, dst_offset, alpha, beta, cast_x->vec_, this->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::ScaleAdd2(const ValueType alpha, const BaseVector<ValueType> &x,
                                                const ValueType beta,  const BaseVector<ValueType> &y,
                                                const ValueType gamma) {

  if (this->size_ > 0) {

    assert (this->size_ == x.get_size());
    assert (this->size_ == y.get_size());

    const OCLAcceleratorVector<ValueType> *cast_x = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&x);
    const OCLAcceleratorVector<ValueType> *cast_y = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&y);

    assert (cast_x != NULL);
    assert (cast_y != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_scaleadd2",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, alpha, beta, gamma, cast_x->vec_, cast_y->vec_, this->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::Scale(const ValueType alpha) {

  if (this->size_ > 0) {

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_scale",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, alpha, this->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::ExclusiveScan(const BaseVector<ValueType> &x) {

  LOG_INFO("OCLAcceleratorVector::ExclusiveScan() NYI");
  FATAL_ERROR(__FILE__, __LINE__); 

}

template <typename ValueType>
ValueType OCLAcceleratorVector<ValueType>::Dot(const BaseVector<ValueType> &x) const {

  assert (this->size_ == x.get_size());

  const OCLAcceleratorVector<ValueType> *cast_x = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&x);

  assert (cast_x != NULL);

  ValueType res = (ValueType) 0;

  if (this->size_ > 0) {

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = this->local_backend_.OCL_warp_size * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_dotc",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, this->vec_, cast_x->vec_, this->device_buffer_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    ocl_dev2host(this->local_backend_.OCL_warp_size, this->device_buffer_, this->host_buffer_, this->local_backend_.OCL_command_queue);

    for (int i=0; i<this->local_backend_.OCL_warp_size; ++i) {
      res += this->host_buffer_[i];
    }

  }

  return res;

}

template <typename ValueType>
ValueType OCLAcceleratorVector<ValueType>::DotNonConj(const BaseVector<ValueType> &x) const {

  assert (this->size_ == x.get_size());

  const OCLAcceleratorVector<ValueType> *cast_x = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&x);

  assert (cast_x != NULL);

  ValueType res = (ValueType) 0;

  if (this->size_ > 0) {

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = this->local_backend_.OCL_warp_size * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_dot",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, this->vec_, cast_x->vec_, this->device_buffer_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    ocl_dev2host(this->local_backend_.OCL_warp_size, this->device_buffer_, this->host_buffer_, this->local_backend_.OCL_command_queue);

    for (int i=0; i<this->local_backend_.OCL_warp_size; ++i) {
      res += this->host_buffer_[i];
    }

  }

  return res;

}

template <typename ValueType>
ValueType OCLAcceleratorVector<ValueType>::Norm(void) const {

  ValueType res = (ValueType) 0;

  if (this->size_ > 0) {

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = this->local_backend_.OCL_warp_size * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_norm",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, this->vec_, this->device_buffer_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    ocl_dev2host(this->local_backend_.OCL_warp_size, this->device_buffer_, this->host_buffer_, this->local_backend_.OCL_command_queue);

    for (int i=0; i<this->local_backend_.OCL_warp_size; ++i) {
      res += this->host_buffer_[i];
    }

  }

  return sqrt(res);

}

template <>
int OCLAcceleratorVector<int>::Norm(void) const {

  LOG_INFO("What is int OCLAcceleratorVector<ValueType>::Norm(void) const?");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
ValueType OCLAcceleratorVector<ValueType>::Reduce(void) const {

  ValueType res = (ValueType) 0;

  if (this->size_ > 0) {

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = this->local_backend_.OCL_warp_size * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_reduce",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, this->vec_, this->device_buffer_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    ocl_dev2host(this->local_backend_.OCL_warp_size, this->device_buffer_, this->host_buffer_, this->local_backend_.OCL_command_queue);

    for (int i=0; i<this->local_backend_.OCL_warp_size; ++i) {
      res += this->host_buffer_[i];
    }

  }

  return res;

}

template <typename ValueType>
ValueType OCLAcceleratorVector<ValueType>::Asum(void) const {

  ValueType res = (ValueType) 0;

  if (this->size_ > 0) {

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = this->local_backend_.OCL_warp_size * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_asum",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, this->vec_, this->device_buffer_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    ocl_dev2host(this->local_backend_.OCL_warp_size, this->device_buffer_, this->host_buffer_, this->local_backend_.OCL_command_queue);

    for (int i=0; i<this->local_backend_.OCL_warp_size; ++i) {
      res += paralution_abs(this->host_buffer_[i]);
    }

  }

  return res;

}

template <typename ValueType>
int OCLAcceleratorVector<ValueType>::Amax(ValueType &value) const {

  ValueType res = (ValueType) 0;
  int idx = 0;

  if (this->size_ > 0) {

    int LocalSize     = this->local_backend_.OCL_block_size;
    size_t GlobalSize = this->local_backend_.OCL_warp_size * LocalSize;

    int *iDeviceBuffer = NULL;
    int *iHostBuffer = NULL;

    allocate_ocl(LocalSize, this->local_backend_.OCL_context, &iDeviceBuffer);

    cl_int err = ocl_kernel<ValueType>("kernel_amax",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, this->vec_, this->device_buffer_, iDeviceBuffer);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    allocate_host(LocalSize, &iHostBuffer);

    ocl_dev2host(this->local_backend_.OCL_warp_size, this->device_buffer_, this->host_buffer_, this->local_backend_.OCL_command_queue);
    ocl_dev2host(LocalSize, iDeviceBuffer, iHostBuffer, this->local_backend_.OCL_command_queue);
    free_ocl(&iDeviceBuffer);

    for (int i=0; i<this->local_backend_.OCL_warp_size; ++i) {
      ValueType tmp = paralution_abs(this->host_buffer_[i]);
      if (res < tmp) {
        res = tmp;
        idx = iHostBuffer[i];
      }
    }

    free_host(&iHostBuffer);

  }

  value = res;

  return idx;

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::PointWiseMult(const BaseVector<ValueType> &x) {

  if (this->size_ > 0) {

    assert (this->size_ == x.get_size());

    const OCLAcceleratorVector<ValueType> *cast_x = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&x);

    assert (cast_x != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_pointwisemult",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, cast_x->vec_, this->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::PointWiseMult(const BaseVector<ValueType> &x, const BaseVector<ValueType> &y) {

  if (this->size_ > 0) {

    assert (this->size_ == x.get_size());
    assert (this->size_ == y.get_size());

    const OCLAcceleratorVector<ValueType> *cast_x = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&x);
    const OCLAcceleratorVector<ValueType> *cast_y = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&y);

    assert (cast_x != NULL);
    assert (cast_y != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_pointwisemult2",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, cast_x->vec_, cast_y->vec_, this->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::Permute(const BaseVector<int> &permutation) {

  if (this->size_ > 0) {

    assert (this->size_ == permutation.get_size());

    const OCLAcceleratorVector<int> *cast_perm = dynamic_cast<const OCLAcceleratorVector<int>*> (&permutation);

    assert (cast_perm != NULL);

    OCLAcceleratorVector<ValueType> vec_tmp(this->local_backend_);
    vec_tmp.Allocate(this->size_);
    vec_tmp.CopyFrom(*this);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_permute",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, cast_perm->vec_, vec_tmp.vec_, this->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::PermuteBackward(const BaseVector<int> &permutation) {

  if (this->size_ > 0) {

    assert (this->size_ == permutation.get_size());

    const OCLAcceleratorVector<int> *cast_perm = dynamic_cast<const OCLAcceleratorVector<int>*> (&permutation);

    assert (cast_perm != NULL);

    OCLAcceleratorVector<ValueType> vec_tmp(this->local_backend_);
    vec_tmp.Allocate(this->size_);
    vec_tmp.CopyFrom(*this);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_permute_backward",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, cast_perm->vec_, vec_tmp.vec_, this->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::CopyFromPermute(const BaseVector<ValueType> &src,
                                                      const BaseVector<int> &permutation) {

  if (this->size_ > 0) {

    assert (this != &src);

    const OCLAcceleratorVector<ValueType> *cast_vec = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&src);
    const OCLAcceleratorVector<int> *cast_perm      = dynamic_cast<const OCLAcceleratorVector<int>*> (&permutation);

    assert (cast_perm != NULL);
    assert (cast_vec  != NULL);

    assert (cast_vec ->size_ == this->size_);
    assert (cast_perm->size_ == this->size_);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_permute",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, cast_perm->vec_, cast_vec->vec_, this->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::CopyFromPermuteBackward(const BaseVector<ValueType> &src,
                                                              const BaseVector<int> &permutation) {

  if (this->size_ > 0) {

    assert (this != &src);

    const OCLAcceleratorVector<ValueType> *cast_vec = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&src);
    const OCLAcceleratorVector<int> *cast_perm      = dynamic_cast<const OCLAcceleratorVector<int>*> (&permutation);

    assert (cast_perm != NULL);
    assert (cast_vec  != NULL);

    assert (cast_vec ->size_ == this->size_);
    assert (cast_perm->size_ == this->size_);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_permute_backward",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, cast_perm->vec_, cast_vec->vec_, this->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::Power(const double power) {

  if (this->size_ > 0) {

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->size_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_power",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->size_, power, this->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::SetIndexArray(const int size, const int *index) {

  assert (index != NULL);
  assert (size > 0);
  assert (this->size_ >= size);

  this->index_size_ = size;

  allocate_ocl(this->index_size_, this->local_backend_.OCL_context, &this->index_buffer_);
  allocate_ocl(this->index_size_, this->local_backend_.OCL_context, &this->index_array_);

  ocl_host2dev(this->index_size_,  // size
               index,              // src
               this->index_array_, // dst
               this->local_backend_.OCL_command_queue);

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::GetIndexValues(ValueType *values) const {

  assert (values != NULL);

  size_t LocalSize  = this->local_backend_.OCL_block_size;
  size_t GlobalSize = (this->index_size_ / LocalSize + 1) * LocalSize;

  cl_int err = ocl_kernel<ValueType>("kernel_get_index_values",
                                     this->local_backend_.OCL_command_queue,
                                     LocalSize, GlobalSize,
                                     this->index_size_, this->index_array_, this->vec_, this->index_buffer_);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  ocl_dev2host(this->index_size_,   // size
               this->index_buffer_, // src
               values,              // dst
               this->local_backend_.OCL_command_queue);

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::SetIndexValues(const ValueType *values) {

  assert (values != NULL);

  ocl_host2dev(this->index_size_,   // size
               values,              // src
               this->index_buffer_, // dst
               this->local_backend_.OCL_command_queue);

  size_t LocalSize  = this->local_backend_.OCL_block_size;
  size_t GlobalSize = (this->index_size_ / LocalSize + 1) * LocalSize;

  cl_int err = ocl_kernel<ValueType>("kernel_set_index_values",
                                     this->local_backend_.OCL_command_queue,
                                     LocalSize, GlobalSize,
                                     this->index_size_, this->index_array_, this->index_buffer_, this->vec_);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::GetContinuousValues(const int start, const int end, ValueType *values) const {

  assert (start >= 0);
  assert (end >= start);
  assert (end <= this->size_);
  assert (values != NULL);

  ocl_dev2host(end - start,        // size
               this->vec_ + start, // src
               values,             // dst
               this->local_backend_.OCL_command_queue);

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::SetContinuousValues(const int start, const int end, const ValueType *values) {

  assert (start >= 0);
  assert (end >= start);
  assert (end <= this->size_);
  assert (values != NULL);

  ocl_host2dev(end - start,        // size
               values,             // src
               this->vec_ + start, // dst
               this->local_backend_.OCL_command_queue);

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::ExtractCoarseMapping(const int start, const int end, const int *index,
                                                           const int nc, int *size, int *map) const {

  LOG_INFO("ExtractCoarseMapping() NYI for OpenCL");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
void OCLAcceleratorVector<ValueType>::ExtractCoarseBoundary(const int start, const int end, const int *index,
                                                            const int nc, int *size, int *boundary) const {

  LOG_INFO("ExtractCoarseBoundary() NYI for OpenCL");
  FATAL_ERROR(__FILE__, __LINE__);

}


template class OCLAcceleratorVector<double>;
template class OCLAcceleratorVector<float>;
#ifdef SUPPORT_COMPLEX
template class OCLAcceleratorVector<std::complex<double> >;
template class OCLAcceleratorVector<std::complex<float> >;
#endif
template class OCLAcceleratorVector<int>;

}
