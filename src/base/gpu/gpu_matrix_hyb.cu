#include "../../utils/def.hpp"
#include "gpu_matrix_csr.hpp"
#include "gpu_matrix_coo.hpp"
#include "gpu_matrix_ell.hpp"
#include "gpu_matrix_hyb.hpp"
#include "gpu_vector.hpp"
#include "../host/host_matrix_hyb.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "gpu_utils.hpp"
#include "cuda_kernels_general.hpp"
#include "cuda_kernels_hyb.hpp"
#include "cuda_kernels_vector.hpp"
#include "gpu_allocate_free.hpp"
#include "../matrix_formats_ind.hpp"

#include <algorithm>

#include <cuda.h>
#include <cusparse_v2.h>
#include <cuComplex.h>

namespace paralution {

template <typename ValueType>
GPUAcceleratorMatrixHYB<ValueType>::GPUAcceleratorMatrixHYB() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
GPUAcceleratorMatrixHYB<ValueType>::GPUAcceleratorMatrixHYB(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "GPUAcceleratorMatrixHYB::GPUAcceleratorMatrixHYB()",
            "constructor with local_backend");

  this->mat_.ELL.val = NULL;
  this->mat_.ELL.col = NULL;
  this->mat_.ELL.max_row = 0;

  this->mat_.COO.row = NULL;  
  this->mat_.COO.col = NULL;  
  this->mat_.COO.val = NULL;

  this->ell_nnz_ = 0;
  this->coo_nnz_ = 0;

  this->set_backend(local_backend); 

  CHECK_CUDA_ERROR(__FILE__, __LINE__);

}


template <typename ValueType>
GPUAcceleratorMatrixHYB<ValueType>::~GPUAcceleratorMatrixHYB() {

  LOG_DEBUG(this, "GPUAcceleratorMatrixHYB::~GPUAcceleratorMatrixHYB()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void GPUAcceleratorMatrixHYB<ValueType>::info(void) const {

  LOG_INFO("GPUAcceleratorMatrixHYB<ValueType>");

}

template <typename ValueType>
void GPUAcceleratorMatrixHYB<ValueType>::AllocateHYB(const int ell_nnz, const int coo_nnz, const int ell_max_row, 
                                                     const int nrow, const int ncol) {

  assert( ell_nnz   >= 0);
  assert( coo_nnz   >= 0);
  assert( ell_max_row >= 0);

  assert( ncol  >= 0);
  assert( nrow  >= 0);
  
  if (this->get_nnz() > 0)
    this->Clear();

  if (ell_nnz + coo_nnz > 0) {
    
    // ELL
    assert(ell_nnz == ell_max_row*nrow);

    allocate_gpu(ell_nnz, &this->mat_.ELL.val);
    allocate_gpu(ell_nnz, &this->mat_.ELL.col);
    
    set_to_zero_gpu(this->local_backend_.GPU_block_size, 
                    this->local_backend_.GPU_max_threads,
                    ell_nnz, this->mat_.ELL.val);
    set_to_zero_gpu(this->local_backend_.GPU_block_size, 
                    this->local_backend_.GPU_max_threads,
                    ell_nnz, this->mat_.ELL.col);

    this->mat_.ELL.max_row = ell_max_row;
    this->ell_nnz_ = ell_nnz;

    // COO
    allocate_gpu(coo_nnz, &this->mat_.COO.row);
    allocate_gpu(coo_nnz, &this->mat_.COO.col);
    allocate_gpu(coo_nnz, &this->mat_.COO.val);
 
    set_to_zero_gpu(this->local_backend_.GPU_block_size, 
                    this->local_backend_.GPU_max_threads,
                    coo_nnz, this->mat_.COO.row);
    set_to_zero_gpu(this->local_backend_.GPU_block_size, 
                    this->local_backend_.GPU_max_threads,
                    coo_nnz, this->mat_.COO.col);
    set_to_zero_gpu(this->local_backend_.GPU_block_size, 
                    this->local_backend_.GPU_max_threads,
                    coo_nnz, this->mat_.COO.val);
    this->coo_nnz_ = coo_nnz;

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = ell_nnz + coo_nnz;

  }

}


template <typename ValueType>
void GPUAcceleratorMatrixHYB<ValueType>::Clear() {

  if (this->get_nnz() > 0) {
    
    free_gpu(&this->mat_.COO.row);
    free_gpu(&this->mat_.COO.col);
    free_gpu(&this->mat_.COO.val);
    
    free_gpu(&this->mat_.ELL.val);
    free_gpu(&this->mat_.ELL.col);
    
    this->ell_nnz_ = 0;
    this->coo_nnz_ = 0;
    this->mat_.ELL.max_row = 0;
    
    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;
    
  }
  

}

template <typename ValueType>
void GPUAcceleratorMatrixHYB<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to GPU copy
  if ((cast_mat = dynamic_cast<const HostMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateHYB(cast_mat->get_ell_nnz(), cast_mat->get_coo_nnz(), cast_mat->get_ell_max_row(),
                      cast_mat->get_nrow(), cast_mat->get_ncol());

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_ell_nnz() > 0) {

      // ELL
      cudaMemcpy(this->mat_.ELL.col,     // dst
                 cast_mat->mat_.ELL.col, // src
                 this->get_ell_nnz()*sizeof(int), // size
                 cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.ELL.val,     // dst
                 cast_mat->mat_.ELL.val, // src
                 this->get_ell_nnz()*sizeof(ValueType), // size
                 cudaMemcpyHostToDevice);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
    }

    if (this->get_coo_nnz() > 0) {

      // COO
      cudaMemcpy(this->mat_.COO.row,     // dst
                 cast_mat->mat_.COO.row, // src
                 (this->get_coo_nnz())*sizeof(int), // size
                 cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.COO.col,     // dst
                 cast_mat->mat_.COO.col, // src
                 this->get_coo_nnz()*sizeof(int), // size
                 cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.COO.val,     // dst
                 cast_mat->mat_.COO.val, // src
                 this->get_coo_nnz()*sizeof(ValueType), // size
                 cudaMemcpyHostToDevice);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
    }
    
  } else {
    
    LOG_INFO("Error unsupported GPU matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void GPUAcceleratorMatrixHYB<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixHYB<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateHYB(this->get_ell_nnz(), this->get_coo_nnz(), this->get_ell_max_row(),
                      this->get_nrow(), this->get_ncol());

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_ell_nnz() > 0) {
      
      // ELL
      cudaMemcpy(cast_mat->mat_.ELL.col, // dst
                 this->mat_.ELL.col,     // src
                 this->get_ell_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(cast_mat->mat_.ELL.val, // dst
                 this->mat_.ELL.val,     // src
                 this->get_ell_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToHost);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
    }


    if (this->get_coo_nnz() > 0) {

      // COO
      cudaMemcpy(cast_mat->mat_.COO.row, // dst
                 this->mat_.COO.row,     // src
                 this->get_coo_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(cast_mat->mat_.COO.col, // dst
                 this->mat_.COO.col,     // src
                 this->get_coo_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(cast_mat->mat_.COO.val, // dst
                 this->mat_.COO.val,     // src
                 this->get_coo_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToHost);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
    }

  } else {
    
    LOG_INFO("Error unsupported GPU matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void GPUAcceleratorMatrixHYB<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const GPUAcceleratorMatrixHYB<ValueType> *gpu_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<const GPUAcceleratorMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateHYB(gpu_cast_mat->get_ell_nnz(), gpu_cast_mat->get_coo_nnz(), gpu_cast_mat->get_ell_max_row(),
                      gpu_cast_mat->get_nrow(), gpu_cast_mat->get_ncol());

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());


    if (this->get_ell_nnz() > 0) {

      // ELL
      cudaMemcpy(this->mat_.ELL.col,     // dst
                 gpu_cast_mat->mat_.ELL.col, // src
                 this->get_ell_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.ELL.val,     // dst
                 gpu_cast_mat->mat_.ELL.val, // src
                 this->get_ell_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToDevice);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
    }

    if (this->get_coo_nnz() > 0) {

      // COO
      cudaMemcpy(this->mat_.COO.row,     // dst
                 gpu_cast_mat->mat_.COO.row, // src
                 (this->get_coo_nnz())*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.COO.col,     // dst
                 gpu_cast_mat->mat_.COO.col, // src
                 this->get_coo_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.COO.val,     // dst
                 gpu_cast_mat->mat_.COO.val, // src
                 this->get_coo_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToDevice);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     

    }
      
  } else {

    //CPU to GPU
    if ((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*> (&src)) != NULL) {
      
      this->CopyFromHost(*host_cast_mat);
      
    } else {
      
      LOG_INFO("Error unsupported GPU matrix type");
      this->info();
      src.info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }
    
  }

}

template <typename ValueType>
void GPUAcceleratorMatrixHYB<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  GPUAcceleratorMatrixHYB<ValueType> *gpu_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to GPU copy
  if (gpu_cast_mat = dynamic_cast<GPUAcceleratorMatrixHYB<ValueType>*> (dst)) {

    gpu_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    gpu_cast_mat->AllocateHYB(this->get_ell_nnz(), this->get_coo_nnz(), this->get_ell_max_row(),
                      this->get_nrow(), this->get_ncol());

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_ell_nnz() > 0) {

      // ELL
      cudaMemcpy(gpu_cast_mat->mat_.ELL.col, // dst
                 this->mat_.ELL.col,     // src
                 this->get_ell_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(gpu_cast_mat->mat_.ELL.val, // dst
                 this->mat_.ELL.val,     // src
                 this->get_ell_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToDevice);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
    }

    if (this->get_coo_nnz() > 0) {

      // COO
      cudaMemcpy(gpu_cast_mat->mat_.COO.row, // dst
                 this->mat_.COO.row,     // src
                 this->get_coo_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(gpu_cast_mat->mat_.COO.col, // dst
                 this->mat_.COO.col,     // src
                 this->get_coo_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(gpu_cast_mat->mat_.COO.val, // dst
                 this->mat_.COO.val,     // src
                 this->get_coo_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToDevice);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     

    }
   
    
  } else {

    //GPU to CPU
    if (host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst)) {
      
      this->CopyToHost(host_cast_mat);

    } else {
      
      LOG_INFO("Error unsupported GPU matrix type");
      this->info();
      dst->info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }

  }


}


template <typename ValueType>
void GPUAcceleratorMatrixHYB<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  const HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to GPU copy
  if ((cast_mat = dynamic_cast<const HostMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateHYB(cast_mat->get_ell_nnz(), cast_mat->get_coo_nnz(), cast_mat->get_ell_max_row(),
                      cast_mat->get_nrow(), cast_mat->get_ncol());

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_ell_nnz() > 0) {

      // ELL
      cudaMemcpyAsync(this->mat_.ELL.col,     // dst
                      cast_mat->mat_.ELL.col, // src
                      this->get_ell_nnz()*sizeof(int), // size
                      cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(this->mat_.ELL.val,     // dst
                      cast_mat->mat_.ELL.val, // src
                      this->get_ell_nnz()*sizeof(ValueType), // size
                      cudaMemcpyHostToDevice);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
    }

    if (this->get_coo_nnz() > 0) {

      // COO
      cudaMemcpyAsync(this->mat_.COO.row,     // dst
                      cast_mat->mat_.COO.row, // src
                      (this->get_coo_nnz())*sizeof(int), // size
                      cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(this->mat_.COO.col,     // dst
                      cast_mat->mat_.COO.col, // src
                      this->get_coo_nnz()*sizeof(int), // size
                      cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(this->mat_.COO.val,     // dst
                      cast_mat->mat_.COO.val, // src
                      this->get_coo_nnz()*sizeof(ValueType), // size
                      cudaMemcpyHostToDevice);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
    }
    
  } else {
    
    LOG_INFO("Error unsupported GPU matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void GPUAcceleratorMatrixHYB<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixHYB<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateHYB(this->get_ell_nnz(), this->get_coo_nnz(), this->get_ell_max_row(),
                      this->get_nrow(), this->get_ncol());

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_ell_nnz() > 0) {
      
      // ELL
      cudaMemcpyAsync(cast_mat->mat_.ELL.col, // dst
                      this->mat_.ELL.col,     // src
                      this->get_ell_nnz()*sizeof(int), // size
                      cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(cast_mat->mat_.ELL.val, // dst
                      this->mat_.ELL.val,     // src
                      this->get_ell_nnz()*sizeof(ValueType), // size
                      cudaMemcpyDeviceToHost);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
    }


    if (this->get_coo_nnz() > 0) {

      // COO
      cudaMemcpyAsync(cast_mat->mat_.COO.row, // dst
                      this->mat_.COO.row,     // src
                      this->get_coo_nnz()*sizeof(int), // size
                      cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(cast_mat->mat_.COO.col, // dst
                      this->mat_.COO.col,     // src
                      this->get_coo_nnz()*sizeof(int), // size
                      cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(cast_mat->mat_.COO.val, // dst
                      this->mat_.COO.val,     // src
                      this->get_coo_nnz()*sizeof(ValueType), // size
                      cudaMemcpyDeviceToHost);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
    }

  } else {
    
    LOG_INFO("Error unsupported GPU matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void GPUAcceleratorMatrixHYB<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &src) {

  const GPUAcceleratorMatrixHYB<ValueType> *gpu_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<const GPUAcceleratorMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateHYB(gpu_cast_mat->get_ell_nnz(), gpu_cast_mat->get_coo_nnz(), gpu_cast_mat->get_ell_max_row(),
                      gpu_cast_mat->get_nrow(), gpu_cast_mat->get_ncol());

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());


    if (this->get_ell_nnz() > 0) {

      // ELL
      cudaMemcpy(this->mat_.ELL.col,     // dst
                 gpu_cast_mat->mat_.ELL.col, // src
                 this->get_ell_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.ELL.val,     // dst
                 gpu_cast_mat->mat_.ELL.val, // src
                 this->get_ell_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToDevice);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
    }

    if (this->get_coo_nnz() > 0) {

      // COO
      cudaMemcpy(this->mat_.COO.row,     // dst
                 gpu_cast_mat->mat_.COO.row, // src
                 (this->get_coo_nnz())*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.COO.col,     // dst
                 gpu_cast_mat->mat_.COO.col, // src
                 this->get_coo_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.COO.val,     // dst
                 gpu_cast_mat->mat_.COO.val, // src
                 this->get_coo_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToDevice);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     

    }
      
  } else {

    //CPU to GPU
    if ((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*> (&src)) != NULL) {
      
      this->CopyFromHostAsync(*host_cast_mat);
      
    } else {
      
      LOG_INFO("Error unsupported GPU matrix type");
      this->info();
      src.info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }
    
  }

}

template <typename ValueType>
void GPUAcceleratorMatrixHYB<ValueType>::CopyToAsync(BaseMatrix<ValueType> *dst) const {

  GPUAcceleratorMatrixHYB<ValueType> *gpu_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to GPU copy
  if (gpu_cast_mat = dynamic_cast<GPUAcceleratorMatrixHYB<ValueType>*> (dst)) {

    gpu_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    gpu_cast_mat->AllocateHYB(this->get_ell_nnz(), this->get_coo_nnz(), this->get_ell_max_row(),
                      this->get_nrow(), this->get_ncol());

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_ell_nnz() > 0) {

      // ELL
      cudaMemcpy(gpu_cast_mat->mat_.ELL.col, // dst
                 this->mat_.ELL.col,     // src
                 this->get_ell_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(gpu_cast_mat->mat_.ELL.val, // dst
                 this->mat_.ELL.val,     // src
                 this->get_ell_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToDevice);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
    }

    if (this->get_coo_nnz() > 0) {

      // COO
      cudaMemcpy(gpu_cast_mat->mat_.COO.row, // dst
                 this->mat_.COO.row,     // src
                 this->get_coo_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(gpu_cast_mat->mat_.COO.col, // dst
                 this->mat_.COO.col,     // src
                 this->get_coo_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(gpu_cast_mat->mat_.COO.val, // dst
                 this->mat_.COO.val,     // src
                 this->get_coo_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToDevice);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     

    }
   
    
  } else {

    //GPU to CPU
    if (host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst)) {
      
      this->CopyToHostAsync(host_cast_mat);

    } else {
      
      LOG_INFO("Error unsupported GPU matrix type");
      this->info();
      dst->info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }

  }


}


template <typename ValueType>
bool GPUAcceleratorMatrixHYB<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const GPUAcceleratorMatrixHYB<ValueType>   *cast_mat_hyb;
  
  if ((cast_mat_hyb = dynamic_cast<const GPUAcceleratorMatrixHYB<ValueType>*> (&mat)) != NULL) {

    this->CopyFrom(*cast_mat_hyb);
    return true;

  }

  const GPUAcceleratorMatrixCSR<ValueType> *cast_mat_csr;

  if ((cast_mat_csr = dynamic_cast<const GPUAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {

    this->Clear();

    int nrow = cast_mat_csr->get_nrow();
    int ncol = cast_mat_csr->get_ncol();
    int max_row = cast_mat_csr->get_nnz() / nrow;

    // get nnz per row for COO part
    int *nnz_coo = NULL;
    dim3 BlockSize(this->local_backend_.GPU_block_size);
    dim3 GridSize(nrow / this->local_backend_.GPU_block_size + 1);

    allocate_gpu<int>(nrow, &nnz_coo);

    kernel_ell_nnz_coo<int> <<<GridSize, BlockSize>>> (nrow, max_row,
                                                       cast_mat_csr->mat_.row_offset,
                                                       nnz_coo);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    // get nnz for COO part by summing up nnz per row array
    int num_nnz_coo = 0;
    int *d_buffer = NULL;
    int *h_buffer = NULL;

    allocate_gpu<int>(this->local_backend_.GPU_warp, &d_buffer);
    allocate_host(this->local_backend_.GPU_warp, &h_buffer);

    reduce_cuda<int, int, 32, 256>(nrow, nnz_coo, &num_nnz_coo, h_buffer, d_buffer);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    free_gpu<int>(&d_buffer);
    free_host(&h_buffer);

    // allocate ELL and COO matrices
    int num_nnz_ell = max_row * nrow;

    if (num_nnz_ell <= 0 || num_nnz_coo <= 0) {
      free_gpu<int>(&nnz_coo);
      return false;
    }

    this->AllocateHYB(num_nnz_ell, num_nnz_coo, max_row, nrow, ncol);

    cudaMemset(this->mat_.ELL.col, -1, num_nnz_ell*sizeof(int));
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    // copy up to num_cols_per_row values of row i into the ELL
    int *nnz_ell = NULL;

    allocate_gpu<int>(nrow, &nnz_ell);

    kernel_ell_fill_ell<ValueType, int> <<<GridSize, BlockSize>>> (nrow, max_row,
                                                                   cast_mat_csr->mat_.row_offset,
                                                                   cast_mat_csr->mat_.col,
                                                                   cast_mat_csr->mat_.val,
                                                                   this->mat_.ELL.col,
                                                                   this->mat_.ELL.val,
                                                                   nnz_ell);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    // TODO currently performing partial sum on host
    allocate_host(nrow, &h_buffer);
    cudaMemcpy(h_buffer, // dst
               nnz_ell, // src
               nrow*sizeof(int), // size
               cudaMemcpyDeviceToHost);

    for (int i=1; i<nrow; ++i)
      h_buffer[i] += h_buffer[i-1];

    cudaMemcpy(nnz_ell, // dst
               h_buffer, // src
               nrow*sizeof(int), // size
               cudaMemcpyHostToDevice);

    free_host(&h_buffer);
    // end TODO

    // copy any remaining values in row i into the COO

    kernel_ell_fill_coo<ValueType, int> <<<GridSize, BlockSize>>> (nrow, cast_mat_csr->mat_.row_offset,
                                                                   cast_mat_csr->mat_.col,
                                                                   cast_mat_csr->mat_.val,
                                                                   nnz_coo, nnz_ell,
                                                                   this->mat_.COO.row, this->mat_.COO.col,
                                                                   this->mat_.COO.val);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    free_gpu<int>(&nnz_ell);
    free_gpu<int>(&nnz_coo);

    this->nrow_ = cast_mat_csr->get_nrow();
    this->ncol_ = cast_mat_csr->get_ncol();
    this->nnz_  = num_nnz_ell + num_nnz_coo;
    this->mat_.ELL.max_row = max_row;
    this->ell_nnz_ = num_nnz_ell;
    this->coo_nnz_ = num_nnz_coo;

    return true;

  }

  return false;

}

template <typename ValueType>
void GPUAcceleratorMatrixHYB<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {
    
    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());
        
    const GPUAcceleratorVector<ValueType> *cast_in = dynamic_cast<const GPUAcceleratorVector<ValueType>*> (&in) ; 
    GPUAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      GPUAcceleratorVector<ValueType>*> (out) ; 
    
    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    // ELL
    if (this->get_ell_nnz() > 0) {

      int nrow = this->get_nrow();
      int ncol = this->get_ncol();
      int max_row = this->get_ell_max_row();
      dim3 BlockSize(this->local_backend_.GPU_block_size);
      dim3 GridSize(nrow / this->local_backend_.GPU_block_size + 1);
          
      kernel_ell_spmv<<<GridSize, BlockSize>>> (nrow, ncol, max_row,
                                                this->mat_.ELL.col, CUDAPtr(this->mat_.ELL.val),
                                                CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_ ));
      CHECK_CUDA_ERROR(__FILE__, __LINE__);

    }

    // ----------------------------------------------------------
    // Modified and adapted from CUSP 0.3.1, 
    // http://code.google.com/p/cusp-library/
    // NVIDIA, APACHE LICENSE 2.0
    // ----------------------------------------------------------
    // see __spmv_coo_flat(...)
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    if (this->get_coo_nnz() > 0) {

      ValueType one = (ValueType) 1;

      if (this->get_coo_nnz() / this->nrow_ < 2) {

        dim3 BlockSize(this->local_backend_.GPU_block_size);
        dim3 GridSize(this->get_coo_nnz() / this->local_backend_.GPU_block_size + 1);

        kernel_spmv_add_coo<<<GridSize, BlockSize>>> (this->get_coo_nnz(), this->mat_.COO.row, this->mat_.COO.col,
                                                      CUDAPtr(this->mat_.COO.val), CUDAPtr(cast_in->vec_),
                                                      CUDAVal(one), CUDAPtr(cast_out->vec_));
        CHECK_CUDA_ERROR(__FILE__, __LINE__);

      } else {

        // If nnz < warpsize, perform sequential spmv
        if (this->get_coo_nnz() < this->local_backend_.GPU_warp) {

          kernel_spmv_coo_serial<<<1, 1>>> (this->get_coo_nnz(), this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val),
                                            CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_));
          CHECK_CUDA_ERROR(__FILE__, __LINE__);

        } else {

          int warps_per_block = this->local_backend_.GPU_block_size / this->local_backend_.GPU_warp;
          int numWarps = (this->local_backend_.GPU_block_size + (this->local_backend_.GPU_warp - 1)) / this->local_backend_.GPU_warp;

          int ctaLimitThreads = this->local_backend_.GPU_threads_per_proc / this->local_backend_.GPU_block_size;
          int ctaLimitRegs = this->local_backend_.GPU_max_threads / this->local_backend_.GPU_block_size / numWarps;

          int max_blocks = this->local_backend_.GPU_num_procs * std::min(ctaLimitRegs, std::min(ctaLimitThreads, 16));
          int num_units  = this->get_coo_nnz() / this->local_backend_.GPU_warp;
          int num_warps  = (num_units < warps_per_block * max_blocks) ? num_units : warps_per_block * max_blocks;
          int num_blocks = (num_warps + (warps_per_block - 1)) / warps_per_block;
          int num_iters  = (num_units + (num_warps - 1)) / num_warps;

          int interval_size = this->local_backend_.GPU_warp * num_iters;
          int tail = num_units * this->local_backend_.GPU_warp;
          int active_warps = (interval_size == 0) ? 0 : (tail + (interval_size - 1)) / interval_size;

          int *temp_rows = NULL;
          ValueType *temp_vals = NULL;

          allocate_gpu(active_warps, &temp_rows);
          allocate_gpu(active_warps, &temp_vals);

          // Run the appropriate kernel
          if (this->local_backend_.GPU_warp == 32) {
            if (this->local_backend_.GPU_block_size == 512) {
              kernel_spmv_coo_flat<512, 32> <<<num_blocks, 512>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<512> <<<1, 512>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size == 256) {
              kernel_spmv_coo_flat<256, 32> <<<num_blocks, 256>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<256> <<<1, 256>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size == 128) {
              kernel_spmv_coo_flat<128, 32> <<<num_blocks, 128>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<128> <<<1, 128>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size ==  64) {
              kernel_spmv_coo_flat<64, 32> <<<num_blocks, 64>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<64> <<<1, 64>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size ==  32) {
              kernel_spmv_coo_flat<32, 32> <<<num_blocks, 32>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<32> <<<1, 32>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else {
              LOG_INFO("Unsupported CUDA blocksize of " << this->local_backend_.GPU_block_size);
              CHECK_CUDA_ERROR(__FILE__, __LINE__);
            }
          } else if (this->local_backend_.GPU_warp == 64) {
            if (this->local_backend_.GPU_block_size == 512) {
              kernel_spmv_coo_flat<512, 64> <<<num_blocks, 512>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<512> <<<1, 512>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size == 256) {
              kernel_spmv_coo_flat<256, 64> <<<num_blocks, 256>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<256> <<<1, 256>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size == 128) {
              kernel_spmv_coo_flat<128, 64> <<<num_blocks, 128>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<128> <<<1, 128>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size ==  64) {
              kernel_spmv_coo_flat<64, 64> <<<num_blocks, 64>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<64> <<<1, 64>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else {
              LOG_INFO("Unsupported CUDA blocksize of " << this->local_backend_.GPU_block_size);
              CHECK_CUDA_ERROR(__FILE__, __LINE__);
            }
          } else {
            LOG_INFO("Unsupported CUDA warpsize of " << this->local_backend_.GPU_warp);
            CHECK_CUDA_ERROR(__FILE__, __LINE__);
          }

          free_gpu(&temp_rows);
          free_gpu(&temp_vals);

          kernel_spmv_coo_serial<<<1, 1>>>(this->get_coo_nnz()-tail, this->mat_.COO.row+tail, this->mat_.COO.col+tail, CUDAPtr(this->mat_.COO.val+tail), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_));
          CHECK_CUDA_ERROR(__FILE__, __LINE__);

        }

      }

    }

  }

}

template <typename ValueType>
void GPUAcceleratorMatrixHYB<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                                  BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const GPUAcceleratorVector<ValueType> *cast_in = dynamic_cast<const GPUAcceleratorVector<ValueType>*> (&in) ; 
    GPUAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      GPUAcceleratorVector<ValueType>*> (out) ; 

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    // ELL
    if (this->get_ell_nnz() > 0) {

      int nrow = this->get_nrow();
      int ncol = this->get_ncol();
      int max_row = this->get_ell_max_row();
      dim3 BlockSize(this->local_backend_.GPU_block_size);
      dim3 GridSize(nrow / this->local_backend_.GPU_block_size + 1);

      kernel_ell_add_spmv<<<GridSize, BlockSize>>> (nrow, ncol, max_row,
                                                    this->mat_.ELL.col, CUDAPtr(this->mat_.ELL.val),
                                                    CUDAVal(scalar),
                                                    CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_ ));
      CHECK_CUDA_ERROR(__FILE__, __LINE__);

    }

    // ----------------------------------------------------------
    // Modified and adapted from CUSP 0.3.1, 
    // http://code.google.com/p/cusp-library/
    // NVIDIA, APACHE LICENSE 2.0
    // ----------------------------------------------------------
    // see __spmv_coo_flat(...)
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    if (this->get_coo_nnz() > 0) {

      if (this->get_coo_nnz() / this->nrow_ < 2) {

        dim3 BlockSize(this->local_backend_.GPU_block_size);
        dim3 GridSize(this->get_coo_nnz() / this->local_backend_.GPU_block_size + 1);

        kernel_spmv_add_coo<<<GridSize, BlockSize>>> (this->get_coo_nnz(), this->mat_.COO.row, this->mat_.COO.col,
                                                      CUDAPtr(this->mat_.COO.val), CUDAPtr(cast_in->vec_),
                                                      CUDAVal(scalar), CUDAPtr(cast_out->vec_));
        CHECK_CUDA_ERROR(__FILE__, __LINE__);

      } else {

        // If nnz < warpsize, perform sequential spmv
        if (this->get_coo_nnz() < this->local_backend_.GPU_warp) {

          kernel_spmv_coo_serial<<<1, 1>>> (this->get_coo_nnz(), this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val),
                                            CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_));
          CHECK_CUDA_ERROR(__FILE__, __LINE__);

        } else {

          int warps_per_block = this->local_backend_.GPU_block_size / this->local_backend_.GPU_warp;
          int numWarps = (this->local_backend_.GPU_block_size + (this->local_backend_.GPU_warp - 1)) / this->local_backend_.GPU_warp;

          int ctaLimitThreads = this->local_backend_.GPU_threads_per_proc / this->local_backend_.GPU_block_size;
          int ctaLimitRegs = this->local_backend_.GPU_max_threads / this->local_backend_.GPU_block_size / numWarps;

          int max_blocks = this->local_backend_.GPU_num_procs * std::min(ctaLimitRegs, std::min(ctaLimitThreads, 16));
          int num_units  = this->get_coo_nnz() / this->local_backend_.GPU_warp;
          int num_warps  = (num_units < warps_per_block * max_blocks) ? num_units : warps_per_block * max_blocks;
          int num_blocks = (num_warps + (warps_per_block - 1)) / warps_per_block;
          int num_iters  = (num_units + (num_warps - 1)) / num_warps;

          int interval_size = this->local_backend_.GPU_warp * num_iters;
          int tail = num_units * this->local_backend_.GPU_warp;
          int active_warps = (interval_size == 0) ? 0 : (tail + (interval_size - 1)) / interval_size;

          int *temp_rows = NULL;
          ValueType *temp_vals = NULL;

          allocate_gpu(active_warps, &temp_rows);
          allocate_gpu(active_warps, &temp_vals);

          // Run the appropriate kernel
          if (this->local_backend_.GPU_warp == 32) {
            if (this->local_backend_.GPU_block_size == 512) {
              kernel_spmv_coo_flat<512, 32> <<<num_blocks, 512>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<512> <<<1, 512>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size == 256) {
              kernel_spmv_coo_flat<256, 32> <<<num_blocks, 256>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<256> <<<1, 256>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size == 128) {
              kernel_spmv_coo_flat<128, 32> <<<num_blocks, 128>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<128> <<<1, 128>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size ==  64) {
              kernel_spmv_coo_flat<64, 32> <<<num_blocks, 64>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<64> <<<1, 64>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size ==  32) {
              kernel_spmv_coo_flat<32, 32> <<<num_blocks, 32>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<32> <<<1, 32>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else {
              LOG_INFO("Unsupported CUDA blocksize of " << this->local_backend_.GPU_block_size);
              CHECK_CUDA_ERROR(__FILE__, __LINE__);
            }
          } else if (this->local_backend_.GPU_warp == 64) {
            if (this->local_backend_.GPU_block_size == 512) {
              kernel_spmv_coo_flat<512, 64> <<<num_blocks, 512>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<512> <<<1, 512>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size == 256) {
              kernel_spmv_coo_flat<256, 64> <<<num_blocks, 256>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<256> <<<1, 256>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size == 128) {
              kernel_spmv_coo_flat<128, 64> <<<num_blocks, 128>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<128> <<<1, 128>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else if (this->local_backend_.GPU_block_size ==  64) {
              kernel_spmv_coo_flat<64, 64> <<<num_blocks, 64>>>(tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, CUDAPtr(this->mat_.COO.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
              kernel_spmv_coo_reduce_update<64> <<<1, 64>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
            } else {
              LOG_INFO("Unsupported CUDA blocksize of " << this->local_backend_.GPU_block_size);
              CHECK_CUDA_ERROR(__FILE__, __LINE__);
            }
          } else {
            LOG_INFO("Unsupported CUDA warpsize of " << this->local_backend_.GPU_warp);
            CHECK_CUDA_ERROR(__FILE__, __LINE__);
          }

          free_gpu(&temp_rows);
          free_gpu(&temp_vals);

          kernel_spmv_coo_serial<<<1, 1>>>(this->get_coo_nnz()-tail, this->mat_.COO.row+tail, this->mat_.COO.col+tail, CUDAPtr(this->mat_.COO.val+tail), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_));
          CHECK_CUDA_ERROR(__FILE__, __LINE__);

        }

      }

    }

  }

}


template class GPUAcceleratorMatrixHYB<double>;
template class GPUAcceleratorMatrixHYB<float>;
#ifdef SUPPORT_COMPLEX
template class GPUAcceleratorMatrixHYB<std::complex<double> >;
template class GPUAcceleratorMatrixHYB<std::complex<float> >;
#endif

}
