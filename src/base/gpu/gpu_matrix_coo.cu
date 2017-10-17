#include "../../utils/def.hpp"
#include "gpu_matrix_csr.hpp"
#include "gpu_matrix_coo.hpp"
#include "gpu_vector.hpp"
#include "../host/host_matrix_coo.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "gpu_utils.hpp"
#include "cuda_kernels_general.hpp"
#include "cuda_kernels_coo.hpp"
#include "gpu_allocate_free.hpp"
#include "../matrix_formats_ind.hpp"

#include <algorithm>

#include <cuda.h>
#include <cusparse_v2.h>
#include <cuComplex.h>

namespace paralution {

template <typename ValueType>
GPUAcceleratorMatrixCOO<ValueType>::GPUAcceleratorMatrixCOO() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
GPUAcceleratorMatrixCOO<ValueType>::GPUAcceleratorMatrixCOO(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "GPUAcceleratorMatrixCOO::GPUAcceleratorMatrixCOO()",
            "constructor with local_backend");

  this->mat_.row = NULL;  
  this->mat_.col = NULL;  
  this->mat_.val = NULL;
  this->set_backend(local_backend); 

  CHECK_CUDA_ERROR(__FILE__, __LINE__);

}


template <typename ValueType>
GPUAcceleratorMatrixCOO<ValueType>::~GPUAcceleratorMatrixCOO() {

  LOG_DEBUG(this, "GPUAcceleratorMatrixCOO::~GPUAcceleratorMatrixCOO()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void GPUAcceleratorMatrixCOO<ValueType>::info(void) const {

  LOG_INFO("GPUAcceleratorMatrixCOO<ValueType>");

}

template <typename ValueType>
void GPUAcceleratorMatrixCOO<ValueType>::AllocateCOO(const int nnz, const int nrow, const int ncol) {

  assert(nnz >= 0);
  assert(ncol >= 0);
  assert(nrow >= 0);

  if (this->get_nnz() > 0)
    this->Clear();

  if (nnz > 0) {

    allocate_gpu(nnz, &this->mat_.row);
    allocate_gpu(nnz, &this->mat_.col);
    allocate_gpu(nnz, &this->mat_.val);
 
    set_to_zero_gpu(this->local_backend_.GPU_block_size, 
                    this->local_backend_.GPU_max_threads,
                    nnz, this->mat_.row);
    set_to_zero_gpu(this->local_backend_.GPU_block_size, 
                    this->local_backend_.GPU_max_threads,
                    nnz, this->mat_.col);
    set_to_zero_gpu(this->local_backend_.GPU_block_size, 
                    this->local_backend_.GPU_max_threads,
                    nnz, this->mat_.val);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

  }

}

template <typename ValueType>
void GPUAcceleratorMatrixCOO<ValueType>::SetDataPtrCOO(int **row, int **col, ValueType **val,
                                                       const int nnz, const int nrow, const int ncol) {

  assert(*row != NULL);
  assert(*col != NULL);
  assert(*val != NULL);
  assert(nnz > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  this->Clear();

  this->nrow_ = nrow;
  this->ncol_ = ncol;
  this->nnz_  = nnz;

  cudaDeviceSynchronize();

  this->mat_.row = *row;
  this->mat_.col = *col;
  this->mat_.val = *val;

}

template <typename ValueType>
void GPUAcceleratorMatrixCOO<ValueType>::LeaveDataPtrCOO(int **row, int **col, ValueType **val) {

  assert(this->get_nrow() > 0);
  assert(this->get_ncol() > 0);
  assert(this->get_nnz() > 0);

  cudaDeviceSynchronize();

  // see free_host function for details
  *row = this->mat_.row;
  *col = this->mat_.col;
  *val = this->mat_.val;

  this->mat_.row = NULL;
  this->mat_.col = NULL;
  this->mat_.val = NULL;

  this->nrow_ = 0;
  this->ncol_ = 0;
  this->nnz_  = 0;

}

template <typename ValueType>
void GPUAcceleratorMatrixCOO<ValueType>::Clear() {

  if (this->get_nnz() > 0) {

    free_gpu(&this->mat_.row);
    free_gpu(&this->mat_.col);
    free_gpu(&this->mat_.val);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

  }


}

template <typename ValueType>
void GPUAcceleratorMatrixCOO<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixCOO<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to GPU copy
  if ((cast_mat = dynamic_cast<const HostMatrixCOO<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateCOO(src.get_nnz(), src.get_nrow(), src.get_ncol() );

  if (this->get_nnz() > 0) {

      assert(this->get_nnz()  == src.get_nnz());
      assert(this->get_nrow()  == src.get_nrow());
      assert(this->get_ncol()  == src.get_ncol());
      
      cudaMemcpy(this->mat_.row,     // dst
                 cast_mat->mat_.row, // src
                 (this->get_nnz())*sizeof(int), // size
                 cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.col,     // dst
                 cast_mat->mat_.col, // src
                 this->get_nnz()*sizeof(int), // size
                 cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.val,     // dst
                 cast_mat->mat_.val, // src
                 this->get_nnz()*sizeof(ValueType), // size
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
void GPUAcceleratorMatrixCOO<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixCOO<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixCOO<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateCOO(this->get_nnz(), this->get_nrow(), this->get_ncol() );

  if (this->get_nnz() > 0) {

      assert(this->get_nnz()  == dst->get_nnz());
      assert(this->get_nrow() == dst->get_nrow());
      assert(this->get_ncol() == dst->get_ncol());
      
      cudaMemcpy(cast_mat->mat_.row, // dst
                 this->mat_.row,     // src
                 this->get_nnz()*sizeof(int), // size           
                 cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(cast_mat->mat_.col, // dst
                 this->mat_.col,     // src
                 this->get_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(cast_mat->mat_.val, // dst
                 this->mat_.val,     // src
                 this->get_nnz()*sizeof(ValueType), // size
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
void GPUAcceleratorMatrixCOO<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const GPUAcceleratorMatrixCOO<ValueType> *gpu_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<const GPUAcceleratorMatrixCOO<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateCOO(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpy(this->mat_.row,         // dst
                 gpu_cast_mat->mat_.row, // src
                 (this->get_nnz())*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.col,         // dst
                 gpu_cast_mat->mat_.col, // src
                 this->get_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.val,         // dst
                 gpu_cast_mat->mat_.val, // src
                 this->get_nnz()*sizeof(ValueType), // size
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
void GPUAcceleratorMatrixCOO<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  GPUAcceleratorMatrixCOO<ValueType> *gpu_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<GPUAcceleratorMatrixCOO<ValueType>*> (dst)) != NULL) {

    gpu_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    gpu_cast_mat->AllocateCOO(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpy(gpu_cast_mat->mat_.row, // dst
                 this->mat_.row,         // src
                 (this->get_nnz())*sizeof(int), // size
                 cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(gpu_cast_mat->mat_.col, // dst
                 this->mat_.col,         // src
                 this->get_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(gpu_cast_mat->mat_.val, // dst
                 this->mat_.val,         // src
                 this->get_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToHost);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
    }
    
  } else {

    //GPU to CPU
    if ((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst)) != NULL) {
      
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
void GPUAcceleratorMatrixCOO<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  const HostMatrixCOO<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to GPU copy
  if ((cast_mat = dynamic_cast<const HostMatrixCOO<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateCOO(src.get_nnz(), src.get_nrow(), src.get_ncol() );

  if (this->get_nnz() > 0) {

      assert(this->get_nnz()  == src.get_nnz());
      assert(this->get_nrow()  == src.get_nrow());
      assert(this->get_ncol()  == src.get_ncol());
      
      cudaMemcpyAsync(this->mat_.row,     // dst
                      cast_mat->mat_.row, // src
                      (this->get_nnz())*sizeof(int), // size
                      cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(this->mat_.col,     // dst
                      cast_mat->mat_.col, // src
                      this->get_nnz()*sizeof(int), // size
                      cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(this->mat_.val,     // dst
                      cast_mat->mat_.val, // src
                      this->get_nnz()*sizeof(ValueType), // size
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
void GPUAcceleratorMatrixCOO<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  HostMatrixCOO<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixCOO<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateCOO(this->get_nnz(), this->get_nrow(), this->get_ncol() );

  if (this->get_nnz() > 0) {

      assert(this->get_nnz()  == dst->get_nnz());
      assert(this->get_nrow() == dst->get_nrow());
      assert(this->get_ncol() == dst->get_ncol());
      
      cudaMemcpyAsync(cast_mat->mat_.row, // dst
                      this->mat_.row,     // src
                      this->get_nnz()*sizeof(int), // size           
                      cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(cast_mat->mat_.col, // dst
                      this->mat_.col,     // src
                      this->get_nnz()*sizeof(int), // size
                      cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(cast_mat->mat_.val, // dst
                      this->mat_.val,     // src
                      this->get_nnz()*sizeof(ValueType), // size
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
void GPUAcceleratorMatrixCOO<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &src) {

  const GPUAcceleratorMatrixCOO<ValueType> *gpu_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<const GPUAcceleratorMatrixCOO<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateCOO(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpy(this->mat_.row,         // dst
                 gpu_cast_mat->mat_.row, // src
                 (this->get_nnz())*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.col,         // dst
                 gpu_cast_mat->mat_.col, // src
                 this->get_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.val,         // dst
                 gpu_cast_mat->mat_.val, // src
                 this->get_nnz()*sizeof(ValueType), // size
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
void GPUAcceleratorMatrixCOO<ValueType>::CopyToAsync(BaseMatrix<ValueType> *dst) const {

  GPUAcceleratorMatrixCOO<ValueType> *gpu_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<GPUAcceleratorMatrixCOO<ValueType>*> (dst)) != NULL) {

    gpu_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    gpu_cast_mat->AllocateCOO(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpy(gpu_cast_mat->mat_.row, // dst
                 this->mat_.row,         // src
                 (this->get_nnz())*sizeof(int), // size
                 cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(gpu_cast_mat->mat_.col, // dst
                 this->mat_.col,         // src
                 this->get_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToHost);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(gpu_cast_mat->mat_.val, // dst
                 this->mat_.val,         // src
                 this->get_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToHost);    
      CHECK_CUDA_ERROR(__FILE__, __LINE__);     
    }
    
  } else {

    //GPU to CPU
    if ((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst)) != NULL) {
      
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
void GPUAcceleratorMatrixCOO<ValueType>::CopyFromCOO(const int *row, const int *col, const ValueType *val) {

  // assert CSR format
  assert(this->get_mat_format() == COO);

  if (this->get_nnz() > 0) {

    assert(this->nrow_ > 0);
    assert(this->ncol_ > 0);

    cudaMemcpy(this->mat_.row,              // dst
               row,                         // src
               this->get_nnz()*sizeof(int), // size
               cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    cudaMemcpy(this->mat_.col,              // dst
               col,                         // src
               this->get_nnz()*sizeof(int), // size
               cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    cudaMemcpy(this->mat_.val,                    // dst
               val,                               // src
               this->get_nnz()*sizeof(ValueType), // size
               cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void GPUAcceleratorMatrixCOO<ValueType>::CopyToCOO(int *row, int *col, ValueType *val) const {

  // assert CSR format
  assert(this->get_mat_format() == COO);

  if (this->get_nnz() > 0) {

    assert(this->nrow_ > 0);
    assert(this->ncol_ > 0);

    cudaMemcpy(row,                         // dst
               this->mat_.row,              // src
               this->get_nnz()*sizeof(int), // size
               cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    cudaMemcpy(col,                         // dst
               this->mat_.col,              // src
               this->get_nnz()*sizeof(int), // size
               cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    cudaMemcpy(val,                               // dst
               this->mat_.val,                    // src
               this->get_nnz()*sizeof(ValueType), // size
               cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
bool GPUAcceleratorMatrixCOO<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const GPUAcceleratorMatrixCOO<ValueType> *cast_mat_coo;
  if ((cast_mat_coo = dynamic_cast<const GPUAcceleratorMatrixCOO<ValueType>*> (&mat)) != NULL) {

      this->CopyFrom(*cast_mat_coo);
      return true;

  }

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const GPUAcceleratorMatrixCSR<ValueType> *cast_mat_csr;
  if ((cast_mat_csr = dynamic_cast<const GPUAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {

    this->Clear();

    assert(cast_mat_csr->get_nrow() > 0);
    assert(cast_mat_csr->get_ncol() > 0);
    assert(cast_mat_csr->get_nnz() > 0);

    this->AllocateCOO(cast_mat_csr->nnz_, cast_mat_csr->nrow_, cast_mat_csr->ncol_);

    int nrow = cast_mat_csr->nrow_;

    dim3 BlockSize(this->local_backend_.GPU_block_size);
    dim3 GridSize(nrow / this->local_backend_.GPU_block_size + 1);

    kernel_coo_csr_to_coo<int> <<<GridSize, BlockSize>>>(nrow, cast_mat_csr->mat_.row_offset, this->mat_.row);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    cudaMemcpy(this->mat_.col, cast_mat_csr->mat_.col, this->nnz_*sizeof(int), cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    cudaMemcpy(this->mat_.val, cast_mat_csr->mat_.val, this->nnz_*sizeof(ValueType), cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    return true;

  }

  return false;

}

// ----------------------------------------------------------
// Modified and adapted from CUSP 0.5.1, 
// http://cusplibrary.github.io/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// see coo_flat_spmv.h
// ----------------------------------------------------------
// CHANGELOG
// - adapted interface
// ----------------------------------------------------------  
template <typename ValueType>
void GPUAcceleratorMatrixCOO<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const GPUAcceleratorVector<ValueType> *cast_in = dynamic_cast<const GPUAcceleratorVector<ValueType>*> (&in);
    GPUAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      GPUAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    cast_out->Zeros();

    ValueType one = (ValueType) 1;

    // If nnz < warpsize, perform sequential spmv
    if (this->get_nnz() < this->local_backend_.GPU_warp) {

      kernel_spmv_coo_serial<<<1, 1>>> (this->get_nnz(), this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val),
                                        CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_));
      CHECK_CUDA_ERROR(__FILE__, __LINE__);

    } else {

      int warps_per_block = this->local_backend_.GPU_block_size / this->local_backend_.GPU_warp;
      int numWarps = (this->local_backend_.GPU_block_size + (this->local_backend_.GPU_warp - 1)) / this->local_backend_.GPU_warp;

      int ctaLimitThreads = this->local_backend_.GPU_threads_per_proc / this->local_backend_.GPU_block_size;
      int ctaLimitRegs = this->local_backend_.GPU_max_threads / this->local_backend_.GPU_block_size / numWarps;

      int max_blocks = this->local_backend_.GPU_num_procs * std::min(ctaLimitRegs, std::min(ctaLimitThreads, 16));
      int num_units  = this->nnz_ / this->local_backend_.GPU_warp;
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
          kernel_spmv_coo_flat<512, 32> <<<num_blocks, 512>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
          kernel_spmv_coo_reduce_update<512> <<<1, 512>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
        } else if (this->local_backend_.GPU_block_size == 256) {
          kernel_spmv_coo_flat<256, 32> <<<num_blocks, 256>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
          kernel_spmv_coo_reduce_update<256> <<<1, 256>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
        } else if (this->local_backend_.GPU_block_size == 128) {
          kernel_spmv_coo_flat<128, 32> <<<num_blocks, 128>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
          kernel_spmv_coo_reduce_update<128> <<<1, 128>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
        } else if (this->local_backend_.GPU_block_size ==  64) {
          kernel_spmv_coo_flat<64, 32> <<<num_blocks, 64>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
          kernel_spmv_coo_reduce_update<64> <<<1, 64>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
        } else if (this->local_backend_.GPU_block_size ==  32) {
          kernel_spmv_coo_flat<32, 32> <<<num_blocks, 32>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
          kernel_spmv_coo_reduce_update<32> <<<1, 32>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
        } else {
          LOG_INFO("Unsupported CUDA blocksize of " << this->local_backend_.GPU_block_size);
          CHECK_CUDA_ERROR(__FILE__, __LINE__);
        }
      } else if (this->local_backend_.GPU_warp == 64) {
        if (this->local_backend_.GPU_block_size == 512) {
          kernel_spmv_coo_flat<512, 64> <<<num_blocks, 512>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
          kernel_spmv_coo_reduce_update<512> <<<1, 512>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
        } else if (this->local_backend_.GPU_block_size == 256) {
          kernel_spmv_coo_flat<256, 64> <<<num_blocks, 256>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
          kernel_spmv_coo_reduce_update<256> <<<1, 256>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
        } else if (this->local_backend_.GPU_block_size == 128) {
          kernel_spmv_coo_flat<128, 64> <<<num_blocks, 128>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
          kernel_spmv_coo_reduce_update<128> <<<1, 128>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
        } else if (this->local_backend_.GPU_block_size ==  64) {
          kernel_spmv_coo_flat<64, 64> <<<num_blocks, 64>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
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

      kernel_spmv_coo_serial<<<1, 1>>>(this->get_nnz()-tail, this->mat_.row+tail, this->mat_.col+tail, CUDAPtr(this->mat_.val+tail), CUDAVal(one), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_));
      CHECK_CUDA_ERROR(__FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
void GPUAcceleratorMatrixCOO<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                                  BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const GPUAcceleratorVector<ValueType> *cast_in = dynamic_cast<const GPUAcceleratorVector<ValueType>*> (&in);
    GPUAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      GPUAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    // If matrix is very sparse, we do COO via atomics
    if (this->nnz_ / this->nrow_ < 2) {

      dim3 BlockSize(this->local_backend_.GPU_block_size);
      dim3 GridSize(this->get_nnz() / this->local_backend_.GPU_block_size + 1);

      kernel_spmv_add_coo<<<GridSize, BlockSize>>> (this->get_nnz(), this->mat_.row, this->mat_.col,
                                                    CUDAPtr(this->mat_.val), CUDAPtr(cast_in->vec_),
                                                    CUDAVal(scalar), CUDAPtr(cast_out->vec_));
      CHECK_CUDA_ERROR(__FILE__, __LINE__);

    } else {

      // If nnz < warpsize, perform sequential spmv
      if (this->get_nnz() < this->local_backend_.GPU_warp) {

        kernel_spmv_coo_serial<<<1, 1>>> (this->get_nnz(), this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val),
                                          CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_));
        CHECK_CUDA_ERROR(__FILE__, __LINE__);

      } else {

        int warps_per_block = this->local_backend_.GPU_block_size / this->local_backend_.GPU_warp;
        int numWarps = (this->local_backend_.GPU_block_size + (this->local_backend_.GPU_warp - 1)) / this->local_backend_.GPU_warp;

        int ctaLimitThreads = this->local_backend_.GPU_threads_per_proc / this->local_backend_.GPU_block_size;
        int ctaLimitRegs = this->local_backend_.GPU_max_threads / this->local_backend_.GPU_block_size / numWarps;

        int max_blocks = this->local_backend_.GPU_num_procs * std::min(ctaLimitRegs, std::min(ctaLimitThreads, 16));
        int num_units  = this->nnz_ / this->local_backend_.GPU_warp;
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
            kernel_spmv_coo_flat<512, 32> <<<num_blocks, 512>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
            kernel_spmv_coo_reduce_update<512> <<<1, 512>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
          } else if (this->local_backend_.GPU_block_size == 256) {
            kernel_spmv_coo_flat<256, 32> <<<num_blocks, 256>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
            kernel_spmv_coo_reduce_update<256> <<<1, 256>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
          } else if (this->local_backend_.GPU_block_size == 128) {
            kernel_spmv_coo_flat<128, 32> <<<num_blocks, 128>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
            kernel_spmv_coo_reduce_update<128> <<<1, 128>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
          } else if (this->local_backend_.GPU_block_size ==  64) {
            kernel_spmv_coo_flat<64, 32> <<<num_blocks, 64>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
            kernel_spmv_coo_reduce_update<64> <<<1, 64>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
          } else if (this->local_backend_.GPU_block_size ==  32) {
            kernel_spmv_coo_flat<32, 32> <<<num_blocks, 32>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
            kernel_spmv_coo_reduce_update<32> <<<1, 32>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
          } else {
            LOG_INFO("Unsupported CUDA blocksize of " << this->local_backend_.GPU_block_size);
            CHECK_CUDA_ERROR(__FILE__, __LINE__);
          }
        } else if (this->local_backend_.GPU_warp == 64) {
          if (this->local_backend_.GPU_block_size == 512) {
            kernel_spmv_coo_flat<512, 64> <<<num_blocks, 512>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
            kernel_spmv_coo_reduce_update<512> <<<1, 512>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
          } else if (this->local_backend_.GPU_block_size == 256) {
            kernel_spmv_coo_flat<256, 64> <<<num_blocks, 256>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
            kernel_spmv_coo_reduce_update<256> <<<1, 256>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
          } else if (this->local_backend_.GPU_block_size == 128) {
            kernel_spmv_coo_flat<128, 64> <<<num_blocks, 128>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
            kernel_spmv_coo_reduce_update<128> <<<1, 128>>>(active_warps, temp_rows, CUDAPtr(temp_vals), CUDAPtr(cast_out->vec_));
          } else if (this->local_backend_.GPU_block_size ==  64) {
            kernel_spmv_coo_flat<64, 64> <<<num_blocks, 64>>>(tail, interval_size, this->mat_.row, this->mat_.col, CUDAPtr(this->mat_.val), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_), temp_rows, CUDAPtr(temp_vals));
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

        kernel_spmv_coo_serial<<<1, 1>>>(this->get_nnz()-tail, this->mat_.row+tail, this->mat_.col+tail, CUDAPtr(this->mat_.val+tail), CUDAVal(scalar), CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_));
        CHECK_CUDA_ERROR(__FILE__, __LINE__);

      }

    }

  }

}

template <typename ValueType>
bool GPUAcceleratorMatrixCOO<ValueType>::Permute(const BaseVector<int> &permutation) {

  // symmetric permutation only
  assert(permutation.get_size() == this->get_nrow());
  assert(permutation.get_size() == this->get_ncol());

  if (this->get_nnz() > 0) {

    const GPUAcceleratorVector<int> *cast_perm = dynamic_cast<const GPUAcceleratorVector<int>*> (&permutation);
    assert(cast_perm != NULL);

    GPUAcceleratorMatrixCOO<ValueType> src(this->local_backend_);
    src.AllocateCOO(this->get_nnz(), this->get_nrow(), this->get_ncol());
    src.CopyFrom(*this);

    int nnz = this->get_nnz();
    int s = nnz;
    int k = (nnz/this->local_backend_.GPU_block_size)/this->local_backend_.GPU_max_threads + 1;
    if (k > 1) s = nnz / k;

    dim3 BlockSize(this->local_backend_.GPU_block_size);
    dim3 GridSize(s / this->local_backend_.GPU_block_size + 1);

    kernel_coo_permute<ValueType, int> <<<GridSize, BlockSize>>> (nnz,
                                                                  src.mat_.row, src.mat_.col,
                                                                  cast_perm->vec_,
                                                                  this->mat_.row, this->mat_.col);

    CHECK_CUDA_ERROR(__FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool GPUAcceleratorMatrixCOO<ValueType>::PermuteBackward(const BaseVector<int> &permutation) {

  // symmetric permutation only
  assert(permutation.get_size() == this->get_nrow());
  assert(permutation.get_size() == this->get_ncol());

  if (this->get_nnz() > 0) {

    const GPUAcceleratorVector<int> *cast_perm = dynamic_cast<const GPUAcceleratorVector<int>*> (&permutation);
    assert(cast_perm != NULL);

    int *pb = NULL;
    allocate_gpu(this->get_nrow(), &pb);

    int n = this->get_nrow();
    dim3 BlockSize1(this->local_backend_.GPU_block_size);
    dim3 GridSize1(n / this->local_backend_.GPU_block_size + 1);

    kernel_reverse_index<int> <<<GridSize1, BlockSize1>>> (n,
                                                           cast_perm->vec_,
                                                           pb);

    GPUAcceleratorMatrixCOO<ValueType> src(this->local_backend_);
    src.AllocateCOO(this->get_nnz(), this->get_nrow(), this->get_ncol());
    src.CopyFrom(*this);

    int nnz = this->get_nnz();
    int s = nnz;
    int k = (nnz/this->local_backend_.GPU_block_size)/this->local_backend_.GPU_max_threads + 1;
    if (k > 1) s = nnz / k;

    dim3 BlockSize2(this->local_backend_.GPU_block_size);
    dim3 GridSize2(s / this->local_backend_.GPU_block_size + 1);

    kernel_coo_permute<ValueType, int> <<<GridSize2, BlockSize2>>> (nnz,
                                                                    src.mat_.row, src.mat_.col,
                                                                    pb,
                                                                    this->mat_.row, this->mat_.col);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    free_gpu(&pb);

  }

  return true;

}


template class GPUAcceleratorMatrixCOO<double>;
template class GPUAcceleratorMatrixCOO<float>;
#ifdef SUPPORT_COMPLEX
template class GPUAcceleratorMatrixCOO<std::complex<double> >;
template class GPUAcceleratorMatrixCOO<std::complex<float> >;
#endif

}
