#include "../../utils/def.hpp"
#include "gpu_matrix_csr.hpp"
#include "gpu_matrix_mcsr.hpp"
#include "gpu_vector.hpp"
#include "../host/host_matrix_mcsr.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "gpu_utils.hpp"
#include "cuda_kernels_general.hpp"
#include "cuda_kernels_mcsr.hpp"
#include "gpu_allocate_free.hpp"
#include "../matrix_formats_ind.hpp"

#include <cuda.h>
#include <cusparse_v2.h>
#include <cuComplex.h>

namespace paralution {

template <typename ValueType>
GPUAcceleratorMatrixMCSR<ValueType>::GPUAcceleratorMatrixMCSR() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
GPUAcceleratorMatrixMCSR<ValueType>::GPUAcceleratorMatrixMCSR(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "GPUAcceleratorMatrixMCSR::GPUAcceleratorMatrixMCSR()",
            "constructor with local_backend");

  this->mat_.row_offset = NULL;  
  this->mat_.col = NULL;  
  this->mat_.val = NULL;
  this->set_backend(local_backend); 

  CHECK_CUDA_ERROR(__FILE__, __LINE__);

}


template <typename ValueType>
GPUAcceleratorMatrixMCSR<ValueType>::~GPUAcceleratorMatrixMCSR() {

  LOG_DEBUG(this, "GPUAcceleratorMatrixMCSR::~GPUAcceleratorMatrixMCSR()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void GPUAcceleratorMatrixMCSR<ValueType>::info(void) const {

  LOG_INFO("GPUAcceleratorMatrixMCSR<ValueType>");

}

template <typename ValueType>
void GPUAcceleratorMatrixMCSR<ValueType>::AllocateMCSR(const int nnz, const int nrow, const int ncol) {

  assert(nnz >= 0);
  assert(ncol >= 0);
  assert(nrow >= 0);

  if (this->get_nnz() > 0)
    this->Clear();

  if (nnz > 0) {

    allocate_gpu(nrow+1, &this->mat_.row_offset);
    allocate_gpu(nnz,    &this->mat_.col);
    allocate_gpu(nnz,    &this->mat_.val);
    
    set_to_zero_gpu(this->local_backend_.GPU_block_size, 
                    this->local_backend_.GPU_max_threads,
                    nrow+1, mat_.row_offset);
    set_to_zero_gpu(this->local_backend_.GPU_block_size, 
                    this->local_backend_.GPU_max_threads,
                    nnz, mat_.col);
    set_to_zero_gpu(this->local_backend_.GPU_block_size, 
                    this->local_backend_.GPU_max_threads,
                    nnz, mat_.val);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

  }

}

template <typename ValueType>
void GPUAcceleratorMatrixMCSR<ValueType>::SetDataPtrMCSR(int **row_offset, int **col, ValueType **val,
                                                         const int nnz, const int nrow, const int ncol) {

  assert(*row_offset != NULL);
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

  this->mat_.row_offset = *row_offset;
  this->mat_.col = *col;
  this->mat_.val = *val;

}

template <typename ValueType>
void GPUAcceleratorMatrixMCSR<ValueType>::LeaveDataPtrMCSR(int **row_offset, int **col, ValueType **val) {

  assert(this->get_nrow() > 0);
  assert(this->get_ncol() > 0);
  assert(this->get_nnz() > 0);

  cudaDeviceSynchronize();

  // see free_host function for details
  *row_offset = this->mat_.row_offset;
  *col = this->mat_.col;
  *val = this->mat_.val;

  this->mat_.row_offset = NULL;
  this->mat_.col = NULL;
  this->mat_.val = NULL;

  this->nrow_ = 0;
  this->ncol_ = 0;
  this->nnz_  = 0;

}

template <typename ValueType>
void GPUAcceleratorMatrixMCSR<ValueType>::Clear() {

  if (this->get_nnz() > 0) {

    free_gpu(&this->mat_.row_offset);
    free_gpu(&this->mat_.col);
    free_gpu(&this->mat_.val);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

  }


}

template <typename ValueType>
void GPUAcceleratorMatrixMCSR<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixMCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to GPU copy
  if ((cast_mat = dynamic_cast<const HostMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateMCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    cudaMemcpy(this->mat_.row_offset,     // dst
               cast_mat->mat_.row_offset, // src
               (this->get_nrow()+1)*sizeof(int), // size
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
    
    
  } else {
    
    LOG_INFO("Error unsupported GPU matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void GPUAcceleratorMatrixMCSR<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixMCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixMCSR<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateMCSR(this->get_nnz(), this->get_nrow(), this->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    cudaMemcpy(cast_mat->mat_.row_offset, // dst
               this->mat_.row_offset,     // src
               (this->get_nrow()+1)*sizeof(int), // size
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
   
    
  } else {
    
    LOG_INFO("Error unsupported GPU matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void GPUAcceleratorMatrixMCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const GPUAcceleratorMatrixMCSR<ValueType> *gpu_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<const GPUAcceleratorMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateMCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    cudaMemcpy(this->mat_.row_offset,         // dst
               gpu_cast_mat->mat_.row_offset, // src
               (this->get_nrow()+1)*sizeof(int), // size
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
void GPUAcceleratorMatrixMCSR<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  GPUAcceleratorMatrixMCSR<ValueType> *gpu_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<GPUAcceleratorMatrixMCSR<ValueType>*> (dst)) != NULL) {

    gpu_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    gpu_cast_mat->AllocateMCSR(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    cudaMemcpy(gpu_cast_mat->mat_.row_offset, // dst
               this->mat_.row_offset,         // src
               (this->get_nrow()+1)*sizeof(int), // size
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
void GPUAcceleratorMatrixMCSR<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  const HostMatrixMCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to GPU copy
  if ((cast_mat = dynamic_cast<const HostMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateMCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    cudaMemcpyAsync(this->mat_.row_offset,     // dst
                    cast_mat->mat_.row_offset, // src
                    (this->get_nrow()+1)*sizeof(int), // size
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
    
    
  } else {
    
    LOG_INFO("Error unsupported GPU matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void GPUAcceleratorMatrixMCSR<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  HostMatrixMCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixMCSR<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateMCSR(this->get_nnz(), this->get_nrow(), this->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    cudaMemcpyAsync(cast_mat->mat_.row_offset, // dst
                    this->mat_.row_offset,     // src
                    (this->get_nrow()+1)*sizeof(int), // size
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
   
    
  } else {
    
    LOG_INFO("Error unsupported GPU matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void GPUAcceleratorMatrixMCSR<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &src) {

  const GPUAcceleratorMatrixMCSR<ValueType> *gpu_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<const GPUAcceleratorMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateMCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    cudaMemcpy(this->mat_.row_offset,         // dst
               gpu_cast_mat->mat_.row_offset, // src
               (this->get_nrow()+1)*sizeof(int), // size
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
void GPUAcceleratorMatrixMCSR<ValueType>::CopyToAsync(BaseMatrix<ValueType> *dst) const {

  GPUAcceleratorMatrixMCSR<ValueType> *gpu_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<GPUAcceleratorMatrixMCSR<ValueType>*> (dst)) != NULL) {

    gpu_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    gpu_cast_mat->AllocateMCSR(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    cudaMemcpy(gpu_cast_mat->mat_.row_offset, // dst
               this->mat_.row_offset,         // src
               (this->get_nrow()+1)*sizeof(int), // size
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
bool GPUAcceleratorMatrixMCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const GPUAcceleratorMatrixMCSR<ValueType>   *cast_mat_mcsr;
  
  if ((cast_mat_mcsr = dynamic_cast<const GPUAcceleratorMatrixMCSR<ValueType>*> (&mat)) != NULL) {

      this->CopyFrom(*cast_mat_mcsr);
      return true;

  }

  /*
  const GPUAcceleratorMatrixCSR<ValueType>   *cast_mat_csr;
  if ((cast_mat_csr = dynamic_cast<const GPUAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {
    
    this->Clear();
    
    FATAL_ERROR(__FILE__, __LINE__);
    
    this->nrow_ = cast_mat_csr->get_nrow();
    this->ncol_ = cast_mat_csr->get_ncol();
    this->nnz_  = cast_mat_csr->get_nnz();
    
    return true;
    
  }
  */

  return false;

}

template <typename ValueType>
void GPUAcceleratorMatrixMCSR<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const GPUAcceleratorVector<ValueType> *cast_in = dynamic_cast<const GPUAcceleratorVector<ValueType>*> (&in);
    GPUAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      GPUAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    dim3 BlockSize(this->local_backend_.GPU_block_size);
    dim3 GridSize(this->nrow_ / this->local_backend_.GPU_block_size + 1);

    int nnz_per_row = this->nnz_ / this->nrow_;
    int nthreads;

    if      (nnz_per_row <=   8) nthreads =  2;
    else if (nnz_per_row <=  16) nthreads =  4;
    else if (nnz_per_row <=  32) nthreads =  8;
    else if (nnz_per_row <=  64) nthreads = 16;
    else if (nnz_per_row <= 128) nthreads = 32;
    else                         nthreads = 64;

    kernel_mcsr_spmv<256, 32> <<<GridSize, BlockSize>>> (this->nrow_, nthreads,
                                                         this->mat_.row_offset, this->mat_.col,
                                                         CUDAPtr(this->mat_.val),
                                                         CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_));
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void GPUAcceleratorMatrixMCSR<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
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

    dim3 BlockSize(this->local_backend_.GPU_block_size);
    dim3 GridSize(this->nrow_ / this->local_backend_.GPU_block_size + 1);

    int nnz_per_row = this->nnz_ / this->nrow_;
    int nthreads;

    if      (nnz_per_row <=   8) nthreads =  2;
    else if (nnz_per_row <=  16) nthreads =  4;
    else if (nnz_per_row <=  32) nthreads =  8;
    else if (nnz_per_row <=  64) nthreads = 16;
    else if (nnz_per_row <= 128) nthreads = 32;
    else                         nthreads = 64;

    kernel_mcsr_add_spmv<256, 32> <<<GridSize, BlockSize>>> (this->nrow_, nthreads,
                                                             this->mat_.row_offset, this->mat_.col,
                                                             CUDAPtr(this->mat_.val), CUDAVal(scalar),
                                                             CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_));
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

  }

}


template class GPUAcceleratorMatrixMCSR<double>;
template class GPUAcceleratorMatrixMCSR<float>;
#ifdef SUPPORT_COMPLEX
template class GPUAcceleratorMatrixMCSR<std::complex<double> >;
template class GPUAcceleratorMatrixMCSR<std::complex<float> >;
#endif

}
