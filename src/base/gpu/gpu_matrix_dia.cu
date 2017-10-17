#include "../../utils/def.hpp"
#include "gpu_matrix_csr.hpp"
#include "gpu_matrix_dia.hpp"
#include "gpu_vector.hpp"
#include "../host/host_matrix_dia.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "gpu_utils.hpp"
#include "cuda_kernels_general.hpp"
#include "cuda_kernels_dia.hpp"
#include "cuda_kernels_vector.hpp"
#include "gpu_allocate_free.hpp"
#include "../matrix_formats_ind.hpp"

#include <cuda.h>
#include <cusparse_v2.h>
#include <cuComplex.h>

namespace paralution {

template <typename ValueType>
GPUAcceleratorMatrixDIA<ValueType>::GPUAcceleratorMatrixDIA() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
GPUAcceleratorMatrixDIA<ValueType>::GPUAcceleratorMatrixDIA(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "GPUAcceleratorMatrixDIA::GPUAcceleratorMatrixDIA()",
            "constructor with local_backend");

  this->mat_.val = NULL;
  this->mat_.offset = NULL;  
  this->mat_.num_diag = 0 ;
  this->set_backend(local_backend); 

  CHECK_CUDA_ERROR(__FILE__, __LINE__);

}


template <typename ValueType>
GPUAcceleratorMatrixDIA<ValueType>::~GPUAcceleratorMatrixDIA() {

  LOG_DEBUG(this, "GPUAcceleratorMatrixDIA::GPUAcceleratorMatrixDIA()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void GPUAcceleratorMatrixDIA<ValueType>::info(void) const {

  LOG_INFO("GPUAcceleratorMatrixDIA<ValueType> diag=" << this->get_ndiag() << " nnz=" << this->get_nnz() );

}

template <typename ValueType>
void GPUAcceleratorMatrixDIA<ValueType>::AllocateDIA(const int nnz, const int nrow, const int ncol, const int ndiag) {

  assert(nnz >= 0);
  assert(ncol >= 0);
  assert(nrow >= 0);

  if (this->get_nnz() > 0)
    this->Clear();

  if (nnz > 0) {

    assert(ndiag > 0);


    allocate_gpu(nnz, &this->mat_.val);
    allocate_gpu(ndiag, &this->mat_.offset);
 
    set_to_zero_gpu(this->local_backend_.GPU_block_size, 
                    this->local_backend_.GPU_max_threads,
                    nnz, mat_.val);
    
    set_to_zero_gpu(this->local_backend_.GPU_block_size, 
                    this->local_backend_.GPU_max_threads,
                    ndiag, mat_.offset);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;
    this->mat_.num_diag = ndiag;

  }

}

template <typename ValueType>
void GPUAcceleratorMatrixDIA<ValueType>::SetDataPtrDIA(int **offset, ValueType **val,
                                             const int nnz, const int nrow, const int ncol, const int num_diag) {

  assert(*offset != NULL);
  assert(*val != NULL);
  assert(nnz > 0);
  assert(nrow > 0);
  assert(ncol > 0);
  assert(num_diag > 0);

  if (nrow < ncol) {
    assert(nnz == ncol * num_diag);
  } else {
    assert(nnz == nrow * num_diag);
  }

  this->Clear();

  cudaDeviceSynchronize();

  this->mat_.num_diag = num_diag;
  this->nrow_ = nrow;
  this->ncol_ = ncol;
  this->nnz_  = nnz;

  this->mat_.offset = *offset;
  this->mat_.val = *val;

}

template <typename ValueType>
void GPUAcceleratorMatrixDIA<ValueType>::LeaveDataPtrDIA(int **offset, ValueType **val, int &num_diag) {

  assert(this->nrow_ > 0);
  assert(this->ncol_ > 0);
  assert(this->nnz_ > 0);
  assert(this->mat_.num_diag > 0);

  if (this->nrow_ < this->ncol_) {
    assert(this->nnz_ == this->ncol_ * this->mat_.num_diag);
  } else {
    assert(this->nnz_ == this->nrow_ * this->mat_.num_diag);
  }

  cudaDeviceSynchronize();

  // see free_host function for details
  *offset = this->mat_.offset;
  *val = this->mat_.val;

  this->mat_.offset = NULL;
  this->mat_.val = NULL;

  num_diag = this->mat_.num_diag;

  this->mat_.num_diag = 0;
  this->nrow_ = 0;
  this->ncol_ = 0;
  this->nnz_  = 0;

}

template <typename ValueType>
void GPUAcceleratorMatrixDIA<ValueType>::Clear() {

  if (this->get_nnz() > 0) {

    free_gpu(&this->mat_.val);
    free_gpu(&this->mat_.offset);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;
    this->mat_.num_diag = 0 ;

  }


}

template <typename ValueType>
void GPUAcceleratorMatrixDIA<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixDIA<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to GPU copy
  if ((cast_mat = dynamic_cast<const HostMatrixDIA<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateDIA(cast_mat->get_nnz(), cast_mat->get_nrow(), cast_mat->get_ncol(), cast_mat->get_ndiag());

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpy(this->mat_.offset,     // dst
                 cast_mat->mat_.offset, // src
                 this->get_ndiag()*sizeof(int), // size
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
void GPUAcceleratorMatrixDIA<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixDIA<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixDIA<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateDIA(this->get_nnz(), this->get_nrow(), this->get_ncol(), this->get_ndiag());

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpy(cast_mat->mat_.offset, // dst
                 this->mat_.offset,     // src
                 this->get_ndiag()*sizeof(int), // size
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
void GPUAcceleratorMatrixDIA<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const GPUAcceleratorMatrixDIA<ValueType> *gpu_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<const GPUAcceleratorMatrixDIA<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateDIA(gpu_cast_mat->get_nnz(), gpu_cast_mat->get_nrow(), gpu_cast_mat->get_ncol(), gpu_cast_mat->get_ndiag());

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpy(this->mat_.offset,         // dst
                 gpu_cast_mat->mat_.offset, // src
                 this->get_ndiag()*sizeof(int), // size
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
void GPUAcceleratorMatrixDIA<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  GPUAcceleratorMatrixDIA<ValueType> *gpu_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<GPUAcceleratorMatrixDIA<ValueType>*> (dst)) != NULL) {

    gpu_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    gpu_cast_mat->AllocateDIA(gpu_cast_mat->get_nnz(), gpu_cast_mat->get_nrow(), gpu_cast_mat->get_ncol(), gpu_cast_mat->get_ndiag());

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_nnz() > 0) { 

      cudaMemcpy(gpu_cast_mat->mat_.offset, // dst
                 this->mat_.offset,         // src
                 this->get_ndiag()*sizeof(int), // size
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
void GPUAcceleratorMatrixDIA<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  const HostMatrixDIA<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to GPU copy
  if ((cast_mat = dynamic_cast<const HostMatrixDIA<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateDIA(cast_mat->get_nnz(), cast_mat->get_nrow(), cast_mat->get_ncol(), cast_mat->get_ndiag());

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpyAsync(this->mat_.offset,     // dst
                      cast_mat->mat_.offset, // src
                      this->get_ndiag()*sizeof(int), // size
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
void GPUAcceleratorMatrixDIA<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  HostMatrixDIA<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixDIA<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateDIA(this->get_nnz(), this->get_nrow(), this->get_ncol(), this->get_ndiag());

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpyAsync(cast_mat->mat_.offset, // dst
                      this->mat_.offset,     // src
                      this->get_ndiag()*sizeof(int), // size
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
void GPUAcceleratorMatrixDIA<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &src) {

  const GPUAcceleratorMatrixDIA<ValueType> *gpu_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<const GPUAcceleratorMatrixDIA<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateDIA(gpu_cast_mat->get_nnz(), gpu_cast_mat->get_nrow(), gpu_cast_mat->get_ncol(), gpu_cast_mat->get_ndiag());

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpy(this->mat_.offset,         // dst
                 gpu_cast_mat->mat_.offset, // src
                 this->get_ndiag()*sizeof(int), // size
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
void GPUAcceleratorMatrixDIA<ValueType>::CopyToAsync(BaseMatrix<ValueType> *dst) const {

  GPUAcceleratorMatrixDIA<ValueType> *gpu_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<GPUAcceleratorMatrixDIA<ValueType>*> (dst)) != NULL) {

    gpu_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    gpu_cast_mat->AllocateDIA(gpu_cast_mat->get_nnz(), gpu_cast_mat->get_nrow(), gpu_cast_mat->get_ncol(), gpu_cast_mat->get_ndiag());

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_nnz() > 0) { 

      cudaMemcpy(gpu_cast_mat->mat_.offset, // dst
                 this->mat_.offset,         // src
                 this->get_ndiag()*sizeof(int), // size
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
bool GPUAcceleratorMatrixDIA<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const GPUAcceleratorMatrixDIA<ValueType>   *cast_mat_dia;
  
  if ((cast_mat_dia = dynamic_cast<const GPUAcceleratorMatrixDIA<ValueType>*> (&mat)) != NULL) {

      this->CopyFrom(*cast_mat_dia);
      return true;

  }

  const GPUAcceleratorMatrixCSR<ValueType>   *cast_mat_csr;
  if ((cast_mat_csr = dynamic_cast<const GPUAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {

    this->Clear();

    // TODO
    // upper bound (somehow fixed for now)
    //
    //     GROUP_SIZE = ( size_t( ( size_t( nrow+ncol / ( this->local_backend_.GPU_warp * 4 ) ) + 1 ) 
    //                  / this->local_backend_.GPU_block_size ) + 1 ) * this->local_backend_.GPU_block_size;
    //
    if (cast_mat_csr->get_nrow()+cast_mat_csr->get_ncol() > 16842494*4)
      return false;


    int nrow = cast_mat_csr->get_nrow();
    int ncol = cast_mat_csr->get_ncol();
    int *diag_map = NULL;

    // DIA does not support non-squared matrices
    if (cast_mat_csr->nrow_ != cast_mat_csr->ncol_)
      return false;

    // Get diagonal mapping vector
    allocate_gpu<int>(nrow+ncol, &diag_map);

    set_to_zero_gpu(this->local_backend_.GPU_block_size,
                    this->local_backend_.GPU_max_threads,
                    nrow+ncol, diag_map);

    dim3 BlockSize(this->local_backend_.GPU_block_size);
    dim3 GridSize(nrow / this->local_backend_.GPU_block_size + 1);

    kernel_dia_diag_map<int> <<<GridSize, BlockSize>>> (nrow, cast_mat_csr->mat_.row_offset,
                                                        cast_mat_csr->mat_.col, diag_map);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    // Reduction to obtain number of occupied diagonals
    int num_diag = 0;
    int *d_buffer = NULL;
    int *h_buffer = NULL;

    allocate_gpu<int>(this->local_backend_.GPU_warp, &d_buffer);
    allocate_host(this->local_backend_.GPU_warp, &h_buffer);

    reduce_cuda<int, int, 32, 256>(nrow+ncol, diag_map, &num_diag, h_buffer, d_buffer);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    free_gpu<int>(&d_buffer);
    free_host(&h_buffer);

    // Conversion fails if number of diagonal is too large
    if (num_diag > 200) {
      free_gpu<int>(&diag_map);
      return false;
    }

    int nnz_dia;
    if (nrow < ncol)
      nnz_dia = ncol * num_diag;
    else
      nnz_dia = nrow * num_diag;

    // Allocate DIA structure
    this->AllocateDIA(nnz_dia, nrow, ncol, num_diag);

    set_to_zero_gpu(this->local_backend_.GPU_block_size,
                    this->local_backend_.GPU_max_threads,
                    nnz_dia, this->mat_.val);
    set_to_zero_gpu(this->local_backend_.GPU_block_size,
                    this->local_backend_.GPU_max_threads,
                    num_diag, this->mat_.offset);

    // Fill diagonal offset array
    allocate_gpu<int>(nrow+ncol+1, &d_buffer);

    // TODO currently performing partial sum on host
    allocate_host(nrow+ncol+1, &h_buffer);
    cudaMemcpy(h_buffer+1, // dst
               diag_map, // src
               (nrow+ncol)*sizeof(int), // size
               cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    h_buffer[0] = 0;
    for (int i=2; i<nrow+ncol+1; ++i)
      h_buffer[i] += h_buffer[i-1];

    cudaMemcpy(d_buffer, // dst
               h_buffer, // src
               (nrow+ncol)*sizeof(int), // size
               cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    free_host(&h_buffer);
    // end TODO

    // TODO
    // fix the numbers (not hardcoded)
    //
    if (cast_mat_csr->get_nrow()+cast_mat_csr->get_ncol() > 16842494) {
      
      // Large systems
      // 2D indexing

      int d2_bs = 16;
    
      int gsize1 = 65535;
      int gsize2 = ((nrow+ncol)/(65535*d2_bs))/d2_bs + 1;
    
      
      dim3 GridSize3(gsize1, 
                     gsize2);
      
      dim3 BlockSize3(d2_bs, 
                      d2_bs);
      
      kernel_dia_fill_offset<int> <<<GridSize3, BlockSize3>>> (nrow, ncol, diag_map,
                                                               d_buffer, this->mat_.offset);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);

    } else {

      // Small systems
      // 1D indexing

      dim3 GridSize3((nrow+ncol) / this->local_backend_.GPU_block_size + 1);

      kernel_dia_fill_offset<int> <<<GridSize3, BlockSize>>> (nrow, ncol, diag_map,
                                                              d_buffer, this->mat_.offset);
      CHECK_CUDA_ERROR(__FILE__, __LINE__);

    }

    free_gpu<int>(&d_buffer);

    kernel_dia_convert<ValueType, int> <<<GridSize, BlockSize>>> (nrow, num_diag, cast_mat_csr->mat_.row_offset,
                                                                  cast_mat_csr->mat_.col, cast_mat_csr->mat_.val,
                                                                  diag_map, this->mat_.val);
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

    free_gpu<int>(&diag_map);

    this->nrow_ = cast_mat_csr->get_nrow();
    this->ncol_ = cast_mat_csr->get_ncol();
    this->nnz_  = nnz_dia;
    this->mat_.num_diag = num_diag;

    return true;

  }

  return false;

}

template <typename ValueType>
void GPUAcceleratorMatrixDIA<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const GPUAcceleratorVector<ValueType> *cast_in = dynamic_cast<const GPUAcceleratorVector<ValueType>*> (&in);
    GPUAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      GPUAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    int nrow = this->get_nrow();
    int ncol = this->get_ncol();
    int num_diag = this->get_ndiag();
    dim3 BlockSize(this->local_backend_.GPU_block_size);
    dim3 GridSize(nrow / this->local_backend_.GPU_block_size + 1);

    kernel_dia_spmv<<<GridSize, BlockSize>>> (nrow, ncol, num_diag,
                                              this->mat_.offset, CUDAPtr(this->mat_.val),
                                              CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_));
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void GPUAcceleratorMatrixDIA<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
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

    int nrow = this->get_nrow();
    int ncol = this->get_ncol();
    int num_diag = this->get_ndiag();
    dim3 BlockSize(this->local_backend_.GPU_block_size);
    dim3 GridSize(nrow / this->local_backend_.GPU_block_size + 1);

    kernel_dia_add_spmv<<<GridSize, BlockSize>>> (nrow, ncol, num_diag,
                                                  this->mat_.offset, CUDAPtr(this->mat_.val),
                                                  CUDAVal(scalar),
                                                  CUDAPtr(cast_in->vec_), CUDAPtr(cast_out->vec_));
    CHECK_CUDA_ERROR(__FILE__, __LINE__);

  }

}


template class GPUAcceleratorMatrixDIA<double>;
template class GPUAcceleratorMatrixDIA<float>;
#ifdef SUPPORT_COMPLEX
template class GPUAcceleratorMatrixDIA<std::complex<double> >;
template class GPUAcceleratorMatrixDIA<std::complex<float> >;
#endif

}
