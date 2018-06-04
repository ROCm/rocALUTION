#include "../../utils/def.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_matrix_mcsr.hpp"
#include "hip_vector.hpp"
#include "../host/host_matrix_mcsr.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_mcsr.hpp"
#include "hip_allocate_free.hpp"
#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType>
HIPAcceleratorMatrixMCSR<ValueType>::HIPAcceleratorMatrixMCSR() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HIPAcceleratorMatrixMCSR<ValueType>::HIPAcceleratorMatrixMCSR(const Rocalution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "HIPAcceleratorMatrixMCSR::HIPAcceleratorMatrixMCSR()",
            "constructor with local_backend");

  this->mat_.row_offset = NULL;  
  this->mat_.col = NULL;  
  this->mat_.val = NULL;
  this->set_backend(local_backend); 

  CHECK_HIP_ERROR(__FILE__, __LINE__);

}


template <typename ValueType>
HIPAcceleratorMatrixMCSR<ValueType>::~HIPAcceleratorMatrixMCSR() {

  LOG_DEBUG(this, "HIPAcceleratorMatrixMCSR::~HIPAcceleratorMatrixMCSR()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::info(void) const {

  LOG_INFO("HIPAcceleratorMatrixMCSR<ValueType>");

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::AllocateMCSR(const int nnz, const int nrow, const int ncol) {

  assert(nnz >= 0);
  assert(ncol >= 0);
  assert(nrow >= 0);

  if (this->get_nnz() > 0)
    this->Clear();

  if (nnz > 0) {

    allocate_hip(nrow+1, &this->mat_.row_offset);
    allocate_hip(nnz,    &this->mat_.col);
    allocate_hip(nnz,    &this->mat_.val);
    
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    this->local_backend_.HIP_max_threads,
                    nrow+1, mat_.row_offset);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    this->local_backend_.HIP_max_threads,
                    nnz, mat_.col);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    this->local_backend_.HIP_max_threads,
                    nnz, mat_.val);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::SetDataPtrMCSR(int **row_offset, int **col, ValueType **val,
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

  hipDeviceSynchronize();

  this->mat_.row_offset = *row_offset;
  this->mat_.col = *col;
  this->mat_.val = *val;

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::LeaveDataPtrMCSR(int **row_offset, int **col, ValueType **val) {

  assert(this->get_nrow() > 0);
  assert(this->get_ncol() > 0);
  assert(this->get_nnz() > 0);

  hipDeviceSynchronize();

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
void HIPAcceleratorMatrixMCSR<ValueType>::Clear() {

  if (this->get_nnz() > 0) {

    free_hip(&this->mat_.row_offset);
    free_hip(&this->mat_.col);
    free_hip(&this->mat_.val);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

  }


}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixMCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateMCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    hipMemcpy(this->mat_.row_offset,     // dst
               cast_mat->mat_.row_offset, // src
               (this->get_nrow()+1)*sizeof(int), // size
               hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(this->mat_.col,     // dst
               cast_mat->mat_.col, // src
               this->get_nnz()*sizeof(int), // size
               hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(this->mat_.val,     // dst
               cast_mat->mat_.val, // src
               this->get_nnz()*sizeof(ValueType), // size
               hipMemcpyHostToDevice);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
    
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixMCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixMCSR<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateMCSR(this->get_nnz(), this->get_nrow(), this->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    hipMemcpy(cast_mat->mat_.row_offset, // dst
               this->mat_.row_offset,     // src
               (this->get_nrow()+1)*sizeof(int), // size
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(cast_mat->mat_.col, // dst
               this->mat_.col,     // src
               this->get_nnz()*sizeof(int), // size
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(cast_mat->mat_.val, // dst
               this->mat_.val,     // src
               this->get_nnz()*sizeof(ValueType), // size
               hipMemcpyDeviceToHost);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
   
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixMCSR<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateMCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    hipMemcpy(this->mat_.row_offset,         // dst
               hip_cast_mat->mat_.row_offset, // src
               (this->get_nrow()+1)*sizeof(int), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(this->mat_.col,         // dst
               hip_cast_mat->mat_.col, // src
               this->get_nnz()*sizeof(int), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(this->mat_.val,         // dst
               hip_cast_mat->mat_.val, // src
               this->get_nnz()*sizeof(ValueType), // size
               hipMemcpyDeviceToDevice);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
    
  } else {

    //CPU to HIP
    if ((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*> (&src)) != NULL) {
      
      this->CopyFromHost(*host_cast_mat);
      
    } else {
      
      LOG_INFO("Error unsupported HIP matrix type");
      this->info();
      src.info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixMCSR<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixMCSR<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    hip_cast_mat->AllocateMCSR(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    hipMemcpy(hip_cast_mat->mat_.row_offset, // dst
               this->mat_.row_offset,         // src
               (this->get_nrow()+1)*sizeof(int), // size
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(hip_cast_mat->mat_.col, // dst
               this->mat_.col,         // src
               this->get_nnz()*sizeof(int), // size
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(hip_cast_mat->mat_.val, // dst
               this->mat_.val,         // src
               this->get_nnz()*sizeof(ValueType), // size
               hipMemcpyDeviceToHost);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
   
    
  } else {

    //HIP to CPU
    if ((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst)) != NULL) {
      
      this->CopyToHost(host_cast_mat);

    } else {
      
      LOG_INFO("Error unsupported HIP matrix type");
      this->info();
      dst->info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }

  }


}


template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  const HostMatrixMCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateMCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    hipMemcpyAsync(this->mat_.row_offset,     // dst
                    cast_mat->mat_.row_offset, // src
                    (this->get_nrow()+1)*sizeof(int), // size
                    hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpyAsync(this->mat_.col,     // dst
                    cast_mat->mat_.col, // src
                    this->get_nnz()*sizeof(int), // size
                    hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpyAsync(this->mat_.val,     // dst
                    cast_mat->mat_.val, // src
                    this->get_nnz()*sizeof(ValueType), // size
                    hipMemcpyHostToDevice);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
    
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  HostMatrixMCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixMCSR<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateMCSR(this->get_nnz(), this->get_nrow(), this->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    hipMemcpyAsync(cast_mat->mat_.row_offset, // dst
                    this->mat_.row_offset,     // src
                    (this->get_nrow()+1)*sizeof(int), // size
                    hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpyAsync(cast_mat->mat_.col, // dst
                    this->mat_.col,     // src
                    this->get_nnz()*sizeof(int), // size
                    hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpyAsync(cast_mat->mat_.val, // dst
                    this->mat_.val,     // src
                    this->get_nnz()*sizeof(ValueType), // size
                    hipMemcpyDeviceToHost);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
   
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixMCSR<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateMCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    hipMemcpy(this->mat_.row_offset,         // dst
               hip_cast_mat->mat_.row_offset, // src
               (this->get_nrow()+1)*sizeof(int), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(this->mat_.col,         // dst
               hip_cast_mat->mat_.col, // src
               this->get_nnz()*sizeof(int), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(this->mat_.val,         // dst
               hip_cast_mat->mat_.val, // src
               this->get_nnz()*sizeof(ValueType), // size
               hipMemcpyDeviceToDevice);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
    
  } else {

    //CPU to HIP
    if ((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*> (&src)) != NULL) {
      
      this->CopyFromHostAsync(*host_cast_mat);
      
    } else {
      
      LOG_INFO("Error unsupported HIP matrix type");
      this->info();
      src.info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyToAsync(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixMCSR<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixMCSR<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    hip_cast_mat->AllocateMCSR(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    hipMemcpy(hip_cast_mat->mat_.row_offset, // dst
               this->mat_.row_offset,         // src
               (this->get_nrow()+1)*sizeof(int), // size
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(hip_cast_mat->mat_.col, // dst
               this->mat_.col,         // src
               this->get_nnz()*sizeof(int), // size
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(hip_cast_mat->mat_.val, // dst
               this->mat_.val,         // src
               this->get_nnz()*sizeof(ValueType), // size
               hipMemcpyDeviceToHost);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
   
    
  } else {

    //HIP to CPU
    if ((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst)) != NULL) {
      
      this->CopyToHostAsync(host_cast_mat);

    } else {
      
      LOG_INFO("Error unsupported HIP matrix type");
      this->info();
      dst->info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }

  }


}


template <typename ValueType>
bool HIPAcceleratorMatrixMCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const HIPAcceleratorMatrixMCSR<ValueType>   *cast_mat_mcsr;
  
  if ((cast_mat_mcsr = dynamic_cast<const HIPAcceleratorMatrixMCSR<ValueType>*> (&mat)) != NULL) {

      this->CopyFrom(*cast_mat_mcsr);
      return true;

  }

  /*
  const HIPAcceleratorMatrixCSR<ValueType>   *cast_mat_csr;
  if ((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {
    
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
void HIPAcceleratorMatrixMCSR<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in);
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    dim3 BlockSize(512);
    dim3 GridSize(this->nrow_ / 512 + 1);

    int nnz_per_row = this->nnz_ / this->nrow_;

    if (this->local_backend_.HIP_warp == 32)
    {
      if (nnz_per_row < 4)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 2, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 8)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 4, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 16)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 8, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 32)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 16, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val,
                           cast_in->vec_, cast_out->vec_);
      }
      else
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 32, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val,
                           cast_in->vec_, cast_out->vec_);
      }
    }
    else if (this->local_backend_.HIP_warp == 64)
    {
      if (nnz_per_row < 4)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 2, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 8)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 4, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 16)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 8, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 32)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 16, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 64)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 32, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val,
                           cast_in->vec_, cast_out->vec_);
      }
      else
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 64, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val,
                           cast_in->vec_, cast_out->vec_);
      }
    }
    else
    {
      // TODO error message
      FATAL_ERROR(__FILE__, __LINE__);
    }

    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                                   BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in);
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    dim3 BlockSize(512);
    dim3 GridSize(this->nrow_ / 512 + 1);

    int nnz_per_row = this->nnz_ / this->nrow_;

    if (this->local_backend_.HIP_warp == 32)
    {
      if (nnz_per_row < 4)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 2, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 8)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 4, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 16)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 8, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 32)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 16, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 32, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
    }
    else if (this->local_backend_.HIP_warp == 64)
    {
      if (nnz_per_row < 4)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 2, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 8)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 4, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 16)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 8, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 32)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 16, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 64)
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 32, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else
      {
        hipLaunchKernelGGL((kernel_mcsr_spmv<512, 64, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
    }
    else
    {
      // TODO error message
      FATAL_ERROR(__FILE__, __LINE__);
    }

    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}


template class HIPAcceleratorMatrixMCSR<double>;
template class HIPAcceleratorMatrixMCSR<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorMatrixMCSR<std::complex<double> >;
template class HIPAcceleratorMatrixMCSR<std::complex<float> >;
#endif

}
