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
void HIPAcceleratorMatrixMCSR<ValueType>::Info(void) const {

  LOG_INFO("HIPAcceleratorMatrixMCSR<ValueType>");

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::AllocateMCSR(const int nnz, const int nrow, const int ncol) {

  assert(nnz >= 0);
  assert(ncol >= 0);
  assert(nrow >= 0);

  if (this->GetNnz() > 0)
    this->Clear();

  if (nnz > 0) {

    allocate_hip(nrow+1, &this->mat_.row_offset);
    allocate_hip(nnz,    &this->mat_.col);
    allocate_hip(nnz,    &this->mat_.val);
    
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    nrow+1, mat_.row_offset);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    nnz, mat_.col);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
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

  assert(this->GetM() > 0);
  assert(this->GetN() > 0);
  assert(this->GetNnz() > 0);

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

  if (this->GetNnz() > 0) {

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
  assert(this->GetMatFormat() == src.GetMatFormat());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateMCSR(src.GetNnz(), src.GetM(), src.GetN() );

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    hipMemcpy(this->mat_.row_offset,     // dst
               cast_mat->mat_.row_offset, // src
               (this->GetM()+1)*sizeof(int), // size
               hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(this->mat_.col,     // dst
               cast_mat->mat_.col, // src
               this->GetNnz()*sizeof(int), // size
               hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(this->mat_.val,     // dst
               cast_mat->mat_.val, // src
               this->GetNnz()*sizeof(ValueType), // size
               hipMemcpyHostToDevice);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
    
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixMCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixMCSR<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->GetNnz() == 0)
    cast_mat->AllocateMCSR(this->GetNnz(), this->GetM(), this->GetN() );

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    hipMemcpy(cast_mat->mat_.row_offset, // dst
               this->mat_.row_offset,     // src
               (this->GetM()+1)*sizeof(int), // size
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(cast_mat->mat_.col, // dst
               this->mat_.col,     // src
               this->GetNnz()*sizeof(int), // size
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(cast_mat->mat_.val, // dst
               this->mat_.val,     // src
               this->GetNnz()*sizeof(ValueType), // size
               hipMemcpyDeviceToHost);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
   
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->Info();
    dst->Info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixMCSR<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateMCSR(src.GetNnz(), src.GetM(), src.GetN() );

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    hipMemcpy(this->mat_.row_offset,         // dst
               hip_cast_mat->mat_.row_offset, // src
               (this->GetM()+1)*sizeof(int), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(this->mat_.col,         // dst
               hip_cast_mat->mat_.col, // src
               this->GetNnz()*sizeof(int), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(this->mat_.val,         // dst
               hip_cast_mat->mat_.val, // src
               this->GetNnz()*sizeof(ValueType), // size
               hipMemcpyDeviceToDevice);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
    
  } else {

    //CPU to HIP
    if ((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*> (&src)) != NULL) {
      
      this->CopyFromHost(*host_cast_mat);
      
    } else {
      
      LOG_INFO("Error unsupported HIP matrix type");
      this->Info();
      src.Info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixMCSR<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixMCSR<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->GetNnz() == 0)
    hip_cast_mat->AllocateMCSR(dst->GetNnz(), dst->GetM(), dst->GetN() );

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    hipMemcpy(hip_cast_mat->mat_.row_offset, // dst
               this->mat_.row_offset,         // src
               (this->GetM()+1)*sizeof(int), // size
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(hip_cast_mat->mat_.col, // dst
               this->mat_.col,         // src
               this->GetNnz()*sizeof(int), // size
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(hip_cast_mat->mat_.val, // dst
               this->mat_.val,         // src
               this->GetNnz()*sizeof(ValueType), // size
               hipMemcpyDeviceToHost);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
   
    
  } else {

    //HIP to CPU
    if ((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst)) != NULL) {
      
      this->CopyToHost(host_cast_mat);

    } else {
      
      LOG_INFO("Error unsupported HIP matrix type");
      this->Info();
      dst->Info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }

  }


}


template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  const HostMatrixMCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateMCSR(src.GetNnz(), src.GetM(), src.GetN() );

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    hipMemcpyAsync(this->mat_.row_offset,     // dst
                    cast_mat->mat_.row_offset, // src
                    (this->GetM()+1)*sizeof(int), // size
                    hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpyAsync(this->mat_.col,     // dst
                    cast_mat->mat_.col, // src
                    this->GetNnz()*sizeof(int), // size
                    hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpyAsync(this->mat_.val,     // dst
                    cast_mat->mat_.val, // src
                    this->GetNnz()*sizeof(ValueType), // size
                    hipMemcpyHostToDevice);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
    
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  HostMatrixMCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixMCSR<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->GetNnz() == 0)
    cast_mat->AllocateMCSR(this->GetNnz(), this->GetM(), this->GetN() );

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    hipMemcpyAsync(cast_mat->mat_.row_offset, // dst
                    this->mat_.row_offset,     // src
                    (this->GetM()+1)*sizeof(int), // size
                    hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpyAsync(cast_mat->mat_.col, // dst
                    this->mat_.col,     // src
                    this->GetNnz()*sizeof(int), // size
                    hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpyAsync(cast_mat->mat_.val, // dst
                    this->mat_.val,     // src
                    this->GetNnz()*sizeof(ValueType), // size
                    hipMemcpyDeviceToHost);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
   
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->Info();
    dst->Info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixMCSR<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateMCSR(src.GetNnz(), src.GetM(), src.GetN() );

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    hipMemcpy(this->mat_.row_offset,         // dst
               hip_cast_mat->mat_.row_offset, // src
               (this->GetM()+1)*sizeof(int), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(this->mat_.col,         // dst
               hip_cast_mat->mat_.col, // src
               this->GetNnz()*sizeof(int), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(this->mat_.val,         // dst
               hip_cast_mat->mat_.val, // src
               this->GetNnz()*sizeof(ValueType), // size
               hipMemcpyDeviceToDevice);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
    
  } else {

    //CPU to HIP
    if ((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*> (&src)) != NULL) {
      
      this->CopyFromHostAsync(*host_cast_mat);
      
    } else {
      
      LOG_INFO("Error unsupported HIP matrix type");
      this->Info();
      src.Info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::CopyToAsync(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixMCSR<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixMCSR<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->GetNnz() == 0)
    hip_cast_mat->AllocateMCSR(dst->GetNnz(), dst->GetM(), dst->GetN() );

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    hipMemcpy(hip_cast_mat->mat_.row_offset, // dst
               this->mat_.row_offset,         // src
               (this->GetM()+1)*sizeof(int), // size
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(hip_cast_mat->mat_.col, // dst
               this->mat_.col,         // src
               this->GetNnz()*sizeof(int), // size
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);     

    hipMemcpy(hip_cast_mat->mat_.val, // dst
               this->mat_.val,         // src
               this->GetNnz()*sizeof(ValueType), // size
               hipMemcpyDeviceToHost);    
    CHECK_HIP_ERROR(__FILE__, __LINE__);     
   
    
  } else {

    //HIP to CPU
    if ((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst)) != NULL) {
      
      this->CopyToHostAsync(host_cast_mat);

    } else {
      
      LOG_INFO("Error unsupported HIP matrix type");
      this->Info();
      dst->Info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }

  }


}


template <typename ValueType>
bool HIPAcceleratorMatrixMCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.GetNnz() == 0)
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
    
    this->nrow_ = cast_mat_csr->GetM();
    this->ncol_ = cast_mat_csr->GetN();
    this->nnz_  = cast_mat_csr->GetNnz();
    
    return true;
    
  }
  */

  return false;

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->GetNnz() > 0) {

    assert(in.  GetSize() >= 0);
    assert(out->GetSize() >= 0);
    assert(in.  GetSize() == this->GetN());
    assert(out->GetSize() == this->GetM());

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
      LOG_INFO("Unsupported HIP warp size of " << this->local_backend_.HIP_warp);
      FATAL_ERROR(__FILE__, __LINE__);
    }

    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixMCSR<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                                   BaseVector<ValueType> *out) const {

  if (this->GetNnz() > 0) {

    assert(in.  GetSize() >= 0);
    assert(out->GetSize() >= 0);
    assert(in.  GetSize() == this->GetN());
    assert(out->GetSize() == this->GetM());

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
        hipLaunchKernelGGL((kernel_mcsr_add_spmv<512, 2, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 8)
      {
        hipLaunchKernelGGL((kernel_mcsr_add_spmv<512, 4, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 16)
      {
        hipLaunchKernelGGL((kernel_mcsr_add_spmv<512, 8, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 32)
      {
        hipLaunchKernelGGL((kernel_mcsr_add_spmv<512, 16, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else
      {
        hipLaunchKernelGGL((kernel_mcsr_add_spmv<512, 32, ValueType, int>),
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
        hipLaunchKernelGGL((kernel_mcsr_add_spmv<512, 2, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 8)
      {
        hipLaunchKernelGGL((kernel_mcsr_add_spmv<512, 4, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 16)
      {
        hipLaunchKernelGGL((kernel_mcsr_add_spmv<512, 8, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 32)
      {
        hipLaunchKernelGGL((kernel_mcsr_add_spmv<512, 16, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else if (nnz_per_row < 64)
      {
        hipLaunchKernelGGL((kernel_mcsr_add_spmv<512, 32, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
      else
      {
        hipLaunchKernelGGL((kernel_mcsr_add_spmv<512, 64, ValueType, int>),
                           GridSize, BlockSize, 0, 0,
                           this->nrow_,
                           this->mat_.row_offset, this->mat_.col,
                           this->mat_.val, scalar,
                           cast_in->vec_, cast_out->vec_);
      }
    }
    else
    {
      LOG_INFO("Unsupported HIP warp size of " << this->local_backend_.HIP_warp);
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
