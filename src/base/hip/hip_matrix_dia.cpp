#include "../../utils/def.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_matrix_dia.hpp"
#include "hip_vector.hpp"
#include "hip_conversion.hpp"
#include "../host/host_matrix_dia.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_dia.hpp"
#include "hip_kernels_vector.hpp"
#include "hip_allocate_free.hpp"
#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType>
HIPAcceleratorMatrixDIA<ValueType>::HIPAcceleratorMatrixDIA() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HIPAcceleratorMatrixDIA<ValueType>::HIPAcceleratorMatrixDIA(const Rocalution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "HIPAcceleratorMatrixDIA::HIPAcceleratorMatrixDIA()",
            "constructor with local_backend");

  this->mat_.val = NULL;
  this->mat_.offset = NULL;  
  this->mat_.num_diag = 0 ;
  this->set_backend(local_backend); 

  CHECK_HIP_ERROR(__FILE__, __LINE__);

}


template <typename ValueType>
HIPAcceleratorMatrixDIA<ValueType>::~HIPAcceleratorMatrixDIA() {

  LOG_DEBUG(this, "HIPAcceleratorMatrixDIA::HIPAcceleratorMatrixDIA()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void HIPAcceleratorMatrixDIA<ValueType>::Info(void) const {

  LOG_INFO("HIPAcceleratorMatrixDIA<ValueType> diag=" << this->GetNDiag() << " nnz=" << this->GetNnz() );

}

template <typename ValueType>
void HIPAcceleratorMatrixDIA<ValueType>::AllocateDIA(const int nnz, const int nrow, const int ncol, const int ndiag) {

  assert(nnz >= 0);
  assert(ncol >= 0);
  assert(nrow >= 0);

  if (this->GetNnz() > 0)
    this->Clear();

  if (nnz > 0) {

    assert(ndiag > 0);


    allocate_hip(nnz, &this->mat_.val);
    allocate_hip(ndiag, &this->mat_.offset);
 
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    nnz, mat_.val);
    
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    ndiag, mat_.offset);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;
    this->mat_.num_diag = ndiag;

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixDIA<ValueType>::SetDataPtrDIA(int **offset, ValueType **val,
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

  hipDeviceSynchronize();

  this->mat_.num_diag = num_diag;
  this->nrow_ = nrow;
  this->ncol_ = ncol;
  this->nnz_  = nnz;

  this->mat_.offset = *offset;
  this->mat_.val = *val;

}

template <typename ValueType>
void HIPAcceleratorMatrixDIA<ValueType>::LeaveDataPtrDIA(int **offset, ValueType **val, int &num_diag) {

  assert(this->nrow_ > 0);
  assert(this->ncol_ > 0);
  assert(this->nnz_ > 0);
  assert(this->mat_.num_diag > 0);

  if (this->nrow_ < this->ncol_) {
    assert(this->nnz_ == this->ncol_ * this->mat_.num_diag);
  } else {
    assert(this->nnz_ == this->nrow_ * this->mat_.num_diag);
  }

  hipDeviceSynchronize();

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
void HIPAcceleratorMatrixDIA<ValueType>::Clear() {

  if (this->GetNnz() > 0) {

    free_hip(&this->mat_.val);
    free_hip(&this->mat_.offset);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;
    this->mat_.num_diag = 0 ;

  }


}

template <typename ValueType>
void HIPAcceleratorMatrixDIA<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixDIA<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixDIA<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateDIA(cast_mat->GetNnz(), cast_mat->GetM(), cast_mat->GetN(), cast_mat->GetNDiag());

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    if (this->GetNnz() > 0) {

      hipMemcpy(this->mat_.offset,     // dst
                 cast_mat->mat_.offset, // src
                 this->GetNDiag()*sizeof(int), // size
                 hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.val,     // dst
                 cast_mat->mat_.val, // src
                 this->GetNnz()*sizeof(ValueType), // size
                 hipMemcpyHostToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     

    }
      
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixDIA<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixDIA<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixDIA<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->GetNnz() == 0)
    cast_mat->AllocateDIA(this->GetNnz(), this->GetM(), this->GetN(), this->GetNDiag());

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    if (this->GetNnz() > 0) {

      hipMemcpy(cast_mat->mat_.offset, // dst
                 this->mat_.offset,     // src
                 this->GetNDiag()*sizeof(int), // size
                 hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(cast_mat->mat_.val, // dst
                 this->mat_.val,     // src
                 this->GetNnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToHost);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->Info();
    dst->Info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixDIA<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixDIA<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixDIA<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateDIA(hip_cast_mat->GetNnz(), hip_cast_mat->GetM(), hip_cast_mat->GetN(), hip_cast_mat->GetNDiag());

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    if (this->GetNnz() > 0) {

      hipMemcpy(this->mat_.offset,         // dst
                 hip_cast_mat->mat_.offset, // src
                 this->GetNDiag()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.val,         // dst
                 hip_cast_mat->mat_.val, // src
               this->GetNnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

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
void HIPAcceleratorMatrixDIA<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixDIA<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixDIA<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->GetNnz() == 0)
    hip_cast_mat->AllocateDIA(hip_cast_mat->GetNnz(), hip_cast_mat->GetM(), hip_cast_mat->GetN(), hip_cast_mat->GetNDiag());

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    if (this->GetNnz() > 0) { 

      hipMemcpy(hip_cast_mat->mat_.offset, // dst
                 this->mat_.offset,         // src
                 this->GetNDiag()*sizeof(int), // size
                 hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.val, // dst
                 this->mat_.val,         // src
                 this->GetNnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToHost);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }
    
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
void HIPAcceleratorMatrixDIA<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  const HostMatrixDIA<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixDIA<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateDIA(cast_mat->GetNnz(), cast_mat->GetM(), cast_mat->GetN(), cast_mat->GetNDiag());

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    if (this->GetNnz() > 0) {

      hipMemcpyAsync(this->mat_.offset,     // dst
                      cast_mat->mat_.offset, // src
                      this->GetNDiag()*sizeof(int), // size
                      hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(this->mat_.val,     // dst
                      cast_mat->mat_.val, // src
                      this->GetNnz()*sizeof(ValueType), // size
                      hipMemcpyHostToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     

    }
      
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixDIA<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  HostMatrixDIA<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixDIA<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->GetNnz() == 0)
    cast_mat->AllocateDIA(this->GetNnz(), this->GetM(), this->GetN(), this->GetNDiag());

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    if (this->GetNnz() > 0) {

      hipMemcpyAsync(cast_mat->mat_.offset, // dst
                      this->mat_.offset,     // src
                      this->GetNDiag()*sizeof(int), // size
                      hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(cast_mat->mat_.val, // dst
                      this->mat_.val,     // src
                      this->GetNnz()*sizeof(ValueType), // size
                      hipMemcpyDeviceToHost);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->Info();
    dst->Info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixDIA<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixDIA<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixDIA<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateDIA(hip_cast_mat->GetNnz(), hip_cast_mat->GetM(), hip_cast_mat->GetN(), hip_cast_mat->GetNDiag());

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    if (this->GetNnz() > 0) {

      hipMemcpy(this->mat_.offset,         // dst
                 hip_cast_mat->mat_.offset, // src
                 this->GetNDiag()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.val,         // dst
                 hip_cast_mat->mat_.val, // src
               this->GetNnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

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
void HIPAcceleratorMatrixDIA<ValueType>::CopyToAsync(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixDIA<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixDIA<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->GetNnz() == 0)
    hip_cast_mat->AllocateDIA(hip_cast_mat->GetNnz(), hip_cast_mat->GetM(), hip_cast_mat->GetN(), hip_cast_mat->GetNDiag());

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    if (this->GetNnz() > 0) { 

      hipMemcpy(hip_cast_mat->mat_.offset, // dst
                 this->mat_.offset,         // src
                 this->GetNDiag()*sizeof(int), // size
                 hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.val, // dst
                 this->mat_.val,         // src
                 this->GetNnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToHost);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }
    
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
bool HIPAcceleratorMatrixDIA<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.GetNnz() == 0)
    return true;

  const HIPAcceleratorMatrixDIA<ValueType>   *cast_mat_dia;
  
  if ((cast_mat_dia = dynamic_cast<const HIPAcceleratorMatrixDIA<ValueType>*> (&mat)) != NULL) {

      this->CopyFrom(*cast_mat_dia);
      return true;

  }

  const HIPAcceleratorMatrixCSR<ValueType>   *cast_mat_csr;
  if ((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {

    this->Clear();

    int nnz_dia;
    int num_diag;

    if(csr_to_dia_hip(this->local_backend_.HIP_block_size,
                      cast_mat_csr->nnz_,
                      cast_mat_csr->nrow_,
                      cast_mat_csr->ncol_,
                      cast_mat_csr->mat_,
                      &this->mat_,
                      &nnz_dia,
                      &num_diag) == true)
    {

        this->nrow_ = cast_mat_csr->GetM();
        this->ncol_ = cast_mat_csr->GetN();
        this->nnz_  = nnz_dia;
        this->mat_.num_diag = num_diag;

        return true;
    }
  }

  return false;
}

template <typename ValueType>
void HIPAcceleratorMatrixDIA<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->GetNnz() > 0) {

    assert(in.  GetSize() >= 0);
    assert(out->GetSize() >= 0);
    assert(in.  GetSize() == this->GetN());
    assert(out->GetSize() == this->GetM());

    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in);
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    int nrow = this->GetM();
    int ncol = this->GetN();
    int num_diag = this->GetNDiag();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_dia_spmv<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       nrow, ncol, num_diag,
                       this->mat_.offset, this->mat_.val,
                       cast_in->vec_, cast_out->vec_);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixDIA<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
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

    int nrow = this->GetM();
    int ncol = this->GetN();
    int num_diag = this->GetNDiag();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_dia_add_spmv<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       nrow, ncol, num_diag,
                       this->mat_.offset, this->mat_.val,
                       scalar,
                       cast_in->vec_, cast_out->vec_);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}


template class HIPAcceleratorMatrixDIA<double>;
template class HIPAcceleratorMatrixDIA<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorMatrixDIA<std::complex<double> >;
template class HIPAcceleratorMatrixDIA<std::complex<float> >;
#endif

}
