#include "../../utils/def.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_matrix_ell.hpp"
#include "hip_vector.hpp"
#include "../host/host_matrix_ell.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "hip_utils.hpp"
#include "hip_allocate_free.hpp"
#include "../../utils/allocate_free.hpp"
#include "hip_sparse.hpp"
#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType>
HIPAcceleratorMatrixELL<ValueType>::HIPAcceleratorMatrixELL() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HIPAcceleratorMatrixELL<ValueType>::HIPAcceleratorMatrixELL(const Rocalution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "HIPAcceleratorMatrixELL::HIPAcceleratorMatrixELL()",
            "constructor with local_backend");

  this->mat_.val = NULL;
  this->mat_.col = NULL;
  this->mat_.max_row = 0;
  this->set_backend(local_backend); 

  this->mat_descr_ = 0;

  CHECK_HIP_ERROR(__FILE__, __LINE__);

  hipsparseStatus_t stat_t;
  
  stat_t = hipsparseCreateMatDescr(&this->mat_descr_);
  CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);
  
  stat_t = hipsparseSetMatIndexBase(this->mat_descr_, HIPSPARSE_INDEX_BASE_ZERO);
  CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);
  
  stat_t = hipsparseSetMatType(this->mat_descr_, HIPSPARSE_MATRIX_TYPE_GENERAL);
  CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

}


template <typename ValueType>
HIPAcceleratorMatrixELL<ValueType>::~HIPAcceleratorMatrixELL() {

  LOG_DEBUG(this, "HIPAcceleratorMatrixELL::~HIPAcceleratorMatrixELL()",
            "destructor");

  this->Clear();

  hipsparseStatus_t stat_t;

  stat_t = hipsparseDestroyMatDescr(this->mat_descr_);
  CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

}

template <typename ValueType>
void HIPAcceleratorMatrixELL<ValueType>::Info(void) const {

  LOG_INFO("HIPAcceleratorMatrixELL<ValueType>");

}

template <typename ValueType>
void HIPAcceleratorMatrixELL<ValueType>::AllocateELL(const int nnz, const int nrow, const int ncol, const int max_row) {

  assert(nnz   >= 0);
  assert(ncol  >= 0);
  assert(nrow  >= 0);
  assert(max_row >= 0);

  if (this->GetNnz() > 0)
    this->Clear();

  if (nnz > 0) {

    assert(nnz == max_row * nrow);

    allocate_hip(nnz, &this->mat_.val);
    allocate_hip(nnz, &this->mat_.col);
    
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    nnz, this->mat_.val);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    nnz, this->mat_.col);
    
    this->mat_.max_row = max_row;
    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixELL<ValueType>::Clear() {

  if (this->GetNnz() > 0) {

    free_hip(&this->mat_.val);
    free_hip(&this->mat_.col);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixELL<ValueType>::SetDataPtrELL(int **col, ValueType **val,
                                             const int nnz, const int nrow, const int ncol, const int max_row) {

  assert(*col != NULL);
  assert(*val != NULL);
  assert(nnz > 0);
  assert(nrow > 0);
  assert(ncol > 0);
  assert(max_row > 0);
  assert(max_row*nrow == nnz);

  this->Clear();

  hipDeviceSynchronize();

  this->mat_.max_row = max_row;
  this->nrow_ = nrow;
  this->ncol_ = ncol;
  this->nnz_  = nnz;

  this->mat_.col = *col;
  this->mat_.val = *val;

}

template <typename ValueType>
void HIPAcceleratorMatrixELL<ValueType>::LeaveDataPtrELL(int **col, ValueType **val, int &max_row) {

  assert(this->nrow_ > 0);
  assert(this->ncol_ > 0);
  assert(this->nnz_ > 0);
  assert(this->mat_.max_row > 0);
  assert(this->mat_.max_row*this->nrow_ == this->nnz_);

  hipDeviceSynchronize();

  // see free_host function for details
  *col = this->mat_.col;
  *val = this->mat_.val;

  this->mat_.col = NULL;
  this->mat_.val = NULL;

  max_row = this->mat_.max_row;

  this->mat_.max_row = 0;
  this->nrow_ = 0;
  this->ncol_ = 0;
  this->nnz_  = 0;

}




template <typename ValueType>
void HIPAcceleratorMatrixELL<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixELL<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixELL<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateELL(cast_mat->GetNnz(), cast_mat->GetM(), cast_mat->GetN(), cast_mat->get_max_row());

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    if (this->GetNnz() > 0) { 

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
    }
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixELL<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixELL<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixELL<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->GetNnz() == 0)
    cast_mat->AllocateELL(this->GetNnz(), this->GetM(), this->GetN(), this->get_max_row() );

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    if (this->GetNnz() > 0) {

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
    }
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->Info();
    dst->Info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixELL<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixELL<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixELL<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateELL(hip_cast_mat->GetNnz(), hip_cast_mat->GetM(), hip_cast_mat->GetN(), hip_cast_mat->get_max_row() );

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    if (this->GetNnz() > 0) {

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
void HIPAcceleratorMatrixELL<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixELL<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixELL<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->GetNnz() == 0)
    hip_cast_mat->AllocateELL(hip_cast_mat->GetNnz(), hip_cast_mat->GetM(), hip_cast_mat->GetN(), hip_cast_mat->get_max_row() );

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    if (this->GetNnz() > 0) {
      
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
void HIPAcceleratorMatrixELL<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  const HostMatrixELL<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixELL<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateELL(cast_mat->GetNnz(), cast_mat->GetM(), cast_mat->GetN(), cast_mat->get_max_row());

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    if (this->GetNnz() > 0) { 

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
    }
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixELL<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  HostMatrixELL<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixELL<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->GetNnz() == 0)
    cast_mat->AllocateELL(this->GetNnz(), this->GetM(), this->GetN(), this->get_max_row() );

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    if (this->GetNnz() > 0) {

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
    }
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->Info();
    dst->Info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixELL<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixELL<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixELL<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateELL(hip_cast_mat->GetNnz(), hip_cast_mat->GetM(), hip_cast_mat->GetN(), hip_cast_mat->get_max_row() );

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    if (this->GetNnz() > 0) {

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
void HIPAcceleratorMatrixELL<ValueType>::CopyToAsync(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixELL<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixELL<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->GetNnz() == 0)
    hip_cast_mat->AllocateELL(hip_cast_mat->GetNnz(), hip_cast_mat->GetM(), hip_cast_mat->GetN(), hip_cast_mat->get_max_row() );

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    if (this->GetNnz() > 0) {
      
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
bool HIPAcceleratorMatrixELL<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.GetNnz() == 0)
    return true;

  const HIPAcceleratorMatrixELL<ValueType> *cast_mat_ell;

  if ((cast_mat_ell = dynamic_cast<const HIPAcceleratorMatrixELL<ValueType>*> (&mat)) != NULL) {

    this->CopyFrom(*cast_mat_ell);
    return true;

  }

  const HIPAcceleratorMatrixCSR<ValueType>   *cast_mat_csr;
  if ((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {

    this->Clear();

    assert(cast_mat_csr->GetM() > 0);
    assert(cast_mat_csr->GetN() > 0);
    assert(cast_mat_csr->GetNnz() > 0);

    int nrow = cast_mat_csr->GetM();

    hipsparseStatus_t stat_t;

    stat_t = hipsparseXcsr2ellWidth(HIPSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                    nrow,
                                    cast_mat_csr->mat_descr_,
                                    cast_mat_csr->mat_.row_offset,
                                    this->mat_descr_,
                                    &this->mat_.max_row);
    CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    int ell_nnz = this->mat_.max_row * nrow;
    this->AllocateELL(ell_nnz, nrow, cast_mat_csr->GetN(), this->mat_.max_row);

    stat_t = hipsparseTcsr2ell(HIPSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                               nrow,
                               cast_mat_csr->mat_descr_,
                               cast_mat_csr->mat_.val,
                               cast_mat_csr->mat_.row_offset,
                               cast_mat_csr->mat_.col,
                               this->mat_descr_,
                               this->mat_.max_row,
                               this->mat_.val,
                               this->mat_.col);
    CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    this->nrow_ = cast_mat_csr->GetM();
    this->ncol_ = cast_mat_csr->GetN();
    this->nnz_  = ell_nnz;

    return true;

  }

  return false;

}

template <typename ValueType>
void HIPAcceleratorMatrixELL<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->GetNnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->GetN());
    assert(out->get_size() == this->GetM());

    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in);
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    hipsparseStatus_t stat_t;
    stat_t = hipsparseTellmv(HIPSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                             HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             this->GetM(), this->GetN(), &alpha,
                             this->mat_descr_,
                             this->mat_.val, this->mat_.col,
                             this->get_max_row(),
                             cast_in->vec_, &beta,
                             cast_out->vec_);
    CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixELL<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                                  BaseVector<ValueType> *out) const {

  if (this->GetNnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->GetN());
    assert(out->get_size() == this->GetM());

    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in);
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    ValueType beta = 1.0;

    hipsparseStatus_t stat_t;
    stat_t = hipsparseTellmv(HIPSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                             HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             this->GetM(), this->GetN(), &scalar,
                             this->mat_descr_,
                             this->mat_.val, this->mat_.col,
                             this->get_max_row(),
                             cast_in->vec_, &beta,
                             cast_out->vec_);
    CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  }

}


template class HIPAcceleratorMatrixELL<double>;
template class HIPAcceleratorMatrixELL<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorMatrixELL<std::complex<double> >;
template class HIPAcceleratorMatrixELL<std::complex<float> >;
#endif

}
