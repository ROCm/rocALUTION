#include "../../utils/def.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_matrix_coo.hpp"
#include "hip_matrix_ell.hpp"
#include "hip_matrix_hyb.hpp"
#include "hip_vector.hpp"
#include "hip_conversion.hpp"
#include "../host/host_matrix_hyb.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_vector.hpp"
#include "hip_allocate_free.hpp"
#include "hip_sparse.hpp"
#include "../matrix_formats_ind.hpp"

#include <algorithm>

#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType>
HIPAcceleratorMatrixHYB<ValueType>::HIPAcceleratorMatrixHYB() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HIPAcceleratorMatrixHYB<ValueType>::HIPAcceleratorMatrixHYB(const Rocalution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "HIPAcceleratorMatrixHYB::HIPAcceleratorMatrixHYB()",
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

  this->ell_mat_descr_ = 0;
  this->coo_mat_descr_ = 0;

  CHECK_HIP_ERROR(__FILE__, __LINE__);

  rocsparse_status status;
  
  status = rocsparse_create_mat_descr(&this->ell_mat_descr_);
  CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
  
  status = rocsparse_set_mat_index_base(this->ell_mat_descr_, rocsparse_index_base_zero);
  CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
  
  status = rocsparse_set_mat_type(this->ell_mat_descr_, rocsparse_matrix_type_general);
  CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

  status = rocsparse_create_mat_descr(&this->coo_mat_descr_);
  CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
  
  status = rocsparse_set_mat_index_base(this->coo_mat_descr_, rocsparse_index_base_zero);
  CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
  
  status = rocsparse_set_mat_type(this->coo_mat_descr_, rocsparse_matrix_type_general);
  CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

}


template <typename ValueType>
HIPAcceleratorMatrixHYB<ValueType>::~HIPAcceleratorMatrixHYB() {

  LOG_DEBUG(this, "HIPAcceleratorMatrixHYB::~HIPAcceleratorMatrixHYB()",
            "destructor");

  this->Clear();

  rocsparse_status status;

  status = rocsparse_destroy_mat_descr(this->ell_mat_descr_);
  CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

  status = rocsparse_destroy_mat_descr(this->coo_mat_descr_);
  CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

}

template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::Info(void) const {

  LOG_INFO("HIPAcceleratorMatrixHYB<ValueType>");

}

template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::AllocateHYB(const int ell_nnz, const int coo_nnz, const int ell_max_row, 
                                                     const int nrow, const int ncol) {

  assert(ell_nnz   >= 0);
  assert(coo_nnz   >= 0);
  assert(ell_max_row >= 0);

  assert(ncol  >= 0);
  assert(nrow  >= 0);
  
  if (this->GetNnz() > 0)
    this->Clear();

  this->nrow_ = nrow;
  this->ncol_ = ncol;
  this->nnz_ = 0;

  if (ell_nnz > 0)
  {
    // ELL
    assert(ell_nnz == ell_max_row*nrow);

    allocate_hip(ell_nnz, &this->mat_.ELL.val);
    allocate_hip(ell_nnz, &this->mat_.ELL.col);
    
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    ell_nnz, this->mat_.ELL.val);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    ell_nnz, this->mat_.ELL.col);

    this->mat_.ELL.max_row = ell_max_row;
    this->ell_nnz_ = ell_nnz;
    this->nnz_ += ell_nnz;
  }

  if(coo_nnz > 0)
  {
    // COO
    allocate_hip(coo_nnz, &this->mat_.COO.row);
    allocate_hip(coo_nnz, &this->mat_.COO.col);
    allocate_hip(coo_nnz, &this->mat_.COO.val);
 
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    coo_nnz, this->mat_.COO.row);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    coo_nnz, this->mat_.COO.col);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    coo_nnz, this->mat_.COO.val);
    this->coo_nnz_ = coo_nnz;

    this->nnz_ += coo_nnz;
  }

}


template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::Clear() {

  if (this->GetNnz() > 0) {

    if(this->ell_nnz_ > 0)
    {
        free_hip(&this->mat_.ELL.val);
        free_hip(&this->mat_.ELL.col);

        this->ell_nnz_ = 0;
        this->mat_.ELL.max_row = 0;
    }

    if(this->coo_nnz_ > 0)
    {
        free_hip(&this->mat_.COO.row);
        free_hip(&this->mat_.COO.col);
        free_hip(&this->mat_.COO.val);

        this->coo_nnz_ = 0;
    }

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;
    
  }  

}

template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateHYB(cast_mat->GetELLNnz(), cast_mat->GetCOONnz(), cast_mat->GetELLMaxRow(),
                      cast_mat->GetM(), cast_mat->GetN());

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    if (this->GetELLNnz() > 0) {

      // ELL
      hipMemcpy(this->mat_.ELL.col,     // dst
                 cast_mat->mat_.ELL.col, // src
                 this->GetELLNnz()*sizeof(int), // size
                 hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.ELL.val,     // dst
                 cast_mat->mat_.ELL.val, // src
                 this->GetELLNnz()*sizeof(ValueType), // size
                 hipMemcpyHostToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

    if (this->GetCOONnz() > 0) {

      // COO
      hipMemcpy(this->mat_.COO.row,     // dst
                 cast_mat->mat_.COO.row, // src
                 (this->GetCOONnz())*sizeof(int), // size
                 hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.COO.col,     // dst
                 cast_mat->mat_.COO.col, // src
                 this->GetCOONnz()*sizeof(int), // size
                 hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.COO.val,     // dst
                 cast_mat->mat_.COO.val, // src
                 this->GetCOONnz()*sizeof(ValueType), // size
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
void HIPAcceleratorMatrixHYB<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixHYB<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->GetNnz() == 0)
    cast_mat->AllocateHYB(this->GetELLNnz(), this->GetCOONnz(), this->GetELLMaxRow(),
                      this->GetM(), this->GetN());

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    if (this->GetELLNnz() > 0) {
      
      // ELL
      hipMemcpy(cast_mat->mat_.ELL.col, // dst
                 this->mat_.ELL.col,     // src
                 this->GetELLNnz()*sizeof(int), // size
                 hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(cast_mat->mat_.ELL.val, // dst
                 this->mat_.ELL.val,     // src
                 this->GetELLNnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToHost);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }


    if (this->GetCOONnz() > 0) {

      // COO
      hipMemcpy(cast_mat->mat_.COO.row, // dst
                 this->mat_.COO.row,     // src
                 this->GetCOONnz()*sizeof(int), // size
                 hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(cast_mat->mat_.COO.col, // dst
                 this->mat_.COO.col,     // src
                 this->GetCOONnz()*sizeof(int), // size
                 hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(cast_mat->mat_.COO.val, // dst
                 this->mat_.COO.val,     // src
                 this->GetCOONnz()*sizeof(ValueType), // size
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
void HIPAcceleratorMatrixHYB<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixHYB<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateHYB(hip_cast_mat->GetELLNnz(), hip_cast_mat->GetCOONnz(), hip_cast_mat->GetELLMaxRow(),
                      hip_cast_mat->GetM(), hip_cast_mat->GetN());

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());


    if (this->GetELLNnz() > 0) {

      // ELL
      hipMemcpy(this->mat_.ELL.col,     // dst
                 hip_cast_mat->mat_.ELL.col, // src
                 this->GetELLNnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.ELL.val,     // dst
                 hip_cast_mat->mat_.ELL.val, // src
                 this->GetELLNnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

    if (this->GetCOONnz() > 0) {

      // COO
      hipMemcpy(this->mat_.COO.row,     // dst
                 hip_cast_mat->mat_.COO.row, // src
                 (this->GetCOONnz())*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.COO.col,     // dst
                 hip_cast_mat->mat_.COO.col, // src
                 this->GetCOONnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.COO.val,     // dst
                 hip_cast_mat->mat_.COO.val, // src
                 this->GetCOONnz()*sizeof(ValueType), // size
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
void HIPAcceleratorMatrixHYB<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixHYB<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixHYB<ValueType>*> (dst))) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->GetNnz() == 0)
    hip_cast_mat->AllocateHYB(this->GetELLNnz(), this->GetCOONnz(), this->GetELLMaxRow(),
                      this->GetM(), this->GetN());

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    if (this->GetELLNnz() > 0) {

      // ELL
      hipMemcpy(hip_cast_mat->mat_.ELL.col, // dst
                 this->mat_.ELL.col,     // src
                 this->GetELLNnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.ELL.val, // dst
                 this->mat_.ELL.val,     // src
                 this->GetELLNnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

    if (this->GetCOONnz() > 0) {

      // COO
      hipMemcpy(hip_cast_mat->mat_.COO.row, // dst
                 this->mat_.COO.row,     // src
                 this->GetCOONnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.COO.col, // dst
                 this->mat_.COO.col,     // src
                 this->GetCOONnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.COO.val, // dst
                 this->mat_.COO.val,     // src
                 this->GetCOONnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     

    }
   
    
  } else {

    //HIP to CPU
    if ((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst))) {
      
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
void HIPAcceleratorMatrixHYB<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  const HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateHYB(cast_mat->GetELLNnz(), cast_mat->GetCOONnz(), cast_mat->GetELLMaxRow(),
                      cast_mat->GetM(), cast_mat->GetN());

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());

    if (this->GetELLNnz() > 0) {

      // ELL
      hipMemcpyAsync(this->mat_.ELL.col,     // dst
                      cast_mat->mat_.ELL.col, // src
                      this->GetELLNnz()*sizeof(int), // size
                      hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(this->mat_.ELL.val,     // dst
                      cast_mat->mat_.ELL.val, // src
                      this->GetELLNnz()*sizeof(ValueType), // size
                      hipMemcpyHostToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

    if (this->GetCOONnz() > 0) {

      // COO
      hipMemcpyAsync(this->mat_.COO.row,     // dst
                      cast_mat->mat_.COO.row, // src
                      (this->GetCOONnz())*sizeof(int), // size
                      hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(this->mat_.COO.col,     // dst
                      cast_mat->mat_.COO.col, // src
                      this->GetCOONnz()*sizeof(int), // size
                      hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(this->mat_.COO.val,     // dst
                      cast_mat->mat_.COO.val, // src
                      this->GetCOONnz()*sizeof(ValueType), // size
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
void HIPAcceleratorMatrixHYB<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixHYB<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->GetNnz() == 0)
    cast_mat->AllocateHYB(this->GetELLNnz(), this->GetCOONnz(), this->GetELLMaxRow(),
                      this->GetM(), this->GetN());

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    if (this->GetELLNnz() > 0) {
      
      // ELL
      hipMemcpyAsync(cast_mat->mat_.ELL.col, // dst
                      this->mat_.ELL.col,     // src
                      this->GetELLNnz()*sizeof(int), // size
                      hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(cast_mat->mat_.ELL.val, // dst
                      this->mat_.ELL.val,     // src
                      this->GetELLNnz()*sizeof(ValueType), // size
                      hipMemcpyDeviceToHost);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }


    if (this->GetCOONnz() > 0) {

      // COO
      hipMemcpyAsync(cast_mat->mat_.COO.row, // dst
                      this->mat_.COO.row,     // src
                      this->GetCOONnz()*sizeof(int), // size
                      hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(cast_mat->mat_.COO.col, // dst
                      this->mat_.COO.col,     // src
                      this->GetCOONnz()*sizeof(int), // size
                      hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(cast_mat->mat_.COO.val, // dst
                      this->mat_.COO.val,     // src
                      this->GetCOONnz()*sizeof(ValueType), // size
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
void HIPAcceleratorMatrixHYB<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixHYB<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == src.GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->GetNnz() == 0)
    this->AllocateHYB(hip_cast_mat->GetELLNnz(), hip_cast_mat->GetCOONnz(), hip_cast_mat->GetELLMaxRow(),
                      hip_cast_mat->GetM(), hip_cast_mat->GetN());

    assert(this->GetNnz()  == src.GetNnz());
    assert(this->GetM() == src.GetM());
    assert(this->GetN() == src.GetN());


    if (this->GetELLNnz() > 0) {

      // ELL
      hipMemcpy(this->mat_.ELL.col,     // dst
                 hip_cast_mat->mat_.ELL.col, // src
                 this->GetELLNnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.ELL.val,     // dst
                 hip_cast_mat->mat_.ELL.val, // src
                 this->GetELLNnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

    if (this->GetCOONnz() > 0) {

      // COO
      hipMemcpy(this->mat_.COO.row,     // dst
                 hip_cast_mat->mat_.COO.row, // src
                 (this->GetCOONnz())*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.COO.col,     // dst
                 hip_cast_mat->mat_.COO.col, // src
                 this->GetCOONnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.COO.val,     // dst
                 hip_cast_mat->mat_.COO.val, // src
                 this->GetCOONnz()*sizeof(ValueType), // size
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
void HIPAcceleratorMatrixHYB<ValueType>::CopyToAsync(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixHYB<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->GetMatFormat() == dst->GetMatFormat());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixHYB<ValueType>*> (dst))) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->GetNnz() == 0)
    hip_cast_mat->AllocateHYB(this->GetELLNnz(), this->GetCOONnz(), this->GetELLMaxRow(),
                      this->GetM(), this->GetN());

    assert(this->GetNnz()  == dst->GetNnz());
    assert(this->GetM() == dst->GetM());
    assert(this->GetN() == dst->GetN());

    if (this->GetELLNnz() > 0) {

      // ELL
      hipMemcpy(hip_cast_mat->mat_.ELL.col, // dst
                 this->mat_.ELL.col,     // src
                 this->GetELLNnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.ELL.val, // dst
                 this->mat_.ELL.val,     // src
                 this->GetELLNnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

    if (this->GetCOONnz() > 0) {

      // COO
      hipMemcpy(hip_cast_mat->mat_.COO.row, // dst
                 this->mat_.COO.row,     // src
                 this->GetCOONnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.COO.col, // dst
                 this->mat_.COO.col,     // src
                 this->GetCOONnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.COO.val, // dst
                 this->mat_.COO.val,     // src
                 this->GetCOONnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     

    }
   
    
  } else {

    //HIP to CPU
    if ((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst))) {
      
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
bool HIPAcceleratorMatrixHYB<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat)
{
    this->Clear();

    // empty matrix is empty matrix
    if(mat.GetNnz() == 0)
    {
        return true;
    }

    const HIPAcceleratorMatrixHYB<ValueType>   *cast_mat_hyb;
  
    if((cast_mat_hyb = dynamic_cast<const HIPAcceleratorMatrixHYB<ValueType>*> (&mat)) != NULL)
    {
        this->CopyFrom(*cast_mat_hyb);
        return true;
    }

    const HIPAcceleratorMatrixCSR<ValueType> *cast_mat_csr;

    if((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL)
    {
        this->Clear();

        int nnz_hyb;
        int nnz_ell;
        int nnz_coo;

        if(csr_to_hyb_hip(this->local_backend_.HIP_block_size,
                          cast_mat_csr->nnz_,
                          cast_mat_csr->nrow_,
                          cast_mat_csr->ncol_,
                          cast_mat_csr->mat_,
                          &this->mat_,
                          &nnz_hyb,
                          &nnz_ell,
                          &nnz_coo) == true)
        {
            this->nrow_    = cast_mat_csr->nrow_;
            this->ncol_    = cast_mat_csr->ncol_;
            this->nnz_     = nnz_hyb;
            this->ell_nnz_ = nnz_ell;
            this->coo_nnz_ = nnz_coo;

            return true;
        }
    }

    return false;
}

template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->GetNnz() > 0) {
    
    assert(in.  GetSize() >= 0);
    assert(out->GetSize() >= 0);
    assert(in.  GetSize() == this->GetN());
    assert(out->GetSize() == this->GetM());
        
    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in) ; 
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out) ; 
    
    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    ValueType alpha = 1.0;

    // ELL
    if (this->ell_nnz_ > 0) {

        ValueType beta = 0.0;

        rocsparse_status status;
        status = rocsparseTellmv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                 rocsparse_operation_none,
                                 this->GetM(), this->GetN(), &alpha,
                                 this->ell_mat_descr_,
                                 this->mat_.ELL.val, this->mat_.ELL.col, this->GetELLMaxRow(),
                                 cast_in->vec_, &beta,
                                 cast_out->vec_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    }

    // COO
    if (this->coo_nnz_ > 0) {

        // Add to y from ELL part
        ValueType beta = 1.0;

        rocsparse_status status;
        status = rocsparseTcoomv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                 rocsparse_operation_none,
                                 this->GetM(), this->GetN(), this->coo_nnz_, &alpha,
                                 this->coo_mat_descr_,
                                 this->mat_.COO.val, this->mat_.COO.row, this->mat_.COO.col,
                                 cast_in->vec_, &beta,
                                 cast_out->vec_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                                  BaseVector<ValueType> *out) const {

  if (this->GetNnz() > 0) {

    assert(in.  GetSize() >= 0);
    assert(out->GetSize() >= 0);
    assert(in.  GetSize() == this->GetN());
    assert(out->GetSize() == this->GetM());

    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in) ; 
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out) ; 

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    // ELL
    if (this->ell_nnz_ > 0) {

        ValueType beta = 0.0;

        rocsparse_status status;
        status = rocsparseTellmv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                 rocsparse_operation_none,
                                 this->GetM(), this->GetN(), &scalar,
                                 this->ell_mat_descr_,
                                 this->mat_.ELL.val, this->mat_.ELL.col, this->GetELLMaxRow(),
                                 cast_in->vec_, &beta,
                                 cast_out->vec_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    }

    // COO
    if (this->coo_nnz_ > 0) {

        // Add to y from ELL part
        ValueType beta = 1.0;

        rocsparse_status status;
        status = rocsparseTcoomv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                 rocsparse_operation_none,
                                 this->GetM(), this->GetN(), this->coo_nnz_, &scalar,
                                 this->coo_mat_descr_,
                                 this->mat_.COO.val, this->mat_.COO.row, this->mat_.COO.col,
                                 cast_in->vec_, &beta,
                                 cast_out->vec_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

    }

  }

}


template class HIPAcceleratorMatrixHYB<double>;
template class HIPAcceleratorMatrixHYB<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorMatrixHYB<std::complex<double> >;
template class HIPAcceleratorMatrixHYB<std::complex<float> >;
#endif

}
