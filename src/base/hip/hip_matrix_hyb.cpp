#include "../../utils/def.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_matrix_coo.hpp"
#include "hip_matrix_ell.hpp"
#include "hip_matrix_hyb.hpp"
#include "hip_vector.hpp"
#include "../host/host_matrix_hyb.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_hyb.hpp"
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

  hipsparseStatus_t stat_t;
  
  stat_t = hipsparseCreateMatDescr(&this->ell_mat_descr_);
  CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);
  
  stat_t = hipsparseSetMatIndexBase(this->ell_mat_descr_, HIPSPARSE_INDEX_BASE_ZERO);
  CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);
  
  stat_t = hipsparseSetMatType(this->ell_mat_descr_, HIPSPARSE_MATRIX_TYPE_GENERAL);
  CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  stat_t = hipsparseCreateMatDescr(&this->coo_mat_descr_);
  CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);
  
  stat_t = hipsparseSetMatIndexBase(this->coo_mat_descr_, HIPSPARSE_INDEX_BASE_ZERO);
  CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);
  
  stat_t = hipsparseSetMatType(this->coo_mat_descr_, HIPSPARSE_MATRIX_TYPE_GENERAL);
  CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

}


template <typename ValueType>
HIPAcceleratorMatrixHYB<ValueType>::~HIPAcceleratorMatrixHYB() {

  LOG_DEBUG(this, "HIPAcceleratorMatrixHYB::~HIPAcceleratorMatrixHYB()",
            "destructor");

  this->Clear();

  hipsparseStatus_t stat_t;

  stat_t = hipsparseDestroyMatDescr(this->ell_mat_descr_);
  CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  stat_t = hipsparseDestroyMatDescr(this->coo_mat_descr_);
  CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

}

template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::info(void) const {

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
  
  if (this->get_nnz() > 0)
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
                    this->local_backend_.HIP_max_threads,
                    ell_nnz, this->mat_.ELL.val);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    this->local_backend_.HIP_max_threads,
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
                    this->local_backend_.HIP_max_threads,
                    coo_nnz, this->mat_.COO.row);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    this->local_backend_.HIP_max_threads,
                    coo_nnz, this->mat_.COO.col);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    this->local_backend_.HIP_max_threads,
                    coo_nnz, this->mat_.COO.val);
    this->coo_nnz_ = coo_nnz;

    this->nnz_ += coo_nnz;
  }

}


template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::Clear() {

  if (this->get_nnz() > 0) {

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
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateHYB(cast_mat->get_ell_nnz(), cast_mat->get_coo_nnz(), cast_mat->get_ell_max_row(),
                      cast_mat->get_nrow(), cast_mat->get_ncol());

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_ell_nnz() > 0) {

      // ELL
      hipMemcpy(this->mat_.ELL.col,     // dst
                 cast_mat->mat_.ELL.col, // src
                 this->get_ell_nnz()*sizeof(int), // size
                 hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.ELL.val,     // dst
                 cast_mat->mat_.ELL.val, // src
                 this->get_ell_nnz()*sizeof(ValueType), // size
                 hipMemcpyHostToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

    if (this->get_coo_nnz() > 0) {

      // COO
      hipMemcpy(this->mat_.COO.row,     // dst
                 cast_mat->mat_.COO.row, // src
                 (this->get_coo_nnz())*sizeof(int), // size
                 hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.COO.col,     // dst
                 cast_mat->mat_.COO.col, // src
                 this->get_coo_nnz()*sizeof(int), // size
                 hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.COO.val,     // dst
                 cast_mat->mat_.COO.val, // src
                 this->get_coo_nnz()*sizeof(ValueType), // size
                 hipMemcpyHostToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to CPU copy
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
      hipMemcpy(cast_mat->mat_.ELL.col, // dst
                 this->mat_.ELL.col,     // src
                 this->get_ell_nnz()*sizeof(int), // size
                 hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(cast_mat->mat_.ELL.val, // dst
                 this->mat_.ELL.val,     // src
                 this->get_ell_nnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToHost);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }


    if (this->get_coo_nnz() > 0) {

      // COO
      hipMemcpy(cast_mat->mat_.COO.row, // dst
                 this->mat_.COO.row,     // src
                 this->get_coo_nnz()*sizeof(int), // size
                 hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(cast_mat->mat_.COO.col, // dst
                 this->mat_.COO.col,     // src
                 this->get_coo_nnz()*sizeof(int), // size
                 hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(cast_mat->mat_.COO.val, // dst
                 this->mat_.COO.val,     // src
                 this->get_coo_nnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToHost);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
    }

  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixHYB<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateHYB(hip_cast_mat->get_ell_nnz(), hip_cast_mat->get_coo_nnz(), hip_cast_mat->get_ell_max_row(),
                      hip_cast_mat->get_nrow(), hip_cast_mat->get_ncol());

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());


    if (this->get_ell_nnz() > 0) {

      // ELL
      hipMemcpy(this->mat_.ELL.col,     // dst
                 hip_cast_mat->mat_.ELL.col, // src
                 this->get_ell_nnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.ELL.val,     // dst
                 hip_cast_mat->mat_.ELL.val, // src
                 this->get_ell_nnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

    if (this->get_coo_nnz() > 0) {

      // COO
      hipMemcpy(this->mat_.COO.row,     // dst
                 hip_cast_mat->mat_.COO.row, // src
                 (this->get_coo_nnz())*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.COO.col,     // dst
                 hip_cast_mat->mat_.COO.col, // src
                 this->get_coo_nnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.COO.val,     // dst
                 hip_cast_mat->mat_.COO.val, // src
                 this->get_coo_nnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     

    }
      
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
void HIPAcceleratorMatrixHYB<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixHYB<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixHYB<ValueType>*> (dst))) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    hip_cast_mat->AllocateHYB(this->get_ell_nnz(), this->get_coo_nnz(), this->get_ell_max_row(),
                      this->get_nrow(), this->get_ncol());

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_ell_nnz() > 0) {

      // ELL
      hipMemcpy(hip_cast_mat->mat_.ELL.col, // dst
                 this->mat_.ELL.col,     // src
                 this->get_ell_nnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.ELL.val, // dst
                 this->mat_.ELL.val,     // src
                 this->get_ell_nnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

    if (this->get_coo_nnz() > 0) {

      // COO
      hipMemcpy(hip_cast_mat->mat_.COO.row, // dst
                 this->mat_.COO.row,     // src
                 this->get_coo_nnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.COO.col, // dst
                 this->mat_.COO.col,     // src
                 this->get_coo_nnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.COO.val, // dst
                 this->mat_.COO.val,     // src
                 this->get_coo_nnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     

    }
   
    
  } else {

    //HIP to CPU
    if ((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst))) {
      
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
void HIPAcceleratorMatrixHYB<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  const HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateHYB(cast_mat->get_ell_nnz(), cast_mat->get_coo_nnz(), cast_mat->get_ell_max_row(),
                      cast_mat->get_nrow(), cast_mat->get_ncol());

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_ell_nnz() > 0) {

      // ELL
      hipMemcpyAsync(this->mat_.ELL.col,     // dst
                      cast_mat->mat_.ELL.col, // src
                      this->get_ell_nnz()*sizeof(int), // size
                      hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(this->mat_.ELL.val,     // dst
                      cast_mat->mat_.ELL.val, // src
                      this->get_ell_nnz()*sizeof(ValueType), // size
                      hipMemcpyHostToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

    if (this->get_coo_nnz() > 0) {

      // COO
      hipMemcpyAsync(this->mat_.COO.row,     // dst
                      cast_mat->mat_.COO.row, // src
                      (this->get_coo_nnz())*sizeof(int), // size
                      hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(this->mat_.COO.col,     // dst
                      cast_mat->mat_.COO.col, // src
                      this->get_coo_nnz()*sizeof(int), // size
                      hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(this->mat_.COO.val,     // dst
                      cast_mat->mat_.COO.val, // src
                      this->get_coo_nnz()*sizeof(ValueType), // size
                      hipMemcpyHostToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to CPU copy
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
      hipMemcpyAsync(cast_mat->mat_.ELL.col, // dst
                      this->mat_.ELL.col,     // src
                      this->get_ell_nnz()*sizeof(int), // size
                      hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(cast_mat->mat_.ELL.val, // dst
                      this->mat_.ELL.val,     // src
                      this->get_ell_nnz()*sizeof(ValueType), // size
                      hipMemcpyDeviceToHost);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }


    if (this->get_coo_nnz() > 0) {

      // COO
      hipMemcpyAsync(cast_mat->mat_.COO.row, // dst
                      this->mat_.COO.row,     // src
                      this->get_coo_nnz()*sizeof(int), // size
                      hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(cast_mat->mat_.COO.col, // dst
                      this->mat_.COO.col,     // src
                      this->get_coo_nnz()*sizeof(int), // size
                      hipMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpyAsync(cast_mat->mat_.COO.val, // dst
                      this->mat_.COO.val,     // src
                      this->get_coo_nnz()*sizeof(ValueType), // size
                      hipMemcpyDeviceToHost);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
    }

  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixHYB<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateHYB(hip_cast_mat->get_ell_nnz(), hip_cast_mat->get_coo_nnz(), hip_cast_mat->get_ell_max_row(),
                      hip_cast_mat->get_nrow(), hip_cast_mat->get_ncol());

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());


    if (this->get_ell_nnz() > 0) {

      // ELL
      hipMemcpy(this->mat_.ELL.col,     // dst
                 hip_cast_mat->mat_.ELL.col, // src
                 this->get_ell_nnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.ELL.val,     // dst
                 hip_cast_mat->mat_.ELL.val, // src
                 this->get_ell_nnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

    if (this->get_coo_nnz() > 0) {

      // COO
      hipMemcpy(this->mat_.COO.row,     // dst
                 hip_cast_mat->mat_.COO.row, // src
                 (this->get_coo_nnz())*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.COO.col,     // dst
                 hip_cast_mat->mat_.COO.col, // src
                 this->get_coo_nnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(this->mat_.COO.val,     // dst
                 hip_cast_mat->mat_.COO.val, // src
                 this->get_coo_nnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     

    }
      
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
void HIPAcceleratorMatrixHYB<ValueType>::CopyToAsync(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixHYB<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixHYB<ValueType>*> (dst))) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    hip_cast_mat->AllocateHYB(this->get_ell_nnz(), this->get_coo_nnz(), this->get_ell_max_row(),
                      this->get_nrow(), this->get_ncol());

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_ell_nnz() > 0) {

      // ELL
      hipMemcpy(hip_cast_mat->mat_.ELL.col, // dst
                 this->mat_.ELL.col,     // src
                 this->get_ell_nnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.ELL.val, // dst
                 this->mat_.ELL.val,     // src
                 this->get_ell_nnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
    }

    if (this->get_coo_nnz() > 0) {

      // COO
      hipMemcpy(hip_cast_mat->mat_.COO.row, // dst
                 this->mat_.COO.row,     // src
                 this->get_coo_nnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.COO.col, // dst
                 this->mat_.COO.col,     // src
                 this->get_coo_nnz()*sizeof(int), // size
                 hipMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      hipMemcpy(hip_cast_mat->mat_.COO.val, // dst
                 this->mat_.COO.val,     // src
                 this->get_coo_nnz()*sizeof(ValueType), // size
                 hipMemcpyDeviceToDevice);    
      CHECK_HIP_ERROR(__FILE__, __LINE__);     

    }
   
    
  } else {

    //HIP to CPU
    if ((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst))) {
      
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
bool HIPAcceleratorMatrixHYB<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const HIPAcceleratorMatrixHYB<ValueType>   *cast_mat_hyb;
  
  if ((cast_mat_hyb = dynamic_cast<const HIPAcceleratorMatrixHYB<ValueType>*> (&mat)) != NULL) {

    this->CopyFrom(*cast_mat_hyb);
    return true;

  }

  const HIPAcceleratorMatrixCSR<ValueType> *cast_mat_csr;

  if ((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {

    this->Clear();

    int nrow = cast_mat_csr->get_nrow();
    int ncol = cast_mat_csr->get_ncol();
    int nnz  = cast_mat_csr->get_nnz();

    assert(nrow > 0);
    assert(ncol > 0);
    assert(nnz > 0);

    // Determine ELL width by average nnz per row
    int max_row = (nnz - 1) / nrow + 1;

    // ELL nnz is ELL width times nrow
    int ell_nnz = max_row * nrow;
    int coo_nnz = 0;

    // Array to hold COO part nnz per row
    int *coo_row_nnz = NULL;
    allocate_hip(nrow+1, &coo_row_nnz);

    int blocks = (nrow - 1) / this->local_backend_.HIP_block_size + 1;

    // If there is no ELL part, its easy...
    if(ell_nnz == 0)
    {
        coo_nnz = nnz;
        hipMemcpy(coo_row_nnz, cast_mat_csr->mat_.row_offset, sizeof(int)*(nrow+1), hipMemcpyDeviceToDevice);
    }
    else
    {
        // Allocate workspace
        int *workspace = NULL;
        allocate_hip(blocks, &workspace);

        dim3 csr2hyb_blocks(blocks);
        dim3 csr2hyb_threads(this->local_backend_.HIP_block_size);

        hipLaunchKernelGGL((kernel_hyb_coo_nnz_part1<256>),
                           csr2hyb_blocks,
                           csr2hyb_threads,
                           0,
                           0,
                           nrow,
                           max_row,
                           cast_mat_csr->mat_.row_offset,
                           workspace,
                           coo_row_nnz);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipLaunchKernelGGL((kernel_hyb_coo_nnz_part2<256>),
                           dim3(1),
                           csr2hyb_threads,
                           0,
                           0,
                           blocks,
                           workspace);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        hipMemcpy(&coo_nnz, workspace, sizeof(int), hipMemcpyDeviceToHost);

        // TODO coo nnz can be extracted from exclusive scan array instead of reducing...

        // Perform exclusive scan on workspace TODO use rocPRIM
        int *hbuf = NULL;
        allocate_host(nrow+1, &hbuf);

        hipMemcpy(hbuf+1, coo_row_nnz, sizeof(int)*nrow, hipMemcpyDeviceToHost);

        hbuf[0] = 0;
        for(int i=0; i<nrow; ++i)
        {
            hbuf[i+1] += hbuf[i];
        }

        hipMemcpy(coo_row_nnz, hbuf, sizeof(int)*(nrow+1), hipMemcpyHostToDevice);

        free_host(&hbuf);
        free_hip(&workspace);
    }

    // Allocate HYB matrix
    this->AllocateHYB(ell_nnz, coo_nnz, max_row, nrow, ncol);

    // Launch csr2hyb kernel
    dim3 csr2hyb_blocks(blocks);
    dim3 csr2hyb_threads(this->local_backend_.HIP_block_size);

    hipLaunchKernelGGL((kernel_hyb_csr2hyb<ValueType>),
                       csr2hyb_blocks,
                       csr2hyb_threads,
                       0,
                       0,
                       nrow,
                       cast_mat_csr->mat_.val,
                       cast_mat_csr->mat_.row_offset,
                       cast_mat_csr->mat_.col,
                       max_row,
                       this->mat_.ELL.col,
                       this->mat_.ELL.val,
                       this->mat_.COO.row,
                       this->mat_.COO.col,
                       this->mat_.COO.val,
                       coo_row_nnz);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    free_hip(&coo_row_nnz);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = ell_nnz + coo_nnz;
    this->mat_.ELL.max_row = max_row;
    this->ell_nnz_ = ell_nnz;
    this->coo_nnz_ = coo_nnz;

    return true;

  }

  return false;

}

template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {
    
    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());
        
    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in) ; 
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out) ; 
    
    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    ValueType alpha = 1.0;

    // ELL
    if (this->ell_nnz_ > 0) {

        ValueType beta = 0.0;

        hipsparseStatus_t stat_t;
        stat_t = hipsparseTellmv(HIPSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 this->get_nrow(), this->get_ncol(), &alpha,
                                 this->ell_mat_descr_,
                                 this->mat_.ELL.val, this->mat_.ELL.col, this->get_ell_max_row(),
                                 cast_in->vec_, &beta,
                                 cast_out->vec_);
        CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    }

    // COO
    if (this->coo_nnz_ > 0) {

        // Add to y from ELL part
        ValueType beta = 1.0;

        hipsparseStatus_t stat_t;
        stat_t = hipsparseTcoomv(HIPSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 this->get_nrow(), this->get_ncol(), this->coo_nnz_, &alpha,
                                 this->coo_mat_descr_,
                                 this->mat_.COO.val, this->mat_.COO.row, this->mat_.COO.col,
                                 cast_in->vec_, &beta,
                                 cast_out->vec_);
        CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixHYB<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                                  BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in) ; 
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out) ; 

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    // ELL
    if (this->ell_nnz_ > 0) {

        ValueType beta = 0.0;

        hipsparseStatus_t stat_t;
        stat_t = hipsparseTellmv(HIPSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 this->get_nrow(), this->get_ncol(), &scalar,
                                 this->ell_mat_descr_,
                                 this->mat_.ELL.val, this->mat_.ELL.col, this->get_ell_max_row(),
                                 cast_in->vec_, &beta,
                                 cast_out->vec_);
        CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    }

    // COO
    if (this->coo_nnz_ > 0) {

        // Add to y from ELL part
        ValueType beta = 1.0;

        hipsparseStatus_t stat_t;
        stat_t = hipsparseTcoomv(HIPSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 this->get_nrow(), this->get_ncol(), this->coo_nnz_, &scalar,
                                 this->coo_mat_descr_,
                                 this->mat_.COO.val, this->mat_.COO.row, this->mat_.COO.col,
                                 cast_in->vec_, &beta,
                                 cast_out->vec_);
        CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

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
