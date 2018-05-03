#include "../../utils/def.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_matrix_coo.hpp"
#include "hip_matrix_dia.hpp"
#include "hip_matrix_ell.hpp"
#include "hip_matrix_hyb.hpp"
#include "hip_matrix_mcsr.hpp"
#include "hip_matrix_bcsr.hpp"
#include "hip_matrix_dense.hpp"
#include "hip_vector.hpp"
#include "../host/host_matrix_csr.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_csr.hpp"
#include "hip_kernels_vector.hpp"
#include "hip_allocate_free.hpp"
#include "hip_blas.hpp"
//#include "hip_sparse.hpp"
#include "../matrix_formats_ind.hpp"

#include <vector>
#include <hip/hip_runtime.h>

namespace rocalution {

template <typename ValueType>
HIPAcceleratorMatrixCSR<ValueType>::HIPAcceleratorMatrixCSR() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HIPAcceleratorMatrixCSR<ValueType>::HIPAcceleratorMatrixCSR(const Rocalution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "HIPAcceleratorMatrixCSR::HIPAcceleratorMatrixCSR()",
            "constructor with local_backend");

  this->mat_.row_offset = NULL;
  this->mat_.col        = NULL;
  this->mat_.val        = NULL;
  this->set_backend(local_backend);
/*
  this->L_mat_descr_ = 0;
  this->U_mat_descr_ = 0;

  this->L_mat_info_ = 0;
  this->U_mat_info_ = 0;

  this->mat_descr_ = 0;
*/
  this->tmp_vec_ = NULL;

  CHECK_HIP_ERROR(__FILE__, __LINE__);
/*
  cusparseStatus_t stat_t;
  
  stat_t = cusparseCreateMatDescr(&this->mat_descr_);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
  
  stat_t = cusparseSetMatIndexBase(this->mat_descr_, CUSPARSE_INDEX_BASE_ZERO);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
  
  stat_t = cusparseSetMatType(this->mat_descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
}

template <typename ValueType>
HIPAcceleratorMatrixCSR<ValueType>::~HIPAcceleratorMatrixCSR() {

  LOG_DEBUG(this, "HIPAcceleratorMatrixCSR::~HIPAcceleratorMatrixCSR()",
            "destructor");

  this->Clear();
/*
  cusparseStatus_t stat_t;

  stat_t = cusparseDestroyMatDescr(this->mat_descr_);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::info(void) const {

  LOG_INFO("HIPAcceleratorMatrixCSR<ValueType>");

}


template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::AllocateCSR(const int nnz, const int nrow, const int ncol) {

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
void HIPAcceleratorMatrixCSR<ValueType>::SetDataPtrCSR(int **row_offset, int **col, ValueType **val,
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
void HIPAcceleratorMatrixCSR<ValueType>::LeaveDataPtrCSR(int **row_offset, int **col, ValueType **val) {

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
void HIPAcceleratorMatrixCSR<ValueType>::Clear() {

  if (this->get_nnz() > 0) {

    free_hip(&this->mat_.row_offset);
    free_hip(&this->mat_.col);
    free_hip(&this->mat_.val);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

    this->LUAnalyseClear();
    this->LLAnalyseClear();

  }

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::Zeros() {

  if (this->get_nnz() > 0)
    set_to_zero_hip(this->local_backend_.HIP_block_size,
                    this->local_backend_.HIP_max_threads,
                    this->get_nnz(), mat_.val);

  return true;

}


template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());
  
    if (this->get_nnz() > 0) {

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
    }
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  const HostMatrixCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());
  
    if (this->get_nnz() > 0) {

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
    }
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixCSR<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateCSR(this->get_nnz(), this->get_nrow(), this->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_ncol() == dst->get_ncol());

   
    if (this->get_nnz() > 0) {

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
    }

    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}


template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  HostMatrixCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixCSR<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateCSR(this->get_nnz(), this->get_nrow(), this->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_ncol() == dst->get_ncol());
   
    if (this->get_nnz() > 0) {

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
    }

    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}


template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixCSR<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );


     assert(this->get_nnz()  == src.get_nnz());
     assert(this->get_nrow() == src.get_nrow());
     assert(this->get_ncol() == src.get_ncol());


    if (this->get_nnz() > 0) {

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
void HIPAcceleratorMatrixCSR<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixCSR<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );


     assert(this->get_nnz()  == src.get_nnz());
     assert(this->get_nrow() == src.get_nrow());
     assert(this->get_ncol() == src.get_ncol());


    if (this->get_nnz() > 0) {

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
void HIPAcceleratorMatrixCSR<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixCSR<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    hip_cast_mat->AllocateCSR(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_nnz() > 0) {

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
      }
    
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
void HIPAcceleratorMatrixCSR<ValueType>::CopyToAsync(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixCSR<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    hip_cast_mat->AllocateCSR(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());


    if (this->get_nnz() > 0) {

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
      }
    
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
void HIPAcceleratorMatrixCSR<ValueType>::CopyFromCSR(const int *row_offsets, const int *col, const ValueType *val) {

  // assert CSR format
  assert(this->get_mat_format() == CSR);

  if (this->get_nnz() > 0) {

    assert(this->nrow_ > 0);
    assert(this->ncol_ > 0);

    hipMemcpy(this->mat_.row_offset,            // dst
               row_offsets,                      // src
               (this->get_nrow()+1)*sizeof(int), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    hipMemcpy(this->mat_.col,              // dst
               col,                         // src
               this->get_nnz()*sizeof(int), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    hipMemcpy(this->mat_.val,                    // dst
               val,                               // src
               this->get_nnz()*sizeof(ValueType), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyToCSR(int *row_offsets, int *col, ValueType *val) const {

  // assert CSR format
  assert(this->get_mat_format() == CSR);

  if (this->get_nnz() > 0) {

    assert(this->nrow_ > 0);
    assert(this->ncol_ > 0);

    hipMemcpy(row_offsets,                      // dst
               this->mat_.row_offset,            // src
               (this->get_nrow()+1)*sizeof(int), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    hipMemcpy(col,                         // dst
               this->mat_.col,              // src
               this->get_nnz()*sizeof(int), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    hipMemcpy(val,                               // dst
               this->mat_.val,                    // src
               this->get_nnz()*sizeof(ValueType), // size
               hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const HIPAcceleratorMatrixCSR<ValueType>   *cast_mat_csr;
  if ((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {
    
    this->CopyFrom(*cast_mat_csr);
    return true;
    
  }

  /*
  const HIPAcceleratorMatrixCOO<ValueType>   *cast_mat_coo;
  if ((cast_mat_coo = dynamic_cast<const HIPAcceleratorMatrixCOO<ValueType>*> (&mat)) != NULL) {

    this->Clear();


  TODO
  Allocate
  copy colmn
  copy val
  cusparseStatus_t
      cusparseXcoo2csr(cusparseHandle_t handle, const int *cooRowInd,
                       int nnz, int m, int *csrRowPtr, cusparseIndexBase_t
                       idxBase);


    FATAL_ERROR(__FILE__, __LINE__);

    this->nrow_ = cast_mat_coo->get_nrow();
    this->ncol_ = cast_mat_coo->get_ncol();
    this->nnz_  = cast_mat_coo->get_nnz();

    return true;

  }
  */

  /*
  const HIPAcceleratorMatrixDENSE<ValueType> *cast_mat_dense;
  if ((cast_mat_dense = dynamic_cast<const HIPAcceleratorMatrixDENSE<ValueType>*> (&mat)) != NULL) {

    this->Clear();
    int nnz = 0;

    FATAL_ERROR(__FILE__, __LINE__);

    this->nrow_ = cast_mat_dense->get_nrow();
    this->ncol_ = cast_mat_dense->get_ncol();
    this->nnz_  = nnz;

    return true;

  }
  */

  /*
  const HIPAcceleratorMatrixDIA<ValueType>   *cast_mat_dia;
  if ((cast_mat_dia = dynamic_cast<const HIPAcceleratorMatrixDIA<ValueType>*> (&mat)) != NULL) {

    this->Clear();
    int nnz = 0;

    FATAL_ERROR(__FILE__, __LINE__);

    this->nrow_ = cast_mat_dia->get_nrow();
    this->ncol_ = cast_mat_dia->get_ncol();
    this->nnz_  = nnz ;

    return true;

  }
  */

  /*
  const HIPAcceleratorMatrixELL<ValueType>   *cast_mat_ell;
  if ((cast_mat_ell = dynamic_cast<const HIPAcceleratorMatrixELL<ValueType>*> (&mat)) != NULL) {

    this->Clear();
    int nnz = 0;

    FATAL_ERROR(__FILE__, __LINE__);

    this->nrow_ = cast_mat_ell->get_nrow();
    this->ncol_ = cast_mat_ell->get_ncol();
    this->nnz_  = nnz ;

    return true;

  }
  */

  /*
  const HIPAcceleratorMatrixMCSR<ValueType>  *cast_mat_mcsr;
  if ((cast_mat_mcsr = dynamic_cast<const HIPAcceleratorMatrixMCSR<ValueType>*> (&mat)) != NULL) {

    this->Clear();

    FATAL_ERROR(__FILE__, __LINE__);

    this->nrow_ = cast_mat_mcsr->get_nrow();
    this->ncol_ = cast_mat_mcsr->get_ncol();
    this->nnz_  = cast_mat_mcsr->get_nnz();

    return true;

  }
  */


  /*
  const HIPAcceleratorMatrixHYB<ValueType>   *cast_mat_hyb;
  if ((cast_mat_hyb = dynamic_cast<const HIPAcceleratorMatrixHYB<ValueType>*> (&mat)) != NULL) {

    this->Clear();

    FATAL_ERROR(__FILE__, __LINE__);
    int nnz = 0;

    this->nrow_ = cast_mat_hyb->get_nrow();
    this->ncol_ = cast_mat_hyb->get_ncol();
    this->nnz_  = nnz;

    return true;

  }
  */


  return false;

}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::CopyFromHostCSR(const int *row_offset, const int *col, const ValueType *val,
                                                         const int nnz, const int nrow, const int ncol) {

  assert(nnz >= 0);
  assert(ncol >= 0);
  assert(nrow >= 0);
  assert(row_offset != NULL);
  assert(col != NULL);
  assert(val != NULL);

  // Allocate matrix
  if (this->nnz_ > 0)
    this->Clear();

  if (nnz > 0) {

    allocate_hip(nrow+1, &this->mat_.row_offset);
    allocate_hip(nnz,    &this->mat_.col);
    allocate_hip(nnz,    &this->mat_.val);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

    hipMemcpy(this->mat_.row_offset,       // dst
               row_offset,                  // src
               (this->nrow_+1)*sizeof(int), // size
               hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    hipMemcpy(this->mat_.col,         // dst
               col,                    // src
               this->nnz_*sizeof(int), // size
               hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    hipMemcpy(this->mat_.val,               // dst
               val,                          // src
               this->nnz_*sizeof(ValueType), // size
               hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::Permute( const BaseVector<int> &permutation){

  assert(permutation.get_size() == this->get_nrow());
  assert(permutation.get_size() == this->get_ncol());

  if (this->get_nnz() > 0) {

    int *d_nnzr     = NULL;
    int *d_nnzrPerm = NULL;
    int *d_nnzPerm  = NULL;
    int *d_offset   = NULL;
    ValueType *d_data = NULL;

    allocate_hip<int>(this->get_nrow(), &d_nnzr);
    allocate_hip<int>(this->get_nrow(), &d_nnzrPerm);
    allocate_hip<int>((this->get_nrow()+1), &d_nnzPerm);
    allocate_hip<ValueType>(this->get_nnz(), &d_data);
    allocate_hip<int>(this->get_nnz(), &d_offset);

    const HIPAcceleratorVector<int> *cast_perm = dynamic_cast<const HIPAcceleratorVector<int>*> (&permutation);
    assert(cast_perm != NULL);

    int nrow = this->get_nrow();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_calc_row_nnz<int>),
                       GridSize, BlockSize, 0, 0,
                       this->get_nrow(), this->mat_.row_offset, d_nnzr);
    CHECK_HIP_ERROR(__FILE__,__LINE__);

    hipLaunchKernelGGL((kernel_permute_row_nnz<int>),
                       GridSize, BlockSize, 0, 0,
                       this->get_nrow(), d_nnzr, cast_perm->vec_, d_nnzrPerm);
    CHECK_HIP_ERROR(__FILE__,__LINE__);

    //TODO 
    //move in extra file
    cum_sum<int, 256>(d_nnzPerm, d_nnzrPerm, this->get_nrow());

    hipLaunchKernelGGL((kernel_permute_rows<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       this->get_nrow(), this->mat_.row_offset, d_nnzPerm,
                       this->mat_.col, HIPPtr(this->mat_.val), HIPPtr(cast_perm->vec_),
                       d_nnzr, d_offset, HIPPtr(d_data));
    CHECK_HIP_ERROR(__FILE__,__LINE__);

    free_hip<int>(&this->mat_.row_offset);	

    this->mat_.row_offset = d_nnzPerm;

    int *d_buffer = NULL;
    int *h_buffer = NULL;
    int GROUP_SIZE;
    int LOCAL_SIZE;
    int FinalReduceSize;

    allocate_hip<int>(this->local_backend_.HIP_warp * 4, &d_buffer);

    dim3 BlockSize2(this->local_backend_.HIP_block_size);
    dim3 GridSize2(this->local_backend_.HIP_warp * 4);

    GROUP_SIZE = ( size_t( ( size_t( nrow / ( this->local_backend_.HIP_warp * 4 ) ) + 1 ) 
                 / this->local_backend_.HIP_block_size ) + 1 ) * this->local_backend_.HIP_block_size;
    LOCAL_SIZE = GROUP_SIZE / this->local_backend_.HIP_block_size;

    hipLaunchKernelGGL((kernel_max<int, int, 256>),
                       GridSize2, BlockSize2, 0, 0,
                       nrow, d_nnzr, d_buffer, GROUP_SIZE, LOCAL_SIZE);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    FinalReduceSize = this->local_backend_.HIP_warp * 4;
    allocate_host(FinalReduceSize, &h_buffer);

    hipMemcpy(h_buffer, // dst
               d_buffer, // src
               FinalReduceSize*sizeof(int), // size
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    free_hip<int>(&d_buffer);

    int maxnnzrow = 0;
    for (int i=0; i<FinalReduceSize; ++i)
      if (maxnnzrow < h_buffer[i])
        maxnnzrow = h_buffer[i];

    free_host(&h_buffer);

    if (maxnnzrow > 64)
      hipLaunchKernelGGL((kernel_permute_cols_fallback<ValueType, int>),
                         GridSize, BlockSize, 0, 0,
                         this->get_nrow(), this->mat_.row_offset,
                         cast_perm->vec_, d_nnzrPerm, d_offset,
                         HIPPtr(d_data), this->mat_.col, HIPPtr(this->mat_.val));
    else if (maxnnzrow >  32)
      hipLaunchKernelGGL((kernel_permute_cols<64, ValueType, int>),
                         GridSize, BlockSize, 0, 0,
                         this->get_nrow(), this->mat_.row_offset,
                         cast_perm->vec_, d_nnzrPerm, d_offset,
                         HIPPtr(d_data), this->mat_.col, HIPPtr(this->mat_.val));
    else if (maxnnzrow >  16)
      hipLaunchKernelGGL((kernel_permute_cols<32, ValueType, int>),
                         GridSize, BlockSize, 0, 0,
                         this->get_nrow(), this->mat_.row_offset,
                         cast_perm->vec_, d_nnzrPerm, d_offset,
                         HIPPtr(d_data), this->mat_.col, HIPPtr(this->mat_.val));
    else if (maxnnzrow >   8)
      hipLaunchKernelGGL((kernel_permute_cols<16, ValueType, int>),
                         GridSize, BlockSize, 0, 0,
                         this->get_nrow(), this->mat_.row_offset,
                         cast_perm->vec_, d_nnzrPerm, d_offset,
                         HIPPtr(d_data), this->mat_.col, HIPPtr(this->mat_.val));
    else if (maxnnzrow >   4)
      hipLaunchKernelGGL((kernel_permute_cols<8, ValueType, int>),
                         GridSize, BlockSize, 0, 0,
                         this->get_nrow(), this->mat_.row_offset,
                         cast_perm->vec_, d_nnzrPerm, d_offset,
                         HIPPtr(d_data), this->mat_.col, HIPPtr(this->mat_.val));
    else
      hipLaunchKernelGGL((kernel_permute_cols<4, ValueType, int>),
                         GridSize, BlockSize, 0, 0,
                         this->get_nrow(), this->mat_.row_offset,
                         cast_perm->vec_, d_nnzrPerm, d_offset,
                         HIPPtr(d_data), this->mat_.col, HIPPtr(this->mat_.val));
    CHECK_HIP_ERROR(__FILE__,__LINE__);

    free_hip<int>(&d_offset);
    free_hip<ValueType>(&d_data);
    free_hip<int>(&d_nnzrPerm);
    free_hip<int>(&d_nnzr);

  }

  return true;

}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());
    
    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in) ; 
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out) ; 
    
    assert(cast_in != NULL);
    assert(cast_out!= NULL);

//    const ValueType scalar = 1.0;
//    const ValueType beta = 0.0;
/*
    rocsparseStatus_t stat_t;
    stat_t = rocsparseTcsrmv(HIPSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                             rocsparse_operation_none,
                             this->get_nrow(), this->get_ncol(), this->get_nnz(), &scalar,
                             this->mat_descr_,
                             this->mat_.val, 
                             this->mat_.row_offset, this->mat_.col,
                             cast_in->vec_, &beta,
                             cast_out->vec_);
    CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(this->get_nrow() / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_spmv_scalar<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       this->get_nrow(), this->mat_.row_offset, this->mat_.col,
                       this->mat_.val, cast_in->vec_, cast_out->vec_);
    CHECK_HIP_ERROR(__FILE__,__LINE__);

  }
    
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
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

    const ValueType beta = 1.0;
/*
    rocsparseStatus_t stat_t;
    stat_t = rocsparseTcsrmv(HIPSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                             rocsparse_operation_non,
                             this->get_nrow(), this->get_ncol(), this->get_nnz(), &scalar,
                             this->mat_descr_,
                             this->mat_.val, this->mat_.row_offset, this->mat_.col,
                             cast_in->vec_, &beta,
                             cast_out->vec_);
    CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(this->get_nrow() / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_add_spmv_scalar<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       this->get_nrow(), this->mat_.row_offset, this->mat_.col,
                       this->mat_.val, scalar, cast_in->vec_, cast_out->vec_);
    CHECK_HIP_ERROR(__FILE__,__LINE__);

  }

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ILU0Factorize(void) {

  if (this->get_nnz() > 0) {
/*
    cusparseStatus_t stat_t;

    cusparseSolveAnalysisInfo_t infoA = 0;

    stat_t = cusparseCreateSolveAnalysisInfo(&infoA);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseDcsrsv_analysis(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     this->get_nrow(), this->get_nnz(),
                                     this->mat_descr_,
                                     this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                     infoA);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseDcsrilu0(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              this->get_nrow(),
                              this->mat_descr_,
                              this->mat_.val, this->mat_.row_offset, this->mat_.col,
                              infoA);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseDestroySolveAnalysisInfo(infoA);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ICFactorize(BaseVector<ValueType> *inv_diag) {

  if (this->get_nnz() > 0) {
/*
    cusparseStatus_t stat_t;

    cusparseSolveAnalysisInfo_t infoA = 0;

    stat_t = cusparseCreateSolveAnalysisInfo(&infoA);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatType(this->mat_descr_, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatFillMode(this->mat_descr_, CUSPARSE_FILL_MODE_LOWER);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatDiagType(this->mat_descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseDcsrsv_analysis(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle), 
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     this->get_nrow(), this->get_nnz(),
                                     this->mat_descr_,
                                     this->mat_.val, this->mat_.row_offset, this->mat_.col, 
                                     infoA);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseDcsric0(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             this->get_nrow(),
                             this->mat_descr_,
                             this->mat_.val, this->mat_.row_offset, this->mat_.col,
                             infoA);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
  }

  return true;

}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::LUAnalyse(void) {

    this->LUAnalyseClear();
/*
    cusparseStatus_t stat_t;

    // L part
    stat_t = cusparseCreateMatDescr(&this->L_mat_descr_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatType(this->L_mat_descr_,CUSPARSE_MATRIX_TYPE_GENERAL);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatIndexBase(this->L_mat_descr_,CUSPARSE_INDEX_BASE_ZERO);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatFillMode(this->L_mat_descr_, CUSPARSE_FILL_MODE_LOWER);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatDiagType(this->L_mat_descr_, CUSPARSE_DIAG_TYPE_UNIT);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseCreateSolveAnalysisInfo(&this->L_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);


    // U part
    stat_t = cusparseCreateMatDescr(&this->U_mat_descr_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatType(this->U_mat_descr_,CUSPARSE_MATRIX_TYPE_GENERAL);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatIndexBase(this->U_mat_descr_,CUSPARSE_INDEX_BASE_ZERO);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatFillMode(this->U_mat_descr_, CUSPARSE_FILL_MODE_UPPER);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatDiagType(this->U_mat_descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseCreateSolveAnalysisInfo(&this->U_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    // Analysis
    stat_t = cusparseDcsrsv_analysis(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     this->get_nrow(), this->get_nnz(),
                                     this->L_mat_descr_,
                                     this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                     this->L_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseDcsrsv_analysis(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                     this->get_nrow(), this->get_nnz(), 
                                     this->U_mat_descr_, 
                                     this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                     this->U_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
    assert(this->get_ncol() == this->get_nrow());
    assert(this->tmp_vec_ == NULL);
    this->tmp_vec_ = new HIPAcceleratorVector<ValueType>(this->local_backend_);
    assert(this->tmp_vec_ != NULL);

    tmp_vec_->Allocate(this->get_nrow());

}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::LUAnalyseClear(void) {
/*
  cusparseStatus_t stat_t;

  if (this->L_mat_info_ != 0) {
    stat_t = cusparseDestroySolveAnalysisInfo(this->L_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__); 
  }

  if (this->L_mat_descr_ != 0) {
    stat_t = cusparseDestroyMatDescr(this->L_mat_descr_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__); 
  }

  if (this->U_mat_info_ != 0) {
    stat_t = cusparseDestroySolveAnalysisInfo(this->U_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__); 
  }

  if (this->U_mat_descr_ != 0) {
    stat_t = cusparseDestroyMatDescr(this->U_mat_descr_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__); 
  }

  this->L_mat_descr_ = 0;
  this->U_mat_descr_ = 0;
  this->L_mat_info_ = 0;
  this->U_mat_info_ = 0;
*/
  if (this ->tmp_vec_ != NULL) {
    delete this->tmp_vec_ ;
    this->tmp_vec_ = NULL;
  }

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::LUSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {
/*
    assert(this->L_mat_descr_ != 0);
    assert(this->U_mat_descr_ != 0);
    assert(this->L_mat_info_  != 0);
    assert(this->U_mat_info_  != 0);
*/
    assert(in.  get_size()  >= 0);
    assert(out->get_size()  >= 0);
    assert(in.  get_size()  == this->get_ncol());
    assert(out->get_size()  == this->get_nrow());
    assert(this->get_ncol() == this->get_nrow());

    assert(this->tmp_vec_ != NULL);

    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in);
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);
/*
    cusparseStatus_t stat_t;

    ValueType one = ValueType(1.0);

    // Solve L
    stat_t = cusparseScsrsv_solve(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  this->get_nrow(),
                                  &one,
                                  this->L_mat_descr_,
                                  this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                  this->L_mat_info_,
                                  cast_in->vec_,
                                  tmp_vec_->vec_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    // Solve U
    stat_t = cusparseScsrsv_solve(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  this->get_nrow(),
                                  &one,
                                  this->U_mat_descr_,
                                  this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                  this->U_mat_info_,
                                  tmp_vec_->vec_,
                                  cast_out->vec_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
  }

  return true;

}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::LLAnalyse(void) {

    this->LLAnalyseClear();
/*
    cusparseStatus_t stat_t;

    // L part
    stat_t = cusparseCreateMatDescr(&this->L_mat_descr_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatType(this->L_mat_descr_,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatIndexBase(this->L_mat_descr_,CUSPARSE_INDEX_BASE_ZERO);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatFillMode(this->L_mat_descr_, CUSPARSE_FILL_MODE_LOWER);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatDiagType(this->L_mat_descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseCreateSolveAnalysisInfo(&this->L_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    // U part
    stat_t = cusparseCreateMatDescr(&this->U_mat_descr_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatType(this->U_mat_descr_,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatIndexBase(this->U_mat_descr_,CUSPARSE_INDEX_BASE_ZERO);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatFillMode(this->U_mat_descr_, CUSPARSE_FILL_MODE_LOWER);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseSetMatDiagType(this->U_mat_descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseCreateSolveAnalysisInfo(&this->U_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    // Analysis
    stat_t = cusparseDcsrsv_analysis(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     this->get_nrow(), this->get_nnz(),
                                     this->L_mat_descr_,
                                     this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                     this->L_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    stat_t = cusparseDcsrsv_analysis(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                     CUSPARSE_OPERATION_TRANSPOSE,
                                     this->get_nrow(), this->get_nnz(),
                                     this->U_mat_descr_,
                                     this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                     this->U_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
    assert(this->get_ncol() == this->get_nrow());
    assert(this->tmp_vec_ == NULL);
    this->tmp_vec_ = new HIPAcceleratorVector<ValueType>(this->local_backend_);
    assert(this->tmp_vec_ != NULL);

    tmp_vec_->Allocate(this->get_nrow());

}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::LLAnalyseClear(void) {
/*
  cusparseStatus_t stat_t;

  if (this->L_mat_info_ != 0) {
    stat_t = cusparseDestroySolveAnalysisInfo(this->L_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__); 
  }

  if (this->L_mat_descr_ != 0) {
    stat_t = cusparseDestroyMatDescr(this->L_mat_descr_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__); 
  }

  if (this->U_mat_info_ != 0) {
    stat_t = cusparseDestroySolveAnalysisInfo(this->U_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__); 
  }

  if (this->U_mat_descr_ != 0) {
    stat_t = cusparseDestroyMatDescr(this->U_mat_descr_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__); 
  }

  this->L_mat_descr_ = 0;
  this->U_mat_descr_ = 0;
  this->L_mat_info_ = 0;
  this->U_mat_info_ = 0;
*/
  if (this ->tmp_vec_ != NULL) {
    delete this->tmp_vec_ ;
    this->tmp_vec_ = NULL;
  }
    

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::LLSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {
/*
    assert(this->L_mat_descr_ != 0);
    assert(this->U_mat_descr_ != 0);
    assert(this->L_mat_info_  != 0);
    assert(this->U_mat_info_  != 0);

    assert(in.  get_size()  >= 0);
    assert(out->get_size()  >= 0);
    assert(in.  get_size()  == this->get_ncol());
    assert(out->get_size()  == this->get_nrow());
    assert(this->get_ncol() == this->get_nrow());

    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in);
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    cusparseStatus_t stat_t;

    ValueType one = ValueType(1.0);

    // Solve L
    stat_t = cusparseScsrsv_solve(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  this->get_nrow(),
                                  &one,
                                  this->L_mat_descr_,
                                  this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                  this->L_mat_info_,
                                  cast_in->vec_,
                                  this->tmp_vec_->vec_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    // Solve U
    stat_t = cusparseScsrsv_solve(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                  CUSPARSE_OPERATION_TRANSPOSE,
                                  this->get_nrow(),
                                  &one,
                                  this->U_mat_descr_,
                                  this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                  this->U_mat_info_,
                                  this->tmp_vec_->vec_,
                                  cast_out->vec_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::LLSolve(const BaseVector<ValueType> &in, const BaseVector<ValueType> &inv_diag,
                                                 BaseVector<ValueType> *out) const {

  return LLSolve(in, out);

}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::LAnalyse(const bool diag_unit) {
/*
  cusparseStatus_t stat_t;

  // L part
  stat_t = cusparseCreateMatDescr(&this->L_mat_descr_);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  stat_t = cusparseSetMatType(this->L_mat_descr_,CUSPARSE_MATRIX_TYPE_GENERAL);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  stat_t = cusparseSetMatIndexBase(this->L_mat_descr_,CUSPARSE_INDEX_BASE_ZERO);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  stat_t = cusparseSetMatFillMode(this->L_mat_descr_, CUSPARSE_FILL_MODE_LOWER);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  if (diag_unit == true) {

    stat_t = cusparseSetMatDiagType(this->L_mat_descr_, CUSPARSE_DIAG_TYPE_UNIT);

  } else {

    stat_t = cusparseSetMatDiagType(this->L_mat_descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);

  }

  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  stat_t = cusparseCreateSolveAnalysisInfo(&this->L_mat_info_);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  // Analysis
  stat_t = cusparseDcsrsv_analysis(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   this->get_nrow(), this->get_nnz(),
                                   this->L_mat_descr_,
                                   this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                   this->L_mat_info_);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::UAnalyse(const bool diag_unit) {
/*
  cusparseStatus_t stat_t;

  // U upart
  stat_t = cusparseCreateMatDescr(&this->U_mat_descr_);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  stat_t = cusparseSetMatType(this->U_mat_descr_,CUSPARSE_MATRIX_TYPE_GENERAL);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  stat_t = cusparseSetMatIndexBase(this->U_mat_descr_,CUSPARSE_INDEX_BASE_ZERO);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  stat_t = cusparseSetMatFillMode(this->U_mat_descr_, CUSPARSE_FILL_MODE_UPPER);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  if (diag_unit == true) {

    stat_t = cusparseSetMatDiagType(this->U_mat_descr_, CUSPARSE_DIAG_TYPE_UNIT);

  } else {

    stat_t = cusparseSetMatDiagType(this->U_mat_descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);

  }

  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  stat_t = cusparseCreateSolveAnalysisInfo(&this->U_mat_info_);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  // Analysis
  stat_t = cusparseDcsrsv_analysis(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   this->get_nrow(), this->get_nnz(),
                                   this->U_mat_descr_,
                                   this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                   this->U_mat_info_);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::LAnalyseClear(void) {
/*
  cusparseStatus_t stat_t;

  if (this->L_mat_info_ != 0) {
    stat_t = cusparseDestroySolveAnalysisInfo(this->L_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__); 
  }

  if (this->L_mat_descr_ != 0) {
    stat_t = cusparseDestroyMatDescr(this->L_mat_descr_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__); 
  }

  this->L_mat_descr_ = 0;
  this->L_mat_info_ = 0;
*/
}

template <typename ValueType>
void HIPAcceleratorMatrixCSR<ValueType>::UAnalyseClear(void) {
/*
  cusparseStatus_t stat_t;

  if (this->U_mat_info_ != 0) {
    stat_t = cusparseDestroySolveAnalysisInfo(this->U_mat_info_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__); 
  }

  if (this->U_mat_descr_ != 0) {
    stat_t = cusparseDestroyMatDescr(this->U_mat_descr_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__); 
  }

  this->U_mat_descr_ = 0;
  this->U_mat_info_ = 0;
*/
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::LSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {
/*
    assert(this->L_mat_descr_ != 0);
    assert(this->U_mat_descr_ != 0);
    assert(this->L_mat_info_  != 0);
    assert(this->U_mat_info_  != 0);
*/
    assert(in.  get_size()  >= 0);
    assert(out->get_size()  >= 0);
    assert(in.  get_size()  == this->get_ncol());
    assert(out->get_size()  == this->get_nrow());
    assert(this->get_ncol() == this->get_nrow());

    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in);
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);
/*
    cusparseStatus_t stat_t;

    ValueType one = ValueType(1.0);

    // Solve L
    stat_t = cusparseDcsrsv_solve(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  this->get_nrow(),
                                  &one,
                                  this->L_mat_descr_,
                                  this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                  this->L_mat_info_,
                                  cast_in->vec_,
                                  cast_out->vec_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::USolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {
/*
    assert(this->L_mat_descr_ != 0);
    assert(this->U_mat_descr_ != 0);
    assert(this->L_mat_info_  != 0);
    assert(this->U_mat_info_  != 0);
*/
    assert(in.  get_size()  >= 0);
    assert(out->get_size()  >= 0);
    assert(in.  get_size()  == this->get_ncol());
    assert(out->get_size()  == this->get_nrow());
    assert(this->get_ncol() == this->get_nrow()); 

    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in);
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);
/*
    cusparseStatus_t stat_t;

    ValueType one = ValueType(1.0);

    // Solve U
    stat_t = cusparseDcsrsv_solve(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  this->get_nrow(),
                                  &one,
                                  this->U_mat_descr_,
                                  this->mat_.val, this->mat_.row_offset, this->mat_.col,
                                  this->U_mat_info_,
                                  cast_in->vec_,
                                  cast_out->vec_);
    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractDiagonal(BaseVector<ValueType> *vec_diag) const {

  if (this->get_nnz() > 0)  {

    assert(vec_diag != NULL);
    assert(vec_diag->get_size() == this->get_nrow());

    HIPAcceleratorVector<ValueType> *cast_vec_diag  = dynamic_cast<HIPAcceleratorVector<ValueType>*> (vec_diag);

    int nrow = this->get_nrow();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_extract_diag<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       nrow, this->mat_.row_offset, this->mat_.col,
                       HIPPtr(this->mat_.val), HIPPtr(cast_vec_diag->vec_));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractInverseDiagonal(BaseVector<ValueType> *vec_inv_diag) const {

  if (this->get_nnz() > 0) {

    assert(vec_inv_diag != NULL);
    assert(vec_inv_diag->get_size() == this->get_nrow());

    HIPAcceleratorVector<ValueType> *cast_vec_inv_diag  = dynamic_cast<HIPAcceleratorVector<ValueType>*> (vec_inv_diag);

    int nrow = this->get_nrow();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_extract_inv_diag<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       nrow, this->mat_.row_offset, this->mat_.col,
                       HIPPtr(this->mat_.val), HIPPtr(cast_vec_inv_diag->vec_));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractSubMatrix(const int row_offset,
                                                          const int col_offset,
                                                          const int row_size,
                                                          const int col_size,
                                                          BaseMatrix<ValueType> *mat) const {
  assert(mat != NULL);

  assert(row_offset >= 0);
  assert(col_offset >= 0);

  assert(this->get_nrow() > 0);
  assert(this->get_ncol() > 0);

  HIPAcceleratorMatrixCSR<ValueType> *cast_mat  = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*> (mat) ; 
  assert(cast_mat != NULL);

  int mat_nnz = 0;

  int *row_nnz = NULL;  
  //int *red_row_nnz  (int *) malloc(sizeof(int)*(row_size+1));
  int *sub_nnz = NULL;
  allocate_hip<int>(row_size+1, &sub_nnz);
  allocate_hip(row_size+1, &row_nnz);

  // compute the nnz per row in the new matrix

  dim3 BlockSize(this->local_backend_.HIP_block_size);
  dim3 GridSize(row_size / this->local_backend_.HIP_block_size + 1);
  
  hipLaunchKernelGGL((kernel_csr_extract_submatrix_row_nnz<ValueType, int>),
                     GridSize, BlockSize, 0, 0,
                     this->mat_.row_offset, this->mat_.col, this->mat_.val,
                     row_offset, col_offset,
                     row_size, col_size,
                     row_nnz);
    
  CHECK_HIP_ERROR(__FILE__, __LINE__);      

  // compute the new nnz by reduction 
  
  
  // CPU reduction
  /*
  hipMemcpy(red_row_nnz, // dst
             row_nnz,  // src
             (row_size+1)*sizeof(int), // size
             hipMemcpyDeviceToHost);

  int sum=0;
  for (int i=0; i<row_size; ++i) {
    int tmp = red_row_nnz[i];
    red_row_nnz[i] = sum;
    sum += tmp;
  }

  mat_nnz = red_row_nnz[row_size] = sum ;
  */

  //TODO
  //move in extra file
  cum_sum<int, 256>(sub_nnz, row_nnz, row_size);
  
  hipMemcpy(&mat_nnz, &sub_nnz[row_size],
             sizeof(int), hipMemcpyDeviceToHost);

  // not empty submatrix
  if (mat_nnz > 0) {

    cast_mat->AllocateCSR(mat_nnz, row_size, col_size);

    // part of the CPU reduction section
    /*
    hipMemcpy(cast_mat->mat_.row_offset, // dst
               red_row_nnz,  // src
               (row_size+1)*sizeof(int), // size
               hipMemcpyHostToDevice);
    */
    
    free_hip<int>(&cast_mat->mat_.row_offset);
    cast_mat->mat_.row_offset = sub_nnz;
    // copying the sub matrix
    
    hipLaunchKernelGGL((kernel_csr_extract_submatrix_copy<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       this->mat_.row_offset, this->mat_.col, this->mat_.val,
                       row_offset, col_offset,
                       row_size, col_size,
                       cast_mat->mat_.row_offset, cast_mat->mat_.col, cast_mat->mat_.val);
    CHECK_HIP_ERROR(__FILE__, __LINE__);      

  }

  free_hip(&row_nnz);

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractL(BaseMatrix<ValueType> *L) const {
  
  assert(L != NULL);
  
  assert(this->get_nrow() > 0);
  assert(this->get_ncol() > 0);
  
  HIPAcceleratorMatrixCSR<ValueType> *cast_L = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*> (L);
  
  assert(cast_L != NULL);
  
  cast_L->Clear();
  
  // compute nnz per row
  int nrow = this->get_nrow();
  
  allocate_hip<int>(nrow+1, &cast_L->mat_.row_offset);
  
  dim3 BlockSize(this->local_backend_.HIP_block_size);
  dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);
  
  hipLaunchKernelGGL((kernel_csr_slower_nnz_per_row<int>),
                     GridSize, BlockSize, 0, 0,
                     nrow, this->mat_.row_offset,
                     this->mat_.col, cast_L->mat_.row_offset+1);
  CHECK_HIP_ERROR(__FILE__,__LINE__);

  // partial sum row_nnz to obtain row_offset vector
  // TODO currently performing partial sum on host
  int *h_buffer = NULL;
  allocate_host(nrow+1, &h_buffer);
  hipMemcpy(h_buffer+1, // dst
             cast_L->mat_.row_offset+1, // src
             nrow*sizeof(int), // size
             hipMemcpyDeviceToHost);
  
  h_buffer[0] = 0;
  for (int i=1; i<nrow+1; ++i)
    h_buffer[i] += h_buffer[i-1];
  
  int nnz_L = h_buffer[nrow];
  
  hipMemcpy(cast_L->mat_.row_offset, // dst
             h_buffer, // src
             (nrow+1)*sizeof(int), // size
             hipMemcpyHostToDevice);
  
  free_host(&h_buffer);
  // end TODO
  
  // allocate lower triangular part structure
  allocate_hip<int>(nnz_L, &cast_L->mat_.col);
  allocate_hip<ValueType>(nnz_L, &cast_L->mat_.val);
  
  // fill lower triangular part
  hipLaunchKernelGGL((kernel_csr_extract_l_triangular<ValueType, int>),
                     GridSize, BlockSize, 0, 0,
                     nrow, this->mat_.row_offset,
                     this->mat_.col, this->mat_.val,
                     cast_L->mat_.row_offset,
                     cast_L->mat_.col,
                     cast_L->mat_.val);
  CHECK_HIP_ERROR(__FILE__,__LINE__);
  
  cast_L->nrow_ = this->get_nrow();
  cast_L->ncol_ = this->get_ncol();
  cast_L->nnz_ = nnz_L;
  
  return true;
  
}


template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractLDiagonal(BaseMatrix<ValueType> *L) const {

  assert(L != NULL);
  
  assert(this->get_nrow() > 0);
  assert(this->get_ncol() > 0);
  
  HIPAcceleratorMatrixCSR<ValueType> *cast_L = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*> (L);
  
  assert(cast_L != NULL);
  
  cast_L->Clear();
  
  // compute nnz per row
  int nrow = this->get_nrow();
  
  allocate_hip<int>(nrow+1, &cast_L->mat_.row_offset);
  
  dim3 BlockSize(this->local_backend_.HIP_block_size);
  dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

  hipLaunchKernelGGL((kernel_csr_lower_nnz_per_row<int>),
                     GridSize, BlockSize, 0, 0,
                     nrow, this->mat_.row_offset,
                     this->mat_.col, cast_L->mat_.row_offset+1);
  CHECK_HIP_ERROR(__FILE__,__LINE__);

  // partial sum row_nnz to obtain row_offset vector
  // TODO currently performing partial sum on host
  int *h_buffer = NULL;
  allocate_host(nrow+1, &h_buffer);
  hipMemcpy(h_buffer+1, // dst
             cast_L->mat_.row_offset+1, // src
             nrow*sizeof(int), // size
             hipMemcpyDeviceToHost);

  h_buffer[0] = 0;
  for (int i=1; i<nrow+1; ++i)
    h_buffer[i] += h_buffer[i-1];
  
  int nnz_L = h_buffer[nrow];

  hipMemcpy(cast_L->mat_.row_offset, // dst
             h_buffer, // src
             (nrow+1)*sizeof(int), // size
             hipMemcpyHostToDevice);
  
  free_host(&h_buffer);
  // end TODO
  
  // allocate lower triangular part structure
  allocate_hip<int>(nnz_L, &cast_L->mat_.col);
  allocate_hip<ValueType>(nnz_L, &cast_L->mat_.val);
  
  // fill lower triangular part
  hipLaunchKernelGGL((kernel_csr_extract_l_triangular<ValueType, int>),
                     GridSize, BlockSize, 0, 0,
                     nrow, this->mat_.row_offset,
                     this->mat_.col, this->mat_.val,
                     cast_L->mat_.row_offset,
                     cast_L->mat_.col,
                     cast_L->mat_.val);
  CHECK_HIP_ERROR(__FILE__,__LINE__);
  
  cast_L->nrow_ = this->get_nrow();
  cast_L->ncol_ = this->get_ncol();
  cast_L->nnz_ = nnz_L;
  
  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractU(BaseMatrix<ValueType> *U) const {
  
  assert(U != NULL);
  
  assert(this->get_nrow() > 0);
  assert(this->get_ncol() > 0);
  
  HIPAcceleratorMatrixCSR<ValueType> *cast_U = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*> (U);
  
  assert(cast_U != NULL);
  
  cast_U->Clear();
  
  // compute nnz per row
  int nrow = this->get_nrow();
  
  allocate_hip<int>(nrow+1, &cast_U->mat_.row_offset);
  
  dim3 BlockSize(this->local_backend_.HIP_block_size);
  dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

  hipLaunchKernelGGL((kernel_csr_supper_nnz_per_row<int>),
                     GridSize, BlockSize, 0, 0,
                     nrow, this->mat_.row_offset,
                     this->mat_.col, cast_U->mat_.row_offset+1);
  CHECK_HIP_ERROR(__FILE__,__LINE__);

  // partial sum row_nnz to obtain row_offset vector
  // TODO currently performing partial sum on host
  int *h_buffer = NULL;
  allocate_host(nrow+1, &h_buffer);
  hipMemcpy(h_buffer+1, // dst
             cast_U->mat_.row_offset+1, // src
             nrow*sizeof(int), // size
             hipMemcpyDeviceToHost);
  
  h_buffer[0] = 0;
  for (int i=1; i<nrow+1; ++i)
    h_buffer[i] += h_buffer[i-1];
  
  int nnz_L = h_buffer[nrow];
  
  hipMemcpy(cast_U->mat_.row_offset, // dst
             h_buffer, // src
             (nrow+1)*sizeof(int), // size
             hipMemcpyHostToDevice);
  
  free_host(&h_buffer);
  // end TODO
  
  // allocate lower triangular part structure
  allocate_hip<int>(nnz_L, &cast_U->mat_.col);
  allocate_hip<ValueType>(nnz_L, &cast_U->mat_.val);
  
  // fill upper triangular part
  hipLaunchKernelGGL((kernel_csr_extract_u_triangular<ValueType, int>),
                     GridSize, BlockSize, 0, 0,
                     nrow, this->mat_.row_offset,
                     this->mat_.col, this->mat_.val,
                     cast_U->mat_.row_offset,
                     cast_U->mat_.col,
                     cast_U->mat_.val);
  CHECK_HIP_ERROR(__FILE__,__LINE__);

  cast_U->nrow_ = this->get_nrow();
  cast_U->ncol_ = this->get_ncol();
  cast_U->nnz_ = nnz_L;

  return true;
  
}


template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractUDiagonal(BaseMatrix<ValueType> *U) const {

  assert(U != NULL);
  
  assert(this->get_nrow() > 0);
  assert(this->get_ncol() > 0);
  
  HIPAcceleratorMatrixCSR<ValueType> *cast_U = dynamic_cast<HIPAcceleratorMatrixCSR<ValueType>*> (U);
  
  assert(cast_U != NULL);
  
  cast_U->Clear();
  
  // compute nnz per row
  int nrow = this->get_nrow();

  allocate_hip<int>(nrow+1, &cast_U->mat_.row_offset);

  dim3 BlockSize(this->local_backend_.HIP_block_size);
  dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

  hipLaunchKernelGGL((kernel_csr_upper_nnz_per_row<int>),
                     GridSize, BlockSize, 0, 0,
                     nrow, this->mat_.row_offset,
                     this->mat_.col, cast_U->mat_.row_offset+1);
  CHECK_HIP_ERROR(__FILE__,__LINE__);

  // partial sum row_nnz to obtain row_offset vector
  // TODO currently performing partial sum on host
  int *h_buffer = NULL;
  allocate_host(nrow+1, &h_buffer);
  hipMemcpy(h_buffer+1, // dst
             cast_U->mat_.row_offset+1, // src
             nrow*sizeof(int), // size
             hipMemcpyDeviceToHost);

  h_buffer[0] = 0;
  for (int i=1; i<nrow+1; ++i)
    h_buffer[i] += h_buffer[i-1];
  
  int nnz_L = h_buffer[nrow];

  hipMemcpy(cast_U->mat_.row_offset, // dst
             h_buffer, // src
             (nrow+1)*sizeof(int), // size
             hipMemcpyHostToDevice);
  
  free_host(&h_buffer);
  // end TODO

  // allocate lower triangular part structure
  allocate_hip<int>(nnz_L, &cast_U->mat_.col);
  allocate_hip<ValueType>(nnz_L, &cast_U->mat_.val);

  // fill lower triangular part
  hipLaunchKernelGGL((kernel_csr_extract_u_triangular<ValueType, int>),
                     GridSize, BlockSize, 0, 0,
                     nrow, this->mat_.row_offset,
                     this->mat_.col, this->mat_.val,
                     cast_U->mat_.row_offset,
                     cast_U->mat_.col,
                     cast_U->mat_.val);
  CHECK_HIP_ERROR(__FILE__,__LINE__);
  
  cast_U->nrow_ = this->get_nrow();
  cast_U->ncol_ = this->get_ncol();
  cast_U->nnz_ = nnz_L;
  
  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::MaximalIndependentSet(int &size,
                                                               BaseVector<int> *permutation) const {
  assert(permutation != NULL);
  HIPAcceleratorVector<int> *cast_perm = dynamic_cast<HIPAcceleratorVector<int>*> (permutation);
  assert(cast_perm != NULL);
  assert(this->get_nrow() == this->get_ncol());

  int *h_row_offset = NULL;
  int *h_col = NULL;

  allocate_host(this->get_nrow()+1, &h_row_offset);
  allocate_host(this->get_nnz(), &h_col);

  hipMemcpy(h_row_offset, this->mat_.row_offset, (this->get_nrow()+1)*sizeof(int), hipMemcpyDeviceToHost);
  hipMemcpy(h_col, this->mat_.col, this->get_nnz()*sizeof(int), hipMemcpyDeviceToHost);

  int *mis = NULL;
  allocate_host(this->get_nrow(), &mis);
  memset(mis, 0, sizeof(int)*this->get_nrow());

  size = 0 ;

  for (int ai=0; ai<this->get_nrow(); ++ai) {

    if (mis[ai] == 0) {

      // set the node
      mis[ai] = 1;
      ++size ;

      //remove all nbh nodes (without diagonal)
      for (int aj=h_row_offset[ai]; aj<h_row_offset[ai+1]; ++aj)
        if (ai != h_col[aj])
          mis[h_col[aj]] = -1 ;
      
    }
  }

  int *h_perm = NULL;
  allocate_host(this->get_nrow(), &h_perm);

  int pos = 0;
  for (int ai=0; ai<this->get_nrow(); ++ai) {

    if (mis[ai] == 1) {

      h_perm[ai] = pos;
      ++pos;

    } else {

      h_perm[ai] = size + ai - pos;

    }

  }
  
  // Check the permutation
  //
  //  for (int ai=0; ai<this->get_nrow(); ++ai) {
  //    assert( h_perm[ai] >= 0 );
  //    assert( h_perm[ai] < this->get_nrow() );
  //  }


  cast_perm->Allocate(this->get_nrow());
  hipMemcpy(cast_perm->vec_, h_perm, permutation->get_size()*sizeof(int), hipMemcpyHostToDevice);

  free_host(&h_row_offset);
  free_host(&h_col);

  free_host(&h_perm);
  free_host(&mis);

  return true;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::MultiColoring(int &num_colors,
                                                       int **size_colors,
                                                       BaseVector<int> *permutation) const {

  assert(permutation != NULL);
  HIPAcceleratorVector<int> *cast_perm = dynamic_cast<HIPAcceleratorVector<int>*> (permutation);
  assert(cast_perm != NULL);

  // node colors (init value = 0 i.e. no color)
  int *color = NULL;
  int *h_row_offset = NULL;
  int *h_col = NULL;
  int size = this->get_nrow();
  allocate_host(size, &color);
  allocate_host(this->get_nrow()+1, &h_row_offset);
  allocate_host(this->get_nnz(), &h_col);

  hipMemcpy(h_row_offset, this->mat_.row_offset, (this->get_nrow()+1)*sizeof(int), hipMemcpyDeviceToHost);
  hipMemcpy(h_col, this->mat_.col, this->get_nnz()*sizeof(int), hipMemcpyDeviceToHost);

  memset(color, 0, size*sizeof(int));
  num_colors = 0;
  std::vector<bool> row_col;

  for (int ai=0; ai<this->get_nrow(); ++ai) {
    color[ai] = 1;
    row_col.clear();
    row_col.assign(num_colors+2, false);

    for (int aj=h_row_offset[ai]; aj<h_row_offset[ai+1]; ++aj)
      if (ai != h_col[aj])
        row_col[color[h_col[aj]]] = true;

    for (int aj=h_row_offset[ai]; aj<h_row_offset[ai+1]; ++aj)
      if (row_col[color[ai]] == true)
        ++color[ai];

    if (color[ai] > num_colors)
      num_colors = color[ai];

  }

  free_host(&h_row_offset);
  free_host(&h_col);

  allocate_host(num_colors, size_colors);
  set_to_zero_host(num_colors, *size_colors);

  int *offsets_color = NULL;
  allocate_host(num_colors, &offsets_color);
  memset(offsets_color, 0, sizeof(int)*num_colors);

  for (int i=0; i<this->get_nrow(); ++i) 
    ++(*size_colors)[color[i]-1];

  int total=0;
  for (int i=1; i<num_colors; ++i) {

    total += (*size_colors)[i-1];
    offsets_color[i] = total; 
    //   LOG_INFO("offsets = " << total);

  }

  int *h_perm = NULL;
  allocate_host(this->get_nrow(), &h_perm);

  for (int i=0; i<this->get_nrow(); ++i) {

    h_perm[i] = offsets_color[ color[i]-1 ] ;
    ++offsets_color[color[i]-1];

  }

  cast_perm->Allocate(this->get_nrow());
  hipMemcpy(cast_perm->vec_, h_perm, permutation->get_size()*sizeof(int), hipMemcpyHostToDevice);

  free_host(&h_perm);
  free_host(&color);
  free_host(&offsets_color);

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::Scale(const ValueType alpha) {

  if (this->get_nnz() > 0) {

    hipblasStatus_t stat_t;
    stat_t = hipblasTscal(HIPBLAS_HANDLE(this->local_backend_.HIP_blas_handle),
                          this->get_nnz(), &alpha, this->mat_.val, 1);
    CHECK_HIPBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ScaleDiagonal(const ValueType alpha) {

  if (this->get_nnz() > 0) {

    int nrow = this->get_nrow();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_scale_diagonal<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       nrow, this->mat_.row_offset, this->mat_.col,
                       HIPVal(alpha), HIPPtr(this->mat_.val));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ScaleOffDiagonal(const ValueType alpha) {

  if (this->get_nnz() > 0) {

    int nrow = this->get_nrow();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_scale_offdiagonal<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       nrow, this->mat_.row_offset, this->mat_.col,
                       HIPVal(alpha), HIPPtr(this->mat_.val));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::AddScalarDiagonal(const ValueType alpha) {

  if (this->get_nnz() > 0) {

    int nrow = this->get_nrow();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_add_diagonal<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       nrow, this->mat_.row_offset, this->mat_.col,
                       HIPVal(alpha), HIPPtr(this->mat_.val));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::AddScalarOffDiagonal(const ValueType alpha) {

  if (this->get_nnz() > 0) {

    int nrow = this->get_nrow();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_add_offdiagonal<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       nrow, this->mat_.row_offset, this->mat_.col,
                       HIPVal(alpha), HIPPtr(this->mat_.val));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::AddScalar(const ValueType alpha) {

  if (this->get_nnz() > 0) {

    int nnz = this->get_nnz();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nnz / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_buffer_addscalar<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       nnz, HIPVal(alpha), HIPPtr(this->mat_.val));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::DiagonalMatrixMultR(const BaseVector<ValueType> &diag) {

  assert(diag.get_size() == this->get_ncol());

  const HIPAcceleratorVector<ValueType> *cast_diag = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&diag);
  assert(cast_diag!= NULL);

  if (this->get_nnz() > 0) {

    int nrow = this->get_nrow();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_diagmatmult_r<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       nrow, this->mat_.row_offset, this->mat_.col,
                       HIPPtr(cast_diag->vec_), HIPPtr(this->mat_.val));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::DiagonalMatrixMultL(const BaseVector<ValueType> &diag) {

  assert(diag.get_size() == this->get_ncol());

  const HIPAcceleratorVector<ValueType> *cast_diag = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&diag);
  assert(cast_diag!= NULL);

  if (this->get_nnz() > 0) {

    int nrow = this->get_nrow();
    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_diagmatmult_l<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       nrow, this->mat_.row_offset,
                       HIPPtr(cast_diag->vec_), HIPPtr(this->mat_.val));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::MatMatMult(const BaseMatrix<ValueType> &A, const BaseMatrix<ValueType> &B) {

  assert(A.get_ncol() == B.get_nrow());
  assert(A.get_nrow() > 0);
  assert(B.get_ncol() > 0);
  assert(B.get_nrow() > 0);

  const HIPAcceleratorMatrixCSR<ValueType> *cast_mat_A = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*> (&A);
  const HIPAcceleratorMatrixCSR<ValueType> *cast_mat_B = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*> (&B);
  assert(cast_mat_A != NULL);
  assert(cast_mat_B != NULL);

  this->Clear();

  int m = cast_mat_A->get_nrow();
  int n = cast_mat_B->get_ncol();
  int k = cast_mat_B->get_nrow();
  int nnzC = 0;

  allocate_hip(m+1, &this->mat_.row_offset);
  CHECK_HIP_ERROR(__FILE__, __LINE__);
/*
  cusparseStatus_t stat_t;

  stat_t = cusparseSetPointerMode(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                  CUSPARSE_POINTER_MODE_HOST);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

  stat_t = cusparseXcsrgemmNnz(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               m, n, k,
                               cast_mat_A->mat_descr_, cast_mat_A->get_nnz(),
                               cast_mat_A->mat_.row_offset, cast_mat_A->mat_.col,
                               cast_mat_B->mat_descr_, cast_mat_B->get_nnz(),
                               cast_mat_B->mat_.row_offset, cast_mat_B->mat_.col,
                               this->mat_descr_, this->mat_.row_offset,
                               &nnzC);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
  allocate_hip(nnzC, &this->mat_.col);
  CHECK_HIP_ERROR(__FILE__, __LINE__);

  allocate_hip(nnzC, &this->mat_.val);
  CHECK_HIP_ERROR(__FILE__, __LINE__);

  this->nrow_ = m;
  this->ncol_ = n;
  this->nnz_  = nnzC;
/*
  stat_t = __cusparseXcsrgemm__(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                m, n, k,
                                // A
                                cast_mat_A->mat_descr_, cast_mat_A->get_nnz(),
                                cast_mat_A->mat_.val,
                                cast_mat_A->mat_.row_offset, cast_mat_A->mat_.col,
                                // B
                                cast_mat_B->mat_descr_, cast_mat_B->get_nnz(),
                                cast_mat_B->mat_.val,
                                cast_mat_B->mat_.row_offset, cast_mat_B->mat_.col,
                                // C
                                this->mat_descr_,
                                this->mat_.val,
                                this->mat_.row_offset, this->mat_.col);
  CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::Gershgorin(ValueType &lambda_min,
                                                    ValueType &lambda_max) const {
  return false;
}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::MatrixAdd(const BaseMatrix<ValueType> &mat, const ValueType alpha,
                                                   const ValueType beta, const bool structure) {

  if (this->get_nnz() > 0) {

    const HIPAcceleratorMatrixCSR<ValueType> *cast_mat = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*> (&mat);
    assert(cast_mat != NULL);

    assert(cast_mat->get_nrow() == this->get_nrow());
    assert(cast_mat->get_ncol() == this->get_ncol());
    assert(this    ->get_nnz() > 0);  
    assert(cast_mat->get_nnz() > 0);

    if (structure == false) {

      int nrow = this->get_nrow();
      dim3 BlockSize(this->local_backend_.HIP_block_size);
      dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

      hipLaunchKernelGGL((kernel_csr_add_csr_same_struct<ValueType, int>),
                         GridSize, BlockSize, 0, 0,
                         nrow,
                         this->mat_.row_offset, this->mat_.col,
                         cast_mat->mat_.row_offset,
                         cast_mat->mat_.col, cast_mat->mat_.val,
                         alpha, beta, this->mat_.val);
      CHECK_HIP_ERROR(__FILE__, __LINE__);

    } else {
      // New structure with rocSPARSE routines

      int m = this->get_nrow();
      int n = this->get_ncol();
      int *csrRowPtrC = NULL;
      int *csrColC = NULL;
      ValueType *csrValC = NULL;
      int nnzC;
/*
      allocate_hip(m+1, &csrRowPtrC);

      cusparseStatus_t stat_t;

      cusparseMatDescr_t desc_mat_C = 0;

      stat_t = cusparseCreateMatDescr(&desc_mat_C);
      CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

      stat_t = cusparseSetMatIndexBase(desc_mat_C, CUSPARSE_INDEX_BASE_ZERO);
      CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

      stat_t = cusparseSetMatType(desc_mat_C, CUSPARSE_MATRIX_TYPE_GENERAL);
      CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

      stat_t = cusparseSetPointerMode(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                      CUSPARSE_POINTER_MODE_HOST);
      CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

      stat_t = cusparseXcsrgeamNnz(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                   m, n,
                                   this->mat_descr_, this->get_nnz(),
                                   this->mat_.row_offset, this->mat_.col,
                                   cast_mat->mat_descr_, cast_mat->get_nnz(),
                                   cast_mat->mat_.row_offset, cast_mat->mat_.col,
                                   desc_mat_C, csrRowPtrC,
                                   &nnzC);
      CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

      allocate_hip(nnzC, &csrColC);
      allocate_hip(nnzC, &csrValC);

      stat_t = __cusparseXcsrgeam__(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                                    m, n,
                                    // A
                                    &alpha,
                                    this->mat_descr_, this->get_nnz(),
                                    this->mat_.val,
                                    this->mat_.row_offset, this->mat_.col,
                                    // B
                                    &beta,
                                    cast_mat->mat_descr_, cast_mat->get_nnz(),
                                    cast_mat->mat_.val,
                                    cast_mat->mat_.row_offset, cast_mat->mat_.col,
                                    // C
                                    desc_mat_C,
                                    csrValC,
                                    csrRowPtrC, csrColC);

      CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);

      stat_t = cusparseDestroyMatDescr(desc_mat_C);
      CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
      this->Clear();

      this->mat_.row_offset = csrRowPtrC;
      this->mat_.col = csrColC;
      this->mat_.val = csrValC;

      this->nrow_ = m;
      this->ncol_ = n;
      this->nnz_  = nnzC;

    }

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::Compress(const double drop_off) {

  if (this->get_nnz() > 0) {

    HIPAcceleratorMatrixCSR<ValueType> tmp(this->local_backend_);

    tmp.CopyFrom(*this);

    int mat_nnz = 0;

    int *row_offset = NULL;
    allocate_hip(this->get_nrow()+1, &row_offset);

    int *mat_row_offset = NULL;
    allocate_hip(this->get_nrow()+1, &mat_row_offset);
    
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    this->local_backend_.HIP_max_threads,
                    this->get_nrow()+1, row_offset); 


    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(this->get_nrow() / this->local_backend_.HIP_block_size + 1);
 
    hipLaunchKernelGGL((kernel_csr_compress_count_nrow<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       this->mat_.row_offset, this->mat_.col,
                       HIPPtr(this->mat_.val), this->get_nrow(),
                       drop_off, row_offset);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    // TODO
    cum_sum<int, 256>(mat_row_offset, row_offset, this->get_nrow());

    // get the new mat nnz
    hipMemcpy(&mat_nnz, &mat_row_offset[this->get_nrow()],
               sizeof(int), hipMemcpyDeviceToHost);

    this->AllocateCSR(mat_nnz, this->get_nrow(), this->get_ncol());

    // TODO - just exchange memory pointers
    // copy row_offset
    hipMemcpy(this->mat_.row_offset, mat_row_offset,
               (this->get_nrow()+1)*sizeof(int), hipMemcpyDeviceToDevice);

    // copy col and val

    hipLaunchKernelGGL((kernel_csr_compress_copy<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       tmp.mat_.row_offset, tmp.mat_.col, HIPPtr(tmp.mat_.val),
                       tmp.get_nrow(), drop_off, this->mat_.row_offset,
                       this->mat_.col, HIPPtr(this->mat_.val));
    CHECK_HIP_ERROR(__FILE__, __LINE__);      

    free_hip(&row_offset);
    free_hip(&mat_row_offset);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::Transpose(void) {

  if (this->get_nnz() > 0) {

    HIPAcceleratorMatrixCSR<ValueType> tmp(this->local_backend_);

    tmp.CopyFrom(*this);

    this->Clear();
    this->AllocateCSR(tmp.get_nnz(), tmp.get_ncol(), tmp.get_nrow());
/*
    cusparseStatus_t stat_t;

    stat_t = cusparseDcsr2csc(CUSPARSE_HANDLE(this->local_backend_.HIP_sparse_handle),
                              tmp.get_nrow(), tmp.get_ncol(), tmp.get_nnz(),
                              tmp.mat_.val, tmp.mat_.row_offset, tmp.mat_.col,
                              this->mat_.val, this->mat_.col, this->mat_.row_offset,
                              CUSPARSE_ACTION_NUMERIC,
                              CUSPARSE_INDEX_BASE_ZERO);

    CHECK_CUSPARSE_ERROR(stat_t, __FILE__, __LINE__);
*/
  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ReplaceColumnVector(const int idx, const BaseVector<ValueType> &vec) {

  assert(vec.get_size() == this->nrow_);

  if (this->get_nnz() > 0) {

    const HIPAcceleratorVector<ValueType> *cast_vec = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&vec);
    assert(cast_vec != NULL);

    int *row_offset = NULL;
    int *col = NULL;
    ValueType *val = NULL;

    int nrow = this->get_nrow();
    int ncol = this->get_ncol();

    allocate_hip(nrow+1, &row_offset);

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_replace_column_vector_offset<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       this->mat_.row_offset, this->mat_.col, nrow,
                       idx, HIPPtr(cast_vec->vec_), row_offset);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    int *host_offset = NULL;
    allocate_host(nrow+1, &host_offset);

    hipMemcpy(host_offset,
               row_offset,
               sizeof(int)*(nrow+1),
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    host_offset[0] = 0;
    for (int i=0; i<nrow; ++i)
      host_offset[i+1] += host_offset[i];

    int nnz  = host_offset[nrow];

    hipMemcpy(row_offset,
               host_offset,
               sizeof(int)*(nrow+1),
               hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    allocate_hip(nnz, &col);
    allocate_hip(nnz, &val);

    hipLaunchKernelGGL((kernel_csr_replace_column_vector<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       this->mat_.row_offset, this->mat_.col,
                       HIPPtr(this->mat_.val), nrow, idx,
                       HIPPtr(cast_vec->vec_), row_offset, col,
                       HIPPtr(val));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    this->Clear();
    this->SetDataPtrCSR(&row_offset, &col, &val, nnz, nrow, ncol);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractColumnVector(const int idx, BaseVector<ValueType> *vec) const {

  assert(vec != NULL);
  assert(vec->get_size() == this->nrow_);

  if (this->get_nnz() > 0) {

    HIPAcceleratorVector<ValueType> *cast_vec = dynamic_cast<HIPAcceleratorVector<ValueType>*> (vec);
    assert(cast_vec != NULL);

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(this->get_nrow() / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_extract_column_vector<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       this->mat_.row_offset, this->mat_.col,
                       HIPPtr(this->mat_.val), this->get_nrow(),
                       idx, HIPPtr(cast_vec->vec_));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCSR<ValueType>::ExtractRowVector(const int idx, BaseVector<ValueType> *vec) const {

  assert(vec != NULL);
  assert(vec->get_size() == this->ncol_);

  if (this->get_nnz() > 0) {

    HIPAcceleratorVector<ValueType> *cast_vec = dynamic_cast<HIPAcceleratorVector<ValueType>*> (vec);
    assert(cast_vec != NULL);

    cast_vec->Zeros();

    // Get nnz of row idx
    int nnz[2];

    hipMemcpy(nnz,
               this->mat_.row_offset+idx,
               2*sizeof(int),
               hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    int row_nnz = nnz[1] - nnz[0];

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(row_nnz / this->local_backend_.HIP_block_size + 1);

    hipLaunchKernelGGL((kernel_csr_extract_row_vector<ValueType, int>),
                       GridSize, BlockSize, 0, 0,
                       this->mat_.row_offset, this->mat_.col,
                       HIPPtr(this->mat_.val), row_nnz, idx,
                       HIPPtr(cast_vec->vec_));
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  return true;

}


template class HIPAcceleratorMatrixCSR<double>;
template class HIPAcceleratorMatrixCSR<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorMatrixCSR<std::complex<double> >;
template class HIPAcceleratorMatrixCSR<std::complex<float> >;
#endif

}
