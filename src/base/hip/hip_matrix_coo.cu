#include "../../utils/def.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_matrix_coo.hpp"
#include "hip_vector.hpp"
#include "../host/host_matrix_coo.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_coo.hpp"
#include "hip_allocate_free.hpp"
#include "../matrix_formats_ind.hpp"

#include <algorithm>

#include <cuda.h>
#include <cusparse_v2.h>
#include <cuComplex.h>

namespace paralution {

template <typename ValueType>
HIPAcceleratorMatrixCOO<ValueType>::HIPAcceleratorMatrixCOO() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HIPAcceleratorMatrixCOO<ValueType>::HIPAcceleratorMatrixCOO(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "HIPAcceleratorMatrixCOO::HIPAcceleratorMatrixCOO()",
            "constructor with local_backend");

  this->mat_.row = NULL;  
  this->mat_.col = NULL;  
  this->mat_.val = NULL;
  this->set_backend(local_backend); 

  CHECK_HIP_ERROR(__FILE__, __LINE__);

}


template <typename ValueType>
HIPAcceleratorMatrixCOO<ValueType>::~HIPAcceleratorMatrixCOO() {

  LOG_DEBUG(this, "HIPAcceleratorMatrixCOO::~HIPAcceleratorMatrixCOO()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::info(void) const {

  LOG_INFO("HIPAcceleratorMatrixCOO<ValueType>");

}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::AllocateCOO(const int nnz, const int nrow, const int ncol) {

  assert(nnz >= 0);
  assert(ncol >= 0);
  assert(nrow >= 0);

  if (this->get_nnz() > 0)
    this->Clear();

  if (nnz > 0) {

    allocate_hip(nnz, &this->mat_.row);
    allocate_hip(nnz, &this->mat_.col);
    allocate_hip(nnz, &this->mat_.val);
 
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    this->local_backend_.HIP_max_threads,
                    nnz, this->mat_.row);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    this->local_backend_.HIP_max_threads,
                    nnz, this->mat_.col);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    this->local_backend_.HIP_max_threads,
                    nnz, this->mat_.val);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::SetDataPtrCOO(int **row, int **col, ValueType **val,
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
void HIPAcceleratorMatrixCOO<ValueType>::LeaveDataPtrCOO(int **row, int **col, ValueType **val) {

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
void HIPAcceleratorMatrixCOO<ValueType>::Clear() {

  if (this->get_nnz() > 0) {

    free_hip(&this->mat_.row);
    free_hip(&this->mat_.col);
    free_hip(&this->mat_.val);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

  }


}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixCOO<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to HIP copy
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
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.col,     // dst
                 cast_mat->mat_.col, // src
                 this->get_nnz()*sizeof(int), // size
                 cudaMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.val,     // dst
                 cast_mat->mat_.val, // src
                 this->get_nnz()*sizeof(ValueType), // size
                 cudaMemcpyHostToDevice);    
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
void HIPAcceleratorMatrixCOO<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixCOO<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to CPU copy
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
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(cast_mat->mat_.col, // dst
                 this->mat_.col,     // src
                 this->get_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(cast_mat->mat_.val, // dst
                 this->mat_.val,     // src
                 this->get_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToHost);    
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
void HIPAcceleratorMatrixCOO<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixCOO<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixCOO<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateCOO(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpy(this->mat_.row,         // dst
                 hip_cast_mat->mat_.row, // src
                 (this->get_nnz())*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.col,         // dst
                 hip_cast_mat->mat_.col, // src
                 this->get_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.val,         // dst
                 hip_cast_mat->mat_.val, // src
                 this->get_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToDevice);    
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
void HIPAcceleratorMatrixCOO<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixCOO<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixCOO<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    hip_cast_mat->AllocateCOO(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpy(hip_cast_mat->mat_.row, // dst
                 this->mat_.row,         // src
                 (this->get_nnz())*sizeof(int), // size
                 cudaMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(hip_cast_mat->mat_.col, // dst
                 this->mat_.col,         // src
                 this->get_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(hip_cast_mat->mat_.val, // dst
                 this->mat_.val,         // src
                 this->get_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToHost);    
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
void HIPAcceleratorMatrixCOO<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  const HostMatrixCOO<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to HIP copy
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
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(this->mat_.col,     // dst
                      cast_mat->mat_.col, // src
                      this->get_nnz()*sizeof(int), // size
                      cudaMemcpyHostToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(this->mat_.val,     // dst
                      cast_mat->mat_.val, // src
                      this->get_nnz()*sizeof(ValueType), // size
                      cudaMemcpyHostToDevice);    
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
void HIPAcceleratorMatrixCOO<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  HostMatrixCOO<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to CPU copy
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
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(cast_mat->mat_.col, // dst
                      this->mat_.col,     // src
                      this->get_nnz()*sizeof(int), // size
                      cudaMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpyAsync(cast_mat->mat_.val, // dst
                      this->mat_.val,     // src
                      this->get_nnz()*sizeof(ValueType), // size
                      cudaMemcpyDeviceToHost);    
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
void HIPAcceleratorMatrixCOO<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixCOO<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixCOO<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateCOO(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpy(this->mat_.row,         // dst
                 hip_cast_mat->mat_.row, // src
                 (this->get_nnz())*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.col,         // dst
                 hip_cast_mat->mat_.col, // src
                 this->get_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToDevice);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(this->mat_.val,         // dst
                 hip_cast_mat->mat_.val, // src
                 this->get_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToDevice);    
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
void HIPAcceleratorMatrixCOO<ValueType>::CopyToAsync(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixCOO<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixCOO<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    hip_cast_mat->AllocateCOO(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_nnz() > 0) {

      cudaMemcpy(hip_cast_mat->mat_.row, // dst
                 this->mat_.row,         // src
                 (this->get_nnz())*sizeof(int), // size
                 cudaMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(hip_cast_mat->mat_.col, // dst
                 this->mat_.col,         // src
                 this->get_nnz()*sizeof(int), // size
                 cudaMemcpyDeviceToHost);
      CHECK_HIP_ERROR(__FILE__, __LINE__);     
      
      cudaMemcpy(hip_cast_mat->mat_.val, // dst
                 this->mat_.val,         // src
                 this->get_nnz()*sizeof(ValueType), // size
                 cudaMemcpyDeviceToHost);    
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
void HIPAcceleratorMatrixCOO<ValueType>::CopyFromCOO(const int *row, const int *col, const ValueType *val) {

  // assert CSR format
  assert(this->get_mat_format() == COO);

  if (this->get_nnz() > 0) {

    assert(this->nrow_ > 0);
    assert(this->ncol_ > 0);

    cudaMemcpy(this->mat_.row,              // dst
               row,                         // src
               this->get_nnz()*sizeof(int), // size
               cudaMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    cudaMemcpy(this->mat_.col,              // dst
               col,                         // src
               this->get_nnz()*sizeof(int), // size
               cudaMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    cudaMemcpy(this->mat_.val,                    // dst
               val,                               // src
               this->get_nnz()*sizeof(ValueType), // size
               cudaMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::CopyToCOO(int *row, int *col, ValueType *val) const {

  // assert CSR format
  assert(this->get_mat_format() == COO);

  if (this->get_nnz() > 0) {

    assert(this->nrow_ > 0);
    assert(this->ncol_ > 0);

    cudaMemcpy(row,                         // dst
               this->mat_.row,              // src
               this->get_nnz()*sizeof(int), // size
               cudaMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    cudaMemcpy(col,                         // dst
               this->mat_.col,              // src
               this->get_nnz()*sizeof(int), // size
               cudaMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    cudaMemcpy(val,                               // dst
               this->mat_.val,                    // src
               this->get_nnz()*sizeof(ValueType), // size
               cudaMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
bool HIPAcceleratorMatrixCOO<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const HIPAcceleratorMatrixCOO<ValueType> *cast_mat_coo;
  if ((cast_mat_coo = dynamic_cast<const HIPAcceleratorMatrixCOO<ValueType>*> (&mat)) != NULL) {

      this->CopyFrom(*cast_mat_coo);
      return true;

  }

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const HIPAcceleratorMatrixCSR<ValueType> *cast_mat_csr;
  if ((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {

    this->Clear();

    assert(cast_mat_csr->get_nrow() > 0);
    assert(cast_mat_csr->get_ncol() > 0);
    assert(cast_mat_csr->get_nnz() > 0);

    this->AllocateCOO(cast_mat_csr->nnz_, cast_mat_csr->nrow_, cast_mat_csr->ncol_);

    int nrow = cast_mat_csr->nrow_;

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    kernel_coo_csr_to_coo<int> <<<GridSize, BlockSize>>>(nrow, cast_mat_csr->mat_.row_offset, this->mat_.row);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    cudaMemcpy(this->mat_.col, cast_mat_csr->mat_.col, this->nnz_*sizeof(int), cudaMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    cudaMemcpy(this->mat_.val, cast_mat_csr->mat_.val, this->nnz_*sizeof(ValueType), cudaMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

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
void HIPAcceleratorMatrixCOO<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in);
    HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    cast_out->Zeros();

    ValueType one = (ValueType) 1;

    // If nnz < warpsize, perform sequential spmv
    if (this->get_nnz() < this->local_backend_.HIP_warp) {

      kernel_spmv_coo_serial<<<1, 1>>> (this->get_nnz(), this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val),
                                        HIPVal(one), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_));
      CHECK_HIP_ERROR(__FILE__, __LINE__);

    } else {

      int warps_per_block = this->local_backend_.HIP_block_size / this->local_backend_.HIP_warp;
      int numWarps = (this->local_backend_.HIP_block_size + (this->local_backend_.HIP_warp - 1)) / this->local_backend_.HIP_warp;

      int ctaLimitThreads = this->local_backend_.HIP_threads_per_proc / this->local_backend_.HIP_block_size;
      int ctaLimitRegs = this->local_backend_.HIP_max_threads / this->local_backend_.HIP_block_size / numWarps;

      int max_blocks = this->local_backend_.HIP_num_procs * std::min(ctaLimitRegs, std::min(ctaLimitThreads, 16));
      int num_units  = this->nnz_ / this->local_backend_.HIP_warp;
      int num_warps  = (num_units < warps_per_block * max_blocks) ? num_units : warps_per_block * max_blocks;
      int num_blocks = (num_warps + (warps_per_block - 1)) / warps_per_block;
      int num_iters  = (num_units + (num_warps - 1)) / num_warps;

      int interval_size = this->local_backend_.HIP_warp * num_iters;
      int tail = num_units * this->local_backend_.HIP_warp;
      int active_warps = (interval_size == 0) ? 0 : (tail + (interval_size - 1)) / interval_size;

      int *temp_rows = NULL;
      ValueType *temp_vals = NULL;

      allocate_hip(active_warps, &temp_rows);
      allocate_hip(active_warps, &temp_vals);

      // Run the appropriate kernel
      if (this->local_backend_.HIP_warp == 32) {
        if (this->local_backend_.HIP_block_size == 512) {
          kernel_spmv_coo_flat<512, 32> <<<num_blocks, 512>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(one), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
          kernel_spmv_coo_reduce_update<512> <<<1, 512>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
        } else if (this->local_backend_.HIP_block_size == 256) {
          kernel_spmv_coo_flat<256, 32> <<<num_blocks, 256>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(one), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
          kernel_spmv_coo_reduce_update<256> <<<1, 256>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
        } else if (this->local_backend_.HIP_block_size == 128) {
          kernel_spmv_coo_flat<128, 32> <<<num_blocks, 128>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(one), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
          kernel_spmv_coo_reduce_update<128> <<<1, 128>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
        } else if (this->local_backend_.HIP_block_size ==  64) {
          kernel_spmv_coo_flat<64, 32> <<<num_blocks, 64>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(one), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
          kernel_spmv_coo_reduce_update<64> <<<1, 64>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
        } else if (this->local_backend_.HIP_block_size ==  32) {
          kernel_spmv_coo_flat<32, 32> <<<num_blocks, 32>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(one), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
          kernel_spmv_coo_reduce_update<32> <<<1, 32>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
        } else {
          LOG_INFO("Unsupported HIP blocksize of " << this->local_backend_.HIP_block_size);
          CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
      } else if (this->local_backend_.HIP_warp == 64) {
        if (this->local_backend_.HIP_block_size == 512) {
          kernel_spmv_coo_flat<512, 64> <<<num_blocks, 512>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(one), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
          kernel_spmv_coo_reduce_update<512> <<<1, 512>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
        } else if (this->local_backend_.HIP_block_size == 256) {
          kernel_spmv_coo_flat<256, 64> <<<num_blocks, 256>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(one), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
          kernel_spmv_coo_reduce_update<256> <<<1, 256>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
        } else if (this->local_backend_.HIP_block_size == 128) {
          kernel_spmv_coo_flat<128, 64> <<<num_blocks, 128>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(one), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
          kernel_spmv_coo_reduce_update<128> <<<1, 128>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
        } else if (this->local_backend_.HIP_block_size ==  64) {
          kernel_spmv_coo_flat<64, 64> <<<num_blocks, 64>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(one), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
          kernel_spmv_coo_reduce_update<64> <<<1, 64>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
        } else {
          LOG_INFO("Unsupported HIP blocksize of " << this->local_backend_.HIP_block_size);
          CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
      } else {
        LOG_INFO("Unsupported HIP warpsize of " << this->local_backend_.HIP_warp);
        CHECK_HIP_ERROR(__FILE__, __LINE__);
      }

      free_hip(&temp_rows);
      free_hip(&temp_vals);

      kernel_spmv_coo_serial<<<1, 1>>>(this->get_nnz()-tail, this->mat_.row+tail, this->mat_.col+tail, HIPPtr(this->mat_.val+tail), HIPVal(one), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_));
      CHECK_HIP_ERROR(__FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixCOO<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
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

    // If matrix is very sparse, we do COO via atomics
    if (this->nnz_ / this->nrow_ < 2) {

      dim3 BlockSize(this->local_backend_.HIP_block_size);
      dim3 GridSize(this->get_nnz() / this->local_backend_.HIP_block_size + 1);

      kernel_spmv_add_coo<<<GridSize, BlockSize>>> (this->get_nnz(), this->mat_.row, this->mat_.col,
                                                    HIPPtr(this->mat_.val), HIPPtr(cast_in->vec_),
                                                    HIPVal(scalar), HIPPtr(cast_out->vec_));
      CHECK_HIP_ERROR(__FILE__, __LINE__);

    } else {

      // If nnz < warpsize, perform sequential spmv
      if (this->get_nnz() < this->local_backend_.HIP_warp) {

        kernel_spmv_coo_serial<<<1, 1>>> (this->get_nnz(), this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val),
                                          HIPVal(scalar), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

      } else {

        int warps_per_block = this->local_backend_.HIP_block_size / this->local_backend_.HIP_warp;
        int numWarps = (this->local_backend_.HIP_block_size + (this->local_backend_.HIP_warp - 1)) / this->local_backend_.HIP_warp;

        int ctaLimitThreads = this->local_backend_.HIP_threads_per_proc / this->local_backend_.HIP_block_size;
        int ctaLimitRegs = this->local_backend_.HIP_max_threads / this->local_backend_.HIP_block_size / numWarps;

        int max_blocks = this->local_backend_.HIP_num_procs * std::min(ctaLimitRegs, std::min(ctaLimitThreads, 16));
        int num_units  = this->nnz_ / this->local_backend_.HIP_warp;
        int num_warps  = (num_units < warps_per_block * max_blocks) ? num_units : warps_per_block * max_blocks;
        int num_blocks = (num_warps + (warps_per_block - 1)) / warps_per_block;
        int num_iters  = (num_units + (num_warps - 1)) / num_warps;

        int interval_size = this->local_backend_.HIP_warp * num_iters;
        int tail = num_units * this->local_backend_.HIP_warp;
        int active_warps = (interval_size == 0) ? 0 : (tail + (interval_size - 1)) / interval_size;

        int *temp_rows = NULL;
        ValueType *temp_vals = NULL;

        allocate_hip(active_warps, &temp_rows);
        allocate_hip(active_warps, &temp_vals);

        // Run the appropriate kernel
        if (this->local_backend_.HIP_warp == 32) {
          if (this->local_backend_.HIP_block_size == 512) {
            kernel_spmv_coo_flat<512, 32> <<<num_blocks, 512>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(scalar), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
            kernel_spmv_coo_reduce_update<512> <<<1, 512>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
          } else if (this->local_backend_.HIP_block_size == 256) {
            kernel_spmv_coo_flat<256, 32> <<<num_blocks, 256>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(scalar), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
            kernel_spmv_coo_reduce_update<256> <<<1, 256>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
          } else if (this->local_backend_.HIP_block_size == 128) {
            kernel_spmv_coo_flat<128, 32> <<<num_blocks, 128>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(scalar), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
            kernel_spmv_coo_reduce_update<128> <<<1, 128>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
          } else if (this->local_backend_.HIP_block_size ==  64) {
            kernel_spmv_coo_flat<64, 32> <<<num_blocks, 64>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(scalar), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
            kernel_spmv_coo_reduce_update<64> <<<1, 64>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
          } else if (this->local_backend_.HIP_block_size ==  32) {
            kernel_spmv_coo_flat<32, 32> <<<num_blocks, 32>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(scalar), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
            kernel_spmv_coo_reduce_update<32> <<<1, 32>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
          } else {
            LOG_INFO("Unsupported HIP blocksize of " << this->local_backend_.HIP_block_size);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
          }
        } else if (this->local_backend_.HIP_warp == 64) {
          if (this->local_backend_.HIP_block_size == 512) {
            kernel_spmv_coo_flat<512, 64> <<<num_blocks, 512>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(scalar), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
            kernel_spmv_coo_reduce_update<512> <<<1, 512>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
          } else if (this->local_backend_.HIP_block_size == 256) {
            kernel_spmv_coo_flat<256, 64> <<<num_blocks, 256>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(scalar), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
            kernel_spmv_coo_reduce_update<256> <<<1, 256>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
          } else if (this->local_backend_.HIP_block_size == 128) {
            kernel_spmv_coo_flat<128, 64> <<<num_blocks, 128>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(scalar), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
            kernel_spmv_coo_reduce_update<128> <<<1, 128>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
          } else if (this->local_backend_.HIP_block_size ==  64) {
            kernel_spmv_coo_flat<64, 64> <<<num_blocks, 64>>>(tail, interval_size, this->mat_.row, this->mat_.col, HIPPtr(this->mat_.val), HIPVal(scalar), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_), temp_rows, HIPPtr(temp_vals));
            kernel_spmv_coo_reduce_update<64> <<<1, 64>>>(active_warps, temp_rows, HIPPtr(temp_vals), HIPPtr(cast_out->vec_));
          } else {
            LOG_INFO("Unsupported HIP blocksize of " << this->local_backend_.HIP_block_size);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
          }
        } else {
          LOG_INFO("Unsupported HIP warpsize of " << this->local_backend_.HIP_warp);
          CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        free_hip(&temp_rows);
        free_hip(&temp_vals);

        kernel_spmv_coo_serial<<<1, 1>>>(this->get_nnz()-tail, this->mat_.row+tail, this->mat_.col+tail, HIPPtr(this->mat_.val+tail), HIPVal(scalar), HIPPtr(cast_in->vec_), HIPPtr(cast_out->vec_));
        CHECK_HIP_ERROR(__FILE__, __LINE__);

      }

    }

  }

}

template <typename ValueType>
bool HIPAcceleratorMatrixCOO<ValueType>::Permute(const BaseVector<int> &permutation) {

  // symmetric permutation only
  assert(permutation.get_size() == this->get_nrow());
  assert(permutation.get_size() == this->get_ncol());

  if (this->get_nnz() > 0) {

    const HIPAcceleratorVector<int> *cast_perm = dynamic_cast<const HIPAcceleratorVector<int>*> (&permutation);
    assert(cast_perm != NULL);

    HIPAcceleratorMatrixCOO<ValueType> src(this->local_backend_);
    src.AllocateCOO(this->get_nnz(), this->get_nrow(), this->get_ncol());
    src.CopyFrom(*this);

    int nnz = this->get_nnz();
    int s = nnz;
    int k = (nnz/this->local_backend_.HIP_block_size)/this->local_backend_.HIP_max_threads + 1;
    if (k > 1) s = nnz / k;

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(s / this->local_backend_.HIP_block_size + 1);

    kernel_coo_permute<ValueType, int> <<<GridSize, BlockSize>>> (nnz,
                                                                  src.mat_.row, src.mat_.col,
                                                                  cast_perm->vec_,
                                                                  this->mat_.row, this->mat_.col);

    CHECK_HIP_ERROR(__FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixCOO<ValueType>::PermuteBackward(const BaseVector<int> &permutation) {

  // symmetric permutation only
  assert(permutation.get_size() == this->get_nrow());
  assert(permutation.get_size() == this->get_ncol());

  if (this->get_nnz() > 0) {

    const HIPAcceleratorVector<int> *cast_perm = dynamic_cast<const HIPAcceleratorVector<int>*> (&permutation);
    assert(cast_perm != NULL);

    int *pb = NULL;
    allocate_hip(this->get_nrow(), &pb);

    int n = this->get_nrow();
    dim3 BlockSize1(this->local_backend_.HIP_block_size);
    dim3 GridSize1(n / this->local_backend_.HIP_block_size + 1);

    kernel_reverse_index<int> <<<GridSize1, BlockSize1>>> (n,
                                                           cast_perm->vec_,
                                                           pb);

    HIPAcceleratorMatrixCOO<ValueType> src(this->local_backend_);
    src.AllocateCOO(this->get_nnz(), this->get_nrow(), this->get_ncol());
    src.CopyFrom(*this);

    int nnz = this->get_nnz();
    int s = nnz;
    int k = (nnz/this->local_backend_.HIP_block_size)/this->local_backend_.HIP_max_threads + 1;
    if (k > 1) s = nnz / k;

    dim3 BlockSize2(this->local_backend_.HIP_block_size);
    dim3 GridSize2(s / this->local_backend_.HIP_block_size + 1);

    kernel_coo_permute<ValueType, int> <<<GridSize2, BlockSize2>>> (nnz,
                                                                    src.mat_.row, src.mat_.col,
                                                                    pb,
                                                                    this->mat_.row, this->mat_.col);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    free_hip(&pb);

  }

  return true;

}


template class HIPAcceleratorMatrixCOO<double>;
template class HIPAcceleratorMatrixCOO<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorMatrixCOO<std::complex<double> >;
template class HIPAcceleratorMatrixCOO<std::complex<float> >;
#endif

}
