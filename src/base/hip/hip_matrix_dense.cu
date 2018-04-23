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
#include "../host/host_matrix_dense.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_dense.hpp"
#include "hip_allocate_free.hpp"
#include "../matrix_formats_ind.hpp"

#include <cuda.h>
#include <cuComplex.h>

namespace paralution {

template <typename ValueType>
HIPAcceleratorMatrixDENSE<ValueType>::HIPAcceleratorMatrixDENSE() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HIPAcceleratorMatrixDENSE<ValueType>::HIPAcceleratorMatrixDENSE(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "HIPAcceleratorMatrixDENSE::HIPAcceleratorMatrixDENSE()",
            "constructor with local_backend");

  this->mat_.val = NULL;
  this->set_backend(local_backend); 

  CHECK_HIP_ERROR(__FILE__, __LINE__);

}


template <typename ValueType>
HIPAcceleratorMatrixDENSE<ValueType>::~HIPAcceleratorMatrixDENSE() {

  LOG_DEBUG(this, "HIPAcceleratorMatrixDENSE::~HIPAcceleratorMatrixDENSE()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void HIPAcceleratorMatrixDENSE<ValueType>::info(void) const {

  LOG_INFO("HIPAcceleratorMatrixDENSE<ValueType>");

}

template <typename ValueType>
void HIPAcceleratorMatrixDENSE<ValueType>::AllocateDENSE(const int nrow, const int ncol) {

  assert( ncol  >= 0);
  assert( nrow  >= 0);

  if (this->get_nnz() > 0)
    this->Clear();

  if (nrow*ncol > 0) {

    allocate_hip(nrow*ncol, &this->mat_.val);
    set_to_zero_hip(this->local_backend_.HIP_block_size, 
                    this->local_backend_.HIP_max_threads,
                    nrow*ncol, mat_.val);   

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nrow*ncol;

  }


}

template <typename ValueType>
void HIPAcceleratorMatrixDENSE<ValueType>::Clear() {

  if (this->get_nnz() > 0) {

    free_hip(&this->mat_.val);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

  }

}

template <typename ValueType>
void HIPAcceleratorMatrixDENSE<ValueType>::SetDataPtrDENSE(ValueType **val, const int nrow, const int ncol) {

  assert(*val != NULL);
  assert(nrow > 0);
  assert(ncol > 0);

  this->Clear();

  cudaDeviceSynchronize();

  this->nrow_ = nrow;
  this->ncol_ = ncol;
  this->nnz_  = nrow*ncol;

  this->mat_.val = *val;

}

template <typename ValueType>
void HIPAcceleratorMatrixDENSE<ValueType>::LeaveDataPtrDENSE(ValueType **val) {

  assert(this->nrow_ > 0);
  assert(this->ncol_ > 0);
  assert(this->nnz_ > 0);
  assert(this->nnz_  == this->nrow_*this->ncol_);

  cudaDeviceSynchronize();

  *val = this->mat_.val;

  this->mat_.val = NULL;

  this->nrow_ = 0;
  this->ncol_ = 0;
  this->nnz_  = 0;

}

template <typename ValueType>
void HIPAcceleratorMatrixDENSE<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixDENSE<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixDENSE<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateDENSE(src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_nnz() > 0) {

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
void HIPAcceleratorMatrixDENSE<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixDENSE<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixDENSE<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateDENSE(this->get_nrow(), this->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_nnz() > 0) {
      
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
void HIPAcceleratorMatrixDENSE<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixDENSE<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixDENSE<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateDENSE(src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_nnz() > 0) { 

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
void HIPAcceleratorMatrixDENSE<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixDENSE<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixDENSE<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    hip_cast_mat->AllocateDENSE(dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_nnz() > 0) {

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
void HIPAcceleratorMatrixDENSE<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType> &src) {

  const HostMatrixDENSE<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixDENSE<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateDENSE(src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_nnz() > 0) {

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
void HIPAcceleratorMatrixDENSE<ValueType>::CopyToHostAsync(HostMatrix<ValueType> *dst) const {

  HostMatrixDENSE<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixDENSE<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateDENSE(this->get_nrow(), this->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_nnz() > 0) {
      
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
void HIPAcceleratorMatrixDENSE<ValueType>::CopyFromAsync(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixDENSE<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixDENSE<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateDENSE(src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    if (this->get_nnz() > 0) { 

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
void HIPAcceleratorMatrixDENSE<ValueType>::CopyToAsync(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixDENSE<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixDENSE<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    hip_cast_mat->AllocateDENSE(dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    if (this->get_nnz() > 0) {

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
bool HIPAcceleratorMatrixDENSE<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const HIPAcceleratorMatrixDENSE<ValueType>   *cast_mat_dense;
  
  if ((cast_mat_dense = dynamic_cast<const HIPAcceleratorMatrixDENSE<ValueType>*> (&mat)) != NULL) {

      this->CopyFrom(*cast_mat_dense);
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

template <>
void HIPAcceleratorMatrixDENSE<double>::Apply(const BaseVector<double> &in, BaseVector<double> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const HIPAcceleratorVector<double> *cast_in = dynamic_cast<const HIPAcceleratorVector<double>*> (&in);
    HIPAcceleratorVector<double> *cast_out      = dynamic_cast<      HIPAcceleratorVector<double>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    cublasStatus_t stat_t;

    const double alpha = double(1.0);
    const double beta  = double(0.0);

    if (DENSE_IND_BASE == 0) {

      stat_t = cublasDgemv(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), CUBLAS_OP_N,
                           this->get_nrow(), this->get_ncol(),
                           &alpha,
                           this->mat_.val, this->get_nrow(),
                           cast_in->vec_, 1,
                           &beta, cast_out->vec_, 1);

      CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    } else {

      stat_t = cublasDgemv(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), CUBLAS_OP_T,
                           this->get_ncol(), this->get_nrow(),
                           &alpha,
                           this->mat_.val, this->get_ncol(),
                           cast_in->vec_, 1,
                           &beta, cast_out->vec_, 1);

      CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    }

  }

}

template <>
void HIPAcceleratorMatrixDENSE<float>::Apply(const BaseVector<float> &in, BaseVector<float> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const HIPAcceleratorVector<float> *cast_in = dynamic_cast<const HIPAcceleratorVector<float>*> (&in);
    HIPAcceleratorVector<float> *cast_out      = dynamic_cast<      HIPAcceleratorVector<float>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    cublasStatus_t stat_t;

    const float alpha = float(1.0);
    const float beta  = float(0.0);

    if (DENSE_IND_BASE == 0) {

      stat_t = cublasSgemv(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), CUBLAS_OP_N,
                           this->get_nrow(), this->get_ncol(),
                           &alpha,
                           this->mat_.val, this->get_nrow(),
                           cast_in->vec_, 1,
                           &beta, cast_out->vec_, 1);

      CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    } else {

      stat_t = cublasSgemv(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), CUBLAS_OP_T,
                           this->get_ncol(), this->get_nrow(),
                           &alpha,
                           this->mat_.val, this->get_ncol(),
                           cast_in->vec_, 1,
                           &beta, cast_out->vec_, 1);

      CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    }

  }

}

template <>
void HIPAcceleratorMatrixDENSE<std::complex<double> >::Apply(const BaseVector<std::complex<double> > &in,
                                                             BaseVector<std::complex<double> > *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const HIPAcceleratorVector<std::complex<double> > *cast_in = dynamic_cast<const HIPAcceleratorVector<std::complex<double> >*> (&in);
    HIPAcceleratorVector<std::complex<double> > *cast_out      = dynamic_cast<      HIPAcceleratorVector<std::complex<double> >*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    cublasStatus_t stat_t;
    cublasOperation_t trans;
    trans=CUBLAS_OP_N;

    cuDoubleComplex cu_alpha;
    cuDoubleComplex cu_beta;

    cu_alpha.x = double(1.0);
    cu_alpha.y = double(0.0);
    cu_beta.x  = double(0.0);
    cu_beta.y  = double(0.0);

    stat_t = cublasZgemv(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), trans,
                         this->get_nrow(), this->get_ncol(),
                         &cu_alpha,
                         (cuDoubleComplex*)this->mat_.val, this->get_nrow(),
                         (cuDoubleComplex*)cast_in->vec_, 1,
                         &cu_beta, (cuDoubleComplex*)cast_out->vec_, 1);

    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorMatrixDENSE<std::complex<float> >::Apply(const BaseVector<std::complex<float> > &in,
                                                            BaseVector<std::complex<float> > *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const HIPAcceleratorVector<std::complex<float> > *cast_in = dynamic_cast<const HIPAcceleratorVector<std::complex<float> >*> (&in);
    HIPAcceleratorVector<std::complex<float> > *cast_out      = dynamic_cast<      HIPAcceleratorVector<std::complex<float> >*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    cublasStatus_t stat_t;
    cublasOperation_t trans;
    trans=CUBLAS_OP_N;

    cuFloatComplex cu_alpha;
    cuFloatComplex cu_beta;

    cu_alpha.x = float(1.0);
    cu_alpha.y = float(0.0);
    cu_beta.x  = float(0.0);
    cu_beta.y  = float(0.0);

    stat_t = cublasCgemv(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), trans,
                         this->get_nrow(), this->get_ncol(),
                         &cu_alpha,
                         (cuFloatComplex*)this->mat_.val, this->get_nrow(),
                         (cuFloatComplex*)cast_in->vec_, 1,
                         &cu_beta, (cuFloatComplex*)cast_out->vec_, 1);

    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorMatrixDENSE<double>::ApplyAdd(const BaseVector<double> &in, const double scalar,
                                                 BaseVector<double> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const HIPAcceleratorVector<double> *cast_in = dynamic_cast<const HIPAcceleratorVector<double>*> (&in);
    HIPAcceleratorVector<double> *cast_out      = dynamic_cast<      HIPAcceleratorVector<double>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    cublasStatus_t stat_t;

    const double alpha = scalar;
    const double beta  = double(0.0);

    if (DENSE_IND_BASE == 0) {

      stat_t = cublasDgemv(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), CUBLAS_OP_N,
                           this->get_nrow(), this->get_ncol(),
                           &alpha,
                           this->mat_.val, this->get_nrow(),
                           cast_in->vec_, 1,
                           &beta, cast_out->vec_, 1);

      CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    } else {

      stat_t = cublasDgemv(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), CUBLAS_OP_T,
                           this->get_ncol(), this->get_nrow(),
                           &alpha,
                           this->mat_.val, this->get_ncol(),
                           cast_in->vec_, 1,
                           &beta, cast_out->vec_, 1);

      CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    }

  }

}

template <>
void HIPAcceleratorMatrixDENSE<float>::ApplyAdd(const BaseVector<float> &in, const float scalar,
                                                BaseVector<float> *out) const {
FATAL_ERROR(__FILE__, __LINE__);
  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const HIPAcceleratorVector<float> *cast_in = dynamic_cast<const HIPAcceleratorVector<float>*> (&in);
    HIPAcceleratorVector<float> *cast_out      = dynamic_cast<      HIPAcceleratorVector<float>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    cublasStatus_t stat_t;

    const float alpha = scalar;
    const float beta  = float(0.0);

    if (DENSE_IND_BASE == 0) {

      stat_t = cublasSgemv(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), CUBLAS_OP_N,
                           this->get_nrow(), this->get_ncol(),
                           &alpha,
                           this->mat_.val, this->get_nrow(),
                           cast_in->vec_, 1,
                           &beta, cast_out->vec_, 1);

      CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    } else {

      stat_t = cublasSgemv(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), CUBLAS_OP_T,
                           this->get_ncol(), this->get_nrow(),
                           &alpha,
                           this->mat_.val, this->get_ncol(),
                           cast_in->vec_, 1,
                           &beta, cast_out->vec_, 1);

      CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

    }

  }

}

template <>
void HIPAcceleratorMatrixDENSE<std::complex<double> >::ApplyAdd(const BaseVector<std::complex<double> > &in,
                                                                const std::complex<double> scalar,
                                                                BaseVector<std::complex<double> > *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const HIPAcceleratorVector<std::complex<double> > *cast_in = dynamic_cast<const HIPAcceleratorVector<std::complex<double> >*> (&in);
    HIPAcceleratorVector<std::complex<double> > *cast_out      = dynamic_cast<      HIPAcceleratorVector<std::complex<double> >*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    cublasStatus_t stat_t;
    cublasOperation_t trans;
    trans=CUBLAS_OP_N;

    cuDoubleComplex *cu_alpha = (cuDoubleComplex*)&scalar;
    cuDoubleComplex cu_beta;

    cu_beta.x  = double(0.0);
    cu_beta.y  = double(0.0);

    stat_t = cublasZgemv(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), trans,
                         this->get_nrow(), this->get_ncol(),
                         cu_alpha,
                         (cuDoubleComplex*)this->mat_.val, this->get_nrow(),
                         (cuDoubleComplex*)cast_in->vec_, 1,
                         &cu_beta, (cuDoubleComplex*)cast_out->vec_, 1);

    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

}

template <>
void HIPAcceleratorMatrixDENSE<std::complex<float> >::ApplyAdd(const BaseVector<std::complex<float> > &in,
                                                               const std::complex<float> scalar,
                                                               BaseVector<std::complex<float> > *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());

    const HIPAcceleratorVector<std::complex<float> > *cast_in = dynamic_cast<const HIPAcceleratorVector<std::complex<float> >*> (&in);
    HIPAcceleratorVector<std::complex<float> > *cast_out      = dynamic_cast<      HIPAcceleratorVector<std::complex<float> >*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    cublasStatus_t stat_t;
    cublasOperation_t trans;
    trans=CUBLAS_OP_N;

    cuFloatComplex *cu_alpha = (cuFloatComplex*)&scalar;
    cuFloatComplex cu_beta;

    cu_beta.x  = float(0.0);
    cu_beta.y  = float(0.0);

    stat_t = cublasCgemv(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), trans,
                         this->get_nrow(), this->get_ncol(),
                         cu_alpha,
                         (cuFloatComplex*)this->mat_.val, this->get_nrow(),
                         (cuFloatComplex*)cast_in->vec_, 1,
                         &cu_beta, (cuFloatComplex*)cast_out->vec_, 1);

    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

}

template <>
bool HIPAcceleratorMatrixDENSE<float>::MatMatMult(const BaseMatrix<float> &A, const BaseMatrix<float> &B) {

  assert((this != &A) && (this != &B));

  const HIPAcceleratorMatrixDENSE<float> *cast_mat_A = dynamic_cast<const HIPAcceleratorMatrixDENSE<float>*> (&A);
  const HIPAcceleratorMatrixDENSE<float> *cast_mat_B = dynamic_cast<const HIPAcceleratorMatrixDENSE<float>*> (&B);

  assert(cast_mat_A != NULL);
  assert(cast_mat_B != NULL);
  assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

  cublasStatus_t stat_t;

  const float alpha = float(1.0);
  const float beta  = float(0.0);

  if (DENSE_IND_BASE == 0) {

    stat_t = cublasSgemm(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), CUBLAS_OP_N, CUBLAS_OP_N,
                         cast_mat_A->nrow_, cast_mat_B->ncol_, cast_mat_A->ncol_,
                         &alpha, cast_mat_A->mat_.val, cast_mat_A->nrow_,
                         cast_mat_B->mat_.val, cast_mat_A->ncol_, &beta,
                         this->mat_.val, cast_mat_A->nrow_);

    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  } else {

    stat_t = cublasSgemm(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), CUBLAS_OP_T, CUBLAS_OP_T,
                         cast_mat_A->nrow_, cast_mat_B->ncol_, cast_mat_A->ncol_,
                         &alpha, cast_mat_A->mat_.val, cast_mat_A->ncol_,
                         cast_mat_B->mat_.val, cast_mat_B->ncol_, &beta,
                         this->mat_.val, cast_mat_A->nrow_);

    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return true;

}

template <>
bool HIPAcceleratorMatrixDENSE<double>::MatMatMult(const BaseMatrix<double> &A, const BaseMatrix<double> &B) {

  assert((this != &A) && (this != &B));

  const HIPAcceleratorMatrixDENSE<double> *cast_mat_A = dynamic_cast<const HIPAcceleratorMatrixDENSE<double>*> (&A);
  const HIPAcceleratorMatrixDENSE<double> *cast_mat_B = dynamic_cast<const HIPAcceleratorMatrixDENSE<double>*> (&B);

  assert(cast_mat_A != NULL);
  assert(cast_mat_B != NULL);
  assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

  cublasStatus_t stat_t;

  const double alpha = double(1.0);
  const double beta  = double(0.0);

  if (DENSE_IND_BASE == 0) {

    stat_t = cublasDgemm(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), CUBLAS_OP_N, CUBLAS_OP_N,
                         cast_mat_A->nrow_, cast_mat_B->ncol_, cast_mat_A->ncol_,
                         &alpha, cast_mat_A->mat_.val, cast_mat_A->nrow_,
                         cast_mat_B->mat_.val, cast_mat_A->ncol_, &beta,
                         this->mat_.val, cast_mat_A->nrow_);

    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  } else {

    stat_t = cublasDgemm(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), CUBLAS_OP_T, CUBLAS_OP_T,
                         cast_mat_A->nrow_, cast_mat_B->ncol_, cast_mat_A->ncol_,
                         &alpha, cast_mat_A->mat_.val, cast_mat_A->ncol_,
                         cast_mat_B->mat_.val, cast_mat_B->ncol_, &beta,
                         this->mat_.val, cast_mat_A->nrow_);

    CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  }

  return true;

}

template <>
bool HIPAcceleratorMatrixDENSE<std::complex<float> >::MatMatMult(const BaseMatrix<std::complex<float> > &A, const BaseMatrix<std::complex<float> > &B) {

  assert((this != &A) && (this != &B));

  const HIPAcceleratorMatrixDENSE<std::complex<float> > *cast_mat_A = dynamic_cast<const HIPAcceleratorMatrixDENSE<std::complex<float> >*> (&A);
  const HIPAcceleratorMatrixDENSE<std::complex<float> > *cast_mat_B = dynamic_cast<const HIPAcceleratorMatrixDENSE<std::complex<float> >*> (&B);

  assert(cast_mat_A != NULL);
  assert(cast_mat_B != NULL);
  assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

  cublasStatus_t stat_t;
  cublasOperation_t trans;
  trans=CUBLAS_OP_N;

  cuFloatComplex cu_alpha;
  cuFloatComplex cu_beta;

  cu_alpha.x = float(1.0);
  cu_alpha.y = float(0.0);

  cu_beta.x = float(0.0);
  cu_beta.y = float(0.0);

  stat_t = cublasCgemm(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), trans, trans,
                       cast_mat_A->nrow_, cast_mat_B->ncol_, cast_mat_A->ncol_,
                       &cu_alpha, (cuFloatComplex*)cast_mat_A->mat_.val, cast_mat_A->nrow_,
                       (cuFloatComplex*)cast_mat_B->mat_.val, cast_mat_B->nrow_, &cu_beta,
                       (cuFloatComplex*)this->mat_.val, cast_mat_A->nrow_);

  CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  return true;

}

template <>
bool HIPAcceleratorMatrixDENSE<std::complex<double> >::MatMatMult(const BaseMatrix<std::complex<double> > &A, const BaseMatrix<std::complex<double> > &B) {

  assert((this != &A) && (this != &B));

  const HIPAcceleratorMatrixDENSE<std::complex<double> > *cast_mat_A = dynamic_cast<const HIPAcceleratorMatrixDENSE<std::complex<double> >*> (&A);
  const HIPAcceleratorMatrixDENSE<std::complex<double> > *cast_mat_B = dynamic_cast<const HIPAcceleratorMatrixDENSE<std::complex<double> >*> (&B);

  assert(cast_mat_A != NULL);
  assert(cast_mat_B != NULL);
  assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

  cublasStatus_t stat_t;
  cublasOperation_t trans;
  trans=CUBLAS_OP_N;

  cuDoubleComplex cu_alpha;
  cuDoubleComplex cu_beta;

  cu_alpha.x = double(1.0);
  cu_alpha.y = double(0.0);

  cu_beta.x = double(0.0);
  cu_beta.y = double(0.0);

  stat_t = cublasZgemm(CUBLAS_HANDLE(this->local_backend_.HIP_rocblas_handle), trans, trans,
                       cast_mat_A->nrow_, cast_mat_B->ncol_, cast_mat_A->ncol_,
                       &cu_alpha, (cuDoubleComplex*)cast_mat_A->mat_.val, cast_mat_A->nrow_,
                       (cuDoubleComplex*)cast_mat_B->mat_.val, cast_mat_B->nrow_, &cu_beta,
                       (cuDoubleComplex*)this->mat_.val, cast_mat_A->nrow_);

  CHECK_CUBLAS_ERROR(stat_t, __FILE__, __LINE__);

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixDENSE<ValueType>::ReplaceColumnVector(const int idx, const BaseVector<ValueType> &vec) {

  assert(vec.get_size() == this->get_nrow());

  if (this->get_nnz() > 0) {

    const HIPAcceleratorVector<ValueType> *cast_vec = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&vec);
    assert(cast_vec != NULL);

    const int nrow = this->get_nrow();
    const int ncol = this->get_ncol();

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    kernel_dense_replace_column_vector<ValueType, int> <<<GridSize, BlockSize>>>(cast_vec->vec_,
                                                                                 idx,
                                                                                 nrow,
                                                                                 ncol,
                                                                                 this->mat_.val);

    CHECK_HIP_ERROR(__FILE__,__LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixDENSE<ValueType>::ReplaceRowVector(const int idx, const BaseVector<ValueType> &vec) {

  assert(vec.get_size() == this->get_ncol());

  if (this->get_nnz() > 0) {

    const HIPAcceleratorVector<ValueType> *cast_vec = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&vec);
    assert(cast_vec != NULL);

    const int nrow = this->get_nrow();
    const int ncol = this->get_ncol();

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(ncol / this->local_backend_.HIP_block_size + 1);

    kernel_dense_replace_row_vector<ValueType, int> <<<GridSize, BlockSize>>>(cast_vec->vec_,
                                                                              idx,
                                                                              nrow,
                                                                              ncol,
                                                                              this->mat_.val);

    CHECK_HIP_ERROR(__FILE__,__LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixDENSE<ValueType>::ExtractColumnVector(const int idx, BaseVector<ValueType> *vec) const {

    assert(vec != NULL);
    assert(vec->get_size() == this->get_nrow());

  if (this->get_nnz() > 0) {

    HIPAcceleratorVector<ValueType> *cast_vec = dynamic_cast<HIPAcceleratorVector<ValueType>*> (vec);
    assert(cast_vec != NULL);

    const int nrow = this->get_nrow();
    const int ncol = this->get_ncol();

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

    kernel_dense_extract_column_vector<ValueType, int> <<<GridSize, BlockSize>>>(cast_vec->vec_,
                                                                                 idx,
                                                                                 nrow,
                                                                                 ncol,
                                                                                 this->mat_.val);

    CHECK_HIP_ERROR(__FILE__,__LINE__);

  }

  return true;

}

template <typename ValueType>
bool HIPAcceleratorMatrixDENSE<ValueType>::ExtractRowVector(const int idx, BaseVector<ValueType> *vec) const {

    assert(vec != NULL);
    assert(vec->get_size() == this->get_ncol());

  if (this->get_nnz() > 0) {

    HIPAcceleratorVector<ValueType> *cast_vec = dynamic_cast<HIPAcceleratorVector<ValueType>*> (vec);
    assert(cast_vec != NULL);

    const int nrow = this->get_nrow();
    const int ncol = this->get_ncol();

    dim3 BlockSize(this->local_backend_.HIP_block_size);
    dim3 GridSize(ncol / this->local_backend_.HIP_block_size + 1);

    kernel_dense_extract_row_vector<ValueType, int> <<<GridSize, BlockSize>>>(cast_vec->vec_,
                                                                              idx,
                                                                              nrow,
                                                                              ncol,
                                                                              this->mat_.val);

    CHECK_HIP_ERROR(__FILE__,__LINE__);

  }

  return true;

}


template class HIPAcceleratorMatrixDENSE<double>;
template class HIPAcceleratorMatrixDENSE<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorMatrixDENSE<std::complex<double> >;
template class HIPAcceleratorMatrixDENSE<std::complex<float> >;
#endif

}
