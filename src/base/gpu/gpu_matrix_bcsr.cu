#include "../../utils/def.hpp"
#include "gpu_matrix_csr.hpp"
#include "gpu_matrix_bcsr.hpp"
#include "gpu_vector.hpp"
#include "../host/host_matrix_bcsr.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "gpu_utils.hpp"
#include "cuda_kernels_general.hpp"
#include "cuda_kernels_bcsr.hpp"
#include "gpu_allocate_free.hpp"
#include "../matrix_formats_ind.hpp"

#include <cuda.h>
#include <cusparse_v2.h>
#include <cuComplex.h>

namespace paralution {

template <typename ValueType>
GPUAcceleratorMatrixBCSR<ValueType>::GPUAcceleratorMatrixBCSR() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
GPUAcceleratorMatrixBCSR<ValueType>::GPUAcceleratorMatrixBCSR(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "GPUAcceleratorMatrixBCSR::GPUAcceleratorMatrixBCSR()",
            "constructor with local_backend");

  this->set_backend(local_backend); 

  CHECK_CUDA_ERROR(__FILE__, __LINE__);

  // this is not working anyway...
  FATAL_ERROR(__FILE__, __LINE__);
}


template <typename ValueType>
GPUAcceleratorMatrixBCSR<ValueType>::~GPUAcceleratorMatrixBCSR() {

  LOG_DEBUG(this, "GPUAcceleratorMatrixBCSR::~GPUAcceleratorMatrixBCSR()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void GPUAcceleratorMatrixBCSR<ValueType>::info(void) const {

  LOG_INFO("GPUAcceleratorMatrixBCSR<ValueType>");

}

template <typename ValueType>
void GPUAcceleratorMatrixBCSR<ValueType>::AllocateBCSR(const int nnz, const int nrow, const int ncol) {

  assert(nnz >= 0);
  assert(ncol >= 0);
  assert(nrow >= 0);

  if (this->get_nnz() > 0)
    this->Clear();

  if (nnz > 0) {

    FATAL_ERROR(__FILE__, __LINE__);
   

  }

}

template <typename ValueType>
void GPUAcceleratorMatrixBCSR<ValueType>::Clear() {

  if (this->get_nnz() > 0) {

    FATAL_ERROR(__FILE__, __LINE__);


  }


}

template <typename ValueType>
void GPUAcceleratorMatrixBCSR<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixBCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to GPU copy
  if ((cast_mat = dynamic_cast<const HostMatrixBCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateBCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    cast_mat->get_nnz();

    FATAL_ERROR(__FILE__, __LINE__);    
    
  } else {
    
    LOG_INFO("Error unsupported GPU matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void GPUAcceleratorMatrixBCSR<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixBCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixBCSR<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateBCSR(this->get_nnz(), this->get_nrow(), this->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    FATAL_ERROR(__FILE__, __LINE__);    
   
    
  } else {
    
    LOG_INFO("Error unsupported GPU matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void GPUAcceleratorMatrixBCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const GPUAcceleratorMatrixBCSR<ValueType> *gpu_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<const GPUAcceleratorMatrixBCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateBCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );  

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    gpu_cast_mat->get_nnz();

    FATAL_ERROR(__FILE__, __LINE__);    

    
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
void GPUAcceleratorMatrixBCSR<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  GPUAcceleratorMatrixBCSR<ValueType> *gpu_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // GPU to GPU copy
  if ((gpu_cast_mat = dynamic_cast<GPUAcceleratorMatrixBCSR<ValueType>*> (dst)) != NULL) {

    gpu_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    gpu_cast_mat->AllocateBCSR(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    FATAL_ERROR(__FILE__, __LINE__);    
    
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
bool GPUAcceleratorMatrixBCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;


  const GPUAcceleratorMatrixBCSR<ValueType>   *cast_mat_bcsr;
  if ((cast_mat_bcsr = dynamic_cast<const GPUAcceleratorMatrixBCSR<ValueType>*> (&mat)) != NULL) {

      this->CopyFrom(*cast_mat_bcsr);
      return true;

  }

  /*
  const GPUAcceleratorMatrixCSR<ValueType>   *cast_mat_csr;
  if ((cast_mat_csr = dynamic_cast<const GPUAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {
    
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
void GPUAcceleratorMatrixBCSR<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {
/*
  assert(in.  get_size() >= 0);
  assert(out->get_size() >= 0);
  assert(in.  get_size() == this->get_ncol());
  assert(out->get_size() == this->get_nrow());


  const GPUAcceleratorVector<ValueType> *cast_in = dynamic_cast<const GPUAcceleratorVector<ValueType>*> (&in) ; 
  GPUAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      GPUAcceleratorVector<ValueType>*> (out) ; 

  assert(cast_in != NULL);
  assert(cast_out!= NULL);
*/
  FATAL_ERROR(__FILE__, __LINE__);    

}


template <typename ValueType>
void GPUAcceleratorMatrixBCSR<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                                  BaseVector<ValueType> *out) const {
  FATAL_ERROR(__FILE__, __LINE__);
}


template class GPUAcceleratorMatrixBCSR<double>;
template class GPUAcceleratorMatrixBCSR<float>;
#ifdef SUPPORT_COMPLEX
template class GPUAcceleratorMatrixBCSR<std::complex<double> >;
template class GPUAcceleratorMatrixBCSR<std::complex<float> >;
#endif

}
