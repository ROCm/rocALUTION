#include "../../utils/def.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_matrix_bcsr.hpp"
#include "hip_vector.hpp"
#include "../host/host_matrix_bcsr.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_bcsr.hpp"
#include "hip_allocate_free.hpp"
#include "../matrix_formats_ind.hpp"

#include <cuda.h>
#include <cusparse_v2.h>
#include <cuComplex.h>

namespace paralution {

template <typename ValueType>
HIPAcceleratorMatrixBCSR<ValueType>::HIPAcceleratorMatrixBCSR() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HIPAcceleratorMatrixBCSR<ValueType>::HIPAcceleratorMatrixBCSR(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "HIPAcceleratorMatrixBCSR::HIPAcceleratorMatrixBCSR()",
            "constructor with local_backend");

  this->set_backend(local_backend); 

  CHECK_HIP_ERROR(__FILE__, __LINE__);

  // this is not working anyway...
  FATAL_ERROR(__FILE__, __LINE__);
}


template <typename ValueType>
HIPAcceleratorMatrixBCSR<ValueType>::~HIPAcceleratorMatrixBCSR() {

  LOG_DEBUG(this, "HIPAcceleratorMatrixBCSR::~HIPAcceleratorMatrixBCSR()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void HIPAcceleratorMatrixBCSR<ValueType>::info(void) const {

  LOG_INFO("HIPAcceleratorMatrixBCSR<ValueType>");

}

template <typename ValueType>
void HIPAcceleratorMatrixBCSR<ValueType>::AllocateBCSR(const int nnz, const int nrow, const int ncol) {

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
void HIPAcceleratorMatrixBCSR<ValueType>::Clear() {

  if (this->get_nnz() > 0) {

    FATAL_ERROR(__FILE__, __LINE__);


  }


}

template <typename ValueType>
void HIPAcceleratorMatrixBCSR<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixBCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to HIP copy
  if ((cast_mat = dynamic_cast<const HostMatrixBCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateBCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    cast_mat->get_nnz();

    FATAL_ERROR(__FILE__, __LINE__);    
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixBCSR<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixBCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixBCSR<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateBCSR(this->get_nnz(), this->get_nrow(), this->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    FATAL_ERROR(__FILE__, __LINE__);    
   
    
  } else {
    
    LOG_INFO("Error unsupported HIP matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void HIPAcceleratorMatrixBCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const HIPAcceleratorMatrixBCSR<ValueType> *hip_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixBCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateBCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );  

    assert(this->get_nnz()  == src.get_nnz());
    assert(this->get_nrow() == src.get_nrow());
    assert(this->get_ncol() == src.get_ncol());

    hip_cast_mat->get_nnz();

    FATAL_ERROR(__FILE__, __LINE__);    

    
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
void HIPAcceleratorMatrixBCSR<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  HIPAcceleratorMatrixBCSR<ValueType> *hip_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // HIP to HIP copy
  if ((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixBCSR<ValueType>*> (dst)) != NULL) {

    hip_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    hip_cast_mat->AllocateBCSR(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert(this->get_nnz()  == dst->get_nnz());
    assert(this->get_nrow() == dst->get_nrow());
    assert(this->get_ncol() == dst->get_ncol());

    FATAL_ERROR(__FILE__, __LINE__);    
    
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
bool HIPAcceleratorMatrixBCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;


  const HIPAcceleratorMatrixBCSR<ValueType>   *cast_mat_bcsr;
  if ((cast_mat_bcsr = dynamic_cast<const HIPAcceleratorMatrixBCSR<ValueType>*> (&mat)) != NULL) {

      this->CopyFrom(*cast_mat_bcsr);
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
void HIPAcceleratorMatrixBCSR<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {
/*
  assert(in.  get_size() >= 0);
  assert(out->get_size() >= 0);
  assert(in.  get_size() == this->get_ncol());
  assert(out->get_size() == this->get_nrow());


  const HIPAcceleratorVector<ValueType> *cast_in = dynamic_cast<const HIPAcceleratorVector<ValueType>*> (&in) ; 
  HIPAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      HIPAcceleratorVector<ValueType>*> (out) ; 

  assert(cast_in != NULL);
  assert(cast_out!= NULL);
*/
  FATAL_ERROR(__FILE__, __LINE__);    

}


template <typename ValueType>
void HIPAcceleratorMatrixBCSR<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                                  BaseVector<ValueType> *out) const {
  FATAL_ERROR(__FILE__, __LINE__);
}


template class HIPAcceleratorMatrixBCSR<double>;
template class HIPAcceleratorMatrixBCSR<float>;
#ifdef SUPPORT_COMPLEX
template class HIPAcceleratorMatrixBCSR<std::complex<double> >;
template class HIPAcceleratorMatrixBCSR<std::complex<float> >;
#endif

}
