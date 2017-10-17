#include "../../utils/def.hpp"
#include "mic_matrix_csr.hpp"
#include "mic_matrix_coo.hpp"
#include "mic_matrix_dia.hpp"
#include "mic_matrix_ell.hpp"
#include "mic_matrix_hyb.hpp"
#include "mic_matrix_mcsr.hpp"
#include "mic_matrix_bcsr.hpp"
#include "mic_matrix_dense.hpp"
#include "mic_vector.hpp"
#include "../host/host_matrix_mcsr.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "mic_utils.hpp"
#include "mic_allocate_free.hpp"
#include "mic_matrix_mcsr_kernel.hpp"
#include "../matrix_formats_ind.hpp"




namespace paralution {

template <typename ValueType>
MICAcceleratorMatrixMCSR<ValueType>::MICAcceleratorMatrixMCSR() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
MICAcceleratorMatrixMCSR<ValueType>::MICAcceleratorMatrixMCSR(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "MICAcceleratorMatrixMCSR::MICAcceleratorMatrixMCSR()",
            "constructor with local_backend");

  this->mat_.row_offset = NULL;  
  this->mat_.col = NULL;  
  this->mat_.val = NULL;
  this->set_backend(local_backend); 

}


template <typename ValueType>
MICAcceleratorMatrixMCSR<ValueType>::~MICAcceleratorMatrixMCSR() {

  LOG_DEBUG(this, "MICAcceleratorMatrixMCSR::~MICAcceleratorMatrixMCSR()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void MICAcceleratorMatrixMCSR<ValueType>::info(void) const {

  LOG_INFO("MICAcceleratorMatrixMCSR<ValueType>");

}

template <typename ValueType>
void MICAcceleratorMatrixMCSR<ValueType>::AllocateMCSR(const int nnz, const int nrow, const int ncol) {

  assert(nnz >= 0);
  assert(ncol >= 0);
  assert(nrow >= 0);

  if (this->get_nnz() > 0)
    this->Clear();

  if (nnz > 0) {

    allocate_mic(this->local_backend_.MIC_dev,
		 nrow+1, &this->mat_.row_offset);
    allocate_mic(this->local_backend_.MIC_dev,
		 nnz,    &this->mat_.col);
    allocate_mic(this->local_backend_.MIC_dev,
		 nnz,    &this->mat_.val);
    
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    nrow+1, mat_.row_offset);
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    nnz, mat_.col);
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    nnz, mat_.val);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

  }

}

template <typename ValueType>
void MICAcceleratorMatrixMCSR<ValueType>::Clear() {

  if (this->get_nnz() > 0) {

    free_mic(this->local_backend_.MIC_dev,
	     &this->mat_.row_offset);
    free_mic(this->local_backend_.MIC_dev,
	     &this->mat_.col);
    free_mic(this->local_backend_.MIC_dev,
	     &this->mat_.val);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

  }


}

template <typename ValueType>
void MICAcceleratorMatrixMCSR<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixMCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to MIC copy
  if ((cast_mat = dynamic_cast<const HostMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateMCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert((this->get_nnz()  == src.get_nnz())  &&
	   (this->get_nrow() == src.get_nrow()) &&
	   (this->get_ncol() == src.get_ncol()) );

    if (this->get_nnz() > 0) {

      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_mat->mat_.row_offset, this->mat_.row_offset,
		  this->get_nrow()+1);

      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_mat->mat_.col, this->mat_.col, 
		  this->get_nnz());

      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_mat->mat_.val, this->mat_.val,
		  this->get_nnz());

    }

  } else {
    
    LOG_INFO("Error unsupported MIC matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void MICAcceleratorMatrixMCSR<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixMCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // MIC to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixMCSR<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateMCSR(this->get_nnz(), this->get_nrow(), this->get_ncol() );

    assert((this->get_nnz()  == dst->get_nnz())  &&
	   (this->get_nrow() == dst->get_nrow()) &&
	   (this->get_ncol() == dst->get_ncol()) );

    if (this->get_nnz() > 0) {

      copy_to_host(this->local_backend_.MIC_dev,
		   this->mat_.row_offset, cast_mat->mat_.row_offset,
		   this->get_nrow()+1);

      copy_to_host(this->local_backend_.MIC_dev,
		   this->mat_.col, cast_mat->mat_.col,
		   this->get_nnz());

      copy_to_host(this->local_backend_.MIC_dev,
		   this->mat_.val, cast_mat->mat_.val,
		   this->get_nnz());

    }


  } else {
    
    LOG_INFO("Error unsupported MIC matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void MICAcceleratorMatrixMCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const MICAcceleratorMatrixMCSR<ValueType> *mic_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // MIC to MIC copy
  if ((mic_cast_mat = dynamic_cast<const MICAcceleratorMatrixMCSR<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateMCSR(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert((this->get_nnz()  == src.get_nnz())  &&
	   (this->get_nrow() == src.get_nrow()) &&
	   (this->get_ncol() == src.get_ncol()) );

    if (this->get_nnz() > 0) {

      copy_mic_mic(this->local_backend_.MIC_dev,
		   mic_cast_mat->mat_.row_offset, this->mat_.row_offset,
		   this->get_nrow()+1);

      copy_mic_mic(this->local_backend_.MIC_dev,
		   mic_cast_mat->mat_.col, this->mat_.col, 
		   this->get_nnz());

      copy_mic_mic(this->local_backend_.MIC_dev,
		   mic_cast_mat->mat_.val, this->mat_.val,
		   this->get_nnz());


      }


  } else {

    //CPU to MIC
    if ((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*> (&src)) != NULL) {
      
      this->CopyFromHost(*host_cast_mat);
      
    } else {
      
      LOG_INFO("Error unsupported MIC matrix type");
      this->info();
      src.info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }
    
  }

}

template <typename ValueType>
void MICAcceleratorMatrixMCSR<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  MICAcceleratorMatrixMCSR<ValueType> *mic_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // MIC to MIC copy
  if ((mic_cast_mat = dynamic_cast<MICAcceleratorMatrixMCSR<ValueType>*> (dst)) != NULL) {

    mic_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    mic_cast_mat->AllocateMCSR(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert((this->get_nnz()  == dst->get_nnz())  &&
	   (this->get_nrow() == dst->get_nrow()) &&
	   (this->get_ncol() == dst->get_ncol()) );

    if (this->get_nnz() > 0) {

      copy_mic_mic(this->local_backend_.MIC_dev,
		   this->mat_.row_offset, mic_cast_mat->mat_.row_offset,
		   this->get_nrow()+1);

      copy_mic_mic(this->local_backend_.MIC_dev,
		   this->mat_.col, mic_cast_mat->mat_.col,
		   this->get_nnz());

      copy_mic_mic(this->local_backend_.MIC_dev,
		   this->mat_.val, mic_cast_mat->mat_.val,
		   this->get_nnz());

      }

    
  } else {

    //MIC to CPU
    if ((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst)) != NULL) {
      
      this->CopyToHost(host_cast_mat);

    } else {
      
      LOG_INFO("Error unsupported MIC matrix type");
      this->info();
      dst->info();
      FATAL_ERROR(__FILE__, __LINE__);
      
    }

  }


}


template <typename ValueType>
bool MICAcceleratorMatrixMCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const MICAcceleratorMatrixMCSR<ValueType>   *cast_mat_mcsr;
  
  if ((cast_mat_mcsr = dynamic_cast<const MICAcceleratorMatrixMCSR<ValueType>*> (&mat)) != NULL) {

      this->CopyFrom(*cast_mat_mcsr);
      return true;

  }

  return false;

}

template <typename ValueType>
void MICAcceleratorMatrixMCSR<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());
    
    const MICAcceleratorVector<ValueType> *cast_in = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&in) ; 
    MICAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      MICAcceleratorVector<ValueType>*> (out) ; 
    
    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    spmv_mcsr(this->local_backend_.MIC_dev,
	      this->mat_.row_offset,
	      this->mat_.col,
	      this->mat_.val,
	      this->get_nrow(),
	      cast_in->vec_,
	      cast_out->vec_);
    
  }
}



template <typename ValueType>
void MICAcceleratorMatrixMCSR<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                                  BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());
    
    const MICAcceleratorVector<ValueType> *cast_in = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&in) ; 
    MICAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      MICAcceleratorVector<ValueType>*> (out) ; 
    
    assert(cast_in != NULL);
    assert(cast_out!= NULL);    

    spmv_add_mcsr(this->local_backend_.MIC_dev,
		  this->mat_.row_offset,
		  this->mat_.col,
		  this->mat_.val,
		  this->get_nrow(),
		  scalar,
		  cast_in->vec_,
		  cast_out->vec_);

  }
}

template class MICAcceleratorMatrixMCSR<double>;
template class MICAcceleratorMatrixMCSR<float>;

}
