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
#include "../host/host_matrix_ell.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "mic_utils.hpp"
#include "mic_allocate_free.hpp"
#include "../../utils/allocate_free.hpp"
#include "mic_matrix_ell_kernel.hpp"
#include "../matrix_formats_ind.hpp"


namespace paralution {

template <typename ValueType>
MICAcceleratorMatrixELL<ValueType>::MICAcceleratorMatrixELL() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
MICAcceleratorMatrixELL<ValueType>::MICAcceleratorMatrixELL(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "MICAcceleratorMatrixELL::MICAcceleratorMatrixELL()",
            "constructor with local_backend");

  this->mat_.val = NULL;
  this->mat_.col = NULL;
  this->mat_.max_row = 0;
  this->set_backend(local_backend); 

}


template <typename ValueType>
MICAcceleratorMatrixELL<ValueType>::~MICAcceleratorMatrixELL() {

  LOG_DEBUG(this, "MICAcceleratorMatrixELL::~MICAcceleratorMatrixELL()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void MICAcceleratorMatrixELL<ValueType>::info(void) const {

  LOG_INFO("MICAcceleratorMatrixELL<ValueType>");

}

template <typename ValueType>
void MICAcceleratorMatrixELL<ValueType>::AllocateELL(const int nnz, const int nrow, const int ncol, const int max_row) {

  assert( nnz   >= 0);
  assert( ncol  >= 0);
  assert( nrow  >= 0);
  assert( max_row >= 0);

  if (this->get_nnz() > 0)
    this->Clear();

  if (nnz > 0) {

    assert(nnz == max_row * nrow);

    allocate_mic(this->local_backend_.MIC_dev,
		 nnz, &this->mat_.val);
    allocate_mic(this->local_backend_.MIC_dev,
		 nnz, &this->mat_.col);
    
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    nnz, this->mat_.val);
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    nnz, this->mat_.col);
    
    this->mat_.max_row = max_row;
    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

  }

}

template <typename ValueType>
void MICAcceleratorMatrixELL<ValueType>::Clear() {

  if (this->get_nnz() > 0) {

    free_mic(this->local_backend_.MIC_dev,
	     &this->mat_.val);
    free_mic(this->local_backend_.MIC_dev,
	     &this->mat_.col);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

  }

}

template <typename ValueType>
void MICAcceleratorMatrixELL<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixELL<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to MIC copy
  if ((cast_mat = dynamic_cast<const HostMatrixELL<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateELL(cast_mat->get_nnz(), cast_mat->get_nrow(), cast_mat->get_ncol(), cast_mat->get_max_row());

    assert((this->get_nnz()  == src.get_nnz())  &&
	   (this->get_nrow() == src.get_nrow()) &&
	   (this->get_ncol() == src.get_ncol()) );

    if (this->get_nnz() > 0) { 

      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_mat->mat_.val, this->mat_.val, this->get_nnz());
      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_mat->mat_.col, this->mat_.col, this->get_nnz());

    }
    
  } else {
    
    LOG_INFO("Error unsupported MIC matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void MICAcceleratorMatrixELL<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixELL<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // MIC to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixELL<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateELL(this->get_nnz(), this->get_nrow(), this->get_ncol(), this->get_max_row() );

    assert((this->get_nnz()  == dst->get_nnz())  &&
	   (this->get_nrow() == dst->get_nrow()) &&
	   (this->get_ncol() == dst->get_ncol()) );

    if (this->get_nnz() > 0) {

      copy_to_host(this->local_backend_.MIC_dev,
		   this->mat_.val, cast_mat->mat_.val, this->get_nnz());
      copy_to_host(this->local_backend_.MIC_dev,
		   this->mat_.col, cast_mat->mat_.col, this->get_nnz());

    }
    
  } else {
    
    LOG_INFO("Error unsupported MIC matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void MICAcceleratorMatrixELL<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const MICAcceleratorMatrixELL<ValueType> *mic_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // MIC to MIC copy
  if ((mic_cast_mat = dynamic_cast<const MICAcceleratorMatrixELL<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateELL(mic_cast_mat->get_nnz(), mic_cast_mat->get_nrow(), mic_cast_mat->get_ncol(), mic_cast_mat->get_max_row() );

    assert((this->get_nnz()  == src.get_nnz())  &&
	   (this->get_nrow() == src.get_nrow()) &&
	   (this->get_ncol() == src.get_ncol()) );

    if (this->get_nnz() > 0) {

      copy_mic_mic(this->local_backend_.MIC_dev,
		   mic_cast_mat->mat_.val, this->mat_.val, this->get_nnz());
      copy_mic_mic(this->local_backend_.MIC_dev,
		   mic_cast_mat->mat_.val, this->mat_.val, this->get_nnz());

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
void MICAcceleratorMatrixELL<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  MICAcceleratorMatrixELL<ValueType> *mic_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // MIC to MIC copy
  if ((mic_cast_mat = dynamic_cast<MICAcceleratorMatrixELL<ValueType>*> (dst)) != NULL) {

    mic_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    mic_cast_mat->AllocateELL(mic_cast_mat->get_nnz(), mic_cast_mat->get_nrow(), mic_cast_mat->get_ncol(), mic_cast_mat->get_max_row() );

    assert((this->get_nnz()  == dst->get_nnz())  &&
	   (this->get_nrow() == dst->get_nrow()) &&
	   (this->get_ncol() == dst->get_ncol()) );

    if (this->get_nnz() > 0) {

      copy_mic_mic(this->local_backend_.MIC_dev,
		   this->mat_.val, mic_cast_mat->mat_.val, this->get_nnz());
      copy_mic_mic(this->local_backend_.MIC_dev,
		   this->mat_.val, mic_cast_mat->mat_.val, this->get_nnz());

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
bool MICAcceleratorMatrixELL<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const MICAcceleratorMatrixELL<ValueType>   *cast_mat_ell;
  
  if ((cast_mat_ell = dynamic_cast<const MICAcceleratorMatrixELL<ValueType>*> (&mat)) != NULL) {

    this->CopyFrom(*cast_mat_ell);
    return true;

  }

  return false;

}

template <typename ValueType>
void MICAcceleratorMatrixELL<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {
    
    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());
    
    
    const MICAcceleratorVector<ValueType> *cast_in = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&in) ; 
    MICAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      MICAcceleratorVector<ValueType>*> (out) ; 
    
    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    spmv_ell(this->local_backend_.MIC_dev,
	     this->mat_.col,
	     this->mat_.val,
	     this->get_nrow(),
	     this->get_ncol(),
	     this->get_max_row(),
	     cast_in->vec_,
	     cast_out->vec_);

  }
}


template <typename ValueType>
void MICAcceleratorMatrixELL<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
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

    spmv_add_ell(this->local_backend_.MIC_dev,
		 this->mat_.col,
		 this->mat_.val,
		 this->get_nrow(),
		 this->get_ncol(),
		 this->get_max_row(),
		 scalar,
		 cast_in->vec_,
		 cast_out->vec_);


  }
}


template class MICAcceleratorMatrixELL<double>;
template class MICAcceleratorMatrixELL<float>;

}
