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
#include "../host/host_matrix_coo.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "mic_utils.hpp"
#include "mic_allocate_free.hpp"
#include "mic_matrix_coo_kernel.hpp"
#include "../matrix_formats_ind.hpp"



namespace paralution {

template <typename ValueType>
MICAcceleratorMatrixCOO<ValueType>::MICAcceleratorMatrixCOO() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
MICAcceleratorMatrixCOO<ValueType>::MICAcceleratorMatrixCOO(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "MICAcceleratorMatrixCOO::MICAcceleratorMatrixCOO()",
            "constructor with local_backend");

  this->mat_.row = NULL;  
  this->mat_.col = NULL;  
  this->mat_.val = NULL;
  this->set_backend(local_backend); 

}


template <typename ValueType>
MICAcceleratorMatrixCOO<ValueType>::~MICAcceleratorMatrixCOO() {

  LOG_DEBUG(this, "MICAcceleratorMatrixCOO::~MICAcceleratorMatrixCOO()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void MICAcceleratorMatrixCOO<ValueType>::info(void) const {

  LOG_INFO("MICAcceleratorMatrixCOO<ValueType>");

}

template <typename ValueType>
void MICAcceleratorMatrixCOO<ValueType>::AllocateCOO(const int nnz, const int nrow, const int ncol) {

  assert(nnz >= 0);
  assert(ncol >= 0);
  assert(nrow >= 0);

  if (this->get_nnz() > 0)
    this->Clear();

  if (nnz > 0) {

    allocate_mic(this->local_backend_.MIC_dev,
		 nnz, &this->mat_.row);
    allocate_mic(this->local_backend_.MIC_dev,
		 nnz, &this->mat_.col);
    allocate_mic(this->local_backend_.MIC_dev,
		 nnz, &this->mat_.val);
 
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    nnz, this->mat_.row);
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    nnz, this->mat_.col);
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    nnz, this->mat_.val);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

  }

}

template <typename ValueType>
void MICAcceleratorMatrixCOO<ValueType>::Clear() {

  if (this->get_nnz() > 0) {

    free_mic(this->local_backend_.MIC_dev,
	     &this->mat_.row);
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
void MICAcceleratorMatrixCOO<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixCOO<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to MIC copy
  if ((cast_mat = dynamic_cast<const HostMatrixCOO<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateCOO(src.get_nnz(), src.get_nrow(), src.get_ncol() );

  if (this->get_nnz() > 0) {

      assert((this->get_nnz()  == src.get_nnz())  &&
             (this->get_nrow() == src.get_nrow()) &&
             (this->get_ncol() == src.get_ncol()) );

      
      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_mat->mat_.row, this->mat_.row, this->get_nnz());
      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_mat->mat_.col, this->mat_.col, this->get_nnz());
      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_mat->mat_.val, this->mat_.val, this->get_nnz());
      
    }
    
  } else {
    
    LOG_INFO("Error unsupported MIC matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void MICAcceleratorMatrixCOO<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixCOO<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // MIC to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixCOO<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateCOO(this->get_nnz(), this->get_nrow(), this->get_ncol() );

  if (this->get_nnz() > 0) {

      assert((this->get_nnz()  == dst->get_nnz())  &&
             (this->get_nrow() == dst->get_nrow()) &&
             (this->get_ncol() == dst->get_ncol()) );

      copy_to_host(this->local_backend_.MIC_dev,
		   this->mat_.row, cast_mat->mat_.row, this->get_nnz());
      copy_to_host(this->local_backend_.MIC_dev,
		   this->mat_.col, cast_mat->mat_.col, this->get_nnz());
      copy_to_host(this->local_backend_.MIC_dev,
		   this->mat_.val, cast_mat->mat_.val, this->get_nnz());

    }
    
  } else {
    
    LOG_INFO("Error unsupported MIC matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void MICAcceleratorMatrixCOO<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const MICAcceleratorMatrixCOO<ValueType> *mic_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // MIC to MIC copy
  if ((mic_cast_mat = dynamic_cast<const MICAcceleratorMatrixCOO<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateCOO(src.get_nnz(), src.get_nrow(), src.get_ncol() );

    assert((this->get_nnz()  == src.get_nnz())  &&
	   (this->get_nrow() == src.get_nrow()) &&
	   (this->get_ncol() == src.get_ncol()) );

    if (this->get_nnz() > 0) {

      copy_mic_mic(this->local_backend_.MIC_dev,
		   mic_cast_mat->mat_.row, this->mat_.row, this->get_nnz());
      copy_mic_mic(this->local_backend_.MIC_dev,
		   mic_cast_mat->mat_.col, this->mat_.col, this->get_nnz());
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
void MICAcceleratorMatrixCOO<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  MICAcceleratorMatrixCOO<ValueType> *mic_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // MIC to MIC copy
  if ((mic_cast_mat = dynamic_cast<MICAcceleratorMatrixCOO<ValueType>*> (dst)) != NULL) {

    mic_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    mic_cast_mat->AllocateCOO(dst->get_nnz(), dst->get_nrow(), dst->get_ncol() );

    assert((this->get_nnz()  == dst->get_nnz())  &&
	   (this->get_nrow() == dst->get_nrow()) &&
	   (this->get_ncol() == dst->get_ncol()) );

    if (this->get_nnz() > 0) {

      copy_mic_mic(this->local_backend_.MIC_dev,
		   this->mat_.row, mic_cast_mat->mat_.row, this->get_nnz());
      copy_mic_mic(this->local_backend_.MIC_dev,
		   this->mat_.col, mic_cast_mat->mat_.col, this->get_nnz());
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
bool MICAcceleratorMatrixCOO<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  return false;

}

// TODO
// MKL

template <typename ValueType>
void MICAcceleratorMatrixCOO<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->get_nnz() > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->get_ncol());
    assert(out->get_size() == this->get_nrow());
    
    
    const MICAcceleratorVector<ValueType> *cast_in = dynamic_cast<const MICAcceleratorVector<ValueType>*> (&in) ; 
    MICAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      MICAcceleratorVector<ValueType>*> (out) ; 
    
    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    spmv_coo(this->local_backend_.MIC_dev,
	     this->mat_.row,
	     this->mat_.col,
	     this->mat_.val,
	     this->get_nrow(),
	     this->get_nnz(),
	     cast_in->vec_,
	     cast_out->vec_);

  }

}


template <typename ValueType>
void MICAcceleratorMatrixCOO<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
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

    spmv_add_coo(this->local_backend_.MIC_dev,
		 this->mat_.row,
		 this->mat_.col,
		 this->mat_.val,
		 this->get_nrow(),
		 this->get_nnz(),
		 scalar,
		 cast_in->vec_,
		 cast_out->vec_);

  }

}


template <typename ValueType>
bool MICAcceleratorMatrixCOO<ValueType>::Permute(const BaseVector<int> &permutation) {

  return false;

}


template <typename ValueType>
bool MICAcceleratorMatrixCOO<ValueType>::PermuteBackward(const BaseVector<int> &permutation) {

  return false;

}


template class MICAcceleratorMatrixCOO<double>;
template class MICAcceleratorMatrixCOO<float>;

}
