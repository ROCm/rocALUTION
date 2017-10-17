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
#include "../host/host_matrix_hyb.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../backend_manager.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "mic_utils.hpp"
#include "mic_allocate_free.hpp"
#include "mic_matrix_coo_kernel.hpp"
#include "mic_matrix_ell_kernel.hpp"
#include "../matrix_formats_ind.hpp"



namespace paralution {

template <typename ValueType>
MICAcceleratorMatrixHYB<ValueType>::MICAcceleratorMatrixHYB() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
MICAcceleratorMatrixHYB<ValueType>::MICAcceleratorMatrixHYB(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "MICAcceleratorMatrixHYB::MICAcceleratorMatrixHYB()",
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

}


template <typename ValueType>
MICAcceleratorMatrixHYB<ValueType>::~MICAcceleratorMatrixHYB() {

  LOG_DEBUG(this, "MICAcceleratorMatrixHYB::~MICAcceleratorMatrixHYB()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void MICAcceleratorMatrixHYB<ValueType>::info(void) const {

  LOG_INFO("MICAcceleratorMatrixHYB<ValueType>");

}

template <typename ValueType>
void MICAcceleratorMatrixHYB<ValueType>::AllocateHYB(const int ell_nnz, const int coo_nnz, const int ell_max_row, 
                                                     const int nrow, const int ncol) {

  assert( ell_nnz   >= 0);
  assert( coo_nnz   >= 0);
  assert( ell_max_row >= 0);

  assert( ncol  >= 0);
  assert( nrow  >= 0);
  
  if (this->get_nnz() > 0)
    this->Clear();

   if (ell_nnz + coo_nnz > 0) {

    // ELL
    assert(ell_nnz == ell_max_row*nrow);

    allocate_mic(this->local_backend_.MIC_dev,
		 ell_nnz, &this->mat_.ELL.val);
    allocate_mic(this->local_backend_.MIC_dev,
		 ell_nnz, &this->mat_.ELL.col);
    
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    ell_nnz, this->mat_.ELL.val);
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    ell_nnz, this->mat_.ELL.col);

    this->mat_.ELL.max_row = ell_max_row;
    this->ell_nnz_ = ell_nnz;

    // COO
    allocate_mic(this->local_backend_.MIC_dev,
		 coo_nnz, &this->mat_.COO.row);
    allocate_mic(this->local_backend_.MIC_dev,
		 coo_nnz, &this->mat_.COO.col);
    allocate_mic(this->local_backend_.MIC_dev,
		 coo_nnz, &this->mat_.COO.val);
 
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    coo_nnz, this->mat_.COO.row);
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    coo_nnz, this->mat_.COO.col);
    set_to_zero_mic(this->local_backend_.MIC_dev,
		    coo_nnz, this->mat_.COO.val);
    this->coo_nnz_ = coo_nnz;

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = ell_nnz + coo_nnz;

  }

}


template <typename ValueType>
void MICAcceleratorMatrixHYB<ValueType>::Clear() {

  if (this->get_nnz() > 0) {
    
    free_mic(this->local_backend_.MIC_dev,
	     &this->mat_.COO.row);
    free_mic(this->local_backend_.MIC_dev,
	     &this->mat_.COO.col);
    free_mic(this->local_backend_.MIC_dev,
	     &this->mat_.COO.val);
    
    free_mic(this->local_backend_.MIC_dev,
	     &this->mat_.ELL.val);
    free_mic(this->local_backend_.MIC_dev,
	     &this->mat_.ELL.col);
    
    this->ell_nnz_ = 0;
    this->coo_nnz_ = 0;
    this->mat_.ELL.max_row = 0;
    
    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;
    
  }
  

}

template <typename ValueType>
void MICAcceleratorMatrixHYB<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // CPU to MIC copy
  if ((cast_mat = dynamic_cast<const HostMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateHYB(cast_mat->get_ell_nnz(), cast_mat->get_coo_nnz(), cast_mat->get_ell_max_row(),
                      cast_mat->get_nrow(), cast_mat->get_ncol());

    assert((this->get_nnz()  == src.get_nnz())  &&
	   (this->get_nrow() == src.get_nrow()) &&
	   (this->get_ncol() == src.get_ncol()) );

    if (this->get_ell_nnz() > 0) {

      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_mat->mat_.ELL.val, this->mat_.ELL.val, this->get_ell_nnz());
      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_mat->mat_.ELL.col, this->mat_.ELL.col, this->get_ell_nnz());
    }
    
    if (this->get_coo_nnz() > 0) {

      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_mat->mat_.COO.row, this->mat_.COO.row, this->get_coo_nnz());
      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_mat->mat_.COO.col, this->mat_.COO.col, this->get_coo_nnz());
      copy_to_mic(this->local_backend_.MIC_dev,
		  cast_mat->mat_.COO.val, this->mat_.COO.val, this->get_coo_nnz());

    }

  } else {
    
    LOG_INFO("Error unsupported MIC matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void MICAcceleratorMatrixHYB<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // MIC to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixHYB<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);   

  if (dst->get_nnz() == 0)
    cast_mat->AllocateHYB(this->get_ell_nnz(), this->get_coo_nnz(), this->get_ell_max_row(),
                      this->get_nrow(), this->get_ncol());

    assert((this->get_nnz()  == dst->get_nnz())  &&
	   (this->get_nrow() == dst->get_nrow()) &&
	   (this->get_ncol() == dst->get_ncol()) );
    
    if (this->get_ell_nnz() > 0) {

      copy_to_host(this->local_backend_.MIC_dev,
		   this->mat_.ELL.val, cast_mat->mat_.ELL.val, this->get_ell_nnz());
      copy_to_host(this->local_backend_.MIC_dev,
		   this->mat_.ELL.col, cast_mat->mat_.ELL.col, this->get_ell_nnz());

    }
    
    if (this->get_coo_nnz() > 0) {

      copy_to_host(this->local_backend_.MIC_dev,
		   this->mat_.COO.row, cast_mat->mat_.COO.row, this->get_coo_nnz());
      copy_to_host(this->local_backend_.MIC_dev,
		   this->mat_.COO.col, cast_mat->mat_.COO.col, this->get_coo_nnz());
      copy_to_host(this->local_backend_.MIC_dev,
		   this->mat_.COO.val, cast_mat->mat_.COO.val, this->get_coo_nnz());

    }

  } else {
    
    LOG_INFO("Error unsupported MIC matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);
    
  }

}

template <typename ValueType>
void MICAcceleratorMatrixHYB<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const MICAcceleratorMatrixHYB<ValueType> *mic_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == src.get_mat_format());

  // MIC to MIC copy
  if ((mic_cast_mat = dynamic_cast<const MICAcceleratorMatrixHYB<ValueType>*> (&src)) != NULL) {
    
  if (this->get_nnz() == 0)
    this->AllocateHYB(mic_cast_mat->get_ell_nnz(), mic_cast_mat->get_coo_nnz(), mic_cast_mat->get_ell_max_row(),
                      mic_cast_mat->get_nrow(), mic_cast_mat->get_ncol());

    assert((this->get_nnz()  == src.get_nnz())  &&
	   (this->get_nrow() == src.get_nrow()) &&
	   (this->get_ncol() == src.get_ncol()) );

    if (this->get_ell_nnz() > 0) {
      
      copy_mic_mic(this->local_backend_.MIC_dev,
		   mic_cast_mat->mat_.ELL.val, this->mat_.ELL.val, this->get_ell_nnz());
      copy_mic_mic(this->local_backend_.MIC_dev,
		   mic_cast_mat->mat_.ELL.col, this->mat_.ELL.col, this->get_ell_nnz());

    }
    
    if (this->get_coo_nnz() > 0) {

      copy_mic_mic(this->local_backend_.MIC_dev,
		   mic_cast_mat->mat_.COO.row, this->mat_.COO.row, this->get_coo_nnz());
      copy_mic_mic(this->local_backend_.MIC_dev,
		   mic_cast_mat->mat_.COO.col, this->mat_.COO.col, this->get_coo_nnz());
      copy_mic_mic(this->local_backend_.MIC_dev,
		   mic_cast_mat->mat_.COO.val, this->mat_.COO.val, this->get_coo_nnz());
      
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
void MICAcceleratorMatrixHYB<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  MICAcceleratorMatrixHYB<ValueType> *mic_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert(this->get_mat_format() == dst->get_mat_format());

  // MIC to MIC copy
  if ((mic_cast_mat = dynamic_cast<MICAcceleratorMatrixHYB<ValueType>*> (dst)) != NULL) {

    mic_cast_mat->set_backend(this->local_backend_);       

  if (this->get_nnz() == 0)
    mic_cast_mat->AllocateHYB(this->get_ell_nnz(), this->get_coo_nnz(), this->get_ell_max_row(),
                      this->get_nrow(), this->get_ncol());

    assert((this->get_nnz()  == dst->get_nnz())  &&
	   (this->get_nrow() == dst->get_nrow()) &&
	   (this->get_ncol() == dst->get_ncol()) );

    if (this->get_ell_nnz() > 0) {

      copy_mic_mic(this->local_backend_.MIC_dev,
		   this->mat_.ELL.val, mic_cast_mat->mat_.ELL.val, this->get_ell_nnz());
      copy_mic_mic(this->local_backend_.MIC_dev,
		   this->mat_.ELL.col, mic_cast_mat->mat_.ELL.col, this->get_ell_nnz());


    }
    
    if (this->get_coo_nnz() > 0) {

      copy_mic_mic(this->local_backend_.MIC_dev,
		   this->mat_.COO.row, mic_cast_mat->mat_.COO.row, this->get_coo_nnz());
      copy_mic_mic(this->local_backend_.MIC_dev,
		   this->mat_.COO.col, mic_cast_mat->mat_.COO.col, this->get_coo_nnz());
      copy_mic_mic(this->local_backend_.MIC_dev,
		   this->mat_.COO.val, mic_cast_mat->mat_.COO.val, this->get_coo_nnz());

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
bool MICAcceleratorMatrixHYB<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const MICAcceleratorMatrixHYB<ValueType>   *cast_mat_hyb;
  
  if ((cast_mat_hyb = dynamic_cast<const MICAcceleratorMatrixHYB<ValueType>*> (&mat)) != NULL) {

    this->CopyFrom(*cast_mat_hyb);
    return true;

  }

  return false;

}

template <typename ValueType>
void MICAcceleratorMatrixHYB<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

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
	     this->mat_.ELL.col,
	     this->mat_.ELL.val,
	     this->get_nrow(),
	     this->get_ncol(),
	     this->get_ell_max_row(),
	     cast_in->vec_,
	     cast_out->vec_);

    spmv_add_coo(this->local_backend_.MIC_dev,
		 this->mat_.COO.row,
		 this->mat_.COO.col,
		 this->mat_.COO.val,
		 this->get_nrow(),
		 this->get_coo_nnz(),
		 ValueType(1.0),
		 cast_in->vec_,
		 cast_out->vec_);
 

  }
}


template <typename ValueType>
void MICAcceleratorMatrixHYB<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
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
		 this->mat_.ELL.col,
		 this->mat_.ELL.val,
		 this->get_nrow(),
		 this->get_ncol(),
		 this->get_ell_max_row(),
		 scalar,
		 cast_in->vec_,
		 cast_out->vec_);

    spmv_add_coo(this->local_backend_.MIC_dev,
		 this->mat_.COO.row,
		 this->mat_.COO.col,
		 this->mat_.COO.val,
		 this->get_nrow(),
		 this->get_coo_nnz(),
		 scalar,
		 cast_in->vec_,
		 cast_out->vec_);

  }
}

template class MICAcceleratorMatrixHYB<double>;
template class MICAcceleratorMatrixHYB<float>;

}
