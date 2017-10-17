#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "ocl_allocate_free.hpp"
#include "ocl_utils.hpp"
#include "ocl_matrix_csr.hpp"
#include "ocl_matrix_dia.hpp"
#include "ocl_vector.hpp"
#include "../host/host_matrix_dia.hpp"
#include "../backend_manager.hpp"

#include <complex>

namespace paralution {

template <typename ValueType>
OCLAcceleratorMatrixDIA<ValueType>::OCLAcceleratorMatrixDIA() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
OCLAcceleratorMatrixDIA<ValueType>::OCLAcceleratorMatrixDIA(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "OCLAcceleratorMatrixDIA::OCLAcceleratorMatrixDIA()",
            "constructor with local_backend");

  this->mat_.val      = NULL;
  this->mat_.offset   = NULL;

  this->mat_.num_diag = 0;

  this->set_backend(local_backend);

}

template <typename ValueType>
OCLAcceleratorMatrixDIA<ValueType>::~OCLAcceleratorMatrixDIA() {

  LOG_DEBUG(this, "OCLAcceleratorMatrixDIA::~OCLAcceleratorMatrixDIA()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void OCLAcceleratorMatrixDIA<ValueType>::info(void) const {

  LOG_INFO("OCLAcceleratorMatrixDIA<ValueType> diag=" << this->mat_.num_diag << " nnz=" << this->nnz_);

}

template <typename ValueType>
void OCLAcceleratorMatrixDIA<ValueType>::AllocateDIA(const int nnz, const int nrow, const int ncol, const int ndiag) {

  assert (nnz  >= 0);
  assert (ncol >= 0);
  assert (nrow >= 0);

  if (this->nnz_ > 0)
    this->Clear();

  if (nnz > 0) {

    assert (ndiag > 0);

    allocate_ocl(ndiag, this->local_backend_.OCL_context, &this->mat_.offset);
    allocate_ocl(nnz,   this->local_backend_.OCL_context, &this->mat_.val);

    ocl_set_to(ndiag, (int) 0, this->mat_.offset, this->local_backend_.OCL_command_queue);
    ocl_set_to(nnz, (ValueType) 0, this->mat_.val, this->local_backend_.OCL_command_queue);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;
    this->mat_.num_diag = ndiag;

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixDIA<ValueType>::SetDataPtrDIA(int **offset, ValueType **val,
                                                       const int nnz, const int nrow, const int ncol,
                                                       const int num_diag) {

  assert (*offset != NULL);
  assert (*val != NULL);
  assert (nnz  > 0);
  assert (nrow > 0);
  assert (ncol > 0);
  assert (num_diag > 0);

  if (nrow < ncol) {
    assert (nnz == ncol * num_diag);
  } else {
    assert (nnz == nrow * num_diag);
  }

  this->Clear();

  cl_int err = clFinish((cl_command_queue) this->local_backend_.OCL_command_queue);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  this->mat_.num_diag = num_diag;
  this->nrow_ = nrow;
  this->ncol_ = ncol;
  this->nnz_  = nnz;

  this->mat_.offset = *offset;
  this->mat_.val = *val;

}

template <typename ValueType>
void OCLAcceleratorMatrixDIA<ValueType>::LeaveDataPtrDIA(int **offset, ValueType **val, int &num_diag) {

  assert (this->nrow_ > 0);
  assert (this->ncol_ > 0);
  assert (this->nnz_  > 0);
  assert (this->mat_.num_diag > 0);

  if (this->nrow_ < this->ncol_) {
    assert (this->nnz_ == this->ncol_ * this->mat_.num_diag);
  } else {
    assert (this->nnz_ == this->nrow_ * this->mat_.num_diag);
  }

  cl_int err = clFinish((cl_command_queue) this->local_backend_.OCL_command_queue);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  // see free_host function for details
  *offset = this->mat_.offset;
  *val = this->mat_.val;

  this->mat_.offset = NULL;
  this->mat_.val = NULL;

  num_diag = this->mat_.num_diag;

  this->mat_.num_diag = 0;
  this->nrow_ = 0;
  this->ncol_ = 0;
  this->nnz_  = 0;

}

template <typename ValueType>
void OCLAcceleratorMatrixDIA<ValueType>::Clear(void) {

  if (this->nnz_ > 0) {

    free_ocl(&this->mat_.val);
    free_ocl(&this->mat_.offset);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;
    this->mat_.num_diag = 0 ;

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixDIA<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixDIA<ValueType> *cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == src.get_mat_format());

  // CPU to OCL copy
  if ((cast_mat = dynamic_cast<const HostMatrixDIA<ValueType>*> (&src)) != NULL) {

    if (this->nnz_ == 0)
      this->AllocateDIA(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_, cast_mat->mat_.num_diag);

    assert (this->nnz_  == cast_mat->nnz_);
    assert (this->nrow_ == cast_mat->nrow_);
    assert (this->ncol_ == cast_mat->ncol_);

    if (this->nnz_ > 0) {

      ocl_host2dev(this->mat_.num_diag,   // size
                   cast_mat->mat_.offset, // src
                   this->mat_.offset,     // dst
                   this->local_backend_.OCL_command_queue);

      ocl_host2dev(this->nnz_,         // size
                   cast_mat->mat_.val, // src
                   this->mat_.val,     // dst
                   this->local_backend_.OCL_command_queue);

    }

  } else {

    LOG_INFO("Error unsupported OCL matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixDIA<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  assert (dst != NULL);

  HostMatrixDIA<ValueType> *cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == dst->get_mat_format());

  // OCL to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixDIA<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);

    if (cast_mat->nnz_ == 0)
      cast_mat->AllocateDIA(this->nnz_, this->nrow_, this->ncol_, this->mat_.num_diag);

    assert (this->nnz_  == cast_mat->nnz_);
    assert (this->nrow_ == cast_mat->nrow_);
    assert (this->ncol_ == cast_mat->ncol_);

    if (this->nnz_ > 0) {

      ocl_dev2host(this->mat_.num_diag,   // size
                   this->mat_.offset,     // src
                   cast_mat->mat_.offset, // dst
                   this->local_backend_.OCL_command_queue);

      ocl_dev2host(this->nnz_,         // size
                   this->mat_.val,     // src
                   cast_mat->mat_.val, // dst
                   this->local_backend_.OCL_command_queue);

    }

  } else {

    LOG_INFO("Error unsupported OCL matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixDIA<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const OCLAcceleratorMatrixDIA<ValueType> *ocl_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
//  TODO FIX MATRIX FORMATS!!!
//  assert (this->get_mat_format() == src.get_mat_format());

  // OCL to OCL copy
  if ((ocl_cast_mat = dynamic_cast<const OCLAcceleratorMatrixDIA<ValueType>*> (&src)) != NULL) {

    if (this->nnz_ == 0)
      this->AllocateDIA(ocl_cast_mat->nnz_, ocl_cast_mat->nrow_, ocl_cast_mat->ncol_, ocl_cast_mat->mat_.num_diag);

    assert (this->nnz_  == ocl_cast_mat->nnz_);
    assert (this->nrow_ == ocl_cast_mat->nrow_);
    assert (this->ncol_ == ocl_cast_mat->ncol_);

    if (this->nnz_ > 0) {

      ocl_dev2dev(this->mat_.num_diag,       // size
                  ocl_cast_mat->mat_.offset, // src
                  this->mat_.offset,         // dst
                  this->local_backend_.OCL_command_queue);

      ocl_dev2dev(this->nnz_,             // size
                  ocl_cast_mat->mat_.val, // src
                  this->mat_.val,         // dst
                  this->local_backend_.OCL_command_queue);

    }

  } else {

    //CPU to OCL
    if ((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*> (&src)) != NULL) {

      this->CopyFromHost(*host_cast_mat);

    } else {

      LOG_INFO("Error unsupported OCL matrix type");
      this->info();
      src.info();
      FATAL_ERROR(__FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixDIA<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  assert (dst != NULL);

  OCLAcceleratorMatrixDIA<ValueType> *ocl_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == dst->get_mat_format());

  // OCL to OCL copy
  if ((ocl_cast_mat = dynamic_cast<OCLAcceleratorMatrixDIA<ValueType>*> (dst)) != NULL) {

    ocl_cast_mat->set_backend(this->local_backend_);

    if (this->nnz_ == 0)
      ocl_cast_mat->AllocateDIA(ocl_cast_mat->nnz_, ocl_cast_mat->nrow_, ocl_cast_mat->ncol_, ocl_cast_mat->mat_.num_diag);

    assert (this->nnz_  == ocl_cast_mat->nnz_);
    assert (this->nrow_ == ocl_cast_mat->nrow_);
    assert (this->ncol_ == ocl_cast_mat->ncol_);

    if (this->nnz_ > 0) {

      ocl_dev2dev(this->mat_.num_diag,       // size
                  this->mat_.offset,         // src
                  ocl_cast_mat->mat_.offset, // dst
                  this->local_backend_.OCL_command_queue);

      ocl_dev2dev(this->nnz_,             // size
                  this->mat_.val,         // src
                  ocl_cast_mat->mat_.val, // dst
                  this->local_backend_.OCL_command_queue);

    }

  } else {

    //OCL to CPU
    if ((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*> (dst)) != NULL) {

      this->CopyToHost(host_cast_mat);

    } else {

      LOG_INFO("Error unsupported OCL matrix type");
      this->info();
      dst->info();
      FATAL_ERROR(__FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
bool OCLAcceleratorMatrixDIA<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const OCLAcceleratorMatrixDIA<ValueType> *cast_mat_dia;

  if ((cast_mat_dia = dynamic_cast<const OCLAcceleratorMatrixDIA<ValueType>*> (&mat)) != NULL) {

    this->CopyFrom(*cast_mat_dia);
    return true;

  }

/*
  const OCLAcceleratorMatrixCSR<ValueType>   *cast_mat_csr;
  if ((cast_mat_csr = dynamic_cast<const OCLAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {

    this->Clear();

    FATAL_ERROR(__FILE__, __LINE__);

    this->nrow_ = cast_mat_csr->nrow_;
    this->ncol_ = cast_mat_csr->ncol_;
    this->nnz_  = cast_mat_csr->nnz_;

    return true;

  }
*/

  return false;

}

template <typename ValueType>
void OCLAcceleratorMatrixDIA<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->nnz_ > 0) {

    assert (out != NULL);
    assert (in.  get_size() >= 0);
    assert (out->get_size() >= 0);
    assert (in.  get_size() == this->ncol_);
    assert (out->get_size() == this->nrow_);

    const OCLAcceleratorVector<ValueType> *cast_in = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&in);
    OCLAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      OCLAcceleratorVector<ValueType>*> (out);

    assert (cast_in  != NULL);
    assert (cast_out != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_dia_spmv",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nrow_, this->ncol_, this->mat_.num_diag, this->mat_.offset,
                                       this->mat_.val, cast_in->vec_, cast_out->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixDIA<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                                  BaseVector<ValueType> *out) const {

  if (this->nnz_ > 0) {

    assert (out != NULL);
    assert (in.  get_size() >= 0);
    assert (out->get_size() >= 0);
    assert (in.  get_size() == this->ncol_);
    assert (out->get_size() == this->nrow_);

    const OCLAcceleratorVector<ValueType> *cast_in = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&in);
    OCLAcceleratorVector<ValueType> *cast_out      = dynamic_cast<      OCLAcceleratorVector<ValueType>*> (out);

    assert (cast_in  != NULL);
    assert (cast_out != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_dia_add_spmv",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nrow_, this->ncol_, this->mat_.num_diag, this->mat_.offset,
                                       this->mat_.val, scalar, cast_in->vec_, cast_out->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}


template class OCLAcceleratorMatrixDIA<double>;
template class OCLAcceleratorMatrixDIA<float>;
#ifdef SUPPORT_COMPLEX
template class OCLAcceleratorMatrixDIA<std::complex<double> >;
template class OCLAcceleratorMatrixDIA<std::complex<float> >;
#endif

}
