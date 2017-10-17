#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "ocl_allocate_free.hpp"
#include "ocl_utils.hpp"
#include "ocl_matrix_csr.hpp"
#include "ocl_matrix_dense.hpp"
#include "ocl_vector.hpp"
#include "../host/host_matrix_dense.hpp"
#include "../backend_manager.hpp"
#include "../matrix_formats_ind.hpp"

#include <complex>

namespace paralution {

template <typename ValueType>
OCLAcceleratorMatrixDENSE<ValueType>::OCLAcceleratorMatrixDENSE() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
OCLAcceleratorMatrixDENSE<ValueType>::OCLAcceleratorMatrixDENSE(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "OCLAcceleratorMatrixDENSE::OCLAcceleratorMatrixDENSE()",
            "constructor with local_backend");

  assert (DENSE_IND_BASE == 0);

  this->mat_.val = NULL;

  this->set_backend(local_backend);

}

template <typename ValueType>
OCLAcceleratorMatrixDENSE<ValueType>::~OCLAcceleratorMatrixDENSE() {

  LOG_DEBUG(this, "OCLAcceleratorMatrixDENSE::~OCLAcceleratorMatrixDENSE()",
            "constructor with local_backend");

  this->Clear();

}

template <typename ValueType>
void OCLAcceleratorMatrixDENSE<ValueType>::info(void) const {

  LOG_INFO("OCLAcceleratorMatrixDENSE<ValueType>");

}

template <typename ValueType>
void OCLAcceleratorMatrixDENSE<ValueType>::AllocateDENSE(const int nrow, const int ncol) {

  assert (ncol  >= 0);
  assert (nrow  >= 0);

  if (this->nnz_ > 0)
    this->Clear();

  if (nrow*ncol > 0) {

    allocate_ocl(nrow*ncol, this->local_backend_.OCL_context, &this->mat_.val);

    ocl_set_to(nrow*ncol, (ValueType) 0, this->mat_.val, this->local_backend_.OCL_command_queue);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nrow*ncol;

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixDENSE<ValueType>::SetDataPtrDENSE(ValueType **val, const int nrow, const int ncol) {

  assert (*val != NULL);
  assert (nrow > 0);
  assert (ncol > 0);

  this->Clear();

  cl_int err = clFinish((cl_command_queue) this->local_backend_.OCL_command_queue);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  this->nrow_ = nrow;
  this->ncol_ = ncol;
  this->nnz_  = nrow * ncol;

  this->mat_.val = *val;

}

template <typename ValueType>
void OCLAcceleratorMatrixDENSE<ValueType>::LeaveDataPtrDENSE(ValueType **val) {

  assert (this->nrow_ > 0);
  assert (this->ncol_ > 0);
  assert (this->nnz_  > 0);
  assert (this->nnz_ == this->nrow_ * this->ncol_);

  cl_int err = clFinish((cl_command_queue) this->local_backend_.OCL_command_queue);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  *val = this->mat_.val;

  this->mat_.val = NULL;

  this->nrow_ = 0;
  this->ncol_ = 0;
  this->nnz_  = 0;

}

template <typename ValueType>
void OCLAcceleratorMatrixDENSE<ValueType>::Clear(void) {

  if (this->nnz_ > 0) {

    free_ocl(&this->mat_.val);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixDENSE<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixDENSE<ValueType> *cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == src.get_mat_format());

  // CPU to OCL copy
  if ((cast_mat = dynamic_cast<const HostMatrixDENSE<ValueType>*> (&src)) != NULL) {

    if (this->nnz_ == 0)
      this->AllocateDENSE(cast_mat->nrow_, cast_mat->ncol_);

    assert (this->nnz_  == cast_mat->nnz_);
    assert (this->nrow_ == cast_mat->nrow_);
    assert (this->ncol_ == cast_mat->ncol_);

    if (this->nnz_ > 0) {

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
void OCLAcceleratorMatrixDENSE<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  HostMatrixDENSE<ValueType> *cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == dst->get_mat_format());

  // OCL to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixDENSE<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);

    if (cast_mat->nnz_ == 0)
      cast_mat->AllocateDENSE(this->nrow_, this->ncol_);

    assert (this->nnz_  == cast_mat->nnz_);
    assert (this->nrow_ == cast_mat->nrow_);
    assert (this->ncol_ == cast_mat->ncol_);

    if (this->nnz_ > 0) {

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
void OCLAcceleratorMatrixDENSE<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const OCLAcceleratorMatrixDENSE<ValueType> *ocl_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == src.get_mat_format());

  // OCL to OCL copy
  if ((ocl_cast_mat = dynamic_cast<const OCLAcceleratorMatrixDENSE<ValueType>*> (&src)) != NULL) {

    if (this->nnz_ == 0)
      this->AllocateDENSE(ocl_cast_mat->nrow_, ocl_cast_mat->ncol_);

    assert (this->nnz_  == ocl_cast_mat->nnz_);
    assert (this->nrow_ == ocl_cast_mat->nrow_);
    assert (this->ncol_ == ocl_cast_mat->ncol_);

    if (this->nnz_ > 0) {

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
void OCLAcceleratorMatrixDENSE<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  assert (dst != NULL);

  OCLAcceleratorMatrixDENSE<ValueType> *ocl_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == dst->get_mat_format());

  // OCL to OCL copy
  if ((ocl_cast_mat = dynamic_cast<OCLAcceleratorMatrixDENSE<ValueType>*> (dst)) != NULL) {

    ocl_cast_mat->set_backend(this->local_backend_);

    if (ocl_cast_mat->nnz_ == 0)
      ocl_cast_mat->AllocateDENSE(this->nrow_, this->ncol_);

    assert (this->nnz_  == ocl_cast_mat->nnz_);
    assert (this->nrow_ == ocl_cast_mat->nrow_);
    assert (this->ncol_ == ocl_cast_mat->ncol_);

    if (this->nnz_ > 0) {

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
bool OCLAcceleratorMatrixDENSE<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const OCLAcceleratorMatrixDENSE<ValueType> *cast_mat_dense;

  if ((cast_mat_dense = dynamic_cast<const OCLAcceleratorMatrixDENSE<ValueType>*> (&mat)) != NULL) {

      this->CopyFrom(*cast_mat_dense);
      return true;

  }

/*
  const OCLAcceleratorMatrixCSR<ValueType> *cast_mat_csr;
  if ((cast_mat_csr = dynamic_cast<const OCLAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {

    this->Clear();

    FATAL_ERROR(__FILE__, __LINE__);

    this->nrow_ = cast_mat_csr->nrow_;
    this->ncol_ = cast_mat_csr->ncol_;
    this->nnz_  = cast_mat_csr->nnz_;

    return 0;

  }
*/

  return false;

}

template <typename ValueType>
void OCLAcceleratorMatrixDENSE<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

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

    cl_int err = ocl_kernel<ValueType>("kernel_dense_spmv",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nrow_, this->ncol_, this->mat_.val, cast_in->vec_, cast_out->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixDENSE<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                                    BaseVector<ValueType> *out) const {

  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
bool OCLAcceleratorMatrixDENSE<ValueType>::ReplaceColumnVector(const int idx, const BaseVector<ValueType> &vec) {

  assert (vec.get_size() == this->nrow_);

  if (this->nnz_ > 0) {

    const OCLAcceleratorVector<ValueType> *cast_vec = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&vec);
    assert (cast_vec != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_dense_replace_column_vector",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       cast_vec->vec_, idx, this->nrow_, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixDENSE<ValueType>::ReplaceRowVector(const int idx, const BaseVector<ValueType> &vec) {

  assert (vec.get_size() == this->ncol_);

  if (this->nnz_ > 0) {

    const OCLAcceleratorVector<ValueType> *cast_vec = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&vec);
    assert (cast_vec != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->ncol_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_dense_replace_row_vector",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       cast_vec->vec_, idx, this->nrow_, this->ncol_, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixDENSE<ValueType>::ExtractColumnVector(const int idx, BaseVector<ValueType> *vec) const {

  assert (vec != NULL);
  assert (vec->get_size() == this->nrow_);

  if (this->nnz_ > 0) {

    OCLAcceleratorVector<ValueType> *cast_vec = dynamic_cast<OCLAcceleratorVector<ValueType>*> (vec);
    assert (cast_vec != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_dense_extract_column_vector",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       cast_vec->vec_, idx, this->nrow_, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixDENSE<ValueType>::ExtractRowVector(const int idx, BaseVector<ValueType> *vec) const {

    assert (vec != NULL);
    assert (vec->get_size() == this->ncol_);

  if (this->nnz_ > 0) {

    OCLAcceleratorVector<ValueType> *cast_vec = dynamic_cast<OCLAcceleratorVector<ValueType>*> (vec);
    assert (cast_vec != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->ncol_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_dense_extract_row_vector",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       cast_vec->vec_, idx, this->nrow_, this->ncol_, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}


template class OCLAcceleratorMatrixDENSE<double>;
template class OCLAcceleratorMatrixDENSE<float>;
#ifdef SUPPORT_COMPLEX
template class OCLAcceleratorMatrixDENSE<std::complex<double> >;
template class OCLAcceleratorMatrixDENSE<std::complex<float> >;
#endif

}
