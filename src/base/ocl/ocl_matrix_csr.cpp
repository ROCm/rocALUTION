#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "ocl_allocate_free.hpp"
#include "ocl_utils.hpp"
#include "ocl_matrix_csr.hpp"
#include "ocl_vector.hpp"
#include "../host/host_matrix_csr.hpp"
#include "../backend_manager.hpp"

#include <vector>
#include <complex>

namespace paralution {

template <typename ValueType>
OCLAcceleratorMatrixCSR<ValueType>::OCLAcceleratorMatrixCSR() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
OCLAcceleratorMatrixCSR<ValueType>::OCLAcceleratorMatrixCSR(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "OCLAcceleratorMatrixCSR::OCLAcceleratorMatrixCSR()",
            "constructor with local_backend");

  this->mat_.row_offset = NULL;
  this->mat_.col        = NULL;
  this->mat_.val        = NULL;

  this->set_backend(local_backend);

  this->tmp_vec_ = NULL;

}

template <typename ValueType>
OCLAcceleratorMatrixCSR<ValueType>::~OCLAcceleratorMatrixCSR() {

  LOG_DEBUG(this, "OCLAcceleratorMatrixCSR::~OCLAcceleratorMatrixCSR()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::info(void) const {

  LOG_INFO("OCLAcceleratorMatrixCSR<ValueType>");

}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::AllocateCSR(const int nnz, const int nrow, const int ncol) {

  assert (nnz  >= 0);
  assert (ncol >= 0);
  assert (nrow >= 0);

  if (this->nnz_ > 0)
    this->Clear();

  if (nnz > 0) {

    allocate_ocl(nrow+1, this->local_backend_.OCL_context, &this->mat_.row_offset);
    allocate_ocl(nnz, this->local_backend_.OCL_context, &this->mat_.col);
    allocate_ocl(nnz, this->local_backend_.OCL_context, &this->mat_.val);

    // Set entries of device object to zero
    ocl_set_to(nrow+1, (int) 0, this->mat_.row_offset, this->local_backend_.OCL_command_queue);
    ocl_set_to(nnz, (int) 0, this->mat_.col, this->local_backend_.OCL_command_queue);
    ocl_set_to(nnz, (ValueType) 0, this->mat_.val, this->local_backend_.OCL_command_queue);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::SetDataPtrCSR(int **row_offset, int **col, ValueType **val,
                                                       const int nnz, const int nrow, const int ncol) {

  assert (*row_offset != NULL);
  assert (*col != NULL);
  assert (*val != NULL);
  assert (nnz  > 0);
  assert (nrow > 0);
  assert (ncol > 0);

  this->Clear();

  this->nrow_ = nrow;
  this->ncol_ = ncol;
  this->nnz_  = nnz;

  cl_int err = clFinish((cl_command_queue) this->local_backend_.OCL_command_queue);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  this->mat_.row_offset = *row_offset;
  this->mat_.col = *col;
  this->mat_.val = *val;

}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::LeaveDataPtrCSR(int **row_offset, int **col, ValueType **val) {

  assert (this->nrow_ > 0);
  assert (this->ncol_ > 0);
  assert (this->nnz_  > 0);

  cl_int err = clFinish((cl_command_queue) this->local_backend_.OCL_command_queue);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  // see free_host function for details
  *row_offset = this->mat_.row_offset;
  *col = this->mat_.col;
  *val = this->mat_.val;

  this->mat_.row_offset = NULL;
  this->mat_.col = NULL;
  this->mat_.val = NULL;

  this->nrow_ = 0;
  this->ncol_ = 0;
  this->nnz_  = 0;

}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::Clear(void) {

  if (this->nnz_ > 0) {

    free_ocl(&this->mat_.row_offset);
    free_ocl(&this->mat_.col);
    free_ocl(&this->mat_.val);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

  }

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::Zeros() {

  if (this->nnz_ > 0) {

    // Set entries of device object to zero
    ocl_set_to(this->nnz_, (ValueType) 0, this->mat_.val, this->local_backend_.OCL_command_queue);

  }

  return true;

}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  // copy only in the same format
  assert (this->get_mat_format() == src.get_mat_format());

  const HostMatrixCSR<ValueType> *cast_mat;

  // CPU to OCL copy
  if ((cast_mat = dynamic_cast<const HostMatrixCSR<ValueType>*> (&src)) != NULL) {

    if (this->nnz_ == 0)
      this->AllocateCSR(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);

    assert (this->nnz_  == cast_mat->nnz_);
    assert (this->nrow_ == cast_mat->nrow_);
    assert (this->ncol_ == cast_mat->ncol_);

    if (this->nnz_ > 0) {

      // Copy object from host to device memory
      ocl_host2dev(this->nrow_+1,             // size
                   cast_mat->mat_.row_offset, // src
                   this->mat_.row_offset,     // dst
                   this->local_backend_.OCL_command_queue);

      // Copy object from host to device memory
      ocl_host2dev(this->nnz_,         // size
                   cast_mat->mat_.col, // src
                   this->mat_.col,     // dst
                   this->local_backend_.OCL_command_queue);

      // Copy object from host to device memory
      ocl_host2dev(this->nnz_,         // size
                   cast_mat->mat_.val, // src
                   this->mat_.val,     // dst
                   this->local_backend_.OCL_command_queue);

    }

  } else {

    LOG_INFO("Error unsupported OpenCL matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  assert (dst != NULL);

  HostMatrixCSR<ValueType> *cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == dst->get_mat_format());

  // OCL to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixCSR<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);

    if (cast_mat->nnz_ == 0)
      cast_mat->AllocateCSR(this->nnz_, this->nrow_, this->ncol_ );

    assert (this->nnz_  == cast_mat->nnz_);
    assert (this->nrow_ == cast_mat->nrow_);
    assert (this->ncol_ == cast_mat->ncol_);

    if (this->nnz_ > 0) {

      // Copy object from device to host memory
      ocl_dev2host(this->nrow_+1,             // size
                   this->mat_.row_offset,     // src
                   cast_mat->mat_.row_offset, // dst
                   this->local_backend_.OCL_command_queue);

      // Copy object from device to host memory
      ocl_dev2host(this->nnz_,         // size
                   this->mat_.col,     // src
                   cast_mat->mat_.col, // dst
                   this->local_backend_.OCL_command_queue);

      // Copy object from device to host memory
      ocl_dev2host(this->nnz_,         // size
                   this->mat_.val,     // src
                   cast_mat->mat_.val, // dst
                   this->local_backend_.OCL_command_queue);

    }

  } else {

    LOG_INFO("Error unsupported OpenCL matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const OCLAcceleratorMatrixCSR<ValueType> *ocl_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == src.get_mat_format());

  // OCL to OCL copy
  if ((ocl_cast_mat = dynamic_cast<const OCLAcceleratorMatrixCSR<ValueType>*> (&src)) != NULL) {

    if (this->nnz_ == 0)
      this->AllocateCSR(ocl_cast_mat->nnz_, ocl_cast_mat->nrow_, ocl_cast_mat->ncol_ );

    assert (this->nnz_  == ocl_cast_mat->nnz_);
    assert (this->nrow_ == ocl_cast_mat->nrow_);
    assert (this->ncol_ == ocl_cast_mat->ncol_);

    if (this->nnz_ > 0) {

      // Copy object from device to device memory (internal copy)
      ocl_dev2dev(this->nrow_+1,                 // size
                  ocl_cast_mat->mat_.row_offset, // src
                  this->mat_.row_offset,         // dst
                  this->local_backend_.OCL_command_queue);

      // Copy object from device to device memory (internal copy)
      ocl_dev2dev(this->nnz_,             // size
                  ocl_cast_mat->mat_.col, // src
                  this->mat_.col,         // dst
                  this->local_backend_.OCL_command_queue);

      // Copy object from device to device memory (internal copy)
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

      LOG_INFO("Error unsupported OpenCL matrix type");
      this->info();
      src.info();
      FATAL_ERROR(__FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  assert (dst != NULL);

  OCLAcceleratorMatrixCSR<ValueType> *ocl_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == dst->get_mat_format());

  // OCL to OCL copy
  if ((ocl_cast_mat = dynamic_cast<OCLAcceleratorMatrixCSR<ValueType>*> (dst)) != NULL) {

    ocl_cast_mat->set_backend(this->local_backend_);

    if (ocl_cast_mat->nnz_ == 0)
      ocl_cast_mat->AllocateCSR(this->nnz_, this->nrow_, this->ncol_);

    assert (this->nnz_  == ocl_cast_mat->nnz_);
    assert (this->nrow_ == ocl_cast_mat->nrow_);
    assert (this->ncol_ == ocl_cast_mat->ncol_);

    if (this->nnz_ > 0) {

      // Copy object from device to device memory (internal copy)
      ocl_dev2dev(this->nrow_+1,                 // size
                  this->mat_.row_offset,         // src
                  ocl_cast_mat->mat_.row_offset, // dst
                  this->local_backend_.OCL_command_queue);

      // Copy object from device to device memory (internal copy)
      ocl_dev2dev(this->nnz_,             // size
                  this->mat_.col,         // src
                  ocl_cast_mat->mat_.col, // dst
                  this->local_backend_.OCL_command_queue);

      // Copy object from device to device memory (internal copy)
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

      LOG_INFO("Error unsupported OpenCL matrix type");
      this->info();
      dst->info();
      FATAL_ERROR(__FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  return false;

}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::CopyFromHostCSR(const int *row_offset, const int *col, const ValueType *val,
                                                         const int nnz, const int nrow, const int ncol) {

  assert(nnz >= 0);
  assert(ncol >= 0);
  assert(nrow >= 0);
  assert(row_offset != NULL);
  assert(col != NULL);
  assert(val != NULL);

  // Allocate matrix
  if (this->nnz_ > 0)
    this->Clear();

  if (nnz > 0) {

    allocate_ocl(nrow+1, this->local_backend_.OCL_context, &this->mat_.row_offset);
    allocate_ocl(nnz, this->local_backend_.OCL_context, &this->mat_.col);
    allocate_ocl(nnz, this->local_backend_.OCL_context, &this->mat_.val);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

    // Copy object from host to device memory
    ocl_host2dev(this->nrow_+1,         // size
                 row_offset,            // src
                 this->mat_.row_offset, // dst
                 this->local_backend_.OCL_command_queue);

    // Copy object from host to device memory
    ocl_host2dev(this->nnz_,     // size
                 col,            // src
                 this->mat_.col, // dst
                 this->local_backend_.OCL_command_queue);

    // Copy object from host to device memory
    ocl_host2dev(this->nnz_,     // size
                 val,            // src
                 this->mat_.val, // dst
                 this->local_backend_.OCL_command_queue);

  }

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::Permute(const BaseVector<int> &permutation) {

  // TODO fix error in extrapolation kernel
  return false;

  if (this->nnz_ > 0) {

    assert (permutation.get_size() == this->nrow_);
    assert (permutation.get_size() == this->ncol_);

    const OCLAcceleratorVector<int> *cast_perm = dynamic_cast<const OCLAcceleratorVector<int>*> (&permutation);

    assert (cast_perm != NULL);

    int *d_nnzr       = NULL;
    int *d_nnzrPerm   = NULL;
    int *d_nnzPerm    = NULL;
    int *d_offset     = NULL;
    ValueType *d_data = NULL;

    allocate_ocl(this->nrow_, this->local_backend_.OCL_context, &d_nnzr);
    allocate_ocl(this->nrow_, this->local_backend_.OCL_context, &d_nnzrPerm);
    allocate_ocl((this->nrow_+1), this->local_backend_.OCL_context, &d_nnzPerm);
    allocate_ocl(this->nnz_, this->local_backend_.OCL_context, &d_data);
    allocate_ocl(this->nnz_, this->local_backend_.OCL_context, &d_offset);

    int nrow = this->nrow_;

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (nrow / LocalSize + 1 ) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_calc_row_nnz",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       nrow, this->mat_.row_offset, d_nnzr);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    err = ocl_kernel<ValueType>("kernel_csr_permute_row_nnz",
                                this->local_backend_.OCL_command_queue,
                                LocalSize, GlobalSize,
                                nrow, d_nnzr, cast_perm->vec_, d_nnzrPerm);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    // Set entries of device object to zero
    ocl_set_to(nrow+1, (int) 0, d_nnzPerm, this->local_backend_.OCL_command_queue);

    int *d_temp = NULL;
    allocate_ocl(this->nrow_+1, this->local_backend_.OCL_context, &d_temp);

    // Set entries of device object to zero
    ocl_set_to(nrow+1, (int) 0, d_temp, this->local_backend_.OCL_command_queue);

    int shift = 1;

    err = ocl_kernel<ValueType>("kernel_red_partial_sum",
                                this->local_backend_.OCL_command_queue,
                                LocalSize, GlobalSize,
                                d_nnzPerm, d_nnzrPerm, nrow, shift);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    int nrowp = nrow+1;

    GlobalSize = (nrow / (LocalSize * LocalSize) + 1) * LocalSize;

    err = ocl_kernel<ValueType>("kernel_red_recurse",
                                this->local_backend_.OCL_command_queue,
                                LocalSize, GlobalSize,
                                d_temp, d_nnzPerm, nrowp);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    shift = 1;

    err = ocl_kernel<ValueType>("kernel_red_extrapolate",
                                this->local_backend_.OCL_command_queue,
                                LocalSize, GlobalSize,
                                d_nnzPerm, d_temp, d_nnzrPerm, nrow, shift);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    free_ocl(&d_temp);

    GlobalSize = (nrow / LocalSize + 1) * LocalSize;

    err = ocl_kernel<ValueType>("kernel_csr_permute_rows",
                                this->local_backend_.OCL_command_queue,
                                LocalSize, GlobalSize,
                                nrow, this->mat_.row_offset, d_nnzPerm, this->mat_.col, this->mat_.val,
                                cast_perm->vec_, d_nnzr, d_offset, d_data);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    free_ocl(&this->mat_.row_offset);

    this->mat_.row_offset = d_nnzPerm;

    err = ocl_kernel<ValueType>("kernel_csr_permute_cols",
                                this->local_backend_.OCL_command_queue,
                                LocalSize, GlobalSize,
                                nrow, this->mat_.row_offset, cast_perm->vec_, d_nnzrPerm, d_offset,
                                d_data, this->mat_.col, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    free_ocl(&d_offset);
    free_ocl(&d_data);
    free_ocl(&d_nnzrPerm);
    free_ocl(&d_nnzr);

  }

  return true;

}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

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

    int nnz_per_row = this->nnz_ / this->nrow_;
    int nthreads;

    if      (nnz_per_row <=   8) nthreads =  2;
    else if (nnz_per_row <=  16) nthreads =  4;
    else if (nnz_per_row <=  32) nthreads =  8;
    else if (nnz_per_row <=  64) nthreads = 16;
    else if (nnz_per_row <= 128) nthreads = 32;
    else                         nthreads = 64;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_spmv",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nrow_, nthreads, this->mat_.row_offset, this->mat_.col, this->mat_.val,
                                       cast_in->vec_, cast_out->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
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

    int nnz_per_row = this->nnz_ / this->nrow_;
    int nthreads;

    if      (nnz_per_row <=   8) nthreads =  2;
    else if (nnz_per_row <=  16) nthreads =  4;
    else if (nnz_per_row <=  32) nthreads =  8;
    else if (nnz_per_row <=  64) nthreads = 16;
    else if (nnz_per_row <= 128) nthreads = 32;
    else                         nthreads = 64;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_add_spmv",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nrow_, nthreads, this->mat_.row_offset, this->mat_.col, this->mat_.val,
                                       scalar, cast_in->vec_, cast_out->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::LUAnalyse(void) {
}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::LUAnalyseClear(void) {
}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::LLAnalyse(void) {
}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::LLAnalyseClear(void) {
}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::LAnalyse(const bool diag_unit) {
}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::UAnalyse(const bool diag_unit) {
}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::LAnalyseClear(void) {
}

template <typename ValueType>
void OCLAcceleratorMatrixCSR<ValueType>::UAnalyseClear(void) {
}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::ExtractDiagonal(BaseVector<ValueType> *vec_diag) const {

  if (this->nnz_ > 0) {

    assert (vec_diag != NULL);
    assert (vec_diag->get_size() == this->nrow_);

    OCLAcceleratorVector<ValueType> *cast_diag = dynamic_cast<OCLAcceleratorVector<ValueType>*> (vec_diag);

    assert (cast_diag != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_extract_diag",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nrow_, this->mat_.row_offset, this->mat_.col,
                                       this->mat_.val, cast_diag->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::ExtractInverseDiagonal(BaseVector<ValueType> *vec_inv_diag) const {

  if (this->nnz_ > 0) {

    assert (vec_inv_diag != NULL);
    assert (vec_inv_diag->get_size() == this->nrow_);

    OCLAcceleratorVector<ValueType> *cast_inv_diag = dynamic_cast<OCLAcceleratorVector<ValueType>*> (vec_inv_diag);

    assert (cast_inv_diag != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_extract_inv_diag",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nrow_, this->mat_.row_offset, this->mat_.col, this->mat_.val,
                                       cast_inv_diag->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::ExtractSubMatrix(const int row_offset, const int col_offset,
                                                          const int row_size, const int col_size,
                                                          BaseMatrix<ValueType> *mat) const {

  assert (mat != NULL);

  assert (row_offset >= 0);
  assert (col_offset >= 0);

  assert (this->nrow_ > 0);
  assert (this->ncol_ > 0);

  OCLAcceleratorMatrixCSR<ValueType> *cast_mat = dynamic_cast<OCLAcceleratorMatrixCSR<ValueType>*> (mat);

  assert (cast_mat != NULL);

  int mat_nnz = 0;

  int *row_nnz = NULL;
  int *red_row_nnz = NULL;

  allocate_host(row_size+1, &red_row_nnz);
  allocate_ocl(row_size+1, this->local_backend_.OCL_context, &row_nnz);

  size_t LocalSize  = this->local_backend_.OCL_block_size;
  size_t GlobalSize = ((row_size+1) / LocalSize + 1) * LocalSize;

  // compute the nnz per row in the new matrix
  cl_int err = ocl_kernel<ValueType>("kernel_csr_extract_submatrix_row_nnz",
                                     this->local_backend_.OCL_command_queue,
                                     LocalSize, GlobalSize,
                                     this->mat_.row_offset, this->mat_.col, this->mat_.val,
                                     row_offset, col_offset, row_size, col_size, row_nnz);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  // compute the new nnz by reduction on CPU

  // Copy object from device to host memory
  ocl_dev2host(row_size+1, row_nnz, red_row_nnz, this->local_backend_.OCL_command_queue);

  int sum = 0;
  for (int i=0; i<row_size; ++i) {
    int tmp = red_row_nnz[i];
    red_row_nnz[i] = sum;
    sum += tmp;
  }

  mat_nnz = red_row_nnz[row_size] = sum;

  // TODO
  //  redSubSum

  // not empty submatrix
  if (mat_nnz > 0) {

    cast_mat->AllocateCSR(mat_nnz, row_size, col_size);

    // part of the CPU reduction section
    // Copy object from host to device memory
    ocl_host2dev(row_size+1,                // size
                 red_row_nnz,               // src
                 cast_mat->mat_.row_offset, // dst
                 this->local_backend_.OCL_command_queue);

    // copying the sub matrix
    err = ocl_kernel<ValueType>("kernel_csr_extract_submatrix_copy",
                                this->local_backend_.OCL_command_queue,
                                LocalSize, GlobalSize,
                                this->mat_.row_offset, this->mat_.col, this->mat_.val,
                                row_offset, col_offset, row_size, col_size,
                                cast_mat->mat_.row_offset, cast_mat->mat_.col, cast_mat->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  free_ocl(&row_nnz);
  free_host(&red_row_nnz);

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::ExtractL(BaseMatrix<ValueType> *L) const {

  assert (L != NULL);

  assert (this->nrow_ > 0);
  assert (this->ncol_ > 0);

  OCLAcceleratorMatrixCSR<ValueType> *cast_L = dynamic_cast<OCLAcceleratorMatrixCSR<ValueType>*> (L);

  assert (cast_L != NULL);

  cast_L->Clear();

  // compute nnz per row
  allocate_ocl(this->nrow_+1, this->local_backend_.OCL_context, &cast_L->mat_.row_offset);

  size_t LocalSize  = this->local_backend_.OCL_block_size;
  size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

  // compute the nnz per row in the new matrix
  cl_int err = ocl_kernel<ValueType>("kernel_csr_slower_nnz_per_row",
                                     this->local_backend_.OCL_command_queue,
                                     LocalSize, GlobalSize,
                                     this->nrow_, this->mat_.row_offset, this->mat_.col, cast_L->mat_.row_offset);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  // partial sum row_nnz to obtain row_offset vector
  // TODO currently performing partial sum on host
  int *h_buffer = NULL;
  allocate_host(this->nrow_+1, &h_buffer);
  ocl_dev2host(this->nrow_+1, cast_L->mat_.row_offset, h_buffer, this->local_backend_.OCL_command_queue);

  h_buffer[0] = 0;
  for (int i=1; i<this->nrow_+1; ++i)
    h_buffer[i] += h_buffer[i-1];

  int nnz_L = h_buffer[this->nrow_];

  ocl_host2dev(this->nrow_+1,           // size
               h_buffer,                // src
               cast_L->mat_.row_offset, // dst
               this->local_backend_.OCL_command_queue);

  free_host(&h_buffer);
  // end TODO

  // allocate lower triangular part structure
  allocate_ocl(nnz_L, this->local_backend_.OCL_context, &cast_L->mat_.col);
  allocate_ocl(nnz_L, this->local_backend_.OCL_context, &cast_L->mat_.val);

  // compute the nnz per row in the new matrix
  err = ocl_kernel<ValueType>("kernel_csr_extract_l_triangular",
                              this->local_backend_.OCL_command_queue,
                              LocalSize, GlobalSize,
                              this->nrow_, this->mat_.row_offset, this->mat_.col, this->mat_.val,
                              cast_L->mat_.row_offset, cast_L->mat_.col, cast_L->mat_.val);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  cast_L->nrow_ = this->nrow_;
  cast_L->ncol_ = this->ncol_;
  cast_L->nnz_  = nnz_L;

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::ExtractLDiagonal(BaseMatrix<ValueType> *L) const {

  assert (L != NULL);

  assert (this->nrow_ > 0);
  assert (this->ncol_ > 0);

  OCLAcceleratorMatrixCSR<ValueType> *cast_L = dynamic_cast<OCLAcceleratorMatrixCSR<ValueType>*> (L);

  assert (cast_L != NULL);

  cast_L->Clear();

  // compute nnz per row
  allocate_ocl(this->nrow_+1, this->local_backend_.OCL_context, &cast_L->mat_.row_offset);

  size_t LocalSize  = this->local_backend_.OCL_block_size;
  size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

  // compute the nnz per row in the new matrix
  cl_int err = ocl_kernel<ValueType>("kernel_csr_lower_nnz_per_row",
                                     this->local_backend_.OCL_command_queue,
                                     LocalSize, GlobalSize,
                                     this->nrow_, this->mat_.row_offset, this->mat_.col, cast_L->mat_.row_offset);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  // partial sum row_nnz to obtain row_offset vector
  // TODO currently performing partial sum on host
  int *h_buffer = NULL;
  allocate_host(this->nrow_+1, &h_buffer);

  ocl_dev2host(this->nrow_+1, cast_L->mat_.row_offset, h_buffer, this->local_backend_.OCL_command_queue);

  h_buffer[0] = 0;
  for (int i=1; i<this->nrow_+1; ++i)
    h_buffer[i] += h_buffer[i-1];

  int nnz_L = h_buffer[this->nrow_];

  ocl_host2dev(this->nrow_+1,           // size
               h_buffer,                // src
               cast_L->mat_.row_offset, // dst
               this->local_backend_.OCL_command_queue);

  free_host(&h_buffer);
  // end TODO

  // allocate lower triangular part structure
  allocate_ocl(nnz_L, this->local_backend_.OCL_context, &cast_L->mat_.col);
  allocate_ocl(nnz_L, this->local_backend_.OCL_context, &cast_L->mat_.val);

  err = ocl_kernel<ValueType>("kernel_csr_extract_l_triangular",
                              this->local_backend_.OCL_command_queue,
                              LocalSize, GlobalSize,
                              this->nrow_, this->mat_.row_offset, this->mat_.col, this->mat_.val,
                              cast_L->mat_.row_offset, cast_L->mat_.col, cast_L->mat_.val);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  cast_L->nrow_ = this->nrow_;
  cast_L->ncol_ = this->ncol_;
  cast_L->nnz_ = nnz_L;

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::ExtractU(BaseMatrix<ValueType> *U) const {

  assert (U != NULL);

  assert (this->nrow_ > 0);
  assert (this->ncol_ > 0);

  OCLAcceleratorMatrixCSR<ValueType> *cast_U = dynamic_cast<OCLAcceleratorMatrixCSR<ValueType>*> (U);

  assert (cast_U != NULL);

  cast_U->Clear();

  // compute nnz per row
  allocate_ocl(this->nrow_+1, this->local_backend_.OCL_context, &cast_U->mat_.row_offset);

  size_t LocalSize  = this->local_backend_.OCL_block_size;
  size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

  // compute the nnz per row in the new matrix
  cl_int err = ocl_kernel<ValueType>("kernel_csr_supper_nnz_per_row",
                                     this->local_backend_.OCL_command_queue,
                                     LocalSize, GlobalSize,
                                     this->nrow_, this->mat_.row_offset, this->mat_.col, cast_U->mat_.row_offset);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  // partial sum row_nnz to obtain row_offset vector
  // TODO currently performing partial sum on host
  int *h_buffer = NULL;
  allocate_host(this->nrow_+1, &h_buffer);

  ocl_dev2host(this->nrow_+1, cast_U->mat_.row_offset, h_buffer, this->local_backend_.OCL_command_queue);

  h_buffer[0] = 0;
  for (int i=1; i<this->nrow_+1; ++i)
    h_buffer[i] += h_buffer[i-1];

  int nnz_L = h_buffer[this->nrow_];

  ocl_host2dev(this->nrow_+1,                  // size
               h_buffer,                // src
               cast_U->mat_.row_offset, // dst
               this->local_backend_.OCL_command_queue);

  free_host(&h_buffer);
  // end TODO

  // allocate lower triangular part structure
  allocate_ocl(nnz_L, this->local_backend_.OCL_context, &cast_U->mat_.col);
  allocate_ocl(nnz_L, this->local_backend_.OCL_context, &cast_U->mat_.val);

  err = ocl_kernel<ValueType>("kernel_csr_extract_u_triangular",
                              this->local_backend_.OCL_command_queue,
                              LocalSize, GlobalSize,
                              this->nrow_, this->mat_.row_offset, this->mat_.col, this->mat_.val,
                              cast_U->mat_.row_offset, cast_U->mat_.col, cast_U->mat_.val);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  cast_U->nrow_ = this->nrow_;
  cast_U->ncol_ = this->ncol_;
  cast_U->nnz_  = nnz_L;

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::ExtractUDiagonal(BaseMatrix<ValueType> *U) const {

  assert (U != NULL);

  assert (this->nrow_ > 0);
  assert (this->ncol_ > 0);

  OCLAcceleratorMatrixCSR<ValueType> *cast_U = dynamic_cast<OCLAcceleratorMatrixCSR<ValueType>*> (U);

  assert (cast_U != NULL);

  cast_U->Clear();

  // compute nnz per row
  allocate_ocl(this->nrow_+1, this->local_backend_.OCL_context, &cast_U->mat_.row_offset);

  size_t LocalSize  = this->local_backend_.OCL_block_size;
  size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

  // compute the nnz per row in the new matrix
  cl_int err = ocl_kernel<ValueType>("kernel_csr_upper_nnz_per_row",
                                     this->local_backend_.OCL_command_queue,
                                     LocalSize, GlobalSize,
                                     this->nrow_, this->mat_.row_offset, this->mat_.col, cast_U->mat_.row_offset);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  // partial sum row_nnz to obtain row_offset vector
  // TODO currently performing partial sum on host
  int *h_buffer = NULL;
  allocate_host(this->nrow_+1, &h_buffer);

  ocl_dev2host(this->nrow_+1, cast_U->mat_.row_offset, h_buffer, this->local_backend_.OCL_command_queue);

  h_buffer[0] = 0;
  for (int i=1; i<this->nrow_+1; ++i)
    h_buffer[i] += h_buffer[i-1];

  int nnz_L = h_buffer[this->nrow_];

  ocl_host2dev(this->nrow_+1, h_buffer, cast_U->mat_.row_offset, this->local_backend_.OCL_command_queue);

  free_host(&h_buffer);
  // end TODO

  // allocate lower triangular part structure
  allocate_ocl(nnz_L, this->local_backend_.OCL_context, &cast_U->mat_.col);
  allocate_ocl(nnz_L, this->local_backend_.OCL_context, &cast_U->mat_.val);

  err = ocl_kernel<ValueType>("kernel_csr_extract_u_triangular",
                              this->local_backend_.OCL_command_queue,
                              LocalSize, GlobalSize,
                              this->nrow_, this->mat_.row_offset, this->mat_.col, this->mat_.val,
                              cast_U->mat_.row_offset, cast_U->mat_.col, cast_U->mat_.val);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  cast_U->nrow_ = this->nrow_;
  cast_U->ncol_ = this->ncol_;
  cast_U->nnz_  = nnz_L;

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::MaximalIndependentSet(int &size, BaseVector<int> *permutation) const {

  assert (permutation != NULL);

  OCLAcceleratorVector<int> *cast_perm = dynamic_cast<OCLAcceleratorVector<int>*> (permutation);

  assert (cast_perm != NULL);
  assert (this->nrow_ == this->ncol_);

  int *h_row_offset = NULL;
  int *h_col = NULL;

  allocate_host(this->nrow_+1, &h_row_offset);
  allocate_host(this->nnz_, &h_col);

  ocl_dev2host(this->nrow_+1, this->mat_.row_offset, h_row_offset,
               this->local_backend_.OCL_command_queue);
  ocl_dev2host(this->nnz_, this->mat_.col, h_col,
               this->local_backend_.OCL_command_queue);

  int *mis = NULL;
  allocate_host(this->nrow_, &mis);
  set_to_zero_host(this->nrow_, mis);

  size = 0;

  for (int ai=0; ai<this->nrow_; ++ai) {

    if (mis[ai] == 0) {

      // set the node
      mis[ai] = 1;
      ++size ;

      //remove all nbh nodes (without diagonal)
      for (int aj=h_row_offset[ai]; aj<h_row_offset[ai+1]; ++aj)
        if (ai != h_col[aj])
          mis[h_col[aj]] = -1;

    }

  }

  int *h_perm = NULL;
  allocate_host(this->nrow_, &h_perm);

  int pos = 0;
  for (int ai=0; ai<this->nrow_; ++ai) {

    if (mis[ai] == 1) {

      h_perm[ai] = pos;
      ++pos;

    } else {

      h_perm[ai] = size + ai - pos;

    }

  }

  // Check the permutation
  //
  //  for (int ai=0; ai<this->nrow_; ++ai) {
  //    assert ( h_perm[ai] >= 0 );
  //    assert ( h_perm[ai] < this->nrow_ );
  //  }

  cast_perm->Allocate(this->nrow_);
  ocl_host2dev(permutation->get_size(), // size
               h_perm,                  // src
               cast_perm->vec_,         // dst
               this->local_backend_.OCL_command_queue);

  free_host(&h_row_offset);
  free_host(&h_col);

  free_host(&h_perm);
  free_host(&mis);

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::MultiColoring(int &num_colors, int **size_colors,
                                                       BaseVector<int> *permutation) const {

  assert (permutation != NULL);

  OCLAcceleratorVector<int> *cast_perm = dynamic_cast<OCLAcceleratorVector<int>*> (permutation);

  assert (cast_perm != NULL);

  // node colors (init value = 0 i.e. no color)
  int *color = NULL;
  int *h_row_offset = NULL;
  int *h_col = NULL;
  int size = this->nrow_;
  allocate_host(size, &color);
  allocate_host(this->nrow_+1, &h_row_offset);
  allocate_host(this->nnz_, &h_col);

  ocl_dev2host(this->nrow_+1, this->mat_.row_offset, h_row_offset,
               this->local_backend_.OCL_command_queue);
  ocl_dev2host(this->nnz_, this->mat_.col, h_col,
               this->local_backend_.OCL_command_queue);

  set_to_zero_host(size, color);
  num_colors = 0;
  std::vector<bool> row_col;

  for (int ai=0; ai<this->nrow_; ++ai) {
    color[ai] = 1;
    row_col.clear();
    row_col.assign(num_colors+2, false);

    for (int aj=h_row_offset[ai]; aj<h_row_offset[ai+1]; ++aj)
      if (ai != h_col[aj])
        row_col[color[h_col[aj]]] = true;

    for (int aj=h_row_offset[ai]; aj<h_row_offset[ai+1]; ++aj)
      if (row_col[color[ai]] == true)
        ++color[ai];

    if (color[ai] > num_colors)
      num_colors = color[ai];

  }

  free_host(&h_row_offset);
  free_host(&h_col);

  allocate_host(num_colors, size_colors);
  set_to_zero_host(num_colors, *size_colors);

  int *offsets_color = NULL;
  allocate_host(num_colors, &offsets_color);
  set_to_zero_host(num_colors, offsets_color);

  for (int i=0; i<this->nrow_; ++i)
    ++(*size_colors)[color[i]-1];

  int total=0;
  for (int i=1; i<num_colors; ++i) {

    total += (*size_colors)[i-1];
    offsets_color[i] = total;
    //   LOG_INFO("offsets = " << total);

  }

  int *h_perm = NULL;
  allocate_host(this->nrow_, &h_perm);

  for (int i=0; i<this->nrow_; ++i) {

    h_perm[i] = offsets_color[color[i]-1];
    ++offsets_color[color[i]-1];

  }

  cast_perm->Allocate(this->nrow_);
  ocl_host2dev(permutation->get_size(), // size
               h_perm,                  // src
               cast_perm->vec_,         // dst
               this->local_backend_.OCL_command_queue);

  free_host(&h_perm);
  free_host(&color);
  free_host(&offsets_color);

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::Scale(const ValueType alpha) {

  if (this->nnz_ > 0) {

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nnz_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_scale",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nnz_, alpha, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::ScaleDiagonal(const ValueType alpha) {

  if (this->nnz_ > 0) {

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_scale_diagonal",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nrow_, this->mat_.row_offset, this->mat_.col, alpha, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::ScaleOffDiagonal(const ValueType alpha) {

  if (this->nnz_ > 0) {

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_scale_offdiagonal",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nrow_, this->mat_.row_offset, this->mat_.col, alpha, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::AddScalarDiagonal(const ValueType alpha) {

  if (this->nnz_ > 0) {

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_add_diagonal",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nrow_, this->mat_.row_offset, this->mat_.col, alpha, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::AddScalarOffDiagonal(const ValueType alpha) {

  if (this->nnz_ > 0) {

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_add_offdiagonal",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nrow_, this->mat_.row_offset, this->mat_.col, alpha, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::AddScalar(const ValueType alpha) {

  if (this->nnz_ > 0) {

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nnz_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_buffer_addscalar",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nnz_, alpha, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::DiagonalMatrixMultR(const BaseVector<ValueType> &diag) {

  if (this->nnz_ > 0) {

    assert (diag.get_size() == this->ncol_);

    const OCLAcceleratorVector<ValueType> *cast_diag = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&diag);

    assert (cast_diag != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_diagmatmult_r",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nrow_, this->mat_.row_offset, this->mat_.col, cast_diag->vec_, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::DiagonalMatrixMultL(const BaseVector<ValueType> &diag) {

  if (this->nnz_ > 0) {

    assert (diag.get_size() == this->ncol_);

    const OCLAcceleratorVector<ValueType> *cast_diag = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&diag);

    assert (cast_diag!= NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_diagmatmult_l",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->nrow_, this->mat_.row_offset, this->mat_.col, cast_diag->vec_, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::MatrixAdd(const BaseMatrix<ValueType> &mat, const ValueType alpha,
                                                   const ValueType beta, const bool structure) {

  if (this->nnz_ > 0) {

    const OCLAcceleratorMatrixCSR<ValueType> *cast_mat = dynamic_cast<const OCLAcceleratorMatrixCSR<ValueType>*> (&mat);

    assert (cast_mat != NULL);

    assert (cast_mat->nrow_ == this->nrow_);
    assert (cast_mat->ncol_ == this->ncol_);
    assert (this->nnz_ > 0);
    assert (cast_mat->nnz_ > 0);

    if (structure == false) {

      size_t LocalSize  = this->local_backend_.OCL_block_size;
      size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

      cl_int err = ocl_kernel<ValueType>("kernel_csr_add_csr_same_struct",
                                         this->local_backend_.OCL_command_queue,
                                         LocalSize, GlobalSize,
                                         this->nrow_, this->mat_.row_offset, this->mat_.col,
                                         cast_mat->mat_.row_offset, cast_mat->mat_.col, cast_mat->mat_.val,
                                         alpha, beta, this->mat_.val);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    } else {

      return false;

    }

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::Compress(const double drop_off) {

  if (this->nnz_ > 0) {

    OCLAcceleratorMatrixCSR<ValueType> tmp(this->local_backend_);

    tmp.CopyFrom(*this);

    int mat_nnz = 0;

    int *row_offset = NULL;
    allocate_ocl(this->nrow_+1, this->local_backend_.OCL_context, &row_offset);
    ocl_set_to(this->nrow_+1, (int) 0, row_offset, this->local_backend_.OCL_command_queue);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_compress_count_nrow",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->mat_.row_offset, this->mat_.col, this->mat_.val,
                                       this->nrow_, drop_off, row_offset);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    int *red_row_offset = NULL;
    allocate_host(this->nrow_+1, &red_row_offset);

    // Copy object from device to host memory
    ocl_dev2host(this->nrow_+1, row_offset, red_row_offset, this->local_backend_.OCL_command_queue);

    int sum = 0;
    for (int i=0; i<this->nrow_; ++i) {
      int tmp = red_row_offset[i];
      red_row_offset[i] = sum;
      sum += tmp;
    }

    mat_nnz = red_row_offset[this->nrow_] = sum;

    this->AllocateCSR(mat_nnz, this->nrow_, this->ncol_);

    ocl_host2dev(this->nrow_+1,         // size
                 red_row_offset,        // src
                 this->mat_.row_offset, // dst
                 this->local_backend_.OCL_command_queue);

    free_host(&red_row_offset);

    err = ocl_kernel<ValueType>("kernel_csr_compress_copy",
                                this->local_backend_.OCL_command_queue,
                                LocalSize, GlobalSize,
                                tmp.mat_.row_offset, tmp.mat_.col, tmp.mat_.val,
                                tmp.nrow_, drop_off, this->mat_.row_offset, this->mat_.col, this->mat_.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    free_ocl(&row_offset);

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::ReplaceColumnVector(const int idx, const BaseVector<ValueType> &vec) {

  if (this->nnz_ > 0) {

    assert (idx >= 0);
    assert (vec.get_size() == this->nrow_);

    const OCLAcceleratorVector<ValueType> *cast_vec = dynamic_cast<const OCLAcceleratorVector<ValueType>*> (&vec);

    assert (cast_vec != NULL);

    int *row_offset = NULL;
    int *col = NULL;
    ValueType *val = NULL;

    int nrow = this->nrow_;
    int ncol = this->ncol_;

    allocate_ocl(nrow+1, this->local_backend_.OCL_context, &row_offset);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (nrow / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_replace_column_vector_offset",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->mat_.row_offset, this->mat_.col, nrow, idx, cast_vec->vec_, row_offset);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    int *host_offset = NULL;
    allocate_host(nrow+1, &host_offset);

    ocl_dev2host(nrow+1, row_offset, host_offset, this->local_backend_.OCL_command_queue);

    host_offset[0] = 0;
    for (int i=0; i<nrow; ++i)
      host_offset[i+1] += host_offset[i];

    int nnz = host_offset[nrow];

    ocl_host2dev(nrow+1, host_offset, row_offset, this->local_backend_.OCL_command_queue);

    allocate_ocl(nnz, this->local_backend_.OCL_context, &col);
    allocate_ocl(nnz, this->local_backend_.OCL_context, &val);

    err = ocl_kernel<ValueType>("kernel_csr_replace_column_vector",
                                this->local_backend_.OCL_command_queue,
                                LocalSize, GlobalSize,
                                this->mat_.row_offset, this->mat_.col, this->mat_.val,
                                nrow, idx, cast_vec->vec_, row_offset, col, val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    this->Clear();

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

    this->mat_.row_offset = row_offset;
    this->mat_.col = col;
    this->mat_.val = val;

  }

  return true;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCSR<ValueType>::ExtractColumnVector(const int idx, BaseVector<ValueType> *vec) const {

  if (this->nnz_ > 0) {

    assert (idx >= 0);
    assert (vec != NULL);
    assert (vec->get_size() == this->nrow_);

    OCLAcceleratorVector<ValueType> *cast_vec = dynamic_cast<OCLAcceleratorVector<ValueType>*> (vec);

    assert (cast_vec != NULL);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_csr_extract_column_vector",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       this->mat_.row_offset, this->mat_.col, this->mat_.val,
                                       this->nrow_, idx, cast_vec->vec_);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  }

  return true;

}


template class OCLAcceleratorMatrixCSR<double>;
template class OCLAcceleratorMatrixCSR<float>;
#ifdef SUPPORT_COMPLEX
template class OCLAcceleratorMatrixCSR<std::complex<double> >;
template class OCLAcceleratorMatrixCSR<std::complex<float> >;
#endif

}
