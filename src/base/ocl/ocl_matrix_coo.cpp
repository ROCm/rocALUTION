#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "ocl_allocate_free.hpp"
#include "ocl_utils.hpp"
#include "ocl_matrix_csr.hpp"
#include "ocl_matrix_coo.hpp"
#include "ocl_vector.hpp"
#include "../host/host_matrix_coo.hpp"
#include "../backend_manager.hpp"

#include <algorithm>
#include <complex>
#include <typeinfo>

namespace paralution {

template <typename ValueType>
OCLAcceleratorMatrixCOO<ValueType>::OCLAcceleratorMatrixCOO() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
OCLAcceleratorMatrixCOO<ValueType>::OCLAcceleratorMatrixCOO(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "OCLAcceleratorMatrixCOO::OCLAcceleratorMatrixCOO()",
            "constructor with local_backend");

  this->mat_.row = NULL;
  this->mat_.col = NULL;
  this->mat_.val = NULL;

  this->set_backend(local_backend);

}

template <typename ValueType>
OCLAcceleratorMatrixCOO<ValueType>::~OCLAcceleratorMatrixCOO() {

  LOG_DEBUG(this, "OCLAcceleratorMatrixCOO::~OCLAcceleratorMatrixCOO()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void OCLAcceleratorMatrixCOO<ValueType>::info(void) const {

  LOG_INFO("OCLAcceleratorMatrixCOO<ValueType>");

}

template <typename ValueType>
void OCLAcceleratorMatrixCOO<ValueType>::AllocateCOO(const int nnz, const int nrow, const int ncol) {

  assert (nnz  >= 0);
  assert (ncol >= 0);
  assert (nrow >= 0);

  if (this->nnz_ > 0)
    this->Clear();

  if (nnz > 0) {

    allocate_ocl(nnz, this->local_backend_.OCL_context, &this->mat_.row);
    allocate_ocl(nnz, this->local_backend_.OCL_context, &this->mat_.col);
    allocate_ocl(nnz, this->local_backend_.OCL_context, &this->mat_.val);

    // Set entries of device object to zero
    ocl_set_to(nnz, (int) 0, this->mat_.row, this->local_backend_.OCL_command_queue);
    ocl_set_to(nnz, (int) 0, this->mat_.col, this->local_backend_.OCL_command_queue);
    ocl_set_to(nnz, (ValueType) 0, this->mat_.val, this->local_backend_.OCL_command_queue);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nnz;

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixCOO<ValueType>::SetDataPtrCOO(int **row, int **col, ValueType **val,
                                                       const int nnz, const int nrow, const int ncol) {

  assert (*row != NULL);
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

  this->mat_.row = *row;
  this->mat_.col = *col;
  this->mat_.val = *val;

}

template <typename ValueType>
void OCLAcceleratorMatrixCOO<ValueType>::LeaveDataPtrCOO(int **row, int **col, ValueType **val) {

  assert (this->nrow_ > 0);
  assert (this->ncol_ > 0);
  assert (this->nnz_  > 0);

  cl_int err = clFinish((cl_command_queue) this->local_backend_.OCL_command_queue);
  CHECK_OCL_ERROR(err, __FILE__, __LINE__);

  // see free_host function for details
  *row = this->mat_.row;
  *col = this->mat_.col;
  *val = this->mat_.val;

  this->mat_.row = NULL;
  this->mat_.col = NULL;
  this->mat_.val = NULL;

  this->nrow_ = 0;
  this->ncol_ = 0;
  this->nnz_  = 0;

}

template <typename ValueType>
void OCLAcceleratorMatrixCOO<ValueType>::Clear(void) {

  if (this->nnz_ > 0) {

    free_ocl(&this->mat_.row);
    free_ocl(&this->mat_.col);
    free_ocl(&this->mat_.val);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixCOO<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixCOO<ValueType> *cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == src.get_mat_format());

  // CPU to OCL copy
  if ((cast_mat = dynamic_cast<const HostMatrixCOO<ValueType>*> (&src)) != NULL) {

    if (this->nnz_ == 0)
      this->AllocateCOO(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);

    if (this->nnz_ > 0) {

      assert (this->nnz_  == cast_mat->nnz_);
      assert (this->nrow_ == cast_mat->nrow_);
      assert (this->ncol_ == cast_mat->ncol_);

      // Copy object from host to device memory
      ocl_host2dev(this->nnz_,         // size
                   cast_mat->mat_.row, // src
                   this->mat_.row,     // dst
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

    LOG_INFO("Error unsupported OCL matrix type");
    this->info();
    src.info();
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixCOO<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  assert (dst != NULL);

  HostMatrixCOO<ValueType> *cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == dst->get_mat_format());

  // OCL to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixCOO<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);

    if (cast_mat->nnz_ == 0)
      cast_mat->AllocateCOO(this->nnz_, this->nrow_, this->ncol_);

    if (this->nnz_ > 0) {

      assert (this->nnz_  == cast_mat->nnz_);
      assert (this->nrow_ == cast_mat->nrow_);
      assert (this->ncol_ == cast_mat->ncol_);

      // Copy object from device to host memory
      ocl_dev2host(this->nnz_,         // size
                   this->mat_.row,     // src
                   cast_mat->mat_.row, // dst
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

    LOG_INFO("Error unsupported OCL matrix type");
    this->info();
    dst->info();
    FATAL_ERROR(__FILE__, __LINE__);

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixCOO<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const OCLAcceleratorMatrixCOO<ValueType> *ocl_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == src.get_mat_format());

  // OCL to OCL copy
  if ((ocl_cast_mat = dynamic_cast<const OCLAcceleratorMatrixCOO<ValueType>*> (&src)) != NULL) {

    if (this->nnz_ == 0)
      this->AllocateCOO(ocl_cast_mat->nnz_, ocl_cast_mat->nrow_, ocl_cast_mat->ncol_);

    assert (this->nnz_  == ocl_cast_mat->nnz_);
    assert (this->nrow_ == ocl_cast_mat->nrow_);
    assert (this->ncol_ == ocl_cast_mat->ncol_);

    if (this->nnz_ > 0) {

      // Copy object from device to device memory (internal copy)
      ocl_dev2dev(this->nnz_,             // size
                  ocl_cast_mat->mat_.row, // src
                  this->mat_.row,         // dst
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

      LOG_INFO("Error unsupported OCL matrix type");
      this->info();
      src.info();
      FATAL_ERROR(__FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixCOO<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  assert (dst != NULL);

  OCLAcceleratorMatrixCOO<ValueType> *ocl_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == dst->get_mat_format());

  // OCL to OCL copy
  if ((ocl_cast_mat = dynamic_cast<OCLAcceleratorMatrixCOO<ValueType>*> (dst)) != NULL) {

    ocl_cast_mat->set_backend(this->local_backend_);

    if (ocl_cast_mat->nnz_ == 0)
      ocl_cast_mat->AllocateCOO(this->nnz_, this->nrow_, this->ncol_);

    assert (this->nnz_  == ocl_cast_mat->nnz_);
    assert (this->nrow_ == ocl_cast_mat->nrow_);
    assert (this->ncol_ == ocl_cast_mat->ncol_);

    if (this->nnz_ > 0) {

      // Copy object from device to device memory (internal copy)
      ocl_dev2dev(this->nnz_,             // size
                  this->mat_.row,         // src
                  ocl_cast_mat->mat_.row, // dst
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

      LOG_INFO("Error unsupported OCL matrix type");
      this->info();
      dst->info();
      FATAL_ERROR(__FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
bool OCLAcceleratorMatrixCOO<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const OCLAcceleratorMatrixCOO<ValueType> *cast_mat_coo;

  if ((cast_mat_coo = dynamic_cast<const OCLAcceleratorMatrixCOO<ValueType>*> (&mat)) != NULL) {

      this->CopyFrom(*cast_mat_coo);
      return true;

  }

  const OCLAcceleratorMatrixCSR<ValueType> *cast_mat_csr;
  if ((cast_mat_csr = dynamic_cast<const OCLAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {

    this->Clear();

    assert (cast_mat_csr->nrow_ > 0);
    assert (cast_mat_csr->ncol_ > 0);
    assert (cast_mat_csr->nnz_  > 0);

    this->AllocateCOO(cast_mat_csr->nnz_, cast_mat_csr->nrow_, cast_mat_csr->ncol_);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (cast_mat_csr->nrow_ / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_coo_csr_to_coo",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       cast_mat_csr->nrow_, cast_mat_csr->mat_.row_offset, this->mat_.row);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    ocl_dev2dev(this->nnz_,             // size
                cast_mat_csr->mat_.col, // src
                this->mat_.col,         // dst
                this->local_backend_.OCL_command_queue);

    ocl_dev2dev(this->nnz_,             // size
                cast_mat_csr->mat_.val, // src
                this->mat_.val,         // dst
                this->local_backend_.OCL_command_queue);

    return true;

  }

  return false;

}

template <typename ValueType>
void OCLAcceleratorMatrixCOO<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

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

    cast_out->Zeros();

    cl_int err;

    // If nnz < warpsize, perform squential spmv
    if (this->nnz_ < this->local_backend_.OCL_warp_size) {

      err = ocl_kernel<ValueType>("kernel_coo_spmv_serial",
                                  this->local_backend_.OCL_command_queue,
                                  1, 1,
                                  this->nnz_, this->mat_.row, this->mat_.col, this->mat_.val,
                                  (ValueType) 1, cast_in->vec_, cast_out->vec_, 0);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    } else {

      // ----------------------------------------------------------
      // Modified and adapted from CUSP 0.3.1, 
      // http://code.google.com/p/cusp-library/
      // NVIDIA, APACHE LICENSE 2.0
      // ----------------------------------------------------------
      // see __spmv_coo_flat(...)
      // ----------------------------------------------------------
      // CHANGELOG
      // - adapted interface
      // ----------------------------------------------------------  

      int warps_per_block = this->local_backend_.OCL_block_size / this->local_backend_.OCL_warp_size;
      int numWarps = (this->local_backend_.OCL_block_size + (this->local_backend_.OCL_warp_size - 1)) / this->local_backend_.OCL_warp_size;

      int ctaLimitThreads = this->local_backend_.OCL_threads_per_proc / this->local_backend_.OCL_block_size;
      int ctaLimitRegs = static_cast<int> (this->local_backend_.OCL_regs_per_block) / this->local_backend_.OCL_block_size / numWarps;

      int max_blocks = this->local_backend_.OCL_num_procs * std::min(ctaLimitRegs, std::min(ctaLimitThreads, 16));
      int num_units  = this->nnz_ / this->local_backend_.OCL_warp_size;
      int num_warps  = (num_units < warps_per_block * max_blocks) ? num_units : warps_per_block * max_blocks;
      int num_blocks = (num_warps + (warps_per_block - 1)) / warps_per_block;
      int num_iters  = (num_units + (num_warps - 1)) / num_warps;

      int interval_size = this->local_backend_.OCL_warp_size * num_iters;
      int tail = num_units * this->local_backend_.OCL_warp_size;
      int active_warps = (interval_size == 0) ? 0 : (tail + (interval_size - 1)) / interval_size;

      int *temp_rows = NULL;
      ValueType *temp_vals = NULL;

      allocate_ocl(active_warps, this->local_backend_.OCL_context, &temp_rows);
      allocate_ocl(active_warps, this->local_backend_.OCL_context, &temp_vals);

      size_t LocalSize = this->local_backend_.OCL_block_size;
      size_t GlobalSize = num_blocks * LocalSize;

      err = ocl_kernel<ValueType>("kernel_coo_spmv_flat",
                                  this->local_backend_.OCL_command_queue,
                                  LocalSize, GlobalSize,
                                  tail, interval_size, this->mat_.row, this->mat_.col, this->mat_.val,
                                  (ValueType) 1, cast_in->vec_, cast_out->vec_, temp_rows, temp_vals);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      err = ocl_kernel<ValueType>("kernel_coo_spmv_reduce_update",
                                  this->local_backend_.OCL_command_queue,
                                  LocalSize, LocalSize,
                                  active_warps, temp_rows, temp_vals, cast_out->vec_);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      free_ocl(&temp_rows);
      free_ocl(&temp_vals);

      err = ocl_kernel<ValueType>("kernel_coo_spmv_serial",
                                  this->local_backend_.OCL_command_queue,
                                  1, 1,
                                  this->nnz_, this->mat_.row, this->mat_.col, this->mat_.val,
                                  (ValueType) 1, cast_in->vec_, cast_out->vec_, tail);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixCOO<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
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

    cl_int err;

    // If matrix is very sparse, we do COO via atomics
    if (typeid(ValueType) != typeid(std::complex<float>) &&
        typeid(ValueType) != typeid(std::complex<double>) &&
        this->nnz_ / this->nrow_ < 2) {

      size_t LocalSize = this->local_backend_.OCL_block_size;
      size_t GlobalSize = (this->nnz_ / LocalSize + 1) * LocalSize;

      err = ocl_kernel<ValueType>("kernel_coo_add_spmv",
                                  this->local_backend_.OCL_command_queue,
                                  LocalSize, GlobalSize,
                                  this->nnz_, this->mat_.row, this->mat_.col, this->mat_.val,
                                  scalar, cast_in->vec_, cast_out->vec_);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    } else {

      // If nnz < warpsize, perform squential spmv
      if (this->nnz_ < this->local_backend_.OCL_warp_size) {

        err = ocl_kernel<ValueType>("kernel_coo_spmv_serial",
                                    this->local_backend_.OCL_command_queue,
                                    1, 1,
                                    this->nnz_, this->mat_.row, this->mat_.col, this->mat_.val,
                                    scalar, cast_in->vec_, cast_out->vec_, 0);
        CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      } else {

        // ----------------------------------------------------------
        // Modified and adapted from CUSP 0.3.1, 
        // http://code.google.com/p/cusp-library/
        // NVIDIA, APACHE LICENSE 2.0
        // ----------------------------------------------------------
        // see __spmv_coo_flat(...)
        // ----------------------------------------------------------
        // CHANGELOG
        // - adapted interface
        // ----------------------------------------------------------  

        int warps_per_block = this->local_backend_.OCL_block_size / this->local_backend_.OCL_warp_size;
        int numWarps = (this->local_backend_.OCL_block_size + (this->local_backend_.OCL_warp_size - 1)) / this->local_backend_.OCL_warp_size;

        int ctaLimitThreads = this->local_backend_.OCL_threads_per_proc / this->local_backend_.OCL_block_size;
        int ctaLimitRegs = static_cast<int> (this->local_backend_.OCL_regs_per_block) / this->local_backend_.OCL_block_size / numWarps;

        int max_blocks = this->local_backend_.OCL_num_procs * std::min(ctaLimitRegs, std::min(ctaLimitThreads, 16));
        int num_units  = this->nnz_ / this->local_backend_.OCL_warp_size;
        int num_warps  = (num_units < warps_per_block * max_blocks) ? num_units : warps_per_block * max_blocks;
        int num_blocks = (num_warps + (warps_per_block - 1)) / warps_per_block;
        int num_iters  = (num_units + (num_warps - 1)) / num_warps;

        int interval_size = this->local_backend_.OCL_warp_size * num_iters;
        int tail = num_units * this->local_backend_.OCL_warp_size;
        int active_warps = (interval_size == 0) ? 0 : (tail + (interval_size - 1)) / interval_size;

        int *temp_rows = NULL;
        ValueType *temp_vals = NULL;

        allocate_ocl(active_warps, this->local_backend_.OCL_context, &temp_rows);
        allocate_ocl(active_warps, this->local_backend_.OCL_context, &temp_vals);

        size_t LocalSize = this->local_backend_.OCL_block_size;
        size_t GlobalSize = num_blocks * LocalSize;

        err = ocl_kernel<ValueType>("kernel_coo_spmv_flat",
                                    this->local_backend_.OCL_command_queue,
                                    LocalSize, GlobalSize,
                                    tail, interval_size, this->mat_.row, this->mat_.col, this->mat_.val,
                                    scalar, cast_in->vec_, cast_out->vec_, temp_rows, temp_vals);
        CHECK_OCL_ERROR(err, __FILE__, __LINE__);

        err = ocl_kernel<ValueType>("kernel_coo_spmv_reduce_update",
                                    this->local_backend_.OCL_command_queue,
                                    LocalSize, LocalSize,
                                    active_warps, temp_rows, temp_vals, cast_out->vec_);
        CHECK_OCL_ERROR(err, __FILE__, __LINE__);

        free_ocl(&temp_rows);
        free_ocl(&temp_vals);

        err = ocl_kernel<ValueType>("kernel_coo_spmv_serial",
                                    this->local_backend_.OCL_command_queue,
                                    1, 1,
                                    this->nnz_, this->mat_.row, this->mat_.col, this->mat_.val,
                                    scalar, cast_in->vec_, cast_out->vec_, tail);
        CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      }

    }

  }

}

template <>
void OCLAcceleratorMatrixCOO<std::complex<float> >::ApplyAdd(const BaseVector<std::complex<float> > &in, const std::complex<float> scalar,
                                                             BaseVector<std::complex<float> > *out) const {

  if (this->nnz_ > 0) {

    assert (out != NULL);
    assert (in.  get_size() >= 0);
    assert (out->get_size() >= 0);
    assert (in.  get_size() == this->ncol_);
    assert (out->get_size() == this->nrow_);

    const OCLAcceleratorVector<std::complex<float> > *cast_in = dynamic_cast<const OCLAcceleratorVector<std::complex<float> >*> (&in);
    OCLAcceleratorVector<std::complex<float> > *cast_out      = dynamic_cast<      OCLAcceleratorVector<std::complex<float> >*> (out);

    assert (cast_in  != NULL);
    assert (cast_out != NULL);

    cl_int err;

    // If nnz < warpsize, perform squential spmv
    if (this->nnz_ < this->local_backend_.OCL_warp_size) {

      err = ocl_kernel<std::complex<float> >("kernel_coo_spmv_serial",
                                             this->local_backend_.OCL_command_queue,
                                             1, 1,
                                             this->nnz_, this->mat_.row, this->mat_.col, this->mat_.val,
                                             scalar, cast_in->vec_, cast_out->vec_, 0);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    } else {

      // ----------------------------------------------------------
      // Modified and adapted from CUSP 0.3.1, 
      // http://code.google.com/p/cusp-library/
      // NVIDIA, APACHE LICENSE 2.0
      // ----------------------------------------------------------
      // see __spmv_coo_flat(...)
      // ----------------------------------------------------------
      // CHANGELOG
      // - adapted interface
      // ----------------------------------------------------------  

      int warps_per_block = this->local_backend_.OCL_block_size / this->local_backend_.OCL_warp_size;
      int numWarps = (this->local_backend_.OCL_block_size + (this->local_backend_.OCL_warp_size - 1)) / this->local_backend_.OCL_warp_size;

      int ctaLimitThreads = this->local_backend_.OCL_threads_per_proc / this->local_backend_.OCL_block_size;
      int ctaLimitRegs = static_cast<int> (this->local_backend_.OCL_regs_per_block) / this->local_backend_.OCL_block_size / numWarps;

      int max_blocks = this->local_backend_.OCL_num_procs * std::min(ctaLimitRegs, std::min(ctaLimitThreads, 16));
      int num_units  = this->nnz_ / this->local_backend_.OCL_warp_size;
      int num_warps  = (num_units < warps_per_block * max_blocks) ? num_units : warps_per_block * max_blocks;
      int num_blocks = (num_warps + (warps_per_block - 1)) / warps_per_block;
      int num_iters  = (num_units + (num_warps - 1)) / num_warps;

      int interval_size = this->local_backend_.OCL_warp_size * num_iters;
      int tail = num_units * this->local_backend_.OCL_warp_size;
      int active_warps = (interval_size == 0) ? 0 : (tail + (interval_size - 1)) / interval_size;

      int *temp_rows = NULL;
      std::complex<float> *temp_vals = NULL;

      allocate_ocl(active_warps, this->local_backend_.OCL_context, &temp_rows);
      allocate_ocl(active_warps, this->local_backend_.OCL_context, &temp_vals);

      size_t LocalSize = this->local_backend_.OCL_block_size;
      size_t GlobalSize = num_blocks * LocalSize;

      err = ocl_kernel<std::complex<float> >("kernel_coo_spmv_flat",
                                             this->local_backend_.OCL_command_queue,
                                             LocalSize, GlobalSize,
                                             tail, interval_size, this->mat_.row, this->mat_.col, this->mat_.val,
                                             scalar, cast_in->vec_, cast_out->vec_, temp_rows, temp_vals);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      err = ocl_kernel<std::complex<float> >("kernel_coo_spmv_reduce_update",
                                             this->local_backend_.OCL_command_queue,
                                             LocalSize, LocalSize,
                                             active_warps, temp_rows, temp_vals, cast_out->vec_);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      free_ocl(&temp_rows);
      free_ocl(&temp_vals);

      err = ocl_kernel<std::complex<float> >("kernel_coo_spmv_serial",
                                             this->local_backend_.OCL_command_queue,
                                             1, 1,
                                             this->nnz_, this->mat_.row, this->mat_.col, this->mat_.val,
                                             scalar, cast_in->vec_, cast_out->vec_, tail);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    }

  }

}

template <>
void OCLAcceleratorMatrixCOO<std::complex<double> >::ApplyAdd(const BaseVector<std::complex<double> > &in, const std::complex<double> scalar,
                                                              BaseVector<std::complex<double> > *out) const {

  if (this->nnz_ > 0) {

    assert (out != NULL);
    assert (in.  get_size() >= 0);
    assert (out->get_size() >= 0);
    assert (in.  get_size() == this->ncol_);
    assert (out->get_size() == this->nrow_);

    const OCLAcceleratorVector<std::complex<double> > *cast_in = dynamic_cast<const OCLAcceleratorVector<std::complex<double> >*> (&in);
    OCLAcceleratorVector<std::complex<double> > *cast_out      = dynamic_cast<      OCLAcceleratorVector<std::complex<double> >*> (out);

    assert (cast_in  != NULL);
    assert (cast_out != NULL);

    cl_int err;

    // If nnz < warpsize, perform squential spmv
    if (this->nnz_ < this->local_backend_.OCL_warp_size) {

      err = ocl_kernel<std::complex<double> >("kernel_coo_spmv_serial",
                                              this->local_backend_.OCL_command_queue,
                                              1, 1,
                                              this->nnz_, this->mat_.row, this->mat_.col, this->mat_.val,
                                              scalar, cast_in->vec_, cast_out->vec_, 0);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    } else {

      // ----------------------------------------------------------
      // Modified and adapted from CUSP 0.3.1, 
      // http://code.google.com/p/cusp-library/
      // NVIDIA, APACHE LICENSE 2.0
      // ----------------------------------------------------------
      // see __spmv_coo_flat(...)
      // ----------------------------------------------------------
      // CHANGELOG
      // - adapted interface
      // ----------------------------------------------------------  

      int warps_per_block = this->local_backend_.OCL_block_size / this->local_backend_.OCL_warp_size;
      int numWarps = (this->local_backend_.OCL_block_size + (this->local_backend_.OCL_warp_size - 1)) / this->local_backend_.OCL_warp_size;

      int ctaLimitThreads = this->local_backend_.OCL_threads_per_proc / this->local_backend_.OCL_block_size;
      int ctaLimitRegs = static_cast<int> (this->local_backend_.OCL_regs_per_block) / this->local_backend_.OCL_block_size / numWarps;

      int max_blocks = this->local_backend_.OCL_num_procs * std::min(ctaLimitRegs, std::min(ctaLimitThreads, 16));
      int num_units  = this->nnz_ / this->local_backend_.OCL_warp_size;
      int num_warps  = (num_units < warps_per_block * max_blocks) ? num_units : warps_per_block * max_blocks;
      int num_blocks = (num_warps + (warps_per_block - 1)) / warps_per_block;
      int num_iters  = (num_units + (num_warps - 1)) / num_warps;

      int interval_size = this->local_backend_.OCL_warp_size * num_iters;
      int tail = num_units * this->local_backend_.OCL_warp_size;
      int active_warps = (interval_size == 0) ? 0 : (tail + (interval_size - 1)) / interval_size;

      int *temp_rows = NULL;
      std::complex<double> *temp_vals = NULL;

      allocate_ocl(active_warps, this->local_backend_.OCL_context, &temp_rows);
      allocate_ocl(active_warps, this->local_backend_.OCL_context, &temp_vals);

      size_t LocalSize = this->local_backend_.OCL_block_size;
      size_t GlobalSize = num_blocks * LocalSize;

      err = ocl_kernel<std::complex<double> >("kernel_coo_spmv_flat",
                                              this->local_backend_.OCL_command_queue,
                                              LocalSize, GlobalSize,
                                              tail, interval_size, this->mat_.row, this->mat_.col, this->mat_.val,
                                              scalar, cast_in->vec_, cast_out->vec_, temp_rows, temp_vals);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      err = ocl_kernel<std::complex<double> >("kernel_coo_spmv_reduce_update",
                                              this->local_backend_.OCL_command_queue,
                                              LocalSize, LocalSize,
                                              active_warps, temp_rows, temp_vals, cast_out->vec_);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      free_ocl(&temp_rows);
      free_ocl(&temp_vals);

      err = ocl_kernel<std::complex<double> >("kernel_coo_spmv_serial",
                                              this->local_backend_.OCL_command_queue,
                                              1, 1,
                                              this->nnz_, this->mat_.row, this->mat_.col, this->mat_.val,
                                              scalar, cast_in->vec_, cast_out->vec_, tail);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    }

  }

}

template <typename ValueType>
bool OCLAcceleratorMatrixCOO<ValueType>::Permute(const BaseVector<int> &permutation) {
/*
  // symmetric permutation only
  assert (permutation.get_size() == this->nrow_);
  assert (permutation.get_size() == this->ncol_);

  if (this->nnz_ > 0) {

    const OCLAcceleratorVector<int> *cast_perm = dynamic_cast<const OCLAcceleratorVector<int>*> (&permutation);
    assert (cast_perm != NULL);

    OCLAcceleratorMatrixCOO<ValueType> src(this->local_backend_);
    src.AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
    src.CopyFrom(*this);

    LOG_INFO("OCLAcceleratorMatrixCOO::Permute NYI")
    FATAL_ERROR(__FILE__, __LINE__);

  }
*/
  return false;

}

template <typename ValueType>
bool OCLAcceleratorMatrixCOO<ValueType>::PermuteBackward(const BaseVector<int> &permutation) {
/*
  // symmetric permutation only
  assert (permutation.get_size() == this->nrow_);
  assert (permutation.get_size() == this->ncol_);

  if (this->nnz_ > 0) {

    const OCLAcceleratorVector<int> *cast_perm = dynamic_cast<const OCLAcceleratorVector<int>*> (&permutation);
    assert (cast_perm != NULL);

    LOG_INFO("OCLAcceleratorMatrixCOO::PermuteBackward NYI");
    FATAL_ERROR(__FILE__, __LINE__);

  }
*/
  return false;

}


template class OCLAcceleratorMatrixCOO<double>;
template class OCLAcceleratorMatrixCOO<float>;
#ifdef SUPPORT_COMPLEX
template class OCLAcceleratorMatrixCOO<std::complex<double> >;
template class OCLAcceleratorMatrixCOO<std::complex<float> >;
#endif

}
