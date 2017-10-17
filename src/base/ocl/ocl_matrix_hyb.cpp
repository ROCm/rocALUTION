#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "ocl_allocate_free.hpp"
#include "ocl_utils.hpp"
#include "ocl_matrix_csr.hpp"
#include "ocl_matrix_coo.hpp"
#include "ocl_matrix_ell.hpp"
#include "ocl_matrix_hyb.hpp"
#include "ocl_vector.hpp"
#include "../host/host_matrix_hyb.hpp"
#include "../backend_manager.hpp"

#include <algorithm>
#include <complex>
#include <typeinfo>

namespace paralution {

template <typename ValueType>
OCLAcceleratorMatrixHYB<ValueType>::OCLAcceleratorMatrixHYB() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
OCLAcceleratorMatrixHYB<ValueType>::OCLAcceleratorMatrixHYB(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "OCLAcceleratorMatrixHYB::OCLAcceleratorMatrixHYB()",
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
OCLAcceleratorMatrixHYB<ValueType>::~OCLAcceleratorMatrixHYB() {

  LOG_DEBUG(this, "OCLAcceleratorMatrixHYB::~OCLAcceleratorMatrixHYB()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void OCLAcceleratorMatrixHYB<ValueType>::info(void) const {

  LOG_INFO("OCLAcceleratorMatrixHYB<ValueType>");

}

template <typename ValueType>
void OCLAcceleratorMatrixHYB<ValueType>::AllocateHYB(const int ell_nnz, const int coo_nnz, const int ell_max_row,
                                                     const int nrow, const int ncol) {

  assert (ell_nnz >= 0);
  assert (coo_nnz >= 0);
  assert (ell_max_row >= 0);

  assert (ncol >= 0);
  assert (nrow >= 0);

  if (this->nnz_ > 0)
    this->Clear();

  if (ell_nnz + coo_nnz > 0) {

    // ELL
    assert (ell_nnz == ell_max_row * nrow);

    allocate_ocl(ell_nnz, this->local_backend_.OCL_context, &this->mat_.ELL.col);
    allocate_ocl(ell_nnz, this->local_backend_.OCL_context, &this->mat_.ELL.val);

    ocl_set_to(ell_nnz, (int) 0, this->mat_.ELL.col, this->local_backend_.OCL_command_queue);
    ocl_set_to(ell_nnz, (ValueType) 0, this->mat_.ELL.val, this->local_backend_.OCL_command_queue);

    this->mat_.ELL.max_row = ell_max_row;
    this->ell_nnz_ = ell_nnz;

    // COO
    allocate_ocl(coo_nnz, this->local_backend_.OCL_context, &this->mat_.COO.row);
    allocate_ocl(coo_nnz, this->local_backend_.OCL_context, &this->mat_.COO.col);
    allocate_ocl(coo_nnz, this->local_backend_.OCL_context, &this->mat_.COO.val);

    ocl_set_to(coo_nnz, (int) 0, this->mat_.COO.row, this->local_backend_.OCL_command_queue);
    ocl_set_to(coo_nnz, (int) 0, this->mat_.COO.col, this->local_backend_.OCL_command_queue);
    ocl_set_to(coo_nnz, (ValueType) 0, this->mat_.COO.val, this->local_backend_.OCL_command_queue);

    this->nrow_ = nrow;
    this->ncol_ = ncol;

    this->coo_nnz_ = coo_nnz;
    this->nnz_  = ell_nnz + coo_nnz;

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixHYB<ValueType>::Clear(void) {

  if (this->nnz_ > 0) {

    free_ocl(&this->mat_.COO.row);
    free_ocl(&this->mat_.COO.col);
    free_ocl(&this->mat_.COO.val);

    free_ocl(&this->mat_.ELL.val);
    free_ocl(&this->mat_.ELL.col);

    this->ell_nnz_ = 0;
    this->coo_nnz_ = 0;
    this->mat_.ELL.max_row = 0;

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixHYB<ValueType>::CopyFromHost(const HostMatrix<ValueType> &src) {

  const HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == src.get_mat_format());

  // CPU to OCL copy
  if ((cast_mat = dynamic_cast<const HostMatrixHYB<ValueType>*> (&src)) != NULL) {

    if (this->nnz_ == 0)
      this->AllocateHYB(cast_mat->ell_nnz_, cast_mat->coo_nnz_, cast_mat->mat_.ELL.max_row,
                        cast_mat->nrow_, cast_mat->ncol_);

    assert (this->nnz_  == cast_mat->nnz_);
    assert (this->nrow_ == cast_mat->nrow_);
    assert (this->ncol_ == cast_mat->ncol_);

    if (this->ell_nnz_ > 0) {

      // ELL
      ocl_host2dev(this->ell_nnz_,         // size
                   cast_mat->mat_.ELL.col, // src
                   this->mat_.ELL.col,     // dst
                   this->local_backend_.OCL_command_queue);

      ocl_host2dev(this->ell_nnz_,         // size
                   cast_mat->mat_.ELL.val, // src
                   this->mat_.ELL.val,     // dst
                   this->local_backend_.OCL_command_queue);

    }

    if (this->coo_nnz_ > 0) {

      // COO
      ocl_host2dev(this->coo_nnz_,         // size
                   cast_mat->mat_.COO.row, // src
                   this->mat_.COO.row,     // dst
                   this->local_backend_.OCL_command_queue);

      ocl_host2dev(this->coo_nnz_,         // size
                   cast_mat->mat_.COO.col, // src
                   this->mat_.COO.col,     // dst
                   this->local_backend_.OCL_command_queue);

      ocl_host2dev(this->coo_nnz_,         // size
                   cast_mat->mat_.COO.val, // src
                   this->mat_.COO.val,     // dst
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
void OCLAcceleratorMatrixHYB<ValueType>::CopyToHost(HostMatrix<ValueType> *dst) const {

  assert (dst != NULL);

  HostMatrixHYB<ValueType> *cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == dst->get_mat_format());

  // OCL to CPU copy
  if ((cast_mat = dynamic_cast<HostMatrixHYB<ValueType>*> (dst)) != NULL) {

    cast_mat->set_backend(this->local_backend_);

    if (cast_mat->nnz_ == 0)
      cast_mat->AllocateHYB(this->ell_nnz_, this->coo_nnz_, this->mat_.ELL.max_row,
                            this->nrow_, this->ncol_);

    assert (this->nnz_  == cast_mat->nnz_);
    assert (this->nrow_ == cast_mat->nrow_);
    assert (this->ncol_ == cast_mat->ncol_);

    if (this->ell_nnz_ > 0) {

      // ELL
      ocl_dev2host(this->ell_nnz_,         // size
                   this->mat_.ELL.col,     // src
                   cast_mat->mat_.ELL.col, // dst
                   this->local_backend_.OCL_command_queue);

      ocl_dev2host(this->ell_nnz_,         // size
                   this->mat_.ELL.val,     // src
                   cast_mat->mat_.ELL.val, // dst
                   this->local_backend_.OCL_command_queue);

    }

    if (this->coo_nnz_ > 0) {

      // COO
      ocl_dev2host(this->coo_nnz_,         // size
                   this->mat_.COO.row,     // src
                   cast_mat->mat_.COO.row, // dst
                   this->local_backend_.OCL_command_queue);

      ocl_dev2host(this->coo_nnz_,         // size
                   this->mat_.COO.col,     // src
                   cast_mat->mat_.COO.col, // dst
                   this->local_backend_.OCL_command_queue);

      ocl_dev2host(this->coo_nnz_,         // size
                   this->mat_.COO.val,     // src
                   cast_mat->mat_.COO.val, // dst
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
void OCLAcceleratorMatrixHYB<ValueType>::CopyFrom(const BaseMatrix<ValueType> &src) {

  const OCLAcceleratorMatrixHYB<ValueType> *ocl_cast_mat;
  const HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == src.get_mat_format());

  // OCL to OCL copy
  if ((ocl_cast_mat = dynamic_cast<const OCLAcceleratorMatrixHYB<ValueType>*> (&src)) != NULL) {

    if (this->nnz_ == 0)
      this->AllocateHYB(ocl_cast_mat->ell_nnz_, ocl_cast_mat->coo_nnz_, ocl_cast_mat->mat_.ELL.max_row,
                        ocl_cast_mat->nrow_, ocl_cast_mat->ncol_);

    assert (this->nnz_  == ocl_cast_mat->nnz_);
    assert (this->nrow_ == ocl_cast_mat->nrow_);
    assert (this->ncol_ == ocl_cast_mat->ncol_);

    if (this->ell_nnz_ > 0) {

      // ELL
      // must be within same opencl context
      ocl_dev2dev(this->ell_nnz_,             // size
                  ocl_cast_mat->mat_.ELL.col, // src
                  this->mat_.ELL.col,         // dst
                  this->local_backend_.OCL_command_queue);

      ocl_dev2dev(this->ell_nnz_,             // size
                  ocl_cast_mat->mat_.ELL.val, // src
                  this->mat_.ELL.val,         // dst
                  this->local_backend_.OCL_command_queue);

    }

    if (this->coo_nnz_ > 0) {

      // COO
      // must be within same opencl context
      ocl_dev2dev(this->coo_nnz_,             // size
                  ocl_cast_mat->mat_.COO.row, // src
                  this->mat_.COO.row,         // dst
                  this->local_backend_.OCL_command_queue);

      ocl_dev2dev(this->coo_nnz_,             // size
                  ocl_cast_mat->mat_.COO.col, // src
                  this->mat_.COO.col,         // dst
                  this->local_backend_.OCL_command_queue);

      ocl_dev2dev(this->coo_nnz_,             // size
                  ocl_cast_mat->mat_.COO.val, // src
                  this->mat_.COO.val,         // dst
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
void OCLAcceleratorMatrixHYB<ValueType>::CopyTo(BaseMatrix<ValueType> *dst) const {

  assert (dst != NULL);

  OCLAcceleratorMatrixHYB<ValueType> *ocl_cast_mat;
  HostMatrix<ValueType> *host_cast_mat;

  // copy only in the same format
  assert (this->get_mat_format() == dst->get_mat_format());

  // OCL to OCL copy
  if ((ocl_cast_mat = dynamic_cast<OCLAcceleratorMatrixHYB<ValueType>*> (dst)) != NULL) {

    ocl_cast_mat->set_backend(this->local_backend_);

    if (ocl_cast_mat->nnz_ == 0)
      ocl_cast_mat->AllocateHYB(this->ell_nnz_, this->coo_nnz_, this->mat_.ELL.max_row,
                                this->nrow_, this->ncol_);

    assert (this->nnz_  == ocl_cast_mat->nnz_);
    assert (this->nrow_ == ocl_cast_mat->nrow_);
    assert (this->ncol_ == ocl_cast_mat->ncol_);

    if (this->ell_nnz_ > 0) {

      // ELL
      // must be within same opencl context
      ocl_dev2dev(this->ell_nnz_,             // size
                  this->mat_.ELL.col,         // src
                  ocl_cast_mat->mat_.ELL.col, // dst
                  this->local_backend_.OCL_command_queue);

      ocl_dev2dev(this->ell_nnz_,             // size
                  this->mat_.ELL.val,         // src
                  ocl_cast_mat->mat_.ELL.val, // dst
                  this->local_backend_.OCL_command_queue);

    }

    if (this->coo_nnz_ > 0) {

      // COO
      // must be within same opencl context
      ocl_dev2dev(this->coo_nnz_,             // size
                  this->mat_.COO.row,         // src
                  ocl_cast_mat->mat_.COO.row, // dst
                  this->local_backend_.OCL_command_queue);

      ocl_dev2dev(this->coo_nnz_,             // size
                  this->mat_.COO.col,         // src
                  ocl_cast_mat->mat_.COO.col, // dst
                  this->local_backend_.OCL_command_queue);

      ocl_dev2dev(this->coo_nnz_,             // size
                  this->mat_.COO.val,         // src
                  ocl_cast_mat->mat_.COO.val, // dst
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
bool OCLAcceleratorMatrixHYB<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  const OCLAcceleratorMatrixHYB<ValueType> *cast_mat_hyb;

  if ((cast_mat_hyb = dynamic_cast<const OCLAcceleratorMatrixHYB<ValueType>*> (&mat)) != NULL) {

      this->CopyFrom(*cast_mat_hyb);
      return true;

  }

  const OCLAcceleratorMatrixCSR<ValueType> *cast_mat_csr;

  if ((cast_mat_csr = dynamic_cast<const OCLAcceleratorMatrixCSR<ValueType>*> (&mat)) != NULL) {

    this->Clear();

    assert (cast_mat_csr->nrow_ > 0);
    assert (cast_mat_csr->ncol_ > 0);
    assert (cast_mat_csr->nnz_ > 0);

    int nrow = cast_mat_csr->nrow_;
    int ncol = cast_mat_csr->ncol_;
    int max_row = cast_mat_csr->nnz_ / nrow;

    // get nnz per row for COO part
    int *nnz_coo = NULL;
    allocate_ocl(nrow, this->local_backend_.OCL_context, &nnz_coo);

    size_t LocalSize  = this->local_backend_.OCL_block_size;
    size_t GlobalSize = (nrow / LocalSize + 1) * LocalSize;

    cl_int err = ocl_kernel<ValueType>("kernel_hyb_ell_nnz_coo",
                                       this->local_backend_.OCL_command_queue,
                                       LocalSize, GlobalSize,
                                       nrow, max_row, cast_mat_csr->mat_.row_offset, nnz_coo);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    // TODO fix int kernel for reduce
    int *hostBuffer = NULL;
    allocate_host(nrow, &hostBuffer);
    ocl_dev2host(nrow, nnz_coo, hostBuffer, this->local_backend_.OCL_command_queue);
    int num_nnz_coo = 0;

    for (int i=0; i<nrow; ++i)
      num_nnz_coo += hostBuffer[i];
    free_host(&hostBuffer);
/*
    // get nnz for COO part by summing up nnz per row array
    int reducesize = (int) this->local_backend_.OCL_computeUnits * 4;
    int *deviceBuffer = NULL;
    int *hostBuffer = NULL;

    allocate_ocl(reducesize, this->local_backend_.OCL_context, &deviceBuffer);

    GlobalSize = reducesize * LocalSize;

    int GROUP_SIZE = ((nrow / reducesize + 1) / (int) LocalSize + 1) * (int) LocalSize;
    int LOCAL_SIZE = GROUP_SIZE / (int) LocalSize;

    err = ocl_kernel<int>(CL_KERNEL_REDUCE,
                          this->local_backend_.OCL_command_queue,
                          LocalSize, GlobalSize,
                          nrow, nnz_coo, deviceBuffer, GROUP_SIZE, LOCAL_SIZE);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    allocate_host(reducesize, &hostBuffer);

    ocl_dev2host(reducesize, deviceBuffer, hostBuffer, this->local_backend_.OCL_command_queue);
    free_ocl(&deviceBuffer);

    int num_nnz_coo = 0;
    for (int i=0; i<reducesize; ++i)
      num_nnz_coo += hostBuffer[i];

    free_host(&hostBuffer);
*/
    // END TODO

    // allocate ELL and COO matrices
    int num_nnz_ell = max_row * nrow;

    if (num_nnz_ell <= 0 || num_nnz_coo <= 0) {
      free_ocl(&nnz_coo);
      return false;
    }

    this->AllocateHYB(num_nnz_ell, num_nnz_coo, max_row, nrow, ncol);

    ocl_set_to(num_nnz_ell, (int) -1, this->mat_.ELL.col, this->local_backend_.OCL_command_queue);

    // copy up to num_cols_per_row values of row i into the ELL
    int *nnz_ell = NULL;

    allocate_ocl(nrow, this->local_backend_.OCL_context, &nnz_ell);

    GlobalSize = (nrow / LocalSize + 1) * LocalSize;

    err = ocl_kernel<ValueType>("kernel_hyb_ell_fill_ell",
                                this->local_backend_.OCL_command_queue,
                                LocalSize, GlobalSize,
                                nrow, max_row,
                                cast_mat_csr->mat_.row_offset, cast_mat_csr->mat_.col, cast_mat_csr->mat_.val,
                                this->mat_.ELL.col, this->mat_.ELL.val, nnz_ell);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    // TODO currently performing partial sum on host
    allocate_host(nrow, &hostBuffer);
    ocl_dev2host(nrow, nnz_ell, hostBuffer, this->local_backend_.OCL_command_queue);

    for (int i=1; i<nrow; ++i)
      hostBuffer[i] += hostBuffer[i-1];

    ocl_host2dev(nrow, hostBuffer, nnz_ell, this->local_backend_.OCL_command_queue);

    free_host(&hostBuffer);
    // end TODO

    // copy any remaining values in row i into the COO
    err = ocl_kernel<ValueType>("kernel_hyb_ell_fill_coo",
                                this->local_backend_.OCL_command_queue,
                                LocalSize, GlobalSize,
                                nrow, cast_mat_csr->mat_.row_offset, cast_mat_csr->mat_.col, cast_mat_csr->mat_.val,
                                nnz_coo, nnz_ell, this->mat_.COO.row, this->mat_.COO.col, this->mat_.COO.val);
    CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    free_ocl(&nnz_ell);
    free_ocl(&nnz_coo);

    this->nrow_ = cast_mat_csr->nrow_;
    this->ncol_ = cast_mat_csr->ncol_;
    this->nnz_  = num_nnz_ell + num_nnz_coo;
    this->mat_.ELL.max_row = max_row;
    this->ell_nnz_ = num_nnz_ell;
    this->coo_nnz_ = num_nnz_coo;

    return true;

  }

  return false;

}

template <typename ValueType>
void OCLAcceleratorMatrixHYB<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

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

    // ELL
    if (this->ell_nnz_ > 0) {

      size_t LocalSize  = this->local_backend_.OCL_block_size;
      size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

      err = ocl_kernel<ValueType>("kernel_ell_spmv",
                                  this->local_backend_.OCL_command_queue,
                                  LocalSize, GlobalSize,
                                  this->nrow_, this->ncol_, this->mat_.ELL.max_row, this->mat_.ELL.col, this->mat_.ELL.val,
                                  cast_in->vec_, cast_out->vec_);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    }

    // COO
    if (this->coo_nnz_ > 0) {

      if (typeid(ValueType) != typeid(std::complex<float>) &&
          typeid(ValueType) != typeid(std::complex<double>) &&
          this->coo_nnz_ / this->nrow_ < 2) {

        size_t LocalSize = this->local_backend_.OCL_block_size;
        size_t GlobalSize = (this->coo_nnz_ / LocalSize + 1) * LocalSize;

        err = ocl_kernel<ValueType>("kernel_coo_add_spmv",
                                    this->local_backend_.OCL_command_queue,
                                    LocalSize, GlobalSize,
                                    this->coo_nnz_, this->mat_.COO.row, this->mat_.COO.col, this->mat_.COO.val,
                                    (ValueType) 1, cast_in->vec_, cast_out->vec_);
        CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      } else {

        // If nnz < warpsize, perform squential spmv
        if (this->coo_nnz_ < this->local_backend_.OCL_warp_size) {

          err = ocl_kernel<ValueType>("kernel_coo_spmv_serial",
                                      this->local_backend_.OCL_command_queue,
                                      1, 1,
                                      this->coo_nnz_, this->mat_.COO.row, this->mat_.COO.col, this->mat_.COO.val,
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
          int num_units  = this->coo_nnz_ / this->local_backend_.OCL_warp_size;
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
                                      tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, this->mat_.COO.val,
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
                                      this->coo_nnz_, this->mat_.COO.row, this->mat_.COO.col, this->mat_.COO.val,
                                      (ValueType) 1, cast_in->vec_, cast_out->vec_, tail);
          CHECK_OCL_ERROR(err, __FILE__, __LINE__);

        }

      }

    }

  }

}

template <typename ValueType>
void OCLAcceleratorMatrixHYB<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
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

    // ELL
    if (this->ell_nnz_ > 0) {

      size_t LocalSize  = this->local_backend_.OCL_block_size;
      size_t GlobalSize = (this->nrow_ / LocalSize + 1) * LocalSize;

      err = ocl_kernel<ValueType>("kernel_ell_add_spmv",
                                  this->local_backend_.OCL_command_queue,
                                  LocalSize, GlobalSize,
                                  this->nrow_, this->ncol_, this->mat_.ELL.max_row, this->mat_.ELL.col, this->mat_.ELL.val,
                                  scalar, cast_in->vec_, cast_out->vec_);
      CHECK_OCL_ERROR(err, __FILE__, __LINE__);

    }

    // COO
    if (this->coo_nnz_ > 0) {

      if (typeid(ValueType) != typeid(std::complex<float>) &&
          typeid(ValueType) != typeid(std::complex<double>) &&
          this->coo_nnz_ / this->nrow_ < 2) {

        size_t LocalSize = this->local_backend_.OCL_block_size;
        size_t GlobalSize = (this->coo_nnz_ / LocalSize + 1) * LocalSize;

        err = ocl_kernel<ValueType>("kernel_coo_add_spmv",
                                    this->local_backend_.OCL_command_queue,
                                    LocalSize, GlobalSize,
                                    this->coo_nnz_, this->mat_.COO.row, this->mat_.COO.col, this->mat_.COO.val,
                                    scalar, cast_in->vec_, cast_out->vec_);
        CHECK_OCL_ERROR(err, __FILE__, __LINE__);

      } else {

        // If nnz < warpsize, perform squential spmv
        if (this->coo_nnz_ < this->local_backend_.OCL_warp_size) {

          err = ocl_kernel<ValueType>("kernel_coo_spmv_serial",
                                      this->local_backend_.OCL_command_queue,
                                      1, 1,
                                      this->coo_nnz_, this->mat_.COO.row, this->mat_.COO.col, this->mat_.COO.val,
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
          int num_units  = this->coo_nnz_ / this->local_backend_.OCL_warp_size;
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
                                      tail, interval_size, this->mat_.COO.row, this->mat_.COO.col, this->mat_.COO.val,
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
                                      this->coo_nnz_, this->mat_.COO.row, this->mat_.COO.col, this->mat_.COO.val,
                                      scalar, cast_in->vec_, cast_out->vec_, tail);
          CHECK_OCL_ERROR(err, __FILE__, __LINE__);

        }

      }

    }

  }

}


template class OCLAcceleratorMatrixHYB<double>;
template class OCLAcceleratorMatrixHYB<float>;
#ifdef SUPPORT_COMPLEX
template class OCLAcceleratorMatrixHYB<std::complex<double> >;
template class OCLAcceleratorMatrixHYB<std::complex<float> >;
#endif

}
