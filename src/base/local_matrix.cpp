#include "../utils/def.hpp"
#include "local_matrix.hpp"
#include "local_vector.hpp"
#include "base_vector.hpp"
#include "base_matrix.hpp"
#include "host/host_matrix_csr.hpp"
#include "host/host_matrix_coo.hpp"
#include "host/host_vector.hpp"
#include "backend_manager.hpp"
#include "../utils/log.hpp"
#include "../utils/math_functions.hpp"
#include "../utils/allocate_free.hpp"

#include <algorithm>
#include <sstream>
#include <string.h>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rocalution {

template <typename ValueType>
LocalMatrix<ValueType>::LocalMatrix() {

  LOG_DEBUG(this, "LocalMatrix::LocalMatrix()",
            "default constructor");

  this->object_name_ = "";

  // Create empty matrix on the host
  // CSR is the default format
  this->matrix_host_= new HostMatrixCSR<ValueType>(this->local_backend_);

  this->matrix_accel_= NULL;
  this->matrix_ = this->matrix_host_ ;

}

template <typename ValueType>
LocalMatrix<ValueType>::~LocalMatrix() {

  LOG_DEBUG(this, "LocalMatrix::~LocalMatrix()",
            "default destructor");

  this->Clear();
  delete this->matrix_;

}

template <typename ValueType>
IndexType2 LocalMatrix<ValueType>::GetM(void) const {

  return (IndexType2) this->matrix_->GetM();

}

template <typename ValueType>
IndexType2 LocalMatrix<ValueType>::GetN(void) const {

  return (IndexType2) this->matrix_->GetN();

}

template <typename ValueType>
IndexType2 LocalMatrix<ValueType>::GetNnz(void) const {

  return (IndexType2) this->matrix_->GetNnz();

}

template <typename ValueType>
unsigned int LocalMatrix<ValueType>::GetFormat(void) const {

  return this->matrix_->GetMatFormat();

}

template <typename ValueType>
void LocalMatrix<ValueType>::Clear(void) {

  LOG_DEBUG(this, "LocalMatrix::Clear()",
            "");

  this->matrix_->Clear();

}

template <typename ValueType>
void LocalMatrix<ValueType>::Zeros(void) {

  LOG_DEBUG(this, "LocalMatrix::Zeros()",
            "");

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->Zeros();

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::Zeros() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->Zeros() == false) {
        LOG_INFO("Computation of LocalMatrix::Zeros() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Zeros() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Zeros() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::AllocateCSR(const std::string name, const int nnz, const int nrow, const int ncol) {

  LOG_DEBUG(this, "LocalMatrix::AllocateCSR()",
            "name=" << name << " nnz=" << nnz << " nrow=" << nrow << " ncol=" << ncol);

  assert (nnz >= 0);
  assert (nrow >= 0);
  assert (ncol >= 0);

  this->Clear();
  this->object_name_ = name;
  this->ConvertToCSR();

  if (nnz > 0) {

    assert(nrow > 0);
    assert(ncol > 0);

    Rocalution_Backend_Descriptor backend = this->local_backend_;
    unsigned int mat = this->GetFormat();

    // init host matrix
    if (this->matrix_ == this->matrix_host_) {

      delete this->matrix_host_;
      this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_host_;

    } else {
      // init accel matrix
      assert(this->matrix_ == this->matrix_accel_);

      delete this->matrix_accel_;
      this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_accel_;

    }

    this->matrix_->AllocateCSR(nnz, nrow, ncol);

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::AllocateCOO(const std::string name, const int nnz, const int nrow, const int ncol) {

  LOG_DEBUG(this, "LocalMatrix::AllocateCOO()",
            "name=" << name << " nnz=" << nnz << " nrow=" << nrow << " ncol=" << ncol);

  assert (nnz >= 0);
  assert (nrow >= 0);
  assert (ncol >= 0);

  this->Clear();
  this->object_name_ = name;
  this->ConvertToCOO();

  if (nnz > 0) {

    assert(nrow > 0);
    assert(ncol > 0);

    Rocalution_Backend_Descriptor backend = this->local_backend_;
    unsigned int mat = this->GetFormat();

    // init host matrix
    if (this->matrix_ == this->matrix_host_) {

      delete this->matrix_host_;
      this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(backend,
                                                                        mat);
      this->matrix_ = this->matrix_host_;

    } else {
      // init accel matrix
      assert(this->matrix_ == this->matrix_accel_);

      delete this->matrix_accel_;
      this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_accel_;

    }

    this->matrix_->AllocateCOO(nnz, nrow, ncol);

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::AllocateDIA(const std::string name, const int nnz, const int nrow, const int ncol, const int ndiag) {

  LOG_DEBUG(this, "LocalMatrix::AllocateDIA()",
            "name=" << name << " nnz=" << nnz << " nrow=" << nrow << " ncol=" << ncol << " ndiag=" << ndiag);

  assert (nnz >= 0);
  assert (nrow >= 0);
  assert (ncol >= 0);

  this->Clear();
  this->object_name_ = name;
  this->ConvertToDIA();

  if (nnz > 0) {

    assert(nrow > 0);
    assert(ncol > 0);

    Rocalution_Backend_Descriptor backend = this->local_backend_;
    unsigned int mat = this->GetFormat();

    // init host matrix
    if (this->matrix_ == this->matrix_host_) {

      delete this->matrix_host_;
      this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_host_;

    } else {
      // init accel matrix
      assert(this->matrix_ == this->matrix_accel_);

      delete this->matrix_accel_;
      this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_accel_;

    }

    this->matrix_->AllocateDIA(nnz, nrow, ncol, ndiag);

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::AllocateMCSR(const std::string name, const int nnz, const int nrow, const int ncol) {

  LOG_DEBUG(this, "LocalMatrix::AllocateMCSR()",
            "name=" << name << " nnz=" << nnz << " nrow=" << nrow << " ncol=" << ncol);

  assert (nnz >= 0);
  assert (nrow >= 0);
  assert (ncol >= 0);

  this->Clear();
  this->object_name_ = name;
  this->ConvertToMCSR();

  if (nnz > 0) {

    assert(nrow > 0);
    assert(ncol > 0);

    Rocalution_Backend_Descriptor backend = this->local_backend_;
    unsigned int mat = this->GetFormat();

    // init host matrix
    if (this->matrix_ == this->matrix_host_) {

      delete this->matrix_host_;
      this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_host_;

    } else {
      // init accel matrix
      assert(this->matrix_ == this->matrix_accel_);

      delete this->matrix_accel_;
      this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_accel_;

    }

    this->matrix_->AllocateMCSR(nnz, nrow, ncol);

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::AllocateELL(const std::string name, const int nnz, const int nrow, const int ncol, const int max_row) {

  LOG_DEBUG(this, "LocalMatrix::AllocateELL()",
            "name=" << name << " nnz=" << nnz << " nrow=" << nrow << " ncol=" << ncol << " max_row=" << max_row);

  assert (nnz >= 0);
  assert (nrow >= 0);
  assert (ncol >= 0);

  this->Clear();
  this->object_name_ = name;
  this->ConvertToELL();

  if (nnz > 0) {

    assert(nrow > 0);
    assert(ncol > 0);

    Rocalution_Backend_Descriptor backend = this->local_backend_;
    unsigned int mat = this->GetFormat();

    // init host matrix
    if (this->matrix_ == this->matrix_host_) {

      delete this->matrix_host_;
      this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_host_;

    } else {
      // init accel matrix
      assert(this->matrix_ == this->matrix_accel_);

      delete this->matrix_accel_;
      this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_accel_;

    }

    this->matrix_->AllocateELL(nnz, nrow, ncol, max_row);

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::AllocateHYB(const std::string name, const int ell_nnz, const int coo_nnz, const int ell_max_row,
                                         const int nrow, const int ncol) {

    LOG_DEBUG(this, "LocalMatrix::AllocateHYB()",
              "name=" << name << " ell_nnz=" << ell_nnz << " coo_nnz=" << coo_nnz
              << " nrow=" << nrow << " ncol=" << ncol
              << " ell_max_row=" << ell_max_row);

  assert (ell_nnz >= 0);
  assert (coo_nnz >= 0);
  assert (nrow >= 0);
  assert (ncol >= 0);

  this->Clear();
  this->object_name_ = name;
  this->ConvertToHYB();

  if (ell_nnz + coo_nnz > 0) {

    assert(nrow > 0);
    assert(ncol > 0);

    Rocalution_Backend_Descriptor backend = this->local_backend_;
    unsigned int mat = this->GetFormat();

    // init host matrix
    if (this->matrix_ == this->matrix_host_) {

      delete this->matrix_host_;
      this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_host_;

    } else {
      // init accel matrix
      assert(this->matrix_ == this->matrix_accel_);

      delete this->matrix_accel_;
      this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_accel_;

    }

    this->matrix_->AllocateHYB(ell_nnz, coo_nnz, ell_max_row, nrow, ncol);

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::AllocateDENSE(const std::string name, const int nrow, const int ncol) {

  LOG_DEBUG(this, "LocalMatrix::AllocateDENSE()",
            "name=" << name << " nrow=" << nrow << " ncol=" << ncol);

  assert (nrow >= 0);
  assert (ncol >= 0);

  this->Clear();
  this->object_name_ = name;
  this->ConvertToDENSE();

  if (nrow*ncol > 0) {

    Rocalution_Backend_Descriptor backend = this->local_backend_;
    unsigned int mat = this->GetFormat();

    // init host matrix
    if (this->matrix_ == this->matrix_host_) {

      delete this->matrix_host_;
      this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_host_;

    } else {
      // init accel matrix
      assert(this->matrix_ == this->matrix_accel_);

      delete this->matrix_accel_;
      this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_accel_;

    }

  this->matrix_->AllocateDENSE(nrow, ncol);

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
bool LocalMatrix<ValueType>::Check(void) const {

  LOG_DEBUG(this, "LocalMatrix::Check()",
            "");

  bool check = false;

  if (this->is_accel() == true) {

    LocalMatrix<ValueType> mat_host;
    mat_host.ConvertTo(this->GetFormat());
    mat_host.CopyFrom(*this);

    // Convert to CSR
    mat_host.ConvertToCSR();

    check = mat_host.matrix_->Check();

    if (this->GetFormat() != CSR)
      LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Check() is performed in CSR format");

    LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Check() is performed on the host");

  } else {

    if (this->GetFormat() != CSR) {

      LocalMatrix<ValueType> mat_csr;
      mat_csr.ConvertTo(this->GetFormat());
      mat_csr.CopyFrom(*this);

      // Convert to CSR
      mat_csr.ConvertToCSR();

      check = mat_csr.matrix_->Check();

      LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Check() is performed in CSR format");

    } else {

      check = this->matrix_->Check();

    }

  }

  return check;

}

template <typename ValueType>
void LocalMatrix<ValueType>::SetDataPtrCOO(int **row, int **col, ValueType **val,
                                           std::string name,
                                           const int nnz, const int nrow, const int ncol) {

  LOG_DEBUG(this, "LocalMatrix::SetDataPtrCOO()",
            "name=" << name << " nnz=" << nnz << " nrow=" << nrow << " ncol=" << ncol);

  assert(row != NULL);
  assert(col != NULL);
  assert(val != NULL);
  assert(*row != NULL);
  assert(*col != NULL);
  assert(*val != NULL);
  assert(nnz > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  this->Clear();

  this->object_name_ = name;

  //  this->MoveToHost();
  this->ConvertToCOO();

  this->matrix_->SetDataPtrCOO(row, col, val, nnz, nrow, ncol);

  *row = NULL;
  *col = NULL;
  *val = NULL;

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::LeaveDataPtrCOO(int **row, int **col, ValueType **val) {

  LOG_DEBUG(this, "LocalMatrix::LeaveDataPtrCOO()",
            "");

  assert(*row == NULL);
  assert(*col == NULL);
  assert(*val == NULL);
  assert(this->GetM() > 0);
  assert(this->GetN() > 0);
  assert(this->GetNnz() > 0);

#ifdef DEBUG_MODE
  this->Check();
#endif

  //  this->MoveToHost();
  this->ConvertToCOO();

  this->matrix_->LeaveDataPtrCOO(row, col, val);

}

template <typename ValueType>
void LocalMatrix<ValueType>::SetDataPtrCSR(int **row_offset, int **col, ValueType **val,
                                           std::string name,
                                           const int nnz, const int nrow, const int ncol) {

  LOG_DEBUG(this, "LocalMatrix::SetDataPtrCSR()",
            "name=" << name << " nnz=" << nnz << " nrow=" << nrow << " ncol=" << ncol);

  assert(row_offset != NULL);
  assert(col != NULL);
  assert(val != NULL);
  assert(*row_offset != NULL);
  assert(*col != NULL);
  assert(*val != NULL);
  assert(nnz > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  this->Clear();

  this->object_name_ = name;

  //  this->MoveToHost();
  this->ConvertToCSR();

  this->matrix_->SetDataPtrCSR(row_offset, col, val, nnz, nrow, ncol);

  *row_offset = NULL ;
  *col = NULL;
  *val = NULL;

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::LeaveDataPtrCSR(int **row_offset, int **col, ValueType **val) {

  LOG_DEBUG(this, "LocalMatrix::LeaveDataPtrCSR()",
            "");

  assert(*row_offset == NULL);
  assert(*col == NULL);
  assert(*val == NULL);
  assert(this->GetM() > 0);
  assert(this->GetN() > 0);
  assert(this->GetNnz() > 0);

#ifdef DEBUG_MODE
  this->Check();
#endif

  //  this->MoveToHost();
  this->ConvertToCSR();

  this->matrix_->LeaveDataPtrCSR(row_offset, col, val);

}

template <typename ValueType>
void LocalMatrix<ValueType>::SetDataPtrMCSR(int **row_offset, int **col, ValueType **val,
                                            std::string name,
                                            const int nnz, const int nrow, const int ncol) {

  LOG_DEBUG(this, "LocalMatrix::SetDataPtrMCSR()",
            "name=" << name << " nnz=" << nnz << " nrow=" << nrow << " ncol=" << ncol);

  assert(row_offset != NULL);
  assert(col != NULL);
  assert(val != NULL);
  assert(*row_offset != NULL);
  assert(*col != NULL);
  assert(*val != NULL);
  assert(nnz > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  this->Clear();

  this->object_name_ = name;

  //  this->MoveToHost();
  this->ConvertToMCSR();

  this->matrix_->SetDataPtrMCSR(row_offset, col, val, nnz, nrow, ncol);

  *row_offset = NULL ;
  *col = NULL;
  *val = NULL;

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::LeaveDataPtrMCSR(int **row_offset, int **col, ValueType **val) {

  LOG_DEBUG(this, "LocalMatrix::LeaveDataPtrMCSR()",
            "");

  assert(*row_offset == NULL);
  assert(*col == NULL);
  assert(*val == NULL);
  assert(this->GetM() > 0);
  assert(this->GetN() > 0);
  assert(this->GetNnz() > 0);

#ifdef DEBUG_MODE
  this->Check();
#endif

  //  this->MoveToHost();
  this->ConvertToMCSR();

  this->matrix_->LeaveDataPtrMCSR(row_offset, col, val);

}

template <typename ValueType>
void LocalMatrix<ValueType>::SetDataPtrELL(int **col, ValueType **val,
                                           std::string name,
                                           const int nnz, const int nrow, const int ncol, const int max_row) {

  LOG_DEBUG(this, "LocalMatrix::SetDataPtrELL()",
            "name=" << name << " nnz=" << nnz << " nrow=" << nrow << " ncol=" << ncol << " max_row=" << max_row);

  assert(col != NULL);
  assert(val != NULL);
  assert(*col != NULL);
  assert(*val != NULL);
  assert(nnz > 0);
  assert(nrow > 0);
  assert(max_row > 0);
  assert(max_row*nrow == nnz);

  this->Clear();

  this->object_name_ = name;

  //  this->MoveToHost();
  this->ConvertToELL();

  this->matrix_->SetDataPtrELL(col, val, nnz, nrow, ncol, max_row);

  *col = NULL;
  *val = NULL;

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::LeaveDataPtrELL(int **col, ValueType **val, int &max_row) {

  LOG_DEBUG(this, "LocalMatrix::LeaveDataPtrELL()",
            "");

  assert(*col == NULL);
  assert(*val == NULL);
  assert(this->GetM() > 0);
  assert(this->GetN() > 0);
  assert(this->GetNnz() > 0);

#ifdef DEBUG_MODE
  this->Check();
#endif

  //  this->MoveToHost();
  this->ConvertToELL();

  this->matrix_->LeaveDataPtrELL(col, val, max_row);

}

template <typename ValueType>
void LocalMatrix<ValueType>::SetDataPtrDIA(int **offset, ValueType **val,
                                           std::string name,
                                           const int nnz, const int nrow, const int ncol, const int num_diag) {

  LOG_DEBUG(this, "LocalMatrix::SetDataPtrDIA()",
            "name=" << name << " nnz=" << nnz << " nrow=" << nrow << " ncol=" << ncol << " num_diag=" << num_diag);

  assert(offset != NULL);
  assert(val != NULL);
  assert(*offset != NULL);
  assert(*val != NULL);
  assert(nnz > 0);
  assert(nrow > 0);
  assert(num_diag > 0);

  if (nrow < ncol) {
    assert(nnz == ncol * num_diag);
  } else {
    assert(nnz == nrow * num_diag);
  }

  this->Clear();

  this->object_name_ = name;

  //  this->MoveToHost();
  this->ConvertToDIA();

  this->matrix_->SetDataPtrDIA(offset, val, nnz, nrow, ncol, num_diag);

  *offset = NULL;
  *val = NULL;

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::LeaveDataPtrDIA(int **offset, ValueType **val, int &num_diag) {

  LOG_DEBUG(this, "LocalMatrix::LeaveDataPtrDIA()",
            "");

  assert(*offset == NULL);
  assert(*val == NULL);
  assert(this->GetM() > 0);
  assert(this->GetN() > 0);
  assert(this->GetNnz() > 0);

#ifdef DEBUG_MODE
  this->Check();
#endif

  //  this->MoveToHost();
  this->ConvertToDIA();

  this->matrix_->LeaveDataPtrDIA(offset, val, num_diag);

}

template <typename ValueType>
void LocalMatrix<ValueType>::SetDataPtrDENSE(ValueType **val, std::string name,
                                             const int nrow, const int ncol) {

  LOG_DEBUG(this, "LocalMatrix::SetDataPtrDENSE()",
            "name=" << name << " nrow=" << nrow << " ncol=" << ncol);

  assert(val != NULL);
  assert(*val != NULL);
  assert(nrow > 0);
  assert(ncol > 0);

  this->Clear();

  this->object_name_ = name;

  //  this->MoveToHost();
  this->ConvertToDENSE();

  this->matrix_->SetDataPtrDENSE(val, nrow, ncol);

  *val = NULL;

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::LeaveDataPtrDENSE(ValueType **val) {

  LOG_DEBUG(this, "LocalMatrix::LeaveDataPtrDENSE()",
            "");

  assert(*val == NULL);
  assert(this->GetM() > 0);
  assert(this->GetN() > 0);
  assert(this->GetNnz() > 0);

#ifdef DEBUG_MODE
  this->Check();
#endif

  //  this->MoveToHost();
  this->ConvertToDENSE();

  this->matrix_->LeaveDataPtrDENSE(val);

}

template <typename ValueType>
void LocalMatrix<ValueType>::CopyFromCSR(const int *row_offsets, const int *col, const ValueType *val) {

  LOG_DEBUG(this, "LocalMatrix::CopyFromCSR()",
            "");

  assert(row_offsets != NULL);
  assert(col != NULL);
  assert(val != NULL);
  assert(this->GetFormat() == CSR);

  if (this->GetNnz() > 0)
    this->matrix_->CopyFromCSR(row_offsets, col, val);

  this->object_name_ = "Imported from CSR matrix";

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::CopyToCSR(int *row_offsets, int *col, ValueType *val) const {

  LOG_DEBUG(this, "LocalMatrix::CopyToCSR()",
            "");

  assert(row_offsets != NULL);
  assert(col != NULL);
  assert(val != NULL);
  assert(this->GetFormat() == CSR);

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0)
    this->matrix_->CopyToCSR(row_offsets, col, val);

}

template <typename ValueType>
void LocalMatrix<ValueType>::CopyFromCOO(const int *row, const int *col, const ValueType *val) {

  LOG_DEBUG(this, "LocalMatrix::CopyFromCOO()",
            "");

  assert(row != NULL);
  assert(col != NULL);
  assert(val != NULL);
  assert(this->GetFormat() == COO);

  if (this->GetNnz() > 0)
    this->matrix_->CopyFromCOO(row, col, val);

  this->object_name_ = "Imported from COO matrix";

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::CopyToCOO(int *row, int *col, ValueType *val) const {

  LOG_DEBUG(this, "LocalMatrix::CopyToCOO()",
            "");

  assert(row != NULL);
  assert(col != NULL);
  assert(val != NULL);
  assert(this->GetFormat() == COO);

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0)
    this->matrix_->CopyToCOO(row, col, val);

}

template <typename ValueType>
void LocalMatrix<ValueType>::CopyFromHostCSR(const int *row_offset, const int *col, const ValueType *val,
                                             const std::string name,
                                             const int nnz, const int nrow, const int ncol) {

  LOG_DEBUG(this, "LocalMatrix::CopyFromHostCSR()",
            "name=" << name << " nnz=" << nnz << " nrow=" << nrow << " ncol=" << ncol);

  assert(nnz >= 0);
  assert(nrow >= 0);
  assert(ncol >= 0);
  assert(row_offset != NULL);
  assert(col != NULL);
  assert(val != NULL);

  this->Clear();
  this->object_name_ = name;
  this->ConvertToCSR();

  if (nnz > 0) {

    assert(nrow > 0);
    assert(ncol > 0);

    Rocalution_Backend_Descriptor backend = this->local_backend_;
    unsigned int mat = this->GetFormat();

    // init host matrix
    if (this->matrix_ == this->matrix_host_) {

      delete this->matrix_host_;
      this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_host_;

    } else {
      // init accel matrix
      assert(this->matrix_ == this->matrix_accel_);

      delete this->matrix_accel_;
      this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(backend, mat);
      this->matrix_ = this->matrix_accel_;

    }

    this->matrix_->CopyFromHostCSR(row_offset, col, val, nnz, nrow, ncol);

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::ReadFileMTX(const std::string filename) {

  LOG_DEBUG(this, "LocalMatrix::ReadFileMTX()",
            filename);

  this->Clear();

  bool err = this->matrix_->ReadFileMTX(filename);

  if ((err == false) && (this->is_host() == true) && (this->GetFormat() == COO)) {
    LOG_INFO("Computation of LocalMatrix::ReadFileMTX() failed");
    this->Info();
    FATAL_ERROR(__FILE__, __LINE__);
  }

  if (err == false) {

    // Move to host
    bool is_accel = this->is_accel();
    this->MoveToHost();

    // Convert to COO
    unsigned int format = this->GetFormat();
    this->ConvertToCOO();

    if (this->matrix_->ReadFileMTX(filename) == false) {
      LOG_INFO("Computation of LocalMatrix::ReadFileMTX() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    this->ConvertTo(format);
    this->Check();

    if (is_accel == true)
      this->MoveToAccelerator();

  }

  this->object_name_ = filename;

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::WriteFileMTX(const std::string filename) const {

  LOG_DEBUG(this, "LocalMatrix::WriteFileMTX()",
            filename);

#ifdef DEBUG_MODE
  this->Check();
#endif

  bool err = this->matrix_->WriteFileMTX(filename);

  if ((err == false) && (this->is_host() == true) && (this->GetFormat() == COO)) {
    LOG_INFO("Computation of LocalMatrix::WriteFileMTX() failed");
    this->Info();
    FATAL_ERROR(__FILE__, __LINE__);
  }

  if (err == false) {

    // Move to host
    LocalMatrix<ValueType> mat_host;
    mat_host.ConvertTo(this->GetFormat());
    mat_host.CopyFrom(*this);

    // Convert to COO
    mat_host.ConvertToCOO();

    if (mat_host.matrix_->WriteFileMTX(filename) == false) {
      LOG_INFO("Computation of LocalMatrix::WriteFileMTX() failed");
      mat_host.Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::ReadFileCSR(const std::string filename) {

  LOG_DEBUG(this, "LocalMatrix::ReadFileCSR()",
            filename);

  this->Clear();

  bool err = this->matrix_->ReadFileCSR(filename);

  if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
    LOG_INFO("Computation of LocalMatrix::ReadFileCSR() failed");
    this->Info();
    FATAL_ERROR(__FILE__, __LINE__);
  }

  if (err == false) {

    // Move to host
    bool is_accel = this->is_accel();
    this->MoveToHost();

    // Convert to CSR
    unsigned int format = this->GetFormat();
    this->ConvertToCSR();

    if (this->matrix_->ReadFileCSR(filename) == false) {
      LOG_INFO("Computation of LocalMatrix::ReadFileCSR() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    this->Check();

    this->ConvertTo(format);

    if (is_accel == true)
      this->MoveToAccelerator();

  }

  this->object_name_ = filename;

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::WriteFileCSR(const std::string filename) const {

  LOG_DEBUG(this, "LocalMatrix::WriteFileCSR()",
            filename);

#ifdef DEBUG_MODE
  this->Check();
#endif

  bool err = this->matrix_->WriteFileCSR(filename);

  if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
    LOG_INFO("Computation of LocalMatrix::WriteFileCSR() failed");
    this->Info();
    FATAL_ERROR(__FILE__, __LINE__);
  }

  if (err == false) {

    // Move to host
    LocalMatrix<ValueType> mat_host;
    mat_host.ConvertTo(this->GetFormat());
    mat_host.CopyFrom(*this);

    // Convert to CSR
    mat_host.ConvertToCSR();

    if (mat_host.matrix_->WriteFileCSR(filename) == false) {
      LOG_INFO("Computation of LocalMatrix::WriteFileCSR() failed");
      mat_host.Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::CopyFrom(const LocalMatrix<ValueType> &src) {

  LOG_DEBUG(this, "LocalMatrix::CopyFrom()",
            "");

  assert(&src != NULL);
  assert(this != &src);

  this->matrix_->CopyFrom(*src.matrix_);

}

template <typename ValueType>
void LocalMatrix<ValueType>::CopyFromAsync(const LocalMatrix<ValueType> &src) {

  LOG_DEBUG(this, "LocalMatrix::CopyFromAsync()",
            "");

  assert(&src != NULL);
  assert(this->asyncf == false);
  assert(this != &src);

  this->matrix_->CopyFromAsync(*src.matrix_);

  this->asyncf = true;

}

template <typename ValueType>
void LocalMatrix<ValueType>::CloneFrom(const LocalMatrix<ValueType> &src) {

  LOG_DEBUG(this, "LocalMatrix::CloneFrom()",
            "");

  assert(&src != NULL);
  assert(this != &src);

#ifdef DEBUG_MODE
  src.Check();
#endif

  this->object_name_   = "Cloned from (";
  this->object_name_  += src.object_name_ + ")";
  this->local_backend_ = src.local_backend_; 

  Rocalution_Backend_Descriptor backend = this->local_backend_;

  if (src.matrix_ == src.matrix_host_) {

    // host
    delete this->matrix_host_;
    this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(backend, src.GetFormat());
    this->matrix_ = this->matrix_host_;

  } else {

    // accel
    delete this->matrix_accel_;

    this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(backend, src.GetFormat());
    this->matrix_ = this->matrix_accel_;

  }

  this->matrix_->CopyFrom(*src.matrix_);

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::UpdateValuesCSR(ValueType *val) {

  LOG_DEBUG(this, "LocalMatrix::UpdateValues()",
            "");

  assert(val != NULL);
  assert(this->GetNnz() > 0);
  assert(this->GetM() > 0);
  assert(this->GetN() > 0);
  assert(this->GetFormat() == CSR);

#ifdef DEBUG_MODE
  this->Check();
#endif

  int *mat_row_offset = NULL;
  int *mat_col = NULL;
  ValueType *mat_val = NULL;

  int nrow = this->get_local_nrow();
  int ncol = this->get_local_ncol();
  int nnz  = this->get_local_nnz();

  // Extract matrix pointers
  this->matrix_->LeaveDataPtrCSR(&mat_row_offset, &mat_col, &mat_val);

  // Dummy vector to follow the correct backend
  LocalVector<ValueType> vec;
  vec.MoveToHost();

  vec.SetDataPtr(&val, "dummy", nnz);

  vec.CloneBackend(*this);

  vec.LeaveDataPtr(&mat_val);

  // Set matrix pointers
  this->matrix_->SetDataPtrCSR(&mat_row_offset, &mat_col, &mat_val, nnz, nrow, ncol);

  mat_row_offset = NULL;
  mat_col = NULL;
  mat_val = NULL;
  val = NULL;

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
bool LocalMatrix<ValueType>::is_host(void) const {

  return (this->matrix_ == this->matrix_host_);

}

template <typename ValueType>
bool LocalMatrix<ValueType>::is_accel(void) const {

  return (this->matrix_ == this->matrix_accel_);

}

template <typename ValueType>
void LocalMatrix<ValueType>::Info(void) const {

  std::string current_backend_name;

  if (this->matrix_ == this->matrix_host_) {
    current_backend_name = _rocalution_host_name[0];
  } else {
    assert(this->matrix_ == this->matrix_accel_);
    current_backend_name = _rocalution_backend_name[this->local_backend_.backend];
  }

  LOG_INFO("LocalMatrix"
           << " name=" << this->object_name_ << ";"
           << " rows=" << this->GetM() << ";"
           << " cols=" << this->GetN() << ";"
           << " nnz=" << this->GetNnz() << ";"
           << " prec=" << 8*sizeof(ValueType) << "bit;"
           << " format=" << _matrix_format_names[this->GetFormat()] << ";"
           << " host backend={" << _rocalution_host_name[0] << "};"
           << " accelerator backend={" << _rocalution_backend_name[this->local_backend_.backend] << "};"
           << " current=" << current_backend_name);

  // this->matrix_->Info();

}

template <typename ValueType>
void LocalMatrix<ValueType>::MoveToAccelerator(void) {

  LOG_DEBUG(this, "LocalMatrix::MoveToAccelerator()",
            "");

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (_rocalution_available_accelerator() == false)
    LOG_VERBOSE_INFO(4,"*** info: LocalMatrix::MoveToAccelerator() no accelerator available - doing nothing");

  if ( (_rocalution_available_accelerator()) && (this->matrix_ == this->matrix_host_)) {

    this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(this->local_backend_, this->GetFormat());
    this->matrix_accel_->CopyFrom(*this->matrix_host_);

    this->matrix_ = this->matrix_accel_;
    delete this->matrix_host_;
    this->matrix_host_ = NULL;

    LOG_VERBOSE_INFO(4, "*** info: LocalMatrix::MoveToAccelerator() host to accelerator transfer");

  }

  // if on accelerator - do nothing

}

template <typename ValueType>
void LocalMatrix<ValueType>::MoveToHost(void) {

  LOG_DEBUG(this, "LocalMatrix::MoveToHost()",
            "");

  if ( (_rocalution_available_accelerator()) && (this->matrix_ == this->matrix_accel_)) {

    this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(this->local_backend_, this->GetFormat());
    this->matrix_host_->CopyFrom(*this->matrix_accel_);

    this->matrix_ = this->matrix_host_;
    delete this->matrix_accel_;
    this->matrix_accel_ = NULL;

    LOG_VERBOSE_INFO(4, "*** info: LocalMatrix::MoveToHost() accelerator to host transfer");

  }

  // if on host - do nothing

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::MoveToAcceleratorAsync(void) {

  LOG_DEBUG(this, "LocalMatrix::MoveToAcceleratorAsync()",
            "");

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (_rocalution_available_accelerator() == false)
    LOG_VERBOSE_INFO(4,"*** info: LocalMatrix::MoveToAcceleratorAsync() no accelerator available - doing nothing");

  if ( (_rocalution_available_accelerator()) && (this->matrix_ == this->matrix_host_)) {

    this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(this->local_backend_, this->GetFormat());
    this->matrix_accel_->CopyFromAsync(*this->matrix_host_);
    this->asyncf = true;

    LOG_VERBOSE_INFO(4, "*** info: LocalMatrix::MoveToAcceleratorAsync() host to accelerator transfer (started)");

  }

  // if on accelerator - do nothing

}

template <typename ValueType>
void LocalMatrix<ValueType>::MoveToHostAsync(void) {

  LOG_DEBUG(this, "LocalMatrix::MoveToHostAsync()",
            "");

  if ( (_rocalution_available_accelerator()) && (this->matrix_ == this->matrix_accel_)) {

    this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(this->local_backend_, this->GetFormat());
    this->matrix_host_->CopyFromAsync(*this->matrix_accel_);
    this->asyncf = true;

    LOG_VERBOSE_INFO(4, "*** info: LocalMatrix::MoveToHostAsync() accelerator to host transfer (started)");

  }

  // if on host - do nothing

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::Sync(void) {

  LOG_DEBUG(this, "LocalMatrix::Sync()",
            "");

  // check for active async transfer
  if (this->asyncf == true) {

    // The Move*Async function is active
    if ( (this->matrix_accel_ != NULL) &&
         (this->matrix_host_  != NULL)) {

      // MoveToHostAsync();
      if ( (_rocalution_available_accelerator() == true) && (this->matrix_ == this->matrix_accel_)) {

        _rocalution_sync();

        this->matrix_ = this->matrix_host_;
        delete this->matrix_accel_;
        this->matrix_accel_ = NULL;

        LOG_VERBOSE_INFO(4, "*** info: LocalMatrix::MoveToHostAsync() accelerator to host transfer (synced)");

      }

      // MoveToAcceleratorAsync();
      if ( (_rocalution_available_accelerator() == true) && (this->matrix_ == this->matrix_host_)) {

        _rocalution_sync();

        this->matrix_ = this->matrix_accel_;
        delete this->matrix_host_;
        this->matrix_host_ = NULL;
        LOG_VERBOSE_INFO(4, "*** info: LocalMatrix::MoveToAcceleratorAsync() host to accelerator transfer (synced)");

      }

    } else {

      // The Copy*Async function is active
      _rocalution_sync();
      LOG_VERBOSE_INFO(4, "*** info: LocalMatrix::Copy*Async() transfer (synced)");

    }

  }

  this->asyncf = false;

}

template <typename ValueType>
void LocalMatrix<ValueType>::ConvertToCSR(void) {

  this->ConvertTo(CSR);

}

template <typename ValueType>
void LocalMatrix<ValueType>::ConvertToMCSR(void) {

  this->ConvertTo(MCSR);

}

template <typename ValueType>
void LocalMatrix<ValueType>::ConvertToBCSR(void) {

  this->ConvertTo(BCSR);

}

template <typename ValueType>
void LocalMatrix<ValueType>::ConvertToCOO(void) {

  this->ConvertTo(COO);

}

template <typename ValueType>
void LocalMatrix<ValueType>::ConvertToELL(void) {

  this->ConvertTo(ELL);

}

template <typename ValueType>
void LocalMatrix<ValueType>::ConvertToDIA(void) {

  this->ConvertTo(DIA);

}

template <typename ValueType>
void LocalMatrix<ValueType>::ConvertToHYB(void) {

  this->ConvertTo(HYB);

}

template <typename ValueType>
void LocalMatrix<ValueType>::ConvertToDENSE(void) {

  this->ConvertTo(DENSE);

}

template <typename ValueType>
void LocalMatrix<ValueType>::ConvertTo(const unsigned int matrix_format) {

  LOG_DEBUG(this, "LocalMatrix::ConvertTo()",
            "");

  assert((matrix_format == DENSE) ||
         (matrix_format == CSR) ||
         (matrix_format == MCSR) ||
         (matrix_format == BCSR) ||
         (matrix_format == COO) ||
         (matrix_format == DIA) ||
         (matrix_format == ELL) ||
         (matrix_format == HYB));

  LOG_VERBOSE_INFO(5, "Converting " << _matrix_format_names[matrix_format] << " <- " << _matrix_format_names[this->GetFormat()] );

  if (this->GetFormat() != matrix_format) {

    if ((this->GetFormat() != CSR) && (matrix_format != CSR))
      this->ConvertToCSR();

    // CPU matrix
    if (this->matrix_ == this->matrix_host_) {
      assert(this->matrix_host_ != NULL);

      HostMatrix<ValueType> *new_mat;
      new_mat = _rocalution_init_base_host_matrix<ValueType>(this->local_backend_, matrix_format);
      assert(new_mat != NULL);

      // If conversion fails, try CSR before we give up
      if (new_mat->ConvertFrom(*this->matrix_host_) == false) {
        LOG_VERBOSE_INFO(2, "*** warning: Matrix conversion to " << _matrix_format_names[matrix_format] << " failed, falling back to CSR format");
        delete new_mat;
        new_mat = _rocalution_init_base_host_matrix<ValueType>(this->local_backend_, CSR);
        assert (new_mat != NULL);

        // If CSR conversion fails too, exit with error
        if (new_mat->ConvertFrom(*this->matrix_host_) == false) {
          LOG_INFO("Unsupported (on host) convertion type to CSR");
          this->Info();
          FATAL_ERROR(__FILE__, __LINE__);
        }

      }

      delete this->matrix_host_;

      this->matrix_host_ = new_mat;
      this->matrix_ = this->matrix_host_;

    } else {

      // Accelerator Matrix
      assert(this->matrix_accel_ != NULL);

      AcceleratorMatrix<ValueType> *new_mat;
      new_mat = _rocalution_init_base_backend_matrix<ValueType>(this->local_backend_, matrix_format);
      assert(new_mat != NULL);

      if (new_mat->ConvertFrom(*this->matrix_accel_) == false) {

        delete new_mat;

        this->MoveToHost();
        this->ConvertTo(matrix_format);
        this->MoveToAccelerator();

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ConvertTo() is performed on the host");

      } else {

        delete this->matrix_accel_;

        this->matrix_accel_ = new_mat;
        this->matrix_ = this->matrix_accel_;

     }

    }

    assert(this->GetFormat() == matrix_format || this->GetFormat() == CSR);

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::Apply(const LocalVector<ValueType> &in, LocalVector<ValueType> *out) const {

  LOG_DEBUG(this, "LocalMatrix::Apply()",
            "");

  assert(&in != NULL);
  assert(out != NULL);

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    assert(in.get_size() == this->GetN());
    assert(out->get_size() == this->GetM());

    assert( ( (this->matrix_ == this->matrix_host_)  && (in.vector_ == in.vector_host_) && (out->vector_ == out->vector_host_)) ||
            ( (this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_) && (out->vector_ == out->vector_accel_)) );

    this->matrix_->Apply(*in.vector_, out->vector_);

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::ApplyAdd(const LocalVector<ValueType> &in, const ValueType scalar,
                                      LocalVector<ValueType> *out) const {

  LOG_DEBUG(this, "LocalMatrix::ApplyAdd()",
            "");

  assert(&in != NULL);
  assert(out != NULL);

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    assert(in.get_size() == this->GetN());
    assert(out->get_size() == this->GetM());

    assert( ( (this->matrix_ == this->matrix_host_)  && (in.vector_ == in.vector_host_) && (out->vector_ == out->vector_host_)) ||
            ( (this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_) && (out->vector_ == out->vector_accel_)) );

    this->matrix_->ApplyAdd(*in.vector_, scalar, out->vector_);

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::ExtractDiagonal(LocalVector<ValueType> *vec_diag) const {

  LOG_DEBUG(this, "LocalMatrix::ExtractDiagonal()",
            "");

  assert(vec_diag != NULL);

  assert( ( (this->matrix_ == this->matrix_host_)  && (vec_diag->vector_ == vec_diag->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (vec_diag->vector_ == vec_diag->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    std::string vec_diag_name = "Diagonal elements of " + this->object_name_;
    vec_diag->Allocate(vec_diag_name, std::min(this->get_local_nrow(), this->get_local_ncol()));

    bool err = this->matrix_->ExtractDiagonal(vec_diag->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ExtractDiagonal() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      vec_diag->MoveToHost();

      mat_host.ConvertToCSR();

      if (mat_host.matrix_->ExtractDiagonal(vec_diag->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::ExtractDiagonal() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractDiagonal() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractDiagonal() is performed on the host");

        vec_diag->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::ExtractInverseDiagonal(LocalVector<ValueType> *vec_inv_diag) const {

  LOG_DEBUG(this, "LocalMatrix::ExtractInverseDiagonal()",
            "");

  assert(vec_inv_diag != NULL);

  assert( ( (this->matrix_ == this->matrix_host_)  && (vec_inv_diag->vector_ == vec_inv_diag->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (vec_inv_diag->vector_ == vec_inv_diag->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    std::string vec_inv_diag_name = "Inverse of the diagonal elements of " + this->object_name_;
    vec_inv_diag->Allocate(vec_inv_diag_name, std::min(this->get_local_nrow(), this->get_local_ncol()));

    bool err = this->matrix_->ExtractInverseDiagonal(vec_inv_diag->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ExtractInverseDiagonal() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      vec_inv_diag->MoveToHost();

      mat_host.ConvertToCSR();

      if (mat_host.matrix_->ExtractInverseDiagonal(vec_inv_diag->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::ExtractInverseDiagonal() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractInverseDiagonal() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractInverseDiagonal() is performed on the host");

        vec_inv_diag->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::ExtractSubMatrix(const int row_offset,
                                              const int col_offset,
                                              const int row_size,
                                              const int col_size,
                                              LocalMatrix<ValueType> *mat) const {

  LOG_DEBUG(this, "LocalMatrix::ExtractSubMatrix()",
            "row_offset=" <<  row_offset << " col_offset=" << col_offset
            << " row_size=" << row_size << " col_size=" << col_size);

  assert(this != mat);
  assert(mat != NULL);
  assert(row_size > 0);
  assert(col_size > 0);
  assert((IndexType2) row_offset <= this->GetM());
  assert((IndexType2) col_offset <= this->GetN());

  assert( ( (this->matrix_ == this->matrix_host_)  && (mat->matrix_ == mat->matrix_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (mat->matrix_ == mat->matrix_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  mat->Clear();

  if (this->GetNnz() > 0) {

    // Submatrix should be same format as full matrix
    mat->ConvertTo(this->GetFormat());

    bool err = false;

    // if the sub matrix has only 1 row
    // it is computed on the host
    if ((this->is_host() == true) || (row_size > 1))
      err = this->matrix_->ExtractSubMatrix(row_offset, col_offset,
                                            row_size,   col_size,
                                            mat->matrix_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ExtractSubMatrix() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      mat->MoveToHost();

      mat_host.ConvertToCSR();
      mat->ConvertToCSR();

      if (mat_host.matrix_->ExtractSubMatrix(row_offset, col_offset, row_size, col_size, mat->matrix_) == false) {
        LOG_INFO("Computation of LocalMatrix::ExtractSubMatrix() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR) {

        if (row_size > 1)
          LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractSubMatrix() is performed in CSR format");

        mat->ConvertTo(this->GetFormat());

      }

      if (this->is_accel() == true) {

        if (row_size > 1)
          LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractSubMatrix() is performed on the host");

        mat->MoveToAccelerator();

      }

      if (row_size <= 1)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractSubMatrix() is performed on the host due to size = 1");

    }

    std::string mat_name = "Submatrix of " + this->object_name_ + " "
      + "[" + static_cast<std::ostringstream*>( &(std::ostringstream() << row_offset) )->str()
      + "," + static_cast<std::ostringstream*>( &(std::ostringstream() << col_offset) )->str() + "]-"
      + "[" + static_cast<std::ostringstream*>( &(std::ostringstream() << row_offset+row_size-1) )->str()
      + "," + static_cast<std::ostringstream*>( &(std::ostringstream() << col_offset+row_size-1) )->str() + "]";

    mat->object_name_ = mat_name;

  }

#ifdef DEBUG_MODE
  mat->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::ExtractSubMatrices(const int row_num_blocks,
                                                const int col_num_blocks,
                                                const int *row_offset,
                                                const int *col_offset,
                                                LocalMatrix<ValueType> ***mat) const {

  LOG_DEBUG(this, "LocalMatrix::ExtractSubMatrices()",
            "row_num_blocks=" << row_num_blocks << " col_num_blocks=" << col_num_blocks);

  assert(row_num_blocks > 0);
  assert(col_num_blocks > 0);
  assert(row_offset != NULL);
  assert(col_offset != NULL);
  assert(mat != NULL);
  assert(*mat != NULL);

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    // implementation via ExtractSubMatrix() calls
    //TODO OMP
    //#pragma omp parallel for schedule(dynamic,1) collapse(2)
    for (int i=0; i<row_num_blocks; ++i)
      for (int j=0; j<col_num_blocks; ++j)
        this->ExtractSubMatrix(row_offset[i],
                               col_offset[j],
                               row_offset[i+1] - row_offset[i],
                               col_offset[j+1] - col_offset[j],
                               mat[i][j]);

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::ExtractU(LocalMatrix<ValueType> *U, const bool diag) const {

  LOG_DEBUG(this, "LocalMatrix::ExtractU()",
            diag);

  assert(U != NULL);
  assert(U != this);

  assert( ( (this->matrix_ == this->matrix_host_)  && (U->matrix_ == U->matrix_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (U->matrix_ == U->matrix_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = false;

    if (diag == true) {
      err = this->matrix_->ExtractUDiagonal(U->matrix_);
    } else {
      err = this->matrix_->ExtractU(U->matrix_);
    }

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ExtractU() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      U->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (diag == true) {
        err = mat_host.matrix_->ExtractUDiagonal(U->matrix_);
      } else {
        err = mat_host.matrix_->ExtractU(U->matrix_);
      }

      if (err == false) {
        LOG_INFO("Computation of LocalMatrix::ExtractU() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractU() is performed in CSR format");

        U->ConvertTo(this->GetFormat());

      }

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractU() is performed on the host");

        U->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  U->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::ExtractL(LocalMatrix<ValueType> *L, const bool diag) const {

  LOG_DEBUG(this, "LocalMatrix::ExtractL()",
            diag);

  assert(L != NULL);
  assert(L != this);

  assert( ( (this->matrix_ == this->matrix_host_)  && (L->matrix_ == L->matrix_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (L->matrix_ == L->matrix_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = false;

    if (diag == true) {
      err = this->matrix_->ExtractLDiagonal(L->matrix_);
    } else {
      err = this->matrix_->ExtractL(L->matrix_);
    }

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ExtractL() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      L->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (diag == true) {
        err = mat_host.matrix_->ExtractLDiagonal(L->matrix_);
      } else {
        err = mat_host.matrix_->ExtractL(L->matrix_);
      }

      if (err == false) {
        LOG_INFO("Computation of LocalMatrix::ExtractL() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractL() is performed in CSR format");

        L->ConvertTo(this->GetFormat());

      }

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractL() is performed on the host");

        L->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  L->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::LUAnalyse(void) {

  LOG_DEBUG(this, "LocalMatrix::LUAnalyse()",
            "");

  if (this->GetNnz() > 0) {
    this->matrix_->LUAnalyse();
  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::LUAnalyseClear(void) {

  LOG_DEBUG(this, "LocalMatrix::LUAnalyseClear()",
            "");

  if (this->GetNnz() > 0) {
    this->matrix_->LUAnalyseClear();
  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::LUSolve(const LocalVector<ValueType> &in, LocalVector<ValueType> *out) const {

  LOG_DEBUG(this, "LocalMatrix::LUSolve()",
            "");

  assert(&in != NULL);
  assert(out != NULL);
  assert(in.get_size() == this->GetN());
  assert(out->get_size() == this->GetM());

  assert( ( (this->matrix_ == this->matrix_host_)  && (in.vector_ == in.vector_host_)  && (out->vector_ == out->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_) && (out->vector_ == out->vector_accel_)) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->LUSolve(*in.vector_, out->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::LUSolve() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      LocalVector<ValueType> vec_host;
      vec_host.CopyFrom(in);

      out->MoveToHost();

      // Try again
      err = mat_host.matrix_->LUSolve(*vec_host.vector_, out->vector_);

      if (err == false) {

        mat_host.ConvertToCSR();

        if (mat_host.matrix_->LUSolve(*vec_host.vector_, out->vector_) == false) {
          LOG_INFO("Computation of LocalMatrix::LUSolve() failed");
          mat_host.Info();
          FATAL_ERROR(__FILE__, __LINE__);
        }

        if (this->GetFormat() != CSR)
          LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::LUSolve() is performed in CSR format");

      }

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::LUSolve() is performed on the host");

        out->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::LLAnalyse(void) {

  LOG_DEBUG(this, "LocalMatrix::LLAnalyse()",
            "");

  if (this->GetNnz() > 0) {
    this->matrix_->LLAnalyse();
  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::LLAnalyseClear(void) {

  LOG_DEBUG(this, "LocalMatrix::LLAnalyseClear()",
            "");

  if (this->GetNnz() > 0) {
    this->matrix_->LLAnalyseClear();
  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::LLSolve(const LocalVector<ValueType> &in, LocalVector<ValueType> *out) const {

  LOG_DEBUG(this, "LocalMatrix::LLSolve()",
            "");

  assert(&in != NULL);
  assert(out != NULL);
  assert(in.get_size() == this->GetN());
  assert(out->get_size() == this->GetM());

  assert( ( (this->matrix_ == this->matrix_host_)  && (in.vector_ == in.vector_host_) && (out->vector_ == out->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_) && (out->vector_ == out->vector_accel_)) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->LLSolve(*in.vector_, out->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::LLSolve() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      LocalVector<ValueType> vec_host;
      vec_host.CopyFrom(in);

      out->MoveToHost();

      mat_host.ConvertToCSR();

      if (mat_host.matrix_->LLSolve(*vec_host.vector_, out->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::LLSolve() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::LLSolve() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::LLSolve() is performed on the host");

        out->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::LLSolve(const LocalVector<ValueType> &in, const LocalVector<ValueType> &inv_diag,
                                     LocalVector<ValueType> *out) const {

  LOG_DEBUG(this, "LocalMatrix::LLSolve()",
            "");

  assert(&in != NULL);
  assert(&inv_diag != NULL);
  assert(out != NULL);
  assert(in.get_size() == this->GetN());
  assert(out->get_size() == this->GetM());

  assert( ( (this->matrix_ == this->matrix_host_) &&
            (in.vector_ == in.vector_host_) &&
            (out->vector_ == out->vector_host_) &&
            (inv_diag.vector_ == inv_diag.vector_host_) ) ||
          ( (this->matrix_ == this->matrix_accel_) &&
            (in.vector_ == in.vector_accel_) &&
            (out->vector_ == out->vector_accel_) &&
            (inv_diag.vector_ == inv_diag.vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->LLSolve(*in.vector_, *inv_diag.vector_, out->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::LLSolve() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      LocalVector<ValueType> vec_host;
      vec_host.CopyFrom(in);

      LocalVector<ValueType> inv_diag_host;
      inv_diag_host.CopyFrom(inv_diag);

      out->MoveToHost();

      mat_host.ConvertToCSR();

      if (mat_host.matrix_->LLSolve(*vec_host.vector_, *inv_diag_host.vector_, out->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::LLSolve() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::LLSolve() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::LLSolve() is performed on the host");

        out->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::LAnalyse(const bool diag_unit) {

  LOG_DEBUG(this, "LocalMatrix::LAnalyse()",
            diag_unit);

  if (this->GetNnz() > 0) {
    this->matrix_->LAnalyse(diag_unit);
  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::LAnalyseClear(void) {

  LOG_DEBUG(this, "LocalMatrix::LAnalyseClear()",
            "");

  if (this->GetNnz() > 0) {
    this->matrix_->LAnalyseClear();
  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::LSolve(const LocalVector<ValueType> &in, LocalVector<ValueType> *out) const {

  LOG_DEBUG(this, "LocalMatrix::LSolve()",
            "");

  assert(&in != NULL);
  assert(out != NULL);
  assert(in.get_size() == this->GetN());
  assert(out->get_size() == this->GetM());

  assert( ( (this->matrix_ == this->matrix_host_)  && (in.vector_ == in.vector_host_) && (out->vector_ == out->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_) && (out->vector_ == out->vector_accel_)) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->LSolve(*in.vector_, out->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::LSolve() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      LocalVector<ValueType> vec_host;
      vec_host.CopyFrom(in);

      out->MoveToHost();

      mat_host.ConvertToCSR();

      if (mat_host.matrix_->LSolve(*vec_host.vector_, out->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::LSolve() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::LSolve() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::LSolve() is performed on the host");

        out->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::UAnalyse(const bool diag_unit) {

  LOG_DEBUG(this, "LocalMatrix::UAnalyse()",
            "");

  if (this->GetNnz() > 0) {
    this->matrix_->UAnalyse(diag_unit);
  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::UAnalyseClear(void) {

  LOG_DEBUG(this, "LocalMatrix::UAnalyseClear()",
            "");

  if (this->GetNnz() > 0) {
    this->matrix_->UAnalyseClear();
  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::USolve(const LocalVector<ValueType> &in, LocalVector<ValueType> *out) const {

  LOG_DEBUG(this, "LocalMatrix::USolve()",
            "");

  assert(&in != NULL);
  assert(out != NULL);
  assert(in.get_size() == this->GetN());
  assert(out->get_size() == this->GetM());

  assert( ( (this->matrix_ == this->matrix_host_)  && (in.vector_ == in.vector_host_) && (out->vector_ == out->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_) && (out->vector_ == out->vector_accel_)) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->USolve(*in.vector_, out->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::USolve() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      LocalVector<ValueType> vec_host;
      vec_host.CopyFrom(in);

      out->MoveToHost();

      mat_host.ConvertToCSR();

      if (mat_host.matrix_->USolve(*vec_host.vector_, out->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::USolve() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::USolve() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::USolve() is performed on the host");

        out->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::ILU0Factorize(void) {

  LOG_DEBUG(this, "LocalMatrix::ILU0Factorize()",
            "");

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->ILU0Factorize();

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ILU0Factorize() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->ILU0Factorize() == false) {
        LOG_INFO("Computation of LocalMatrix::ILU0Factorize() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ILU0Factorize() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ILU0Factorize() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::ILUTFactorize(const double t, const int maxrow) {

  LOG_DEBUG(this, "LocalMatrix::ILUTFactorize()",
            "t=" << t << " maxrow=" << maxrow);

#ifdef DEBUG_MODE
  this->Check();
#endif

  assert(maxrow > 0);
  assert(t > double(0.0));

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->ILUTFactorize(t, maxrow);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ILUTFactorize() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->ILUTFactorize(t, maxrow) == false) {
        LOG_INFO("Computation of LocalMatrix::ILUTFactorize() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ILUTFactorize() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ILUTFactorize() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::ILUpFactorize(const int p, const bool level) {

  LOG_DEBUG(this, "LocalMatrix::ILUpFactorize()",
            "p=" << p << " level=" << level);

  assert(p >= 0);

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (p == 0) {

    this->ILU0Factorize();

  } else {

    if (this->GetNnz() > 0) {

      // with control levels
      if (level == true) {

        LocalMatrix structure;
        structure.CloneFrom(*this);
        structure.SymbolicPower(p+1);

        bool err = this->matrix_->ILUpFactorizeNumeric(p, *structure.matrix_);

        if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
          LOG_INFO("Computation of LocalMatrix::ILUpFactorize() failed");
          this->Info();
          FATAL_ERROR(__FILE__, __LINE__);
        }

        if (err == false) {

          // Move to host
          bool is_accel = this->is_accel();
          this->MoveToHost();
          structure.MoveToHost();

          // Convert to CSR
          unsigned int format = this->GetFormat();
          this->ConvertToCSR();
          structure.ConvertToCSR();

          if (this->matrix_->ILUpFactorizeNumeric(p, *structure.matrix_) == false) {
            LOG_INFO("Computation of LocalMatrix::ILUpFactorize() failed");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
          }

          if (format != CSR) {

            LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ILUpFactorize() is performed in CSR format");

            this->ConvertTo(format);

          }

          if (is_accel == true) {

            LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ILUpFactorize() is performed on the host");

            this->MoveToAccelerator();

          }

        }

      // without control levels
      } else {

        LocalMatrix values;
        values.CloneFrom(*this);

        this->SymbolicPower(p+1);
        this->MatrixAdd(values);

        bool err = this->matrix_->ILU0Factorize();

        if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
          LOG_INFO("Computation of LocalMatrix::ILUpFactorize() failed");
          this->Info();
          FATAL_ERROR(__FILE__, __LINE__);
        }

        if (err == false) {

          // Move to host
          bool is_accel = this->is_accel();
          this->MoveToHost();

          // Convert to CSR
          unsigned int format = this->GetFormat();
          this->ConvertToCSR();

          if (this->matrix_->ILU0Factorize() == false) {
            LOG_INFO("Computation of LocalMatrix::ILUpFactorize() failed");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
          }

          if (format != CSR) {

            LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ILUpFactorize() is performed in CSR format");

            this->ConvertTo(format);

          }

          if (is_accel == true) {

            LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ILUpFactorize() is performed on the host");

            this->MoveToAccelerator();

          }

        }

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::ICFactorize(LocalVector<ValueType> *inv_diag) {

  LOG_DEBUG(this, "LocalMatrix::ICFactorize()",
            "");

  assert(inv_diag != NULL);

  assert( ( (this->matrix_ == this->matrix_host_)  && (inv_diag->vector_ == inv_diag->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (inv_diag->vector_ == inv_diag->vector_accel_)) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->ICFactorize(inv_diag->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ICFactorize() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();
      inv_diag->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->ICFactorize(inv_diag->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::ICFactorize() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ICFactorize() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ICFactorize() is performed on the host");

        this->MoveToAccelerator();
        inv_diag->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::MultiColoring(int &num_colors,
                                           int **size_colors,
                                           LocalVector<int> *permutation) const {

  LOG_DEBUG(this, "LocalMatrix::MultiColoring()",
            "");

  assert(*size_colors == NULL);
  assert(permutation != NULL);
  assert(this->GetM() == this->GetN());

  assert( ( (this->matrix_ == this->matrix_host_)  && (permutation->vector_ == permutation->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (permutation->vector_ == permutation->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    std::string vec_perm_name = "MultiColoring permutation of " + this->object_name_;
    permutation->Allocate(vec_perm_name, 0);
    permutation->CloneBackend(*this);

    bool err = this->matrix_->MultiColoring(num_colors, size_colors, permutation->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::MultiColoring() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      permutation->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->MultiColoring(num_colors, size_colors, permutation->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::MultiColoring() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::MultiColoring() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::MultiColoring() is performed on the host");

        permutation->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::MaximalIndependentSet(int &size, LocalVector<int> *permutation) const {

  LOG_DEBUG(this, "LocalMatrix::MaximalIndependentSet()",
            "");

  assert(permutation != NULL);
  assert(this->GetM() == this->GetN());

  assert( ( (this->matrix_ == this->matrix_host_)  && (permutation->vector_ == permutation->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (permutation->vector_ == permutation->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    std::string vec_perm_name = "MaximalIndependentSet permutation of " + this->object_name_;
    permutation->Allocate(vec_perm_name, 0);
    permutation->CloneBackend(*this);

    bool err = this->matrix_->MaximalIndependentSet(size, permutation->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::MaximalIndependentSet() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      permutation->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->MaximalIndependentSet(size, permutation->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::MaximalIndependentSet() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::MaximalIndependentSet() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::MaximalIndependentSet() is performed on the host");

        permutation->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::ZeroBlockPermutation(int &size, LocalVector<int> *permutation) const {

  LOG_DEBUG(this, "LocalMatrix::ZeroBlockPermutation()",
            "");

  assert(permutation != NULL);
  assert(this->GetM() == this->GetN());

  assert( ( (this->matrix_ == this->matrix_host_)  && (permutation->vector_ == permutation->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (permutation->vector_ == permutation->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    std::string vec_perm_name = "ZeroBlockPermutation permutation of " + this->object_name_;
    permutation->Allocate(vec_perm_name, this->get_local_nrow());

    bool err = this->matrix_->ZeroBlockPermutation(size, permutation->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ZeroBlockPermutation() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      permutation->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->ZeroBlockPermutation(size, permutation->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::ZeroBlockPermutation() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ZeroBlockPermutation() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ZeroBlockPermutation() is performed on the host");

        permutation->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::Householder(const int idx, ValueType &beta, LocalVector<ValueType> *vec) const {

  LOG_DEBUG(this, "LocalMatrix::Householder()",
            "");

  assert(idx >= 0);
  assert(vec != NULL);

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->Householder(idx, beta, vec->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == DENSE)) {
      LOG_INFO("Computation of LocalMatrix::Householder() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      vec->MoveToHost();

      // Convert to DENSE
      mat_host.ConvertToDENSE();

      if (mat_host.matrix_->Householder(idx, beta, vec->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::Householder() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != DENSE)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Householder() is performed in DENSE format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Householder() is performed on the host");

        vec->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::QRDecompose(void) {

  LOG_DEBUG(this, "LocalMatrix::QRDecompose()",
            "");

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->QRDecompose();

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == DENSE)) {
      LOG_INFO("Computation of LocalMatrix::QRDecompose() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to DENSE
      unsigned int format = this->GetFormat();
      this->ConvertToDENSE();

      if (this->matrix_->QRDecompose() == false) {
        LOG_INFO("Computation of LocalMatrix::QRDecompose() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != DENSE) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::QRDecompose() is performed in DENSE format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::QRDecompose() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::QRSolve(const LocalVector<ValueType> &in, LocalVector<ValueType> *out) const {

  LOG_DEBUG(this, "LocalMatrix::QRSolve()",
            "");

  assert(&in != NULL);
  assert(out != NULL);
  assert(in.get_size() == this->GetN());
  assert(out->get_size() == this->GetM());

  assert( ( (this->matrix_ == this->matrix_host_)  && (in.vector_ == in.vector_host_)  && (out->vector_ == out->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_) && (out->vector_ == out->vector_accel_)) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->QRSolve(*in.vector_, out->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == DENSE)) {
      LOG_INFO("Computation of LocalMatrix::QRSolve() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      LocalVector<ValueType> vec_host;
      vec_host.CopyFrom(in);

      mat_host.MoveToHost();
      vec_host.MoveToHost();
      out->MoveToHost();

      mat_host.ConvertToDENSE();

      if (mat_host.matrix_->QRSolve(*vec_host.vector_, out->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::QRSolve() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != DENSE)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::QRSolve() is performed in DENSE format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::QRSolve() is performed on the host");

        out->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::Permute(const LocalVector<int> &permutation) {

  LOG_DEBUG(this, "LocalMatrix::Permute()",
            "");

  assert(&permutation != NULL);
  assert((permutation.get_size() == this->GetM()) ||
          (permutation.get_size() == this->GetN()));
  assert(permutation.get_size() > 0);

  assert( ( (this->matrix_ == this->matrix_host_)  && (permutation.vector_ == permutation.vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (permutation.vector_ == permutation.vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->Permute(*permutation.vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::Permute() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalVector<int> perm_host;
      perm_host.CopyFrom(permutation);

      // Move to host
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->Permute(*perm_host.vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::Permute() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Permute() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (permutation.is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Permute() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::PermuteBackward(const LocalVector<int> &permutation) {

  LOG_DEBUG(this, "LocalMatrix::PermuteBackward()",
            "");

  assert(&permutation != NULL);
  assert((permutation.get_size() == this->GetM()) ||
         (permutation.get_size() == this->GetN()));
  assert(permutation.get_size() > 0);

  assert( ( (this->matrix_ == this->matrix_host_)  && (permutation.vector_ == permutation.vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (permutation.vector_ == permutation.vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->PermuteBackward(*permutation.vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == COO)) {
      LOG_INFO("Computation of LocalMatrix::PermuteBackward() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalVector<int> perm_host;
      perm_host.CopyFrom(permutation);

      // Move to host
      this->MoveToHost();

      // Convert to COO
      unsigned int format = this->GetFormat();
      this->ConvertToCOO();

      if (this->matrix_->PermuteBackward(*perm_host.vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::PermuteBackward() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != COO) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::PermuteBackward() is performed in COO format");

        this->ConvertTo(format);

      }

      if (permutation.is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::PermuteBackward() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::CMK(LocalVector<int> *permutation) const {

  LOG_DEBUG(this, "LocalMatrix::CMK()",
            "");

  assert(permutation != NULL);

  assert( ( (this->matrix_ == this->matrix_host_)  && (permutation->vector_ == permutation->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (permutation->vector_ == permutation->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->CMK(permutation->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::CMK() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      permutation->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->CMK(permutation->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::CMK() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::CMK() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::CMK() is performed on the host");

        permutation->MoveToAccelerator();

      }

    }

  }

  std::string vec_name = "CMK permutation of " + this->object_name_;
  permutation->object_name_ = vec_name;

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::RCMK(LocalVector<int> *permutation) const {

  LOG_DEBUG(this, "LocalMatrix::RCMK()",
            "");

  assert(permutation != NULL);

  assert( ( (this->matrix_ == this->matrix_host_)  && (permutation->vector_ == permutation->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (permutation->vector_ == permutation->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->RCMK(permutation->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::RCMK() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      permutation->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->RCMK(permutation->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::RCMK() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::RCMK() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::RCMK() is performed on the host");

        permutation->MoveToAccelerator();

      }

    }

  }

  std::string vec_name = "RCMK permutation of " + this->object_name_;
  permutation->object_name_ = vec_name;

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::ConnectivityOrder(LocalVector<int> *permutation) const {

  LOG_DEBUG(this, "LocalMatrix::ConnectivityOrder()",
            "");

  assert(permutation != NULL);

  assert( ( (this->matrix_ == this->matrix_host_)  && (permutation->vector_ == permutation->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (permutation->vector_ == permutation->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->ConnectivityOrder(permutation->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ConnectivityOrder() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      permutation->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->ConnectivityOrder(permutation->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::ConnectivityOrder() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ConnectivityOrder() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ConnectivityOrder() is performed on the host");

        permutation->MoveToAccelerator();

      }

    }

  }

  std::string vec_name = "ConnectivityOrder permutation of " + this->object_name_;
  permutation->object_name_ = vec_name;

}

template <typename ValueType>
void LocalMatrix<ValueType>::SymbolicPower(const int p) {

  LOG_DEBUG(this, "LocalMatrix::SymbolicPower()",
            p);

  assert(p >= 1);

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->SymbolicPower(p);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::SymbolicPower() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->SymbolicPower(p) == false) {
        LOG_INFO("Computation of LocalMatrix::SymbolicPower() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::SymbolicPower() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::SymbolicPower() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::MatrixAdd(const LocalMatrix<ValueType> &mat, const ValueType alpha,
                                       const ValueType beta, const bool structure) {

  LOG_DEBUG(this, "LocalMatrix::MatrixAdd()",
            "");

  assert(&mat != NULL);
  assert(&mat != this);
  assert(this->GetFormat() == mat.GetFormat());
  assert(this->GetM() == mat.GetM());
  assert(this->GetN() == mat.GetN());

  assert( ( (this->matrix_ == this->matrix_host_)  && (mat.matrix_ == mat.matrix_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (mat.matrix_ == mat.matrix_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
  mat.Check();
#endif

  bool err = this->matrix_->MatrixAdd(*mat.matrix_, alpha, beta, structure);

  if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
    LOG_INFO("Computation of LocalMatrix::MatrixAdd() failed");
    this->Info();
    FATAL_ERROR(__FILE__, __LINE__);
  }

  if (err == false) {

    LocalMatrix<ValueType> mat_host;
    mat_host.ConvertTo(mat.GetFormat());
    mat_host.CopyFrom(mat);

    this->MoveToHost();

    this->ConvertToCSR();
    mat_host.ConvertToCSR();

    if (this->matrix_->MatrixAdd(*mat_host.matrix_, alpha, beta, structure) == false) {
      LOG_INFO("Computation of LocalMatrix::MatrixAdd() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (mat.GetFormat() != CSR) {

      LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::MatrixAdd() is performed in CSR format");

      this->ConvertTo(mat.GetFormat());

    }

    if (mat.is_accel() == true) {

      LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::MatrixAdd() is performed on the host");

      this->MoveToAccelerator();

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::Gershgorin(ValueType &lambda_min, ValueType &lambda_max) const {

  LOG_DEBUG(this, "LocalMatrix::Gershgorin()",
            "");

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->Gershgorin(lambda_min, lambda_max);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::Gershgorin() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->Gershgorin(lambda_min, lambda_max) == false) {
        LOG_INFO("Computation of LocalMatrix::Gershgorin() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Gershgorin() is performed in CSR format");

      if (this->is_accel() == true)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Gershgorin() is performed on the host");

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::Scale(const ValueType alpha) {

  LOG_DEBUG(this, "LocalMatrix::Scale()",
            alpha);

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->Scale(alpha);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::Scale() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->Scale(alpha) == false) {
        LOG_INFO("Computation of LocalMatrix::Scale() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Scale() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Scale() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::ScaleDiagonal(const ValueType alpha) {

  LOG_DEBUG(this, "LocalMatrix::ScaleDiagonal()",
            alpha);

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->ScaleDiagonal(alpha);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ScaleDiagonal() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->ScaleDiagonal(alpha) == false) {
        LOG_INFO("Computation of LocalMatrix::ScaleDiagonal() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ScaleDiagonal() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ScaleDiagonal() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::ScaleOffDiagonal(const ValueType alpha) {

  LOG_DEBUG(this, "LocalMatrix::ScaleOffDiagonal()",
            alpha);

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->ScaleOffDiagonal(alpha);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ScaleOffDiagonal() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->ScaleOffDiagonal(alpha) == false) {
        LOG_INFO("Computation of LocalMatrix::ScaleOffDiagonal() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ScaleOffDiagonal() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ScaleOffDiagonal() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::AddScalar(const ValueType alpha) {

  LOG_DEBUG(this, "LocalMatrix::AddScalar()",
            alpha);

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->AddScalar(alpha);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::AddScalar() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->AddScalar(alpha) == false) {
        LOG_INFO("Computation of LocalMatrix::AddScalar() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AddScalar() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AddScalar() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::AddScalarDiagonal(const ValueType alpha) {

  LOG_DEBUG(this, "LocalMatrix::AddScalarDiagonal()",
            alpha);

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->AddScalarDiagonal(alpha);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::AddScalarDiagonal() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->AddScalarDiagonal(alpha) == false) {
        LOG_INFO("Computation of LocalMatrix::AddScalarDiagonal() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AddScalarDiagonal() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AddScalarDiagonal() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::AddScalarOffDiagonal(const ValueType alpha) {

  LOG_DEBUG(this, "LocalMatrix::AddScalarOffDiagonal()",
            "");

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->AddScalarOffDiagonal(alpha);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::AddScalarOffDiagonal() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->AddScalarOffDiagonal(alpha) == false) {
        LOG_INFO("Computation of LocalMatrix::AddScalarOffDiagonal() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AddScalarOffDiagonal() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AddScalarOffDiagonal() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::MatrixMult(const LocalMatrix<ValueType> &A, const LocalMatrix<ValueType> &B) {

  LOG_DEBUG(this, "LocalMatrix::AddScalarDiagonal()",
            "");

  assert(&A != NULL);
  assert(&B != NULL);
  assert(&A != this);
  assert(&B != this);
  assert(A.GetN() == B.GetM());

  assert(A.GetFormat() == B.GetFormat());

  assert( ( (this->matrix_ == this->matrix_host_)  && (A.matrix_ == A.matrix_host_)  && (B.matrix_ == B.matrix_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (A.matrix_ == A.matrix_accel_) && (B.matrix_ == B.matrix_accel_)) );

#ifdef DEBUG_MODE
  this->Check();
  A.Check();
  B.Check();
#endif

  if (this->GetFormat() == DENSE) {

    if (this->GetNnz() != A.GetNnz()) {

      this->Clear();
      this->AllocateDENSE("", A.get_local_nrow(), B.get_local_ncol());

    }

  } else {

    this->Clear();

  }

  this->object_name_ = A.object_name_ + " x " + B.object_name_;
  this->ConvertTo(A.GetFormat());

  bool err = this->matrix_->MatMatMult(*A.matrix_, *B.matrix_);

  if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
    LOG_INFO("Computation of LocalMatrix::MatMatMult() failed");
    this->Info();
    FATAL_ERROR(__FILE__, __LINE__);
  }

  if (err == false) {

    LocalMatrix<ValueType> A_host;
    LocalMatrix<ValueType> B_host;
    A_host.ConvertTo(A.GetFormat());
    B_host.ConvertTo(B.GetFormat());
    A_host.CopyFrom(A);
    B_host.CopyFrom(B);

    this->MoveToHost();

    A_host.ConvertToCSR();
    B_host.ConvertToCSR();
    this->ConvertToCSR();

    if (this->matrix_->MatMatMult(*A_host.matrix_, *B_host.matrix_) == false) {
      LOG_INFO("Computation of LocalMatrix::MatMatMult() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (A.GetFormat() != CSR) {

      LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::MatMatMult() is performed in CSR format");

      this->ConvertTo(A.GetFormat());

    }

    if (A.is_accel() == true) {

      LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::MatMatMult() is performed on the host");

      this->MoveToAccelerator();

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::DiagonalMatrixMultR(const LocalVector<ValueType> &diag) {

  LOG_DEBUG(this, "LocalMatrix::DiagonalMatrixMultR()",
            "");

  assert(&diag != NULL);
  assert((diag.get_size() == this->GetM()) ||
         (diag.get_size() == this->GetN()));

  assert( ( (this->matrix_ == this->matrix_host_)  && (diag.vector_ == diag.vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (diag.vector_ == diag.vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->DiagonalMatrixMultR(*diag.vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::DiagonalMatrixMultR() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalVector<ValueType> diag_host;
      diag_host.CopyFrom(diag);

      // Move to host
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->DiagonalMatrixMultR(*diag_host.vector_) ==  false) {
        LOG_INFO("Computation of LocalMatrix::DiagonalMatrixMultR() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::DiagonalMatrixMultR() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (diag.is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::DiagonalMatrixMultR() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::DiagonalMatrixMult(const LocalVector<ValueType> &diag) {

  this->DiagonalMatrixMultR(diag);

}

template <typename ValueType>
void LocalMatrix<ValueType>::DiagonalMatrixMultL(const LocalVector<ValueType> &diag) {

  LOG_DEBUG(this, "LocalMatrix::DiagonalMatrixMultL()",
            "");

  assert(&diag != NULL);
  assert((diag.get_size() == this->GetM()) ||
         (diag.get_size() == this->GetN()));

  assert( ( (this->matrix_ == this->matrix_host_)  && (diag.vector_ == diag.vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (diag.vector_ == diag.vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->DiagonalMatrixMultL(*diag.vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::DiagonalMatrixMultL() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalVector<ValueType> diag_host;
      diag_host.CopyFrom(diag);

      // Move to host
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->DiagonalMatrixMultL(*diag_host.vector_) ==  false) {
        LOG_INFO("Computation of LocalMatrix::DiagonalMatrixMultL() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::DiagonalMatrixMultL() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (diag.is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::DiagonalMatrixMultL() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::Compress(const double drop_off) {

  LOG_DEBUG(this, "LocalMatrix::Compress()",
            "");

  assert(rocalution_abs(drop_off) >= double(0.0));

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->Compress(drop_off);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::Compress() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->Compress(drop_off) == false) {
        LOG_INFO("Computation of LocalMatrix::Compress() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Compress() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Compress() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::Transpose(void) {

  LOG_DEBUG(this, "LocalMatrix::Transpose()",
            "");

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->Transpose();

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::Transpose() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->Transpose() == false) {
        LOG_INFO("Computation of LocalMatrix::Transpose() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Transpose() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Transpose() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::Sort(void) {

  LOG_DEBUG(this, "LocalMatrix::Sort()",
            "");

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->Sort();

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::Sort() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->Sort() ==  false) {
        LOG_INFO("Computation of LocalMatrix::Sort() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Sort() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Sort() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::Key(long int &row_key,
                                 long int &col_key,
                                 long int &val_key) const {

  LOG_DEBUG(this, "LocalMatrix::Key()",
            "");

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->Key(row_key,
                                  col_key,
                                  val_key);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::Key() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->Key(row_key,
                                col_key,
                                val_key) == false) {
        LOG_INFO("Computation of LocalMatrix::Key() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Key() is performed in CSR format");

      if (this->is_accel() == true)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Key() is performed on the host");

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::AMGConnect(const ValueType eps, LocalVector<int> *connections) const {

  LOG_DEBUG(this, "LocalMatrix::AMGConnect()",
            eps);

  assert(eps > ValueType(0.0));
  assert(connections != NULL);

  assert( ( (this->matrix_ == this->matrix_host_)  && (connections->vector_ == connections->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (connections->vector_ == connections->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->AMGConnect(eps, connections->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::AMGConnect() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      connections->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->AMGConnect(eps, connections->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::AMGConnect() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AMGConnect() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AMGConnect() is performed on the host");

        connections->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::AMGAggregate(const LocalVector<int> &connections, LocalVector<int> *aggregates) const {

  LOG_DEBUG(this, "LocalMatrix::AMGAggregate()",
            "");

  assert(&connections != NULL);
  assert(aggregates != NULL);

  assert( ( (this->matrix_ == this->matrix_host_)  && (connections.vector_ == connections.vector_host_)  && (aggregates->vector_ == aggregates->vector_host_) ) ||
          ( (this->matrix_ == this->matrix_accel_) && (connections.vector_ == connections.vector_accel_) && (aggregates->vector_ == aggregates->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->AMGAggregate(*connections.vector_, aggregates->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::AMGAggregate() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      LocalVector<int> conn_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);
      conn_host.CopyFrom(connections);

      // Move to host
      aggregates->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->AMGAggregate(*conn_host.vector_, aggregates->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::AMGAggregate() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AMGAggregate() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AMGAggregate() is performed on the host");

        aggregates->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::AMGSmoothedAggregation(const ValueType relax,
                                                    const LocalVector<int> &aggregates,
                                                    const LocalVector<int> &connections,
                                                          LocalMatrix<ValueType> *prolong,
                                                          LocalMatrix<ValueType> *restrict) const {

  LOG_DEBUG(this, "LocalMatrix::AMGSmoothedAggregation()",
            relax);

  assert(&aggregates != NULL);
  assert(&connections != NULL);
  assert(relax > ValueType(0.0));
  assert(prolong != NULL);
  assert(restrict != NULL);
  assert(this != prolong);
  assert(this != restrict);

  assert( ( (this->matrix_ == this->matrix_host_) &&
            (aggregates.vector_ == aggregates.vector_host_) &&
            (connections.vector_ == connections.vector_host_) &&
            (prolong->matrix_ == prolong->matrix_host_) &&
            (restrict->matrix_ == restrict->matrix_host_) ) ||
          ( (this->matrix_ == this->matrix_accel_) &&
            (aggregates.vector_ == aggregates.vector_accel_) &&
            (connections.vector_ == connections.vector_accel_) &&
            (prolong->matrix_ == prolong->matrix_accel_) &&
            (restrict->matrix_ == restrict->matrix_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
  prolong->Check();
  restrict->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->AMGSmoothedAggregation(relax, *aggregates.vector_, *connections.vector_,
                                                     prolong->matrix_, restrict->matrix_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::AMGSmoothedAggregation() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      LocalVector<int> conn_host;
      LocalVector<int> aggr_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);
      conn_host.CopyFrom(connections);
      aggr_host.CopyFrom(aggregates);

      // Move to host
      prolong->MoveToHost();
      restrict->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->AMGSmoothedAggregation(relax, *aggr_host.vector_, *conn_host.vector_,
                                                   prolong->matrix_, restrict->matrix_) == false) {
        LOG_INFO("Computation of LocalMatrix::AMGSmoothedAggregation() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AMGSmoothedAggregation() is performed in CSR format");

        prolong->ConvertTo(this->GetFormat());
        restrict->ConvertTo(this->GetFormat());

      }

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AMGSmoothedAggregation() is performed on the host");

        prolong->MoveToAccelerator();
        restrict->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  prolong->Check();
  restrict->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::AMGAggregation(const LocalVector<int> &aggregates,
                                                  LocalMatrix<ValueType> *prolong,
                                                  LocalMatrix<ValueType> *restrict) const {

  LOG_DEBUG(this, "LocalMatrix::AMGAggregation()",
            "");

  assert(&aggregates != NULL);
  assert(prolong != NULL);
  assert(restrict != NULL);
  assert(this != prolong);
  assert(this != restrict);

  assert( ( (this->matrix_ == this->matrix_host_) &&
            (aggregates.vector_ == aggregates.vector_host_) &&
            (prolong->matrix_ == prolong->matrix_host_) &&
            (restrict->matrix_ == restrict->matrix_host_) ) ||
          ( (this->matrix_ == this->matrix_accel_) &&
            (aggregates.vector_ == aggregates.vector_accel_) &&
            (prolong->matrix_ == prolong->matrix_accel_) &&
            (restrict->matrix_ == restrict->matrix_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
  prolong->Check();
  restrict->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->AMGAggregation(*aggregates.vector_, prolong->matrix_, restrict->matrix_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::AMGAggregation() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      LocalVector<int> aggr_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);
      aggr_host.CopyFrom(aggregates);

      // Move to host
      prolong->MoveToHost();
      restrict->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->AMGAggregation(*aggr_host.vector_, prolong->matrix_, restrict->matrix_) == false) {
        LOG_INFO("Computation of LocalMatrix::AMGAggregation() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AMGAggregation() is performed in CSR format");

        prolong->ConvertTo(this->GetFormat());
        restrict->ConvertTo(this->GetFormat());

      }

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::AMGAggregation() is performed on the host");

        prolong->MoveToAccelerator();
        restrict->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  prolong->Check();
  restrict->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::RugeStueben(const ValueType eps, LocalMatrix<ValueType> *prolong,
                                                              LocalMatrix<ValueType> *restrict) const {

  LOG_DEBUG(this, "LocalMatrix::RugeStueben()",
            "");

  assert(eps < ValueType(1.0));
  assert(eps > ValueType(0.0));
  assert(prolong != NULL);
  assert(restrict != NULL);
  assert(this != prolong);
  assert(this != restrict);

  assert( ( (this->matrix_ == this->matrix_host_) &&
            (prolong->matrix_ == prolong->matrix_host_) &&
            (restrict->matrix_ == restrict->matrix_host_) ) ||
          ( (this->matrix_ == this->matrix_accel_) &&
            (prolong->matrix_ == prolong->matrix_accel_) &&
            (restrict->matrix_ == restrict->matrix_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->RugeStueben(eps, prolong->matrix_, restrict->matrix_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::RugeStueben() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      prolong->MoveToHost();
      restrict->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->RugeStueben(eps, prolong->matrix_, restrict->matrix_) == false) {
        LOG_INFO("Computation of LocalMatrix::RugeStueben() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::RugeStueben() is performed in CSR format");

        prolong->ConvertTo(this->GetFormat());
        restrict->ConvertTo(this->GetFormat());

      }

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::RugeStueben() is performed on the host");

        prolong->MoveToAccelerator();
        restrict->MoveToAccelerator();

      }

    }

  }

  std::string prolong_name = "Prolongation Operator of " + this->object_name_;
  std::string restrict_name = "Restriction Operator of " + this->object_name_;

  prolong->object_name_ = prolong_name;
  restrict->object_name_ = restrict_name;

#ifdef DEBUG_MODE
  prolong->Check();
  restrict->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::InitialPairwiseAggregation(const ValueType beta, int &nc, LocalVector<int> *G, int &Gsize,
                                                        int **rG, int &rGsize, const int ordering) const {

  LOG_DEBUG(this, "LocalMatrix::InitialPairwiseAggregation()",
            "");

  assert(*rG == NULL);
  assert(beta > ValueType(0.0));
  assert(G != NULL);

  assert( ( (this->matrix_ == this->matrix_host_)  && (G->vector_ == G->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (G->vector_ == G->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->InitialPairwiseAggregation(beta, nc, G->vector_, Gsize, rG, rGsize, ordering);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::InitialPairwiseAggregation() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      G->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->InitialPairwiseAggregation(beta, nc, G->vector_, Gsize, rG, rGsize, ordering) == false) {
        LOG_INFO("Computation of LocalMatrix::InitialPairwiseAggregation() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::InitialPairwiseAggregation() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::InitialPairwiseAggregation() is performed on the host");

        G->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::InitialPairwiseAggregation(const LocalMatrix<ValueType> &mat, const ValueType beta,
                                                        int &nc, LocalVector<int> *G, int &Gsize, int **rG,
                                                        int &rGsize, const int ordering) const {

  LOG_DEBUG(this, "LocalMatrix::InitialPairwiseAggregation()",
            "");

  assert(&mat != NULL);
  assert(*rG == NULL);
  assert(&mat != this);
  assert(beta > ValueType(0.0));
  assert(G != NULL);

  assert( ( (this->matrix_ == this->matrix_host_)  &&
            (mat.matrix_ == mat.matrix_host_) &&
            (G->vector_ == G->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) &&
            (mat.matrix_ == mat.matrix_accel_) &&
            (G->vector_ == G->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
  mat.Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->InitialPairwiseAggregation(*mat.matrix_, beta, nc, G->vector_,
                                                         Gsize, rG, rGsize, ordering);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::InitialPairwiseAggregation() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      LocalMatrix<ValueType> mat2_host;
      mat_host.ConvertTo(this->GetFormat());
      mat2_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);
      mat2_host.CopyFrom(mat);

      // Move to host
      G->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();
      mat2_host.ConvertToCSR();

      if (mat_host.matrix_->InitialPairwiseAggregation(*mat2_host.matrix_, beta, nc, G->vector_,
                                                       Gsize, rG, rGsize, ordering) == false) {
        LOG_INFO("Computation of LocalMatrix::InitialPairwiseAggregation() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::InitialPairwiseAggregation() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::InitialPairwiseAggregation() is performed on the host");

        G->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::FurtherPairwiseAggregation(const ValueType beta, int &nc, LocalVector<int> *G, int &Gsize,
                                                        int **rG, int &rGsize, const int ordering) const {

  LOG_DEBUG(this, "LocalMatrix::FurtherPairwiseAggregation()",
            "");

  assert(*rG == NULL);
  assert(beta > ValueType(0.0));
  assert(G != NULL);

  assert( ( (this->matrix_ == this->matrix_host_)  && (G->vector_ == G->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (G->vector_ == G->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->FurtherPairwiseAggregation(beta, nc, G->vector_, Gsize, rG, rGsize, ordering);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::FurtherPairwiseAggregation() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      G->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->FurtherPairwiseAggregation(beta, nc, G->vector_, Gsize, rG, rGsize, ordering) == false) {
        LOG_INFO("Computation of LocalMatrix::FurtherPairwiseAggregation() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::FurtherPairwiseAggregation() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::FurtherPairwiseAggregation() is performed on the host");

        G->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::FurtherPairwiseAggregation(const LocalMatrix<ValueType> &mat, const ValueType beta,
                                                        int &nc, LocalVector<int> *G, int &Gsize, int **rG,
                                                        int &rGsize, const int ordering) const {

  LOG_DEBUG(this, "LocalMatrix::FurtherPairwiseAggregation()",
            "");

  assert(&mat != NULL);
  assert(*rG == NULL);
  assert(&mat != this);
  assert(beta > ValueType(0.0));
  assert(G != NULL);

  assert( ( (this->matrix_ == this->matrix_host_)  &&
            (mat.matrix_ == mat.matrix_host_) &&
            (G->vector_ == G->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) &&
            (mat.matrix_ == mat.matrix_accel_) &&
            (G->vector_ == G->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
  mat.Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->FurtherPairwiseAggregation(*mat.matrix_, beta, nc, G->vector_,
                                                         Gsize, rG, rGsize, ordering);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::FurtherPairwiseAggregation() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      LocalMatrix<ValueType> mat2_host;
      mat_host.ConvertTo(this->GetFormat());
      mat2_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);
      mat2_host.CopyFrom(mat);

      // Move to host
      G->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->FurtherPairwiseAggregation(*mat2_host.matrix_, beta, nc, G->vector_,
                                                       Gsize, rG, rGsize, ordering) == false) {
        LOG_INFO("Computation of LocalMatrix::FurtherPairwiseAggregation() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::FurtherPairwiseAggregation() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::FurtherPairwiseAggregation() is performed on the host");

        G->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::CoarsenOperator(LocalMatrix<ValueType> *Ac, const int nrow, const int ncol, const LocalVector<int> &G,
                                             const int Gsize, const int *rG, const int rGsize) const {

  LOG_DEBUG(this, "LocalMatrix::CoarsenOperator()",
            "");

  assert(Ac != NULL);
  assert(Ac != this);
  assert(&G != NULL);
  assert(nrow > 0);
  assert(ncol > 0);
  assert(rG != NULL);
  assert(Gsize > 0);
  assert(rGsize > 0);

  assert( ( (this->matrix_ == this->matrix_host_)  && (Ac->matrix_ == Ac->matrix_host_)  && (G.vector_ == G.vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (Ac->matrix_ == Ac->matrix_accel_) && (G.vector_ == G.vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->CoarsenOperator(Ac->matrix_, nrow, ncol, *G.vector_, Gsize, rG, rGsize);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::CoarsenOperator() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      LocalVector<int> vec_host;
      vec_host.CopyFrom(G);

      // Move to host
      Ac->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();
      Ac->ConvertToCSR();

      if (mat_host.matrix_->CoarsenOperator(Ac->matrix_, nrow, ncol, *vec_host.vector_, Gsize, rG, rGsize) == false) {
        LOG_INFO("Computation of LocalMatrix::CoarsenOperator() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR) {

        // Adding COO due to MPI using COO ghost matrix
        if (this->GetFormat() != COO)
          LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::CoarsenOperator() is performed in CSR format");

        Ac->ConvertTo(this->GetFormat());

      }

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::CoarsenOperator() is performed on the host");

        Ac->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  Ac->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::CreateFromMap(const LocalVector<int> &map, const int n, const int m) {

  LOG_DEBUG(this, "LocalMatrix::CreateFromMap()",
            n << " " << m);

  assert(&map != NULL);
  assert(map.get_size() == (IndexType2) n);
  assert(m > 0);

  assert( ( (this->matrix_ == this->matrix_host_)  && (map.vector_ == map.vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (map.vector_ == map.vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->CreateFromMap(*map.vector_, n, m);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::CreateFromMap() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalVector<int> map_host;
      map_host.CopyFrom(map);

      // Move to host
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->CreateFromMap(*map_host.vector_, n, m) == false) {
        LOG_INFO("Computation of LocalMatrix::CreateFromMap() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::CreateFromMap() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (map.is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::CreateFromMap() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::CreateFromMap(const LocalVector<int> &map, const int n, const int m, LocalMatrix<ValueType> *pro) {

  LOG_DEBUG(this, "LocalMatrix::CreateFromMap()",
            n << " " << m);

  assert(&map != NULL);
  assert(pro != NULL);
  assert(this != pro);
  assert(map.get_size() == (IndexType2) n);
  assert(m > 0);

  assert( ( (this->matrix_ == this->matrix_host_)  && (map.vector_ == map.vector_host_)  && (pro->matrix_ == pro->matrix_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (map.vector_ == map.vector_accel_) && (pro->matrix_ == pro->matrix_accel_) ) );

  this->Clear();
  pro->Clear();

  bool err = this->matrix_->CreateFromMap(*map.vector_, n , m, pro->matrix_);

  if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
    LOG_INFO("Computation of LocalMatrix::CreateFromMap() failed");
    this->Info();
    FATAL_ERROR(__FILE__, __LINE__);
  }

  if (err == false) {

    LocalVector<int> map_host;
    map_host.CopyFrom(map);

    // Move to host
    this->MoveToHost();
    pro->MoveToHost();

    // Convert to CSR
    unsigned int format = this->GetFormat();
    this->ConvertToCSR();

    if (this->matrix_->CreateFromMap(*map_host.vector_, n, m, pro->matrix_) == false) {
      LOG_INFO("Computation of LocalMatrix::CreateFromMap() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (format != CSR) {

      LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::CreateFromMap() is performed in CSR format");

      this->ConvertTo(format);
      pro->ConvertTo(format);

    }

    if (map.is_accel() == true) {

      LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::CreateFromMap() is performed on the host");

      this->MoveToAccelerator();
      pro->MoveToAccelerator();

    }

  }

#ifdef DEBUG_MODE
  this->Check();
  pro->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::LUFactorize(void) {

  LOG_DEBUG(this, "LocalMatrix::LUFactorize()",
            "");

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->LUFactorize();

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == DENSE)) {
      LOG_INFO("Computation of LocalMatrix::LUFactorize() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to DENSE
      unsigned int format = this->GetFormat();
      this->ConvertToDENSE();

      if (this->matrix_->LUFactorize() == false) {
        LOG_INFO("Computation of LocalMatrix::LUFactorize() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != DENSE) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::LUFactorize() is performed in DENSE format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::LUFactorize() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::FSAI(const int power, const LocalMatrix<ValueType> *pattern) {

  LOG_DEBUG(this, "LocalMatrix::FSAI()",
            power);

  assert(power > 0);
  assert(pattern != this);
  assert(this->GetM() == this->GetN());

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err;

    if (pattern != NULL) {

      assert( ( (this->matrix_ == this->matrix_host_)  && (pattern->matrix_ == pattern->matrix_host_)) ||
              ( (this->matrix_ == this->matrix_accel_) && (pattern->matrix_ == pattern->matrix_accel_) ) );
      err = this->matrix_->FSAI(power, pattern->matrix_);

    } else {

      err = this->matrix_->FSAI(power, NULL);

    }

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::FSAI() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (pattern != NULL) {

        LocalMatrix<ValueType> pattern_host;
        pattern_host.CopyFrom(*pattern);

        if (this->matrix_->FSAI(power, pattern_host.matrix_) == false) {
          LOG_INFO("Computation of LocalMatrix::FSAI() failed");
          this->Info();
          FATAL_ERROR(__FILE__, __LINE__);
        }

      } else {

        if (this->matrix_->FSAI(power, NULL) == false) {
          LOG_INFO("Computation of LocalMatrix::FSAI() failed");
          this->Info();
          FATAL_ERROR(__FILE__, __LINE__);
        }

      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::FSAI() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::FSAI() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::SPAI(void) {

  LOG_DEBUG(this, "LocalMatrix::SPAI()",
            "");

  assert(this->GetM() == this->GetN());

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->SPAI();

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::SPAI() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to CSR
      unsigned int format = this->GetFormat();
      this->ConvertToCSR();

      if (this->matrix_->SPAI() == false) {
        LOG_INFO("Computation of LocalMatrix::SPAI() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != CSR) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::SPAI() is performed in CSR format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::SPAI() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::Invert(void) {

  LOG_DEBUG(this, "LocalMatrix::Invert()",
            "");

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->Invert();

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == DENSE)) {
      LOG_INFO("Computation of LocalMatrix::Invert() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      // Move to host
      bool is_accel = this->is_accel();
      this->MoveToHost();

      // Convert to DENSE
      unsigned int format = this->GetFormat();
      this->ConvertToDENSE();

      if (this->matrix_->Invert() == false) {
        LOG_INFO("Computation of LocalMatrix::Invert() failed");
        this->Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (format != DENSE) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Invert() is performed in DENSE format");

        this->ConvertTo(format);

      }

      if (is_accel == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Invert() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::ReplaceColumnVector(const int idx, const LocalVector<ValueType> &vec) {

  LOG_DEBUG(this, "LocalMatrix::ReplaceColumnVector()",
            idx);

  assert(&vec != NULL);
  assert(vec.get_size() == this->GetM());
  assert(idx >= 0);

  assert( ( (this->matrix_ == this->matrix_host_)  && (vec.vector_ == vec.vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (vec.vector_ == vec.vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->ReplaceColumnVector(idx, *vec.vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ReplaceColumnVector() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalVector<ValueType> vec_host;
      vec_host.CopyFrom(vec);

      // Move to host
      this->MoveToHost();

      // try again
      err = this->matrix_->ReplaceColumnVector(idx, *vec_host.vector_);

      if (err == false) {

        // Convert to CSR
        unsigned int format = this->GetFormat();
        this->ConvertToCSR();

        if (this->matrix_->ReplaceColumnVector(idx, *vec_host.vector_) == false) {
          LOG_INFO("Computation of LocalMatrix::ReplaceColumnVector() failed");
          this->Info();
          FATAL_ERROR(__FILE__, __LINE__);
        }

        if (format != CSR) {

          LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ReplaceColumnVector() is performed in CSR format");

          this->ConvertTo(format);

        }

      }

      if (vec.is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ReplaceColumnVector() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::ExtractColumnVector(const int idx, LocalVector<ValueType> *vec) const {

  LOG_DEBUG(this, "LocalMatrix::ExtractColumnVector()",
            idx);

  assert(vec != NULL);
  assert(vec->get_size() == this->GetM());
  assert(idx >= 0);

  assert( ( (this->matrix_ == this->matrix_host_)  && (vec->vector_ == vec->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (vec->vector_ == vec->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->ExtractColumnVector(idx, vec->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ExtractColumnVector() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      vec->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->ExtractColumnVector(idx, vec->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::ExtractColumnVector() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractColumnVector() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractColumnVector() is performed on the host");

        vec->MoveToAccelerator();

      }

    }

  }

}

template <typename ValueType>
void LocalMatrix<ValueType>::ReplaceRowVector(const int idx, const LocalVector<ValueType> &vec) {

  LOG_DEBUG(this, "LocalMatrix::ReplaceRowVector()",
            idx);

  assert(&vec != NULL);
  assert(vec.get_size() == this->GetN());
  assert(idx >= 0);

  assert( ( (this->matrix_ == this->matrix_host_)  && (vec.vector_ == vec.vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (vec.vector_ == vec.vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->ReplaceRowVector(idx, *vec.vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ReplaceRowVector() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalVector<ValueType> vec_host;
      vec_host.CopyFrom(vec);

      // Move to host
      this->MoveToHost();

      // try again
      err = this->matrix_->ReplaceRowVector(idx, *vec_host.vector_);

      if (err == false) {

        // Convert to CSR
        unsigned int format = this->GetFormat();
        this->ConvertToCSR();

        if (this->matrix_->ReplaceRowVector(idx, *vec_host.vector_) == false) {
          LOG_INFO("Computation of LocalMatrix::ReplaceRowVector() failed");
          this->Info();
          FATAL_ERROR(__FILE__, __LINE__);
        }

        if (format != CSR) {

          LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ReplaceRowVector() is performed in CSR format");

          this->ConvertTo(format);

        }

      }

      if (vec.is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ReplaceRowVector() is performed on the host");

        this->MoveToAccelerator();

      }

    }

  }

#ifdef DEBUG_MODE
  this->Check();
#endif

}

template <typename ValueType>
void LocalMatrix<ValueType>::ExtractRowVector(const int idx, LocalVector<ValueType> *vec) const {

  LOG_DEBUG(this, "LocalMatrix::ExtractRowVector()",
            idx);

  assert(vec != NULL);
  assert(vec->get_size() == this->GetN());
  assert(idx >= 0);

  assert( ( (this->matrix_ == this->matrix_host_)  && (vec->vector_ == vec->vector_host_)) ||
          ( (this->matrix_ == this->matrix_accel_) && (vec->vector_ == vec->vector_accel_) ) );

#ifdef DEBUG_MODE
  this->Check();
#endif

  if (this->GetNnz() > 0) {

    bool err = this->matrix_->ExtractRowVector(idx, vec->vector_);

    if ((err == false) && (this->is_host() == true) && (this->GetFormat() == CSR)) {
      LOG_INFO("Computation of LocalMatrix::ExtractRowVector() failed");
      this->Info();
      FATAL_ERROR(__FILE__, __LINE__);
    }

    if (err == false) {

      LocalMatrix<ValueType> mat_host;
      mat_host.ConvertTo(this->GetFormat());
      mat_host.CopyFrom(*this);

      // Move to host
      vec->MoveToHost();

      // Convert to CSR
      mat_host.ConvertToCSR();

      if (mat_host.matrix_->ExtractRowVector(idx, vec->vector_) == false) {
        LOG_INFO("Computation of LocalMatrix::ExtractRowVector() failed");
        mat_host.Info();
        FATAL_ERROR(__FILE__, __LINE__);
      }

      if (this->GetFormat() != CSR)
        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractRowVector() is performed in CSR format");

      if (this->is_accel() == true) {

        LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::ExtractRowVector() is performed on the host");

        vec->MoveToAccelerator();

      }

    }

  }

}


template class LocalMatrix<double>;
template class LocalMatrix<float>;
#ifdef SUPPORT_COMPLEX
template class LocalMatrix<std::complex<double> >;
template class LocalMatrix<std::complex<float> >;
#endif

}
