#include "../../utils/def.hpp"
#include "host_matrix_dense.hpp"
#include "host_matrix_csr.hpp"
#include "host_conversion.hpp"
#include "host_vector.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/math_functions.hpp"
#include "../matrix_formats_ind.hpp"

#include <math.h>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#endif

#ifdef SUPPORT_MKL
#include <mkl.h>
#include <mkl_spblas.h>
#endif

namespace paralution {

template <typename ValueType>
HostMatrixDENSE<ValueType>::HostMatrixDENSE() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HostMatrixDENSE<ValueType>::HostMatrixDENSE(const Paralution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "HostMatrixDENSE::HostMatrixDENSE()",
            "constructor with local_backend");

  this->mat_.val = NULL;
  this->set_backend(local_backend);

}

template <typename ValueType>
HostMatrixDENSE<ValueType>::~HostMatrixDENSE() {

  LOG_DEBUG(this, "HostMatrixDENSE::~HostMatrixDENSE()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void HostMatrixDENSE<ValueType>::info(void) const {

  LOG_INFO("HostMatrixDENSE<ValueType>");

  if (DENSE_IND_BASE == 0) {

    LOG_INFO("Dense matrix - row-based");
    
  } else {

    assert(DENSE_IND_BASE == 1);
    LOG_INFO("Dense matrix - column-based");
      
  }

}

template <typename ValueType>
void HostMatrixDENSE<ValueType>::Clear() {

  if (this->nnz_ > 0) {

    free_host(&this->mat_.val);

    this->nrow_ = 0;
    this->ncol_ = 0;
    this->nnz_  = 0;

  }

}

template <typename ValueType>
void HostMatrixDENSE<ValueType>::AllocateDENSE(const int nrow, const int ncol) {

  assert( ncol  >= 0);
  assert( nrow  >= 0);

  if (this->nnz_ > 0)
    this->Clear();

  if (nrow*ncol > 0) {

    allocate_host(nrow*ncol, &this->mat_.val);
    set_to_zero_host(nrow*ncol, mat_.val);

    this->nrow_ = nrow;
    this->ncol_ = ncol;
    this->nnz_  = nrow*ncol;

  }

}

template <typename ValueType>
void HostMatrixDENSE<ValueType>::SetDataPtrDENSE(ValueType **val, const int nrow, const int ncol) {

  assert(*val != NULL);
  assert(nrow > 0);
  assert(ncol > 0);

  this->Clear();

  this->nrow_ = nrow;
  this->ncol_ = ncol;
  this->nnz_  = nrow*ncol;

  this->mat_.val = *val;

}

template <typename ValueType>
void HostMatrixDENSE<ValueType>::LeaveDataPtrDENSE(ValueType **val) {

  assert(this->nrow_ > 0);
  assert(this->ncol_ > 0);
  assert(this->nnz_ > 0);
  assert(this->nnz_  == this->nrow_*this->ncol_);

  *val = this->mat_.val;

  this->mat_.val = NULL;

  this->nrow_ = 0;
  this->ncol_ = 0;
  this->nnz_  = 0;

}

template <typename ValueType>
void HostMatrixDENSE<ValueType>::CopyFrom(const BaseMatrix<ValueType> &mat) {

  // copy only in the same format
  assert(this->get_mat_format() == mat.get_mat_format());

  if (const HostMatrixDENSE<ValueType> *cast_mat = dynamic_cast<const HostMatrixDENSE<ValueType>*> (&mat)) {

    this->AllocateDENSE(cast_mat->nrow_, cast_mat->ncol_);

    assert((this->nnz_  == cast_mat->nnz_)  &&
           (this->nrow_ == cast_mat->nrow_) &&
           (this->ncol_ == cast_mat->ncol_) );

    if (this->nnz_ > 0) {

      _set_omp_backend_threads(this->local_backend_, this->nnz_);

#pragma omp parallel for
      for (int j=0; j<this->nnz_; ++j)
        this->mat_.val[j] = cast_mat->mat_.val[j];

    }

  } else {

    // Host matrix knows only host matrices
    // -> dispatching
    mat.CopyTo(this);

  }

}

template <typename ValueType>
void HostMatrixDENSE<ValueType>::CopyTo(BaseMatrix<ValueType> *mat) const {

  mat->CopyFrom(*this);

}

template <typename ValueType>
bool HostMatrixDENSE<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  if (const HostMatrixDENSE<ValueType> *cast_mat = dynamic_cast<const HostMatrixDENSE<ValueType>*> (&mat)) {

    this->CopyFrom(*cast_mat);
    return true;

  }

  if (const HostMatrixCSR<ValueType> *cast_mat = dynamic_cast<const HostMatrixCSR<ValueType>*> (&mat)) {

    this->Clear();

    if (csr_to_dense(this->local_backend_.OpenMP_threads,
                     cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_, cast_mat->mat_, &this->mat_) == true) {

      this->nrow_ = cast_mat->nrow_;
      this->ncol_ = cast_mat->ncol_;
      this->nnz_ = this->nrow_ * this->ncol_;

      return true;

    }

  }

  return false;

}

#ifdef SUPPORT_MKL

template <>
void HostMatrixDENSE<double>::Apply(const BaseVector<double> &in, BaseVector<double> *out) const {

  assert(in.  get_size() >= 0);
  assert(out->get_size() >= 0);
  assert(in.  get_size() == this->ncol_);
  assert(out->get_size() == this->nrow_);

  const HostVector<double> *cast_in = dynamic_cast<const HostVector<double>*> (&in);
  HostVector<double> *cast_out      = dynamic_cast<      HostVector<double>*> (out);

  assert(cast_in != NULL);
  assert(cast_out!= NULL);

  double alpha = double(1.0);
  double beta = double(0.0);
  int nrow = this->nrow_;
  int ncol = this->ncol_;

  if (DENSE_IND_BASE == 0) {

    cblas_dgemv(CblasColMajor, CblasNoTrans,
                nrow, ncol,
                alpha, this->mat_.val,
                nrow,
                cast_in->vec_, 1, beta,
                cast_out->vec_, 1);

  } else {

    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                nrow, ncol,
                alpha, this->mat_.val,
                ncol,
                cast_in->vec_, 1, beta,
                cast_out->vec_, 1);

  }

}

template <>
void HostMatrixDENSE<float>::Apply(const BaseVector<float> &in, BaseVector<float> *out) const {

  assert(in.  get_size() >= 0);
  assert(out->get_size() >= 0);
  assert(in.  get_size() == this->ncol_);
  assert(out->get_size() == this->nrow_);

  const HostVector<float> *cast_in = dynamic_cast<const HostVector<float>*> (&in);
  HostVector<float> *cast_out      = dynamic_cast<      HostVector<float>*> (out);

  assert(cast_in != NULL);
  assert(cast_out!= NULL);

  float alpha = float(1.0);
  float beta = float(0.0);
  int nrow = this->nrow_;
  int ncol = this->ncol_;

  if (DENSE_IND_BASE == 0) {

    cblas_sgemv(CblasColMajor, CblasNoTrans,
                nrow, ncol,
                alpha, this->mat_.val,
                nrow,
                cast_in->vec_, 1, beta,
                cast_out->vec_, 1);

  } else {

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                nrow, ncol,
                alpha, this->mat_.val,
                ncol,
                cast_in->vec_, 1, beta,
                cast_out->vec_, 1);

  }

}

template <>
void HostMatrixDENSE<std::complex<double> >::Apply(const BaseVector<std::complex<double> > &in, BaseVector<std::complex<double> > *out) const {

  assert(in.  get_size() >= 0);
  assert(out->get_size() >= 0);
  assert(in.  get_size() == this->ncol_);
  assert(out->get_size() == this->nrow_);

  const HostVector<std::complex<double> > *cast_in = dynamic_cast<const HostVector<std::complex<double> >*> (&in);
  HostVector<std::complex<double> > *cast_out      = dynamic_cast<      HostVector<std::complex<double> >*> (out);

  assert(cast_in != NULL);
  assert(cast_out!= NULL);

  std::complex<double> alpha = std::complex<double>(1.0, 0.0);
  std::complex<double> beta = std::complex<double>(0.0, 0.0);
  int nrow = this->nrow_;
  int ncol = this->ncol_;

  if (DENSE_IND_BASE == 0) {

    cblas_zgemv(CblasColMajor, CblasNoTrans,
                nrow, ncol,
                &alpha, this->mat_.val,
                nrow,
                cast_in->vec_, 1, &beta,
                cast_out->vec_, 1);

  } else {

    cblas_zgemv(CblasRowMajor, CblasNoTrans,
                nrow, ncol,
                &alpha, this->mat_.val,
                ncol,
                cast_in->vec_, 1, &beta,
                cast_out->vec_, 1);

  }

}

template <>
void HostMatrixDENSE<std::complex<float> >::Apply(const BaseVector<std::complex<float> > &in, BaseVector<std::complex<float> > *out) const {

  assert(in.  get_size() >= 0);
  assert(out->get_size() >= 0);
  assert(in.  get_size() == this->ncol_);
  assert(out->get_size() == this->nrow_);

  const HostVector<std::complex<float> > *cast_in = dynamic_cast<const HostVector<std::complex<float> >*> (&in);
  HostVector<std::complex<float> > *cast_out      = dynamic_cast<      HostVector<std::complex<float> >*> (out);

  assert(cast_in != NULL);
  assert(cast_out!= NULL);

  std::complex<float> alpha = std::complex<float>(1.0, 0.0);
  std::complex<float> beta = std::complex<float>(0.0, 0.0);
  int nrow = this->nrow_;
  int ncol = this->ncol_;

  if (DENSE_IND_BASE == 0) {

    cblas_cgemv(CblasColMajor, CblasNoTrans,
                nrow, ncol,
                &alpha, this->mat_.val,
                nrow,
                cast_in->vec_, 1, &beta,
                cast_out->vec_, 1);

  } else {

    cblas_cgemv(CblasRowMajor, CblasNoTrans,
                nrow, ncol,
                &alpha, this->mat_.val,
                ncol,
                cast_in->vec_, 1, &beta,
                cast_out->vec_, 1);

  }

}

#else

template <typename ValueType>
void HostMatrixDENSE<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  assert(in.  get_size() >= 0);
  assert(out->get_size() >= 0);
  assert(in.  get_size() == this->ncol_);
  assert(out->get_size() == this->nrow_);

  const HostVector<ValueType> *cast_in = dynamic_cast<const HostVector<ValueType>*> (&in);
  HostVector<ValueType> *cast_out      = dynamic_cast<      HostVector<ValueType>*> (out);

  assert(cast_in != NULL);
  assert(cast_out!= NULL);

  _set_omp_backend_threads(this->local_backend_, this->nnz_);

#pragma omp parallel for
  for (int ai=0; ai<this->nrow_; ++ai) {
    cast_out->vec_[ai] = ValueType(0.0);
      for (int aj=0; aj<this->ncol_; ++aj)
        cast_out->vec_[ai] += this->mat_.val[DENSE_IND(ai,aj,this->nrow_,this->ncol_)] * cast_in->vec_[aj];
  }

}

#endif

#ifdef SUPPORT_MKL

template <>
void HostMatrixDENSE<double>::ApplyAdd(const BaseVector<double> &in, const double scalar,
                                       BaseVector<double> *out) const {

  assert(in.  get_size() >= 0);
  assert(out->get_size() >= 0);
  assert(in.  get_size() == this->ncol_);
  assert(out->get_size() == this->nrow_);

  const HostVector<double> *cast_in = dynamic_cast<const HostVector<double>*> (&in);
  HostVector<double> *cast_out      = dynamic_cast<      HostVector<double>*> (out);

  assert(cast_in != NULL);
  assert(cast_out!= NULL);

  double beta = double(1.0);
  int nrow = this->nrow_;
  int ncol = this->ncol_;

  if (DENSE_IND_BASE == 0) {

    cblas_dgemv(CblasColMajor, CblasNoTrans,
                nrow, ncol,
                scalar, this->mat_.val,
                nrow,
                cast_in->vec_, 1, beta,
                cast_out->vec_, 1);

  } else {

    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                nrow, ncol,
                scalar, this->mat_.val,
                ncol,
                cast_in->vec_, 1, beta,
                cast_out->vec_, 1);

  }

}

template <>
void HostMatrixDENSE<float>::ApplyAdd(const BaseVector<float> &in, const float scalar,
                                      BaseVector<float> *out) const {

  assert(in.  get_size() >= 0);
  assert(out->get_size() >= 0);
  assert(in.  get_size() == this->ncol_);
  assert(out->get_size() == this->nrow_);

  const HostVector<float> *cast_in = dynamic_cast<const HostVector<float>*> (&in);
  HostVector<float> *cast_out      = dynamic_cast<      HostVector<float>*> (out);

  assert(cast_in != NULL);
  assert(cast_out!= NULL);

  float beta = float(1.0);
  int nrow = this->nrow_;
  int ncol = this->ncol_;

  if (DENSE_IND_BASE == 0) {

    cblas_sgemv(CblasColMajor, CblasNoTrans,
                nrow, ncol,
                scalar, this->mat_.val,
                nrow,
                cast_in->vec_, 1, beta,
                cast_out->vec_, 1);

  } else {

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                nrow, ncol,
                scalar, this->mat_.val,
                ncol,
                cast_in->vec_, 1, beta,
                cast_out->vec_, 1);

  }

}

template <>
void HostMatrixDENSE<std::complex<double> >::ApplyAdd(const BaseVector<std::complex<double> > &in,
                                                      const std::complex<double> scalar,
                                                      BaseVector<std::complex<double> > *out) const {

  assert(in.  get_size() >= 0);
  assert(out->get_size() >= 0);
  assert(in.  get_size() == this->ncol_);
  assert(out->get_size() == this->nrow_);

  const HostVector<std::complex<double> > *cast_in = dynamic_cast<const HostVector<std::complex<double> >*> (&in);
  HostVector<std::complex<double> > *cast_out      = dynamic_cast<      HostVector<std::complex<double> >*> (out);

  assert(cast_in != NULL);
  assert(cast_out!= NULL);

  std::complex<double> beta = std::complex<double>(1.0, 0.0);
  int nrow = this->nrow_;
  int ncol = this->ncol_;

  if (DENSE_IND_BASE == 0) {

    cblas_zgemv(CblasColMajor, CblasNoTrans,
                nrow, ncol,
                &scalar, this->mat_.val,
                nrow,
                cast_in->vec_, 1, &beta,
                cast_out->vec_, 1);

  } else {

    cblas_zgemv(CblasRowMajor, CblasNoTrans,
                nrow, ncol,
                &scalar, this->mat_.val,
                ncol,
                cast_in->vec_, 1, &beta,
                cast_out->vec_, 1);

  }

}

template <>
void HostMatrixDENSE<std::complex<float> >::ApplyAdd(const BaseVector<std::complex<float> > &in,
                                                     const std::complex<float> scalar,
                                                     BaseVector<std::complex<float> > *out) const {

  assert(in.  get_size() >= 0);
  assert(out->get_size() >= 0);
  assert(in.  get_size() == this->ncol_);
  assert(out->get_size() == this->nrow_);

  const HostVector<std::complex<float> > *cast_in = dynamic_cast<const HostVector<std::complex<float> >*> (&in);
  HostVector<std::complex<float> > *cast_out      = dynamic_cast<      HostVector<std::complex<float> >*> (out);

  assert(cast_in != NULL);
  assert(cast_out!= NULL);

  std::complex<float> beta = std::complex<float>(1.0, 0.0);
  int nrow = this->nrow_;
  int ncol = this->ncol_;

  if (DENSE_IND_BASE == 0) {

    cblas_cgemv(CblasColMajor, CblasNoTrans,
                nrow, ncol,
                &scalar, this->mat_.val,
                nrow,
                cast_in->vec_, 1, &beta,
                cast_out->vec_, 1);

  } else {

    cblas_cgemv(CblasRowMajor, CblasNoTrans,
                nrow, ncol,
                &scalar, this->mat_.val,
                ncol,
                cast_in->vec_, 1, &beta,
                cast_out->vec_, 1);

  }

}

#else

template <typename ValueType>
void HostMatrixDENSE<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                                          BaseVector<ValueType> *out) const {

  if (this->nnz_ > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->ncol_);
    assert(out->get_size() == this->nrow_);

    const HostVector<ValueType> *cast_in = dynamic_cast<const HostVector<ValueType>*> (&in);
    HostVector<ValueType> *cast_out      = dynamic_cast<      HostVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

  _set_omp_backend_threads(this->local_backend_, this->nnz_);

#pragma omp parallel for
  for (int ai=0; ai<this->nrow_; ++ai)
    for (int aj=0; aj<this->ncol_; ++aj)
      cast_out->vec_[ai] += scalar * this->mat_.val[DENSE_IND(ai,aj,this->nrow_,this->ncol_)] * cast_in->vec_[aj];

  }

}

#endif

#ifdef SUPPORT_MKL

template <>
bool HostMatrixDENSE<double>::MatMatMult(const BaseMatrix<double> &A, const BaseMatrix<double> &B) {

  assert((this != &A) && (this != &B));

  const HostMatrixDENSE<double> *cast_mat_A = dynamic_cast<const HostMatrixDENSE<double>*> (&A);
  const HostMatrixDENSE<double> *cast_mat_B = dynamic_cast<const HostMatrixDENSE<double>*> (&B);

  assert(cast_mat_A != NULL);
  assert(cast_mat_B != NULL);
  assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

  int m = cast_mat_A->nrow_;
  int n = cast_mat_B->ncol_;
  int k = cast_mat_A->ncol_;

  double alpha = double(1.0);
  double beta  = double(0.0);

  if (DENSE_IND_BASE == 0) {

    cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k, alpha, cast_mat_A->mat_.val, m,
                 cast_mat_B->mat_.val, k, beta, this->mat_.val, m);

  } else {

    cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k, alpha, cast_mat_A->mat_.val, k,
                 cast_mat_B->mat_.val, n, beta, this->mat_.val, m);

  }

  return true;

}

template <>
bool HostMatrixDENSE<float>::MatMatMult(const BaseMatrix<float> &A, const BaseMatrix<float> &B) {

  assert((this != &A) && (this != &B));

  const HostMatrixDENSE<float> *cast_mat_A = dynamic_cast<const HostMatrixDENSE<float>*> (&A);
  const HostMatrixDENSE<float> *cast_mat_B = dynamic_cast<const HostMatrixDENSE<float>*> (&B);

  assert(cast_mat_A != NULL);
  assert(cast_mat_B != NULL);
  assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

  int m = cast_mat_A->nrow_;
  int n = cast_mat_B->ncol_;
  int k = cast_mat_A->ncol_;

  float alpha = float(1.0);
  float beta  = float(0.0);

  if (DENSE_IND_BASE == 0) {

    cblas_sgemm (CblasColMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k, alpha, cast_mat_A->mat_.val, m,
                 cast_mat_B->mat_.val, k, beta, this->mat_.val, m);

  } else {

    cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k, alpha, cast_mat_A->mat_.val, k,
                 cast_mat_B->mat_.val, n, beta, this->mat_.val, m);

  }

  return true;

}

template <>
bool HostMatrixDENSE<std::complex<double> >::MatMatMult(const BaseMatrix<std::complex<double> > &A,
                                                        const BaseMatrix<std::complex<double> > &B) {

  assert((this != &A) && (this != &B));

  const HostMatrixDENSE<std::complex<double> > *cast_mat_A = dynamic_cast<const HostMatrixDENSE<std::complex<double> >*> (&A);
  const HostMatrixDENSE<std::complex<double> > *cast_mat_B = dynamic_cast<const HostMatrixDENSE<std::complex<double> >*> (&B);

  assert(cast_mat_A != NULL);
  assert(cast_mat_B != NULL);
  assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

  int m = cast_mat_A->nrow_;
  int n = cast_mat_B->ncol_;
  int k = cast_mat_A->ncol_;

  std::complex<double> alpha = std::complex<double>(1.0, 0.0);
  std::complex<double> beta  = std::complex<double>(0.0, 0.0);

  if (DENSE_IND_BASE == 0) {

    cblas_zgemm (CblasColMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k, &alpha, cast_mat_A->mat_.val, m,
                 cast_mat_B->mat_.val, k, &beta, this->mat_.val, m);

  } else {

    cblas_zgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k, &alpha, cast_mat_A->mat_.val, k,
                 cast_mat_B->mat_.val, n, &beta, this->mat_.val, m);

  }

  return true;

}

template <>
bool HostMatrixDENSE<std::complex<float> >::MatMatMult(const BaseMatrix<std::complex<float> > &A,
                                                       const BaseMatrix<std::complex<float> > &B) {

  assert((this != &A) && (this != &B));

  const HostMatrixDENSE<std::complex<float> > *cast_mat_A = dynamic_cast<const HostMatrixDENSE<std::complex<float> >*> (&A);
  const HostMatrixDENSE<std::complex<float> > *cast_mat_B = dynamic_cast<const HostMatrixDENSE<std::complex<float> >*> (&B);

  assert(cast_mat_A != NULL);
  assert(cast_mat_B != NULL);
  assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

  int m = cast_mat_A->nrow_;
  int n = cast_mat_B->ncol_;
  int k = cast_mat_A->ncol_;

  std::complex<float> alpha = std::complex<float>(1.0, 0.0);
  std::complex<float> beta  = std::complex<float>(0.0, 0.0);

  if (DENSE_IND_BASE == 0) {

    cblas_cgemm (CblasColMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k, &alpha, cast_mat_A->mat_.val, m,
                 cast_mat_B->mat_.val, k, &beta, this->mat_.val, m);

  } else {

    cblas_cgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k, &alpha, cast_mat_A->mat_.val, k,
                 cast_mat_B->mat_.val, n, &beta, this->mat_.val, m);

  }

  return true;

}

#else

template <typename ValueType>
bool HostMatrixDENSE<ValueType>::MatMatMult(const BaseMatrix<ValueType> &A, const BaseMatrix<ValueType> &B) {

  assert((this != &A) && (this != &B));

  const HostMatrixDENSE<ValueType> *cast_mat_A = dynamic_cast<const HostMatrixDENSE<ValueType>*> (&A);
  const HostMatrixDENSE<ValueType> *cast_mat_B = dynamic_cast<const HostMatrixDENSE<ValueType>*> (&B);

  assert(cast_mat_A != NULL);
  assert(cast_mat_B != NULL);
  assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

#pragma omp parallel for
  for (int i=0; i<cast_mat_A->nrow_; ++i) {

    for (int j=0; j<cast_mat_B->ncol_; ++j) {

      ValueType sum = ValueType(0.0);

      for (int k=0; k<cast_mat_A->ncol_; ++k) {

        sum += cast_mat_A->mat_.val[DENSE_IND(i,k,cast_mat_A->nrow_,cast_mat_A->ncol_)]
             * cast_mat_B->mat_.val[DENSE_IND(k,j,cast_mat_B->nrow_,cast_mat_B->ncol_)];

      }

      this->mat_.val[DENSE_IND(i,j,cast_mat_A->nrow_,cast_mat_B->ncol_)] = sum;

    }

  }

  return true;

}

#endif

template <typename ValueType>
bool HostMatrixDENSE<ValueType>::Householder(const int idx, ValueType &beta, BaseVector<ValueType> *vec) const {

  HostVector<ValueType> *cast_vec = dynamic_cast<HostVector<ValueType>*> (vec);
  assert(cast_vec != NULL);
  assert(cast_vec->get_size() >= this->nrow_-idx);

  ValueType s  = ValueType(0.0);

  for (int i=1; i<this->nrow_-idx; ++i)
    cast_vec->vec_[i] = this->mat_.val[DENSE_IND(i+idx, idx, this->nrow_, this->ncol_)];

  for (int i=idx+1; i<this->nrow_; ++i)
    s += cast_vec->vec_[i-idx] * cast_vec->vec_[i-idx];

  if (s == ValueType(0.0)) {

    beta = ValueType(0.0);

  } else {

    ValueType aii = this->mat_.val[DENSE_IND(idx, idx, this->nrow_, this->ncol_)];

    if (aii <= ValueType(0.0))
      aii -= sqrt(aii * aii + s);
    else
      aii += sqrt(aii * aii + s);

    ValueType squared = aii * aii;
    beta = ValueType(2.0) * squared / (s + squared);

    aii = ValueType(1.0) / aii;
    for (int i=1; i<this->nrow_-idx; ++i)
      cast_vec->vec_[i] *= aii;

  }

  return true;

}

template <typename ValueType>
bool HostMatrixDENSE<ValueType>::QRDecompose(void) {

  assert(this->nrow_ > 0);
  assert(this->ncol_ > 0);
  assert(this->nnz_ > 0);

  int size = (this->nrow_ < this->ncol_) ? this->nrow_ : this->ncol_;
  ValueType beta;
  HostVector<ValueType> v(this->local_backend_);
  v.Allocate(this->nrow_);

  for (int i=0; i<size; ++i) {

    this->Householder(i, beta, &v);

    if (beta != ValueType(0.0)) {

      for (int aj=i; aj<this->ncol_; ++aj) {

        ValueType sum = this->mat_.val[DENSE_IND(i, aj, this->nrow_, this->ncol_)];
        for (int ai=i+1; ai<this->nrow_; ++ai)
          sum += v.vec_[ai-i] * this->mat_.val[DENSE_IND(ai, aj, this->nrow_, this->ncol_)];
        sum *= beta;

        this->mat_.val[DENSE_IND(i, aj, this->nrow_, this->ncol_)] -= sum;
        for (int ai=i+1; ai<this->nrow_; ++ai)
          this->mat_.val[DENSE_IND(ai, aj, this->nrow_, this->ncol_)] -= sum * v.vec_[ai-i];

      }

      for (int k=i+1; k<this->nrow_; ++k)
        this->mat_.val[DENSE_IND(k, i, this->nrow_, this->ncol_)] = v.vec_[k-i];

    }

  }

  return true;

}

template <typename ValueType>
bool HostMatrixDENSE<ValueType>::QRSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  assert(in.  get_size() >= 0);
  assert(out->get_size() >= 0);
  assert(in.  get_size() == this->nrow_);
  assert(out->get_size() == this->ncol_);

  HostVector<ValueType> *cast_out = dynamic_cast<HostVector<ValueType>*>(out);

  assert(cast_out!= NULL);

  HostVector<ValueType> copy_in(this->local_backend_);
  copy_in.CopyFrom(in);

  int size = (this->nrow_ < this->ncol_) ? this->nrow_ : this->ncol_;

  // Apply Q^T on copy_in
  for (int i=0; i<size; ++i) {

    ValueType sum = ValueType(1.0);
    for (int j=i+1; j<this->nrow_; ++j) {
      sum += this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)]
           * this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)];
    }

    sum = ValueType(2.0) / sum;

    if (sum != ValueType(2.0)) {

      ValueType sum2 = copy_in.vec_[i];
      for (int j=i+1; j<this->nrow_; ++j)
        sum2 += this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)]
              * copy_in.vec_[j];

      sum2 *= sum;
      copy_in.vec_[i] -= sum2;
      for (int j=i+1; j<this->nrow_; ++j)
        copy_in.vec_[j] -= sum2 * this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)];

    }

  }

  // Backsolve Rx = Q^T b
  for (int i=size-1; i>=0; --i) {

    ValueType sum = ValueType(0.0);
    for (int j=i+1; j<this->ncol_; ++j)
      sum += this->mat_.val[DENSE_IND(i, j, this->nrow_, this->ncol_)] * cast_out->vec_[j];

    cast_out->vec_[i] = (copy_in.vec_[i] - sum) / this->mat_.val[DENSE_IND(i, i, this->nrow_, this->ncol_)];

  }

  return true;

}

template <typename ValueType>
bool HostMatrixDENSE<ValueType>::Invert(void) {

  assert(this->nrow_ > 0);
  assert(this->ncol_ > 0);
  assert(this->nnz_ > 0);
  assert(this->nrow_ == this->ncol_);

  ValueType *val = NULL;
  allocate_host(this->nrow_ * this->ncol_, &val);

  this->QRDecompose();

#pragma omp parallel for
  for (int i=0; i<this->nrow_; ++i) {

    HostVector<ValueType> sol(this->local_backend_);
    HostVector<ValueType> rhs(this->local_backend_);
    sol.Allocate(this->nrow_);
    rhs.Allocate(this->nrow_);

    rhs.vec_[i] = ValueType(1.0);

    this->QRSolve(rhs, &sol);

    for (int j=0; j<this->ncol_; ++j)
      val[DENSE_IND(j, i, this->nrow_, this->ncol_)] = sol.vec_[j];

  }

  free_host(&this->mat_.val);
  this->mat_.val = val;

  return true;

}

template <typename ValueType>
bool HostMatrixDENSE<ValueType>::LUFactorize(void) {

  assert(this->nrow_ > 0);
  assert(this->ncol_ > 0);
  assert(this->nnz_ > 0);
  assert(this->nrow_ == this->ncol_);

  for (int i=0; i<this->nrow_-1; ++i) {
    for (int j=i+1; j<this->nrow_; ++j) {

      this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)] /=
      this->mat_.val[DENSE_IND(i, i, this->nrow_, this->ncol_)];

      for (int k=i+1; k<this->ncol_; ++k)
        this->mat_.val[DENSE_IND(j, k, this->nrow_, this->ncol_)] -=
        this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)] *
        this->mat_.val[DENSE_IND(i, k, this->nrow_, this->ncol_)];

    }
  }

  return true;

}

template <typename ValueType>
bool HostMatrixDENSE<ValueType>::LUSolve(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  assert(in.  get_size() >= 0);
  assert(out->get_size() >= 0);
  assert(in.  get_size() == this->nrow_);
  assert(out->get_size() == this->ncol_);

  HostVector<ValueType> *cast_out = dynamic_cast<HostVector<ValueType>*>(out);
  const HostVector<ValueType> *cast_in = dynamic_cast<const HostVector<ValueType>*>(&in);

  assert(cast_out!= NULL);

  // fill solution vector
  for (int i=0; i<this->nrow_; ++i)
    cast_out->vec_[i] = cast_in->vec_[i];

  // forward sweeps
  for (int i=0; i<this->nrow_-1; ++i) {
    for (int j=i+1; j<this->nrow_; ++j)
      cast_out->vec_[j] -= cast_out->vec_[i] * this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)];
  }

  // backward sweeps
  for (int i=this->nrow_-1; i>=0; --i) {
    cast_out->vec_[i] /= this->mat_.val[DENSE_IND(i, i, this->nrow_, this->ncol_)];
    for (int j=0; j<i; ++j)
      cast_out->vec_[j] -= cast_out->vec_[i] * this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)];
  }

  return true;

}

template <typename ValueType>
bool HostMatrixDENSE<ValueType>::ReplaceColumnVector(const int idx, const BaseVector<ValueType> &vec) {

  assert(vec.get_size() == this->nrow_);

  if (this->get_nnz() > 0) {

    const HostVector<ValueType> *cast_vec = dynamic_cast<const HostVector<ValueType>*> (&vec);
    assert(cast_vec != NULL);

    _set_omp_backend_threads(this->local_backend_, this->nrow_);

#pragma omp parallel for
    for (int i=0; i<this->nrow_; ++i)
      this->mat_.val[DENSE_IND(i, idx, this->nrow_, this->ncol_)] = cast_vec->vec_[i];

  }

  return true;

}

template <typename ValueType>
bool HostMatrixDENSE<ValueType>::ReplaceRowVector(const int idx, const BaseVector<ValueType> &vec) {

  assert(vec.get_size() == this->ncol_);

  if (this->get_nnz() > 0) {

    const HostVector<ValueType> *cast_vec = dynamic_cast<const HostVector<ValueType>*> (&vec);
    assert(cast_vec != NULL);

    _set_omp_backend_threads(this->local_backend_, this->ncol_);

#pragma omp parallel for
    for (int i=0; i<this->ncol_; ++i)
      this->mat_.val[DENSE_IND(idx, i, this->nrow_, this->ncol_)] = cast_vec->vec_[i];

  }

  return true;

}

template <typename ValueType>
bool HostMatrixDENSE<ValueType>::ExtractColumnVector(const int idx, BaseVector<ValueType> *vec) const {

  assert(vec != NULL);
  assert(vec->get_size() == this->nrow_);

  if (this->get_nnz() > 0) {

    HostVector<ValueType> *cast_vec = dynamic_cast<HostVector<ValueType>*> (vec);
    assert(cast_vec != NULL);

    _set_omp_backend_threads(this->local_backend_, this->nrow_);

#pragma omp parallel for
    for (int i=0; i<this->nrow_; ++i)
      cast_vec->vec_[i] = this->mat_.val[DENSE_IND(i, idx, this->nrow_, this->ncol_)];

  }

  return true;

}

template <typename ValueType>
bool HostMatrixDENSE<ValueType>::ExtractRowVector(const int idx, BaseVector<ValueType> *vec) const {

  assert(vec != NULL);
  assert(vec->get_size() == this->ncol_);

  if (this->get_nnz() > 0) {

    HostVector<ValueType> *cast_vec = dynamic_cast<HostVector<ValueType>*> (vec);
    assert(cast_vec != NULL);

    _set_omp_backend_threads(this->local_backend_, this->nrow_);

#pragma omp parallel for
    for (int i=0; i<this->nrow_; ++i)
      cast_vec->vec_[i] = this->mat_.val[DENSE_IND(idx, i, this->nrow_, this->ncol_)];

  }

  return true;

}


template class HostMatrixDENSE<double>;
template class HostMatrixDENSE<float>;
#ifdef SUPPORT_COMPLEX
template class HostMatrixDENSE<std::complex<double> >;
template class HostMatrixDENSE<std::complex<float> >;
#endif

}
