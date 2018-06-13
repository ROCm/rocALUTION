#include "../../utils/def.hpp"
#include "host_matrix_hyb.hpp"
#include "host_matrix_csr.hpp"
#include "host_conversion.hpp"
#include "host_vector.hpp"
#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "../matrix_formats_ind.hpp"

#include <complex>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#endif

namespace rocalution {

template <typename ValueType>
HostMatrixHYB<ValueType>::HostMatrixHYB() {

  // no default constructors
  LOG_INFO("no default constructor");
  FATAL_ERROR(__FILE__, __LINE__);

}

template <typename ValueType>
HostMatrixHYB<ValueType>::HostMatrixHYB(const Rocalution_Backend_Descriptor local_backend) {

  LOG_DEBUG(this, "HostMatrixHYB::HostMatrixHYB()",
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
HostMatrixHYB<ValueType>::~HostMatrixHYB() {

  LOG_DEBUG(this, "HostMatrixHYB::~HostMatrixHYB()",
            "destructor");

  this->Clear();

}

template <typename ValueType>
void HostMatrixHYB<ValueType>::Info(void) const {

  LOG_INFO("HostMatrixHYB<ValueType>" <<
           " ELL nnz=" << this->ell_nnz_ <<
           " ELL max row=" << this->mat_.ELL.max_row <<
           " COO nnz=" << this->coo_nnz_);

}

template <typename ValueType>
void HostMatrixHYB<ValueType>::Clear() {

  if (this->nnz_ > 0) {

    if (this->ell_nnz_ > 0)
    {
      free_host(&this->mat_.ELL.val);
      free_host(&this->mat_.ELL.col);

      this->mat_.ELL.max_row = 0;
      this->ell_nnz_ = 0;
    }

    if (this->coo_nnz_ > 0)
    {
      free_host(&this->mat_.COO.row);
      free_host(&this->mat_.COO.col);
      free_host(&this->mat_.COO.val);

      this->coo_nnz_ = 0;
    }
  }

  this->nrow_ = 0;
  this->ncol_ = 0;
  this->nnz_  = 0;
}

template <typename ValueType>
void HostMatrixHYB<ValueType>::AllocateHYB(const int ell_nnz, const int coo_nnz, const int ell_max_row,
                                           const int nrow, const int ncol) {

  assert(ell_nnz   >= 0);
  assert(coo_nnz   >= 0);
  assert(ell_max_row >= 0);

  assert(ncol  >= 0);
  assert(nrow  >= 0);

  if (this->nnz_ > 0)
    this->Clear();

  this->nnz_ = 0;

  if (ell_nnz > 0)
  {
    // ELL
    assert(ell_nnz == ell_max_row*nrow);

    allocate_host(ell_nnz, &this->mat_.ELL.val);
    allocate_host(ell_nnz, &this->mat_.ELL.col);

    set_to_zero_host(ell_nnz, this->mat_.ELL.val);
    set_to_zero_host(ell_nnz, this->mat_.ELL.col);

    this->mat_.ELL.max_row = ell_max_row;
    this->ell_nnz_ = ell_nnz;
    this->nnz_ += ell_nnz;
  }

  if (coo_nnz > 0)
  {
    // COO
    allocate_host(coo_nnz, &this->mat_.COO.row);
    allocate_host(coo_nnz, &this->mat_.COO.col);
    allocate_host(coo_nnz, &this->mat_.COO.val);

    set_to_zero_host(coo_nnz, this->mat_.COO.row);
    set_to_zero_host(coo_nnz, this->mat_.COO.col);
    set_to_zero_host(coo_nnz, this->mat_.COO.val);

    this->coo_nnz_ = coo_nnz;
    this->nnz_ += coo_nnz;
  }

  this->nrow_ = nrow;
  this->ncol_ = ncol;
}

template <typename ValueType>
void HostMatrixHYB<ValueType>::CopyFrom(const BaseMatrix<ValueType> &mat) {

  // copy only in the same format
  assert(this->GetMatFormat() == mat.GetMatFormat());

  if (const HostMatrixHYB<ValueType> *cast_mat = dynamic_cast<const HostMatrixHYB<ValueType>*> (&mat)) {

    this->AllocateHYB(cast_mat->ell_nnz_, cast_mat->coo_nnz_, cast_mat->mat_.ELL.max_row,
                      cast_mat->nrow_, cast_mat->ncol_);

    assert((this->nnz_      == cast_mat->nnz_) &&
           (this->ell_nnz_  == cast_mat->ell_nnz_) &&
           (this->coo_nnz_  == cast_mat->coo_nnz_) &&
           (this->nrow_     == cast_mat->nrow_) &&
           (this->ncol_     == cast_mat->ncol_) );

    if (this->ell_nnz_ > 0) {

      // ELL

      for (int i=0; i<this->ell_nnz_; ++i)
        this->mat_.ELL.col[i] = cast_mat->mat_.ELL.col[i];

      for (int i=0; i<this->ell_nnz_; ++i)
        this->mat_.ELL.val[i] = cast_mat->mat_.ELL.val[i];

    }

    if (this->coo_nnz_ > 0) {

      // COO
      for (int i=0; i<this->coo_nnz_; ++i)
        this->mat_.COO.row[i] = cast_mat->mat_.COO.row[i];

      for (int i=0; i<this->coo_nnz_; ++i)
        this->mat_.COO.col[i] = cast_mat->mat_.COO.col[i];

      for (int i=0; i<this->coo_nnz_; ++i)
        this->mat_.COO.val[i] = cast_mat->mat_.COO.val[i];

    }

  } else {

    // Host matrix knows only host matrices
    // -> dispatching
    mat.CopyTo(this);

  }

}

template <typename ValueType>
void HostMatrixHYB<ValueType>::CopyTo(BaseMatrix<ValueType> *mat) const {

  mat->CopyFrom(*this);

}

template <typename ValueType>
bool HostMatrixHYB<ValueType>::ConvertFrom(const BaseMatrix<ValueType> &mat) {

  this->Clear();

  // empty matrix is empty matrix
  if (mat.get_nnz() == 0)
    return true;

  if (const HostMatrixHYB<ValueType> *cast_mat = dynamic_cast<const HostMatrixHYB<ValueType>*> (&mat)) {

    this->CopyFrom(*cast_mat);
    return true;

  }

  if (const HostMatrixCSR<ValueType> *cast_mat = dynamic_cast<const HostMatrixCSR<ValueType>*> (&mat)) {

    this->Clear();

    int nnz = 0;
    int coo_nnz = 0;
    int ell_nnz = 0;

    if (csr_to_hyb(this->local_backend_.OpenMP_threads,
                   cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_,
                   cast_mat->mat_, &this->mat_,
                   &nnz, &ell_nnz, &coo_nnz) == true) {

      this->nrow_    = cast_mat->nrow_;
      this->ncol_    = cast_mat->ncol_;
      this->nnz_     = nnz;
      this->ell_nnz_ = ell_nnz;
      this->coo_nnz_ = coo_nnz;

      return true;

    }

  }

  return false;

}

template <typename ValueType>
void HostMatrixHYB<ValueType>::Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const {

  if (this->nnz_ > 0) {

    assert(in.  get_size() >= 0);
    assert(out->get_size() >= 0);
    assert(in.  get_size() == this->ncol_);
    assert(out->get_size() == this->nrow_);

    const HostVector<ValueType> *cast_in = dynamic_cast<const HostVector<ValueType>*> (&in);
    HostVector<ValueType> *cast_out      = dynamic_cast<      HostVector<ValueType>*> (out);

    assert(cast_in != NULL);
    assert(cast_out!= NULL);

    _set_omp_backend_threads(this->local_backend_, this->nrow_);

    // ELL
    if (this->ell_nnz_ > 0) {

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int ai=0; ai<this->nrow_; ++ai) {
        cast_out->vec_[ai] = ValueType(0.0);

        for (int n=0; n<this->mat_.ELL.max_row; ++n) {

          int aj = ELL_IND(ai, n, this->nrow_, this->mat_.ELL.max_row);

          if ((this->mat_.ELL.col[aj] >= 0) && (this->mat_.ELL.col[aj] < this->ncol_))
            cast_out->vec_[ai] += this->mat_.ELL.val[aj] * cast_in->vec_[ this->mat_.ELL.col[aj] ];

        }

      }

    }

    // COO
    if (this->coo_nnz_ > 0) {

      for (int i=0; i<this->coo_nnz_; ++i)
        cast_out->vec_[this->mat_.COO.row[i] ] += this->mat_.COO.val[i] * cast_in->vec_[ this->mat_.COO.col[i] ];

    }

  }

}

template <typename ValueType>
void HostMatrixHYB<ValueType>::ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
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

    _set_omp_backend_threads(this->local_backend_, this->nrow_);

    // ELL
    if (this->ell_nnz_ > 0) {

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int ai=0; ai<this->nrow_; ++ai) {

        for (int n=0; n<this->mat_.ELL.max_row; ++n) {

          int aj = ELL_IND(ai, n, this->nrow_, this->mat_.ELL.max_row);

          if ((this->mat_.ELL.col[aj] >= 0) && (this->mat_.ELL.col[aj] < this->ncol_))
            cast_out->vec_[ai] += scalar*this->mat_.ELL.val[aj] * cast_in->vec_[ this->mat_.ELL.col[aj] ];

        }

      }

    }

    // COO
    if (this->coo_nnz_ > 0) {

      for (int i=0; i<this->coo_nnz_; ++i)
        cast_out->vec_[this->mat_.COO.row[i] ] += scalar*this->mat_.COO.val[i] * cast_in->vec_[ this->mat_.COO.col[i] ];

    }

  }

}


template class HostMatrixHYB<double>;
template class HostMatrixHYB<float>;
#ifdef SUPPORT_COMPLEX
template class HostMatrixHYB<std::complex<float> >;
template class HostMatrixHYB<std::complex<double> >;
#endif

}
