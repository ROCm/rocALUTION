#include "../../utils/def.hpp"
#include "host_conversion.hpp"
#include "../matrix_formats.hpp"
#include "../matrix_formats_ind.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/log.hpp"

#include <stdlib.h>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#endif

namespace rocalution {

template <typename ValueType, typename IndexType>
bool csr_to_dense(const int omp_threads,
                  const IndexType nnz, const IndexType nrow, const IndexType ncol,
                  const MatrixCSR<ValueType, IndexType> &src,
                  MatrixDENSE<ValueType> *dst) {

  assert(nnz  > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  omp_set_num_threads(omp_threads);

  allocate_host(nrow*ncol, &dst->val);
  set_to_zero_host(nrow*ncol, dst->val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (IndexType i = 0; i < nrow; ++i)
    for(IndexType j = src.row_offset[i]; j < src.row_offset[i+1]; ++j)
      dst->val[DENSE_IND(i,src.col[j],nrow,ncol)] = src.val[j];

  return true;

}

template <typename ValueType, typename IndexType>
bool dense_to_csr(const int omp_threads,
                  const IndexType nrow, const IndexType ncol,
                  const MatrixDENSE<ValueType> &src,
                  MatrixCSR<ValueType, IndexType> *dst,
                  IndexType *nnz) {

  assert(nrow > 0);
  assert(ncol > 0);

  omp_set_num_threads(omp_threads);

  allocate_host(nrow+1, &dst->row_offset);
  set_to_zero_host(nrow+1, dst->row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (IndexType i = 0; i < nrow; ++i)
    for(IndexType j = 0;  j < ncol; ++j)
      if (src.val[DENSE_IND(i,j,nrow,ncol)] != ValueType(0.0))
        dst->row_offset[i] += 1;

  *nnz = 0;
  for (IndexType i = 0; i < nrow; ++i) {
    IndexType tmp = dst->row_offset[i];
    dst->row_offset[i] = *nnz;
    *nnz += tmp;
  }

  dst->row_offset[nrow] = *nnz;

  allocate_host(*nnz, &dst->col);
  allocate_host(*nnz, &dst->val);

  set_to_zero_host(*nnz, dst->col);
  set_to_zero_host(*nnz, dst->val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (IndexType i = 0; i < nrow; ++i) {

    IndexType ind = dst->row_offset[i];

    for(IndexType j = 0;  j < ncol; ++j)
      if (src.val[DENSE_IND(i,j,nrow,ncol)] != ValueType(0.0)) {
        dst->val[ind] = src.val[DENSE_IND(i,j,nrow,ncol)];
        dst->col[ind] = j;
        ++ind;
      }

  }

  return true;

}

template <typename ValueType, typename IndexType>
bool csr_to_mcsr(const int omp_threads,
                 const IndexType nnz, const IndexType nrow, const IndexType ncol,
                 const MatrixCSR<ValueType, IndexType> &src,
                 MatrixMCSR<ValueType, IndexType> *dst) {

  assert(nnz  > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  // No support for non-squared matrices
  if (nrow != ncol)
    return false;

  omp_set_num_threads(omp_threads);

  // Pre-analysing step to check zero diagonal entries
  IndexType diag_entries = 0;

  for (int i=0; i<nrow; ++i)
    for (int j=src.row_offset[i]; j<src.row_offset[i+1]; ++j)
      if (i == src.col[j])
        ++diag_entries;

  IndexType zero_diag_entries = nrow - diag_entries;

  // MCSR does not support zero diagonal entries (yet)
  if (zero_diag_entries > 0)
    return false;

  allocate_host(nrow+1, &dst->row_offset);
  allocate_host(nnz, &dst->col);
  allocate_host(nnz, &dst->val);

  set_to_zero_host(nrow+1, dst->row_offset);
  set_to_zero_host(nnz, dst->col);
  set_to_zero_host(nnz, dst->val);

  for (IndexType ai = 0; ai < nrow+1; ++ai)
    dst->row_offset[ai] = nrow + src.row_offset[ai] - ai;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (IndexType ai=0; ai<nrow; ++ai) {

    IndexType correction = ai;
    for (IndexType aj=src.row_offset[ai]; aj<src.row_offset[ai+1]; ++aj)
      if (ai != src.col[aj]) {

        IndexType ind = nrow + aj - correction;

        // non-diag
        dst->col[ind] = src.col[aj];
        dst->val[ind] = src.val[aj];

      } else {

        // diag
        dst->val[ai] = src.val[aj];
        ++correction;

      }

  }

  if (dst->row_offset[nrow] != src.row_offset[nrow])
    return false;

  return true;

}

template <typename ValueType, typename IndexType>
bool mcsr_to_csr(const int omp_threads,
                 const IndexType nnz, const IndexType nrow, const IndexType ncol,
                 const MatrixMCSR<ValueType, IndexType> &src,
                 MatrixCSR<ValueType, IndexType> *dst) {

  assert(nnz  > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  // Only support square matrices
  if (nrow != ncol)
    return false;

  omp_set_num_threads(omp_threads);

  allocate_host(nrow+1, &dst->row_offset);
  allocate_host(nnz, &dst->col);
  allocate_host(nnz, &dst->val);

  set_to_zero_host(nrow+1, dst->row_offset);
  set_to_zero_host(nnz, dst->col);
  set_to_zero_host(nnz, dst->val);

  for (IndexType ai = 0; ai < nrow+1; ++ai)
    dst->row_offset[ai] = src.row_offset[ai] - nrow + ai;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (IndexType ai=0; ai<nrow; ++ai) {

    for (IndexType aj=src.row_offset[ai]; aj<src.row_offset[ai+1]; ++aj) {

      IndexType ind = aj - nrow + ai;

      // non-diag
      dst->col[ind] = src.col[aj];
      dst->val[ind] = src.val[aj];

    }

    IndexType diag_ind = src.row_offset[ai+1] - nrow + ai;

    // diag
    dst->val[diag_ind] = src.val[ai];
    dst->col[diag_ind] = ai;

  }

  if (dst->row_offset[nrow] != src.row_offset[nrow])
    return false;

  // Sorting the col (per row)
  // Bubble sort algorithm

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (IndexType i=0; i<nrow; ++i)
    for (IndexType j=dst->row_offset[i]; j<dst->row_offset[i+1]; ++j)
      for (IndexType jj=dst->row_offset[i]; jj<dst->row_offset[i+1]-1; ++jj)
        if (dst->col[jj] > dst->col[jj+1]) {
          //swap elements

          IndexType ind = dst->col[jj];
          ValueType val = dst->val[jj];

          dst->col[jj] = dst->col[jj+1];
          dst->val[jj] = dst->val[jj+1];

          dst->col[jj+1] = ind;
          dst->val[jj+1] = val;

        }

  return true;

}

template <typename ValueType, typename IndexType>
bool csr_to_coo(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixCSR<ValueType, IndexType> &src,
                MatrixCOO<ValueType, IndexType> *dst) {

  assert(nnz  > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  omp_set_num_threads(omp_threads);

  allocate_host(nnz, &dst->row);
  allocate_host(nnz, &dst->col);
  allocate_host(nnz, &dst->val);

  set_to_zero_host(nnz, dst->row);
  set_to_zero_host(nnz, dst->col);
  set_to_zero_host(nnz, dst->val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (IndexType i = 0; i < nrow; ++i)
    for(IndexType j = src.row_offset[i]; j < src.row_offset[i+1]; ++j)
      dst->row[j] = i;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (IndexType i = 0; i < nnz; ++i)
    dst->col[i] = src.col[i];

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (IndexType i = 0; i < nnz; ++i)
    dst->val[i] = src.val[i];

  return true;

}

template <typename ValueType, typename IndexType>
bool csr_to_ell(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixCSR<ValueType, IndexType> &src,
                MatrixELL<ValueType, IndexType> *dst, IndexType *nnz_ell) {

  assert(nnz  > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  omp_set_num_threads(omp_threads);
  
  dst->max_row = 0;
  for (IndexType i = 0; i < nrow; ++i) {
    IndexType max_row = src.row_offset[i+1] - src.row_offset[i];
    if (max_row > dst->max_row)
      dst->max_row = max_row;
  }

  *nnz_ell = dst->max_row * nrow;

  allocate_host(*nnz_ell, &dst->val);
  allocate_host(*nnz_ell, &dst->col);

  set_to_zero_host(*nnz_ell, dst->val);
  set_to_zero_host(*nnz_ell, dst->col);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (IndexType i = 0; i < nrow; ++i) {

    IndexType n = 0;

    for(IndexType j = src.row_offset[i]; j < src.row_offset[i+1]; ++j) {

      IndexType ind = ELL_IND(i, n, nrow, dst->max_row);

      dst->val[ind] = src.val[j];
      dst->col[ind] = src.col[j];
      ++n;

    }

    for (IndexType j = src.row_offset[i+1]-src.row_offset[i]; j < dst->max_row; ++j) {

      IndexType ind = ELL_IND(i, n, nrow, dst->max_row);

      dst->val[ind] = ValueType(0.0);
      dst->col[ind] = IndexType(-1);
      ++n;

    }

  }

  return true;

}

template <typename ValueType, typename IndexType>
bool ell_to_csr(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixELL<ValueType, IndexType> &src,
                MatrixCSR<ValueType, IndexType> *dst, IndexType *nnz_csr) {

  assert(nnz  > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  omp_set_num_threads(omp_threads);

  allocate_host(nrow+1, &dst->row_offset);
  set_to_zero_host(nrow+1, dst->row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (IndexType ai=0; ai<nrow; ++ai) {

    for (IndexType n=0; n<src.max_row; ++n) {

      IndexType aj = ELL_IND(ai, n, nrow, src.max_row);

      if ((src.col[aj] >= 0) && (src.col[aj] < ncol))
        dst->row_offset[ai] += 1;

    }

  }

  *nnz_csr = 0;
  for (IndexType i = 0; i < nrow; ++i) {
    IndexType tmp = dst->row_offset[i];
    dst->row_offset[i] = *nnz_csr;
    *nnz_csr += tmp;
  }

  dst->row_offset[nrow] = *nnz_csr;

  allocate_host(*nnz_csr, &dst->col);
  allocate_host(*nnz_csr, &dst->val);

  set_to_zero_host(*nnz_csr, dst->col);
  set_to_zero_host(*nnz_csr, dst->val);  

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (IndexType ai=0; ai<nrow; ++ai) {

    IndexType ind = dst->row_offset[ai];

    for (IndexType n=0; n<src.max_row; ++n) {

      IndexType aj = ELL_IND(ai, n, nrow, src.max_row);

        if ((src.col[aj] >= 0) && (src.col[aj] < ncol)) {

          dst->col[ind] = src.col[aj];
          dst->val[ind] = src.val[aj];
          ++ind;

        }

    }

  }

  return true;

}

template <typename ValueType, typename IndexType>
bool hyb_to_csr(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const IndexType nnz_ell, const IndexType nnz_coo,
                const MatrixHYB<ValueType, IndexType> &src,
                MatrixCSR<ValueType, IndexType> *dst, IndexType *nnz_csr) {

  assert(nnz  > 0);
  assert(nnz  == nnz_ell + nnz_coo);
  assert(nrow > 0);
  assert(ncol > 0);

  omp_set_num_threads(omp_threads);

  allocate_host(nrow+1, &dst->row_offset);
  set_to_zero_host(nrow+1, dst->row_offset);

  IndexType start;
  start = 0;

  // TODO
//#ifdef _OPENMP
// #pragma omp parallel for private(start)
//#endif
  for (IndexType ai=0 ; ai<nrow; ++ai) {

    // ELL
    for (IndexType n=0; n<src.ELL.max_row; ++n) {

      IndexType aj = ELL_IND(ai, n, nrow, src.ELL.max_row);

      if ((src.ELL.col[aj] >= 0) && (src.ELL.col[aj] < ncol))
        dst->row_offset[ai] += 1;

    }

    // COO
    for (IndexType i = start; i < nnz_coo; ++i) {
      if (src.COO.row[i] == ai) {
        dst->row_offset[ai] += 1;
        ++start;
      }
      if (src.COO.row[i] > ai)
        break;
    }

  }

  *nnz_csr = 0;
  for (IndexType i = 0; i < nrow; ++i) {
    IndexType tmp = dst->row_offset[i];
    dst->row_offset[i] = *nnz_csr;
    *nnz_csr += tmp;
  }

  dst->row_offset[nrow] = *nnz_csr;

  allocate_host(*nnz_csr, &dst->col);
  allocate_host(*nnz_csr, &dst->val);

  set_to_zero_host(*nnz_csr, dst->col);
  set_to_zero_host(*nnz_csr, dst->val);

  start = 0 ;

  // TODO
//#ifdef _OPENMP
//#pragma omp parallel for private(start)
//#endif
  for (IndexType ai=0; ai<nrow; ++ai) {

    IndexType ind = dst->row_offset[ai];

    // ELL
    for (IndexType n=0; n<src.ELL.max_row; ++n) {

      IndexType aj = ELL_IND(ai, n, nrow, src.ELL.max_row);

      if ((src.ELL.col[aj] >= 0) && (src.ELL.col[aj] < ncol)) {
         dst->col[ind] = src.ELL.col[aj];
         dst->val[ind] = src.ELL.val[aj];
         ++ind;
       }

    }

    // COO
    for (IndexType i = start; i < nnz_coo; ++i) {
      if (src.COO.row[i] == ai) {
        dst->col[ind] = src.COO.col[i];
        dst->val[ind] = src.COO.val[i];
        ++ind;
        ++start;
      }
      if (src.COO.row[i] > ai)
        break;
    }

  }

  return true;

}

// ----------------------------------------------------------
// function coo_to_csr(...)
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.3.1,
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - adapted interface
// - Bubble sort each column after convertion
// - OpenMP pragma for
// ----------------------------------------------------------
template <typename ValueType, typename IndexType>
bool coo_to_csr(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixCOO<ValueType, IndexType> &src,
                MatrixCSR<ValueType, IndexType> *dst) {

  assert(nnz  > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  omp_set_num_threads(omp_threads);

  allocate_host(nrow+1, &dst->row_offset);
  allocate_host(nnz, &dst->col);
  allocate_host(nnz, &dst->val);

  set_to_zero_host(nrow+1, dst->row_offset);
  set_to_zero_host(nnz, dst->col);
  set_to_zero_host(nnz, dst->val);

  // compute nnz entries per row of CSR
  for (IndexType n = 0; n < nnz; ++n)
    dst->row_offset[src.row[n]] = dst->row_offset[src.row[n]] + 1;

  // cumsum the num_entries per row to get dst->row_offsets[]
  IndexType cumsum = 0;
  for(IndexType i = 0; i < nrow; ++i) {
    IndexType temp = dst->row_offset[i];
    dst->row_offset[i] = cumsum;
    cumsum += temp;
  }

  dst->row_offset[nrow] = cumsum;

  // write Aj,Ax IndexTypeo dst->column_indices,dst->values
  for(IndexType n = 0; n < nnz; ++n) {
    IndexType row  = src.row[n];
    IndexType dest = dst->row_offset[row];

    dst->col[dest] = src.col[n];
    dst->val[dest] = src.val[n];

    dst->row_offset[row] = dst->row_offset[row] + 1;
  }

  IndexType last = 0;
  for(IndexType i = 0; i <= nrow; ++i) {
    IndexType temp = dst->row_offset[i];
    dst->row_offset[i]  = last;
    last   = temp;
  }

  // Sorting the col (per row)
  // Bubble sort algorithm

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (IndexType i=0; i<nrow; ++i)
    for (IndexType j=dst->row_offset[i]; j<dst->row_offset[i+1]; ++j)
      for (IndexType jj=dst->row_offset[i]; jj<dst->row_offset[i+1]-1; ++jj)
        if (dst->col[jj] > dst->col[jj+1]) {
          //swap elements

          IndexType ind = dst->col[jj];
          ValueType val = dst->val[jj];

          dst->col[jj] = dst->col[jj+1];
          dst->val[jj] = dst->val[jj+1];

          dst->col[jj+1] = ind;
          dst->val[jj+1] = val;
        }

  return true;

}

// ----------------------------------------------------------
// function csr_to_dia(...)
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.3.1,
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - adapted interface
// - OpenMP pragma for
// ----------------------------------------------------------
template <typename ValueType, typename IndexType>
bool csr_to_dia(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixCSR<ValueType, IndexType> &src,
                MatrixDIA<ValueType, IndexType> *dst, IndexType *nnz_dia) {

  assert(nnz  > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  if (nrow != ncol)
    return false;

  omp_set_num_threads(omp_threads);

  // compute number of occupied diagonals and enumerate them
  dst->num_diag = 0;

  IndexType *diag_map = NULL;

  allocate_host(nrow+ncol, &diag_map);
  set_to_zero_host(nrow+ncol, diag_map);

  for(IndexType i = 0; i < nrow; i++) {
    for(IndexType jj = src.row_offset[i]; jj < src.row_offset[i+1]; jj++) {
      IndexType j         = src.col[jj];
      IndexType map_index = (nrow - i) + j; //offset shifted by + num_rows

      if(diag_map[map_index] == 0) {
        diag_map[map_index] = 1;
        ++dst->num_diag;
      }
    }
  }

  if (nrow < ncol) {
    *nnz_dia = ncol * dst->num_diag;
  } else {
    *nnz_dia = nrow * dst->num_diag;
  }

  // Conversion fails if numbers of diagonal is too large
  if (dst->num_diag > 200)
    return false;

  //allocate DIA structure
  allocate_host(dst->num_diag, &dst->offset);
  set_to_zero_host(dst->num_diag, dst->offset);

  allocate_host(*nnz_dia, &dst->val);
  set_to_zero_host(*nnz_dia, dst->val);

  // fill in diagonal_offsets array
  for(IndexType n = 0, diag = 0 ; n < nrow + ncol; n++) {
    if(diag_map[n] == 1) {
      diag_map[n] = diag;
      dst->offset[diag] = IndexType(n) - IndexType(nrow);
      diag++;
    }
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(IndexType i = 0; i < nrow; i++) {
    for(IndexType jj = src.row_offset[i]; jj < src.row_offset[i+1]; jj++) {
      IndexType j = src.col[jj];
      IndexType map_index = (nrow - i) + j; //offset shifted by + num_rows
      IndexType diag = diag_map[map_index];

      dst->val[DIA_IND(i, diag, nrow, dst->num_diag)] = src.val[jj];

    }
  }

  free_host(&diag_map);

  return true;

}

// ----------------------------------------------------------
// function dia_to_csr(...)
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.3.1,
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - adapted interface
// ----------------------------------------------------------
template <typename ValueType, typename IndexType>
bool dia_to_csr(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixDIA<ValueType, IndexType> &src,
                MatrixCSR<ValueType, IndexType> *dst, IndexType *nnz_csr) {

  assert(nnz  > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  omp_set_num_threads(omp_threads);

  *nnz_csr = 0;

  // count nonzero entries
  for(IndexType i = 0; i < nrow; i++) {
    for(IndexType n = 0; n < src.num_diag; n++) {
      const IndexType j = i + src.offset[n];

      //      if(j >= IndexType(0) && j < IndexType(ncol) && src.val[i*src.num_diag+n] != ValueType(0.0))
      if(j >= IndexType(0) && j < IndexType(ncol) && src.val[DIA_IND(i, n, nrow, src.num_diag)] != ValueType(0.0))
        ++(*nnz_csr);
    }
  }

  allocate_host(nrow+1, &dst->row_offset);
  allocate_host(*nnz_csr, &dst->col);
  allocate_host(*nnz_csr, &dst->val);

  set_to_zero_host(nrow+1, dst->row_offset);
  set_to_zero_host(*nnz_csr, dst->col);
  set_to_zero_host(*nnz_csr, dst->val);

  *nnz_csr = 0;
  dst->row_offset[0] = 0;

  // copy nonzero entries to CSR structure
  for(IndexType i = 0; i < nrow; i++) {

    for(IndexType n = 0; n < src.num_diag; n++) {

      const IndexType j = i + src.offset[n];

      if(j >= IndexType(0) && j < IndexType(ncol)) {

        IndexType ind = DIA_IND(i, n, nrow, src.num_diag);
        const ValueType value = src.val[ind];

        if (value != ValueType(0.0)) {
          dst->col[*nnz_csr] = j;
          dst->val[*nnz_csr] = value;
          ++(*nnz_csr);
        }

      }

    }

    dst->row_offset[i + 1] = *nnz_csr;

  }

  return true;

}

// ----------------------------------------------------------
// function csr_to_hyb(...)
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.3.1,
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - adapted interface
// - the entries per row are defined out side or as nnz/nrow
// ----------------------------------------------------------
template <typename ValueType, typename IndexType>
bool csr_to_hyb(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixCSR<ValueType, IndexType> &src,
                MatrixHYB<ValueType, IndexType> *dst, 
                IndexType *nnz_hyb, IndexType *nnz_ell, IndexType *nnz_coo) {

  assert(nnz  > 0);
  assert(nrow > 0);
  assert(ncol > 0);

  omp_set_num_threads(omp_threads);

  if (dst->ELL.max_row == 0)
    dst->ELL.max_row = (nnz / nrow);

  *nnz_coo = 0;
  for (IndexType i=0; i<nrow; ++i) {

    IndexType nnz_per_row = src.row_offset[i+1] - src.row_offset[i];

    if (nnz_per_row > dst->ELL.max_row)
      *nnz_coo += nnz_per_row - dst->ELL.max_row;

  }

  *nnz_ell = dst->ELL.max_row * nrow;
  *nnz_hyb = *nnz_coo + *nnz_ell;

  if (*nnz_coo <= 0 || *nnz_ell <= 0 || *nnz_hyb <= 0)
    return false;

  // ELL
  allocate_host(*nnz_ell, &dst->ELL.val);
  allocate_host(*nnz_ell, &dst->ELL.col);

  set_to_zero_host(*nnz_ell, dst->ELL.val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(IndexType i = 0; i < *nnz_ell; i++) 
    dst->ELL.col[i] = IndexType(-1);

  // COO
  allocate_host(*nnz_coo, &dst->COO.row);
  allocate_host(*nnz_coo, &dst->COO.col);
  allocate_host(*nnz_coo, &dst->COO.val);

  set_to_zero_host(*nnz_coo, dst->COO.row);
  set_to_zero_host(*nnz_coo, dst->COO.col);
  set_to_zero_host(*nnz_coo, dst->COO.val);

  IndexType *nnzcoo = NULL;
  IndexType *nnzell = NULL;
  allocate_host(nrow, &nnzcoo);
  allocate_host(nrow, &nnzell);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  // copy up to num_cols_per_row values of row i into the ELL
  for (IndexType i=0; i<nrow; ++i) {

    IndexType n = 0;

    nnzcoo[i] = 0;

    for (IndexType j=src.row_offset[i]; j<src.row_offset[i+1]; ++j) {

      if (n >= dst->ELL.max_row) {
        nnzcoo[i] = src.row_offset[i+1] - j;
        break;
      }

      IndexType ind = ELL_IND(i, n, nrow, dst->ELL.max_row);
      dst->ELL.col[ind] = src.col[j];
      dst->ELL.val[ind] = src.val[j];
      ++n;

    }

    nnzell[i] = n;

  }

  for (int i=1; i<nrow; ++i)
    nnzell[i] += nnzell[i-1];

#ifdef _OPENMP
#pragma omp parallel for
#endif
  // copy any remaining values in row i into the COO
  for (IndexType i=0; i<nrow; ++i) {

    for (IndexType j=src.row_offset[i+1] - nnzcoo[i]; j<src.row_offset[i+1]; ++j) {

      dst->COO.row[j-nnzell[i]] = i;
      dst->COO.col[j-nnzell[i]] = src.col[j];
      dst->COO.val[j-nnzell[i]] = src.val[j];

    }

  }

  free_host(&nnzcoo);
  free_host(&nnzell);

/*
  for(IndexType i = 0, coo_nnz = 0; i < nrow; i++) {
    IndexType n = 0;
    IndexType jj = src.row_offset[i];
    
    // copy up to num_cols_per_row values of row i into the ELL
    while(jj < src.row_offset[i+1] && n < dst->ELL.max_row) {
      IndexType ind = ELL_IND(i, n, nrow, dst->ELL.max_row);
      dst->ELL.col[ind] = src.col[jj];
      dst->ELL.val[ind] = src.val[jj];
      jj++, n++;
    }

    // copy any remaining values in row i into the COO
    while(jj < src.row_offset[i+1]) {
      dst->COO.row[coo_nnz] = i;
      dst->COO.col[coo_nnz] = src.col[jj];
      dst->COO.val[coo_nnz] = src.val[jj];
      jj++; coo_nnz++;
    }
  }
*/

  return true;

}

template bool csr_to_coo(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<double, int> &src,
                         MatrixCOO<double, int> *dst);

template bool csr_to_coo(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<float, int> &src,
                         MatrixCOO<float, int> *dst);

#ifdef SUPPORT_COMPLEX
template bool csr_to_coo(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<std::complex<double>, int> &src,
                         MatrixCOO<std::complex<double>, int> *dst);

template bool csr_to_coo(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<std::complex<float>, int> &src,
                         MatrixCOO<std::complex<float>, int> *dst);
#endif

template bool csr_to_coo(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<int, int> &src,
                         MatrixCOO<int, int> *dst);

template bool csr_to_mcsr(const int omp_threads,
                          const int nnz, const int nrow, const int ncol,
                          const MatrixCSR<double, int> &src,
                          MatrixMCSR<double, int> *dst);

template bool csr_to_mcsr(const int omp_threads,
                          const int nnz, const int nrow, const int ncol,
                          const MatrixCSR<float, int> &src,
                          MatrixMCSR<float, int> *dst);

#ifdef SUPPORT_COMPLEX
template bool csr_to_mcsr(const int omp_threads,
                          const int nnz, const int nrow, const int ncol,
                          const MatrixCSR<std::complex<double>, int> &src,
                          MatrixMCSR<std::complex<double>, int> *dst);

template bool csr_to_mcsr(const int omp_threads,
                          const int nnz, const int nrow, const int ncol,
                          const MatrixCSR<std::complex<float>, int> &src,
                          MatrixMCSR<std::complex<float>, int> *dst);
#endif

template bool csr_to_mcsr(const int omp_threads,
                          const int nnz, const int nrow, const int ncol,
                          const MatrixCSR<int, int> &src,
                          MatrixMCSR<int, int> *dst);

template bool mcsr_to_csr(const int omp_threads,
                          const int nnz, const int nrow, const int ncol,
                          const MatrixMCSR<double, int> &src,
                          MatrixCSR<double, int> *dst);

template bool mcsr_to_csr(const int omp_threads,
                          const int nnz, const int nrow, const int ncol,
                          const MatrixMCSR<float, int> &src,
                          MatrixCSR<float, int> *dst);

#ifdef SUPPORT_COMPLEX
template bool mcsr_to_csr(const int omp_threads,
                          const int nnz, const int nrow, const int ncol,
                          const MatrixMCSR<std::complex<double>, int> &src,
                          MatrixCSR<std::complex<double>, int> *dst);

template bool mcsr_to_csr(const int omp_threads,
                          const int nnz, const int nrow, const int ncol,
                          const MatrixMCSR<std::complex<float>, int> &src,
                          MatrixCSR<std::complex<float>, int> *dst);
#endif

template bool mcsr_to_csr(const int omp_threads,
                          const int nnz, const int nrow, const int ncol,
                          const MatrixMCSR<int, int> &src,
                          MatrixCSR<int, int> *dst);

template bool csr_to_dia(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<double, int> &src,
                         MatrixDIA<double, int> *dst, int *nnz_dia);

template bool csr_to_dia(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<float, int> &src,
                         MatrixDIA<float, int> *dst, int *nnz_dia);

#ifdef SUPPORT_COMPLEX
template bool csr_to_dia(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<std::complex<double>, int> &src,
                         MatrixDIA<std::complex<double>, int> *dst, int *nnz_dia);

template bool csr_to_dia(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<std::complex<float>, int> &src,
                         MatrixDIA<std::complex<float>, int> *dst, int *nnz_dia);
#endif

template bool csr_to_dia(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<int, int> &src,
                         MatrixDIA<int, int> *dst, int *nnz_dia);

template bool csr_to_hyb(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<double, int> &src,
                         MatrixHYB<double, int> *dst, 
                         int *nnz_hyb, int *nnz_ell, int *nnz_coo);

template bool csr_to_hyb(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<float, int> &src,
                         MatrixHYB<float, int> *dst, 
                         int *nnz_hyb, int *nnz_ell, int *nnz_coo);

#ifdef SUPPORT_COMPLEX
template bool csr_to_hyb(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<std::complex<double>, int> &src,
                         MatrixHYB<std::complex<double>, int> *dst, 
                         int *nnz_hyb, int *nnz_ell, int *nnz_coo);

template bool csr_to_hyb(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<std::complex<float>, int> &src,
                         MatrixHYB<std::complex<float>, int> *dst, 
                         int *nnz_hyb, int *nnz_ell, int *nnz_coo);
#endif

template bool csr_to_hyb(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<int, int> &src,
                         MatrixHYB<int, int> *dst, 
                         int *nnz_hyb, int *nnz_ell, int *nnz_coo);

template bool csr_to_ell(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<double, int> &src,
                         MatrixELL<double, int> *dst, int *nnz_ell);

template bool csr_to_ell(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<float, int> &src,
                         MatrixELL<float, int> *dst, int *nnz_ell);

#ifdef SUPPORT_COMPLEX
template bool csr_to_ell(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<std::complex<double>, int> &src,
                         MatrixELL<std::complex<double>, int> *dst, int *nnz_ell);

template bool csr_to_ell(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<std::complex<float>, int> &src,
                         MatrixELL<std::complex<float>, int> *dst, int *nnz_ell);
#endif

template bool csr_to_ell(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCSR<int, int> &src,
                         MatrixELL<int, int> *dst, int *nnz_ell);

template bool csr_to_dense(const int omp_threads,
                           const int nnz, const int nrow, const int ncol,
                           const MatrixCSR<double, int> &src,
                           MatrixDENSE<double> *dst);

template bool csr_to_dense(const int omp_threads,
                           const int nnz, const int nrow, const int ncol,
                           const MatrixCSR<float, int> &src,
                           MatrixDENSE<float> *dst);

#ifdef SUPPORT_COMPLEX
template bool csr_to_dense(const int omp_threads,
                           const int nnz, const int nrow, const int ncol,
                           const MatrixCSR<std::complex<double>, int> &src,
                           MatrixDENSE<std::complex<double> > *dst);

template bool csr_to_dense(const int omp_threads,
                           const int nnz, const int nrow, const int ncol,
                           const MatrixCSR<std::complex<float>, int> &src,
                           MatrixDENSE<std::complex<float> > *dst);
#endif

template bool csr_to_dense(const int omp_threads,
                           const int nnz, const int nrow, const int ncol,
                           const MatrixCSR<int, int> &src,
                           MatrixDENSE<int> *dst);

template bool dense_to_csr(const int omp_threads,
                           const int nrow, const int ncol,
                           const MatrixDENSE<double> &src,
                           MatrixCSR<double, int> *dst,
                           int *nnz);

template bool dense_to_csr(const int omp_threads,
                           const int nrow, const int ncol,
                           const MatrixDENSE<float> &src,
                           MatrixCSR<float, int> *dst,
                           int *nnz);

#ifdef SUPPORT_COMPLEX
template bool dense_to_csr(const int omp_threads,
                           const int nrow, const int ncol,
                           const MatrixDENSE<std::complex<double> > &src,
                           MatrixCSR<std::complex<double>, int> *dst,
                           int *nnz);

template bool dense_to_csr(const int omp_threads,
                           const int nrow, const int ncol,
                           const MatrixDENSE<std::complex<float> > &src,
                           MatrixCSR<std::complex<float>, int> *dst,
                           int *nnz);
#endif

template bool dense_to_csr(const int omp_threads,
                           const int nrow, const int ncol,
                           const MatrixDENSE<int> &src,
                           MatrixCSR<int, int> *dst,
                           int *nnz);

template bool dia_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixDIA<double, int> &src,
                         MatrixCSR<double, int> *dst, int *nnz_csr);

template bool dia_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixDIA<float, int> &src,
                         MatrixCSR<float, int> *dst, int *nnz_csr);

#ifdef SUPPORT_COMPLEX
template bool dia_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixDIA<std::complex<double>, int> &src,
                         MatrixCSR<std::complex<double>, int> *dst, int *nnz_csr);

template bool dia_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixDIA<std::complex<float>, int> &src,
                         MatrixCSR<std::complex<float>, int> *dst, int *nnz_csr);
#endif

template bool dia_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixDIA<int, int> &src,
                         MatrixCSR<int, int> *dst, int *nnz_csr);

template bool ell_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixELL<double, int> &src,
                         MatrixCSR<double, int> *dst, int *nnz_csr);

template bool ell_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixELL<float, int> &src,
                         MatrixCSR<float, int> *dst, int *nnz_csr);

#ifdef SUPPORT_COMPLEX
template bool ell_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixELL<std::complex<double>, int> &src,
                         MatrixCSR<std::complex<double>, int> *dst, int *nnz_csr);

template bool ell_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixELL<std::complex<float>, int> &src,
                         MatrixCSR<std::complex<float>, int> *dst, int *nnz_csr);
#endif

template bool ell_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixELL<int, int> &src,
                         MatrixCSR<int, int> *dst, int *nnz_csr);

template bool coo_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCOO<double, int> &src,
                         MatrixCSR<double, int> *dst);

template bool coo_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCOO<float, int> &src,
                         MatrixCSR<float, int> *dst);

#ifdef SUPPORT_COMPLEX
template bool coo_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCOO<std::complex<double>, int> &src,
                         MatrixCSR<std::complex<double>, int> *dst);

template bool coo_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCOO<std::complex<float>, int> &src,
                         MatrixCSR<std::complex<float>, int> *dst);
#endif

template bool coo_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const MatrixCOO<int, int> &src,
                         MatrixCSR<int, int> *dst);

template bool hyb_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const int nnz_ell, const int nnz_coo,
                         const MatrixHYB<double, int> &src,
                         MatrixCSR<double, int> *dst, int *nnz_csr);

template bool hyb_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const int nnz_ell, const int nnz_coo,
                         const MatrixHYB<float, int> &src,
                         MatrixCSR<float, int> *dst, int *nnz_csr);

#ifdef SUPPORT_COMPLEX
template bool hyb_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const int nnz_ell, const int nnz_coo,
                         const MatrixHYB<std::complex<double>, int> &src,
                         MatrixCSR<std::complex<double>, int> *dst, int *nnz_csr);

template bool hyb_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const int nnz_ell, const int nnz_coo,
                         const MatrixHYB<std::complex<float>, int> &src,
                         MatrixCSR<std::complex<float>, int> *dst, int *nnz_csr);
#endif

template bool hyb_to_csr(const int omp_threads,
                         const int nnz, const int nrow, const int ncol,
                         const int nnz_ell, const int nnz_coo,
                         const MatrixHYB<int, int> &src,
                         MatrixCSR<int, int> *dst, int *nnz_csr);

}
