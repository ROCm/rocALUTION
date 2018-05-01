#ifndef ROCALUTION_HOST_CONVERSION_HPP_
#define ROCALUTION_HOST_CONVERSION_HPP_

#include "../matrix_formats.hpp"

namespace rocalution {

template <typename ValueType, typename IndexType>
bool csr_to_coo(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixCSR<ValueType, IndexType> &src,
                MatrixCOO<ValueType, IndexType> *dst);

template <typename ValueType, typename IndexType>
bool csr_to_mcsr(const int omp_threads,
                 const IndexType nnz, const IndexType nrow, const IndexType ncol,
                 const MatrixCSR<ValueType, IndexType> &src,
                 MatrixMCSR<ValueType, IndexType> *dst);

template <typename ValueType, typename IndexType>
bool csr_to_dia(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixCSR<ValueType, IndexType> &src,
                MatrixDIA<ValueType, IndexType> *dst, IndexType *nnz_dia);

template <typename ValueType, typename IndexType>
bool csr_to_dense(const int omp_threads,
                  const IndexType nnz, const IndexType nrow, const IndexType ncol,
                  const MatrixCSR<ValueType, IndexType> &src,
                  MatrixDENSE<ValueType> *dst);

template <typename ValueType, typename IndexType>
bool csr_to_ell(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixCSR<ValueType, IndexType> &src,
                MatrixELL<ValueType, IndexType> *dst, IndexType *nnz_ell);

template <typename ValueType, typename IndexType>
bool csr_to_hyb(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixCSR<ValueType, IndexType> &src,
                MatrixHYB<ValueType, IndexType> *dst,
                IndexType *nnz_hyb, IndexType *nnz_ell, IndexType *nnz_coo);

template <typename ValueType, typename IndexType>
bool dense_to_csr(const int omp_threads,
                  const IndexType nrow, const IndexType ncol,
                  const MatrixDENSE<ValueType> &src,
                  MatrixCSR<ValueType, IndexType> *dst,
                  IndexType *nnz);

template <typename ValueType, typename IndexType>
bool dia_to_csr(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixDIA<ValueType, IndexType> &src,
                MatrixCSR<ValueType, IndexType> *dst, IndexType *nnz_csr);

template <typename ValueType, typename IndexType>
bool ell_to_csr(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixELL<ValueType, IndexType> &src,
                MatrixCSR<ValueType, IndexType> *dst, IndexType *nnz_csr);

template <typename ValueType, typename IndexType>
bool coo_to_csr(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const MatrixCOO<ValueType, IndexType> &src,
                MatrixCSR<ValueType, IndexType> *dst);

template <typename ValueType, typename IndexType>
bool mcsr_to_csr(const int omp_threads,
                 const IndexType nnz, const IndexType nrow, const IndexType ncol,
                 const MatrixMCSR<ValueType, IndexType> &src,
                 MatrixCSR<ValueType, IndexType> *dst);

template <typename ValueType, typename IndexType>
bool hyb_to_csr(const int omp_threads,
                const IndexType nnz, const IndexType nrow, const IndexType ncol,
                const IndexType nnz_ell, const IndexType nnz_coo,
                const MatrixHYB<ValueType, IndexType> &src,
                MatrixCSR<ValueType, IndexType> *dst, IndexType *nnz_csr);


}

#endif // ROCALUTION_HOST_CONVERSION_HPP_
