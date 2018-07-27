#ifndef ROCALUTION_HIP_CONVERSION_HPP_
#define ROCALUTION_HIP_CONVERSION_HPP_

#include "../backend_manager.hpp"
#include "../matrix_formats.hpp"

#include <hipsparse.h>

namespace rocalution {

template <typename ValueType, typename IndexType>
bool csr_to_coo_hip(const hipsparseHandle_t handle,
                    IndexType nnz, IndexType nrow, IndexType ncol,
                    const MatrixCSR<ValueType, IndexType>& src,
                    MatrixCOO<ValueType, IndexType>* dst);

template <typename ValueType, typename IndexType>
bool coo_to_csr_hip(const hipsparseHandle_t handle,
                    IndexType nnz, IndexType nrow, IndexType ncol,
                    const MatrixCOO<ValueType, IndexType>& src,
                    MatrixCSR<ValueType, IndexType>* dst);

template <typename ValueType, typename IndexType>
bool csr_to_ell_hip(const hipsparseHandle_t handle,
                    IndexType nnz, IndexType nrow, IndexType ncol,
                    const MatrixCSR<ValueType, IndexType>& src,
                    const hipsparseMatDescr_t src_descr,
                    MatrixELL<ValueType, IndexType>* dst,
                    const hipsparseMatDescr_t dst_descr,
                    IndexType* nnz_ell);

template <typename ValueType, typename IndexType>
bool csr_to_dia_hip(int blocksize,
                    IndexType nnz, IndexType nrow, IndexType ncol,
                    const MatrixCSR<ValueType, IndexType>& src,
                    MatrixDIA<ValueType, IndexType>* dst,
                    IndexType* nnz_dia,
                    IndexType* num_diag);

template <typename ValueType, typename IndexType>
bool csr_to_hyb_hip(int blocksize,
                    IndexType nnz, IndexType nrow, IndexType ncol,
                    const MatrixCSR<ValueType, IndexType>& src,
                    MatrixHYB<ValueType, IndexType>* dst,
                    IndexType* nnz_hyb, IndexType* nnz_ell, IndexType* nnz_coo);

} // namespace rocalution

#endif // ROCALUTION_HIP_CONVERSION_HPP_
