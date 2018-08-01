#include "../../utils/def.hpp"
#include "hip_allocate_free.hpp"
#include "hip_conversion.hpp"
#include "hip_sparse.hpp"
#include "hip_utils.hpp"
#include "hip_kernels_conversion.hpp"
#include "../matrix_formats.hpp"

#include <hip/hip_runtime_api.h>
#include <hipcub/hipcub.hpp>

namespace rocalution {

template <typename ValueType, typename IndexType>
bool csr_to_coo_hip(const hipsparseHandle_t handle,
                    const IndexType nnz, const IndexType nrow, const IndexType ncol,
                    const MatrixCSR<ValueType, IndexType> &src,
                    MatrixCOO<ValueType, IndexType> *dst)
{
    assert(nnz  > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    assert(dst != NULL);
    assert(handle != NULL);

    allocate_hip(nnz, &dst->row);
    allocate_hip(nnz, &dst->col);
    allocate_hip(nnz, &dst->val);

    hipMemcpyAsync(dst->col, src.col, sizeof(IndexType) * nnz, hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    hipMemcpyAsync(dst->val, src.val, sizeof(ValueType) * nnz, hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    hipsparseStatus_t stat_t = hipsparseXcsr2coo(HIPSPARSE_HANDLE(handle),
                                                 src.row_offset,
                                                 nnz,
                                                 nrow,
                                                 dst->row,
                                                 HIPSPARSE_INDEX_BASE_ZERO);
    CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    // Sync memcopy
    hipDeviceSynchronize();

    return true;
}

template <typename ValueType, typename IndexType>
bool coo_to_csr_hip(const hipsparseHandle_t handle,
                    const IndexType nnz, const IndexType nrow, const IndexType ncol,
                    const MatrixCOO<ValueType, IndexType> &src,
                    MatrixCSR<ValueType, IndexType> *dst)
{
    assert(nnz  > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    assert(dst != NULL);
    assert(handle != NULL);

    allocate_hip(nrow + 1, &dst->row_offset);
    allocate_hip(nnz, &dst->col);
    allocate_hip(nnz, &dst->val);

    hipMemcpyAsync(dst->col, src.col, sizeof(IndexType) * nnz, hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    hipMemcpyAsync(dst->val, src.val, sizeof(ValueType) * nnz, hipMemcpyDeviceToDevice);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    hipsparseStatus_t stat_t = hipsparseXcoo2csr(HIPSPARSE_HANDLE(handle),
                                                 src.row,
                                                 nnz,
                                                 nrow,
                                                 dst->row_offset,
                                                 HIPSPARSE_INDEX_BASE_ZERO);
    CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    // Sync memcopy
    hipDeviceSynchronize();

    return true;
}

template <typename ValueType, typename IndexType>
bool csr_to_ell_hip(const hipsparseHandle_t handle,
                    IndexType nnz, IndexType nrow, IndexType ncol,
                    const MatrixCSR<ValueType, IndexType>& src,
                    const hipsparseMatDescr_t src_descr,
                    MatrixELL<ValueType, IndexType>* dst,
                    const hipsparseMatDescr_t dst_descr,
                    IndexType* nnz_ell)
{
    assert(nnz  > 0);
    assert(nrow > 0);
    assert(ncol > 0);

    assert(dst != NULL);
    assert(nnz_ell != NULL);
    assert(handle != NULL);
    assert(src_descr != NULL);
    assert(dst_descr != NULL);

    hipsparseStatus_t stat_t;

    // Determine ELL width
    stat_t = hipsparseXcsr2ellWidth(HIPSPARSE_HANDLE(handle),
                                    nrow,
                                    src_descr,
                                    src.row_offset,
                                    dst_descr,
                                    &dst->max_row);
    CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    // Limit ELL size to 5 times CSR nnz
    if(dst->max_row > 5 * nnz / nrow)
    {
        return false;
    }

    // Compute ELL non-zeros
    *nnz_ell = dst->max_row * nrow;

    // Allocate ELL matrix
    allocate_hip(*nnz_ell, &dst->col);
    allocate_hip(*nnz_ell, &dst->val);

    // Conversion
    stat_t = hipsparseTcsr2ell(HIPSPARSE_HANDLE(handle),
                               nrow,
                               src_descr,
                               src.val,
                               src.row_offset,
                               src.col,
                               dst_descr,
                               dst->max_row,
                               dst->val,
                               dst->col);
    CHECK_HIPSPARSE_ERROR(stat_t, __FILE__, __LINE__);

    return true;
}





template <typename ValueType, typename IndexType>
bool csr_to_dia_hip(int blocksize,
                    IndexType nnz, IndexType nrow, IndexType ncol,
                    const MatrixCSR<ValueType, IndexType>& src,
                    MatrixDIA<ValueType, IndexType>* dst,
                    IndexType* nnz_dia, IndexType* num_diag)
{
    assert(nnz  > 0);
    assert(nrow > 0);
    assert(ncol > 0);
    assert(blocksize > 0);

    assert(dst != NULL);
    assert(nnz_dia != NULL);
    assert(num_diag != NULL);

    // Get diagonal mapping vector
    IndexType* diag_idx = NULL;
    allocate_hip(nrow + ncol, &diag_idx);
    set_to_zero_hip(blocksize, nrow + ncol, diag_idx);

    dim3 diag_blocks((nrow - 1) / blocksize + 1);
    dim3 diag_threads(blocksize);

    hipLaunchKernelGGL((kernel_dia_diag_idx<IndexType>),
                       diag_blocks,
                       diag_threads,
                       0,
                       0,
                       nrow,
                       src.row_offset,
                       src.col,
                       diag_idx);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    // Reduction to obtain number of occupied diagonals
    IndexType* d_num_diag = NULL;
    allocate_hip(1, &d_num_diag);

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // Get reduction buffer size
    hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, diag_idx, d_num_diag, nrow + ncol);

    // Allocate hipcub buffer
    hipMalloc(&d_temp_storage, temp_storage_bytes);

    // Do reduction
    hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, diag_idx, d_num_diag, nrow + ncol);

    // Clear hipcub buffer
    hipFree(d_temp_storage);

    // Copy result to host
    hipMemcpy(num_diag, d_num_diag, sizeof(IndexType), hipMemcpyDeviceToHost);

    // Free device memory
    free_hip(&d_num_diag);

    // Conversion fails if DIA nnz exceeds 5 times CSR nnz
    IndexType size = (nrow > ncol) ? nrow : ncol;
    if(*num_diag > 5 * nnz / size)
    {
        free_hip(&diag_idx);
        return false;
    }

    *nnz_dia = *num_diag * size;

    // Allocate DIA matrix
    allocate_hip(*num_diag, &dst->offset);
    allocate_hip(*nnz_dia, &dst->val);

    // Initialize values with zero
    set_to_zero_hip(blocksize, *num_diag, dst->offset);
    set_to_zero_hip(blocksize, *nnz_dia, dst->val);

    // Inclusive sum to obtain diagonal offsets
    IndexType* work = NULL;
    allocate_hip(nrow + ncol, &work);
    d_temp_storage = NULL;
    temp_storage_bytes = 0;

/*
    // Obtain hipcub buffer size
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, diag_idx, work, nrow + ncol);

    // Allocate hipcub buffer
    hipMalloc(&d_temp_storage, temp_storage_bytes);

    // Do inclusive sum
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, diag_idx, work, nrow + ncol);

    // Clear hipcub buffer
    hipFree(d_temp_storage);
*/

    // TODO
    std::vector<int> tmp(nrow+ncol);
    hipMemcpy(&tmp[1], diag_idx, sizeof(int)*(nrow+ncol-1), hipMemcpyDeviceToHost);
    tmp[0] = 0;
    for(int i=1; i<nrow+ncol; ++i)
        tmp[i] += tmp[i-1];
    hipMemcpy(work, tmp.data(), sizeof(int)*(nrow+ncol), hipMemcpyHostToDevice);
    // TODO END

    // Fill DIA structures
    dim3 fill_blocks((nrow + ncol) / blocksize + 1);
    dim3 fill_threads(blocksize);

    hipLaunchKernelGGL((kernel_dia_fill_offset<IndexType>),
                       fill_blocks,
                       fill_threads,
                       0,
                       0,
                       nrow,
                       ncol,
                       diag_idx,
                       work,
                       dst->offset);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    free_hip(&work);

    dim3 conv_blocks((nrow - 1) / blocksize + 1);
    dim3 conv_threads(blocksize);

    hipLaunchKernelGGL((kernel_dia_convert<ValueType, IndexType>),
                       conv_blocks,
                       conv_threads,
                       0,
                       0,
                       nrow,
                       *num_diag,
                       src.row_offset,
                       src.col,
                       src.val,
                       diag_idx,
                       dst->val);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    // Clear
    free_hip(&diag_idx);

    return true;
}

template <typename ValueType, typename IndexType>
bool csr_to_hyb_hip(int blocksize,
                    IndexType nnz, IndexType nrow, IndexType ncol,
                    const MatrixCSR<ValueType, IndexType>& src,
                    MatrixHYB<ValueType, IndexType>* dst,
                    IndexType* nnz_hyb, IndexType* nnz_ell, IndexType* nnz_coo)
{
    assert(nnz  > 0);
    assert(nrow > 0);
    assert(ncol > 0);
    assert(blocksize > 0);

    assert(dst != NULL);
    assert(nnz_hyb != NULL);
    assert(nnz_ell != NULL);
    assert(nnz_coo != NULL);

    // Determine ELL width by average nnz per row
    if(dst->ELL.max_row == 0)
    {
        dst->ELL.max_row = (nnz - 1) / nrow + 1;
    }

    // ELL nnz is ELL width times nrow
    *nnz_ell = dst->ELL.max_row * nrow;
    *nnz_coo = 0;

    // Allocate ELL part
    allocate_hip(*nnz_ell, &dst->ELL.col);
    allocate_hip(*nnz_ell, &dst->ELL.val);

    // Array to gold COO part nnz per row
    IndexType *coo_row_nnz = NULL;
    allocate_hip(nrow + 1, &coo_row_nnz);

    // If there is no ELL part, its easy
    if(*nnz_ell == 0)
    {
        *nnz_coo = nnz;
        hipMemcpy(coo_row_nnz, src.row_offset, sizeof(IndexType) * (nrow + 1), hipMemcpyDeviceToDevice);
    }
    else
    {
        dim3 blocks((nrow - 1) / blocksize + 1);
        dim3 threads(blocksize);

        hipLaunchKernelGGL((kernel_hyb_coo_nnz),
                           blocks,
                           threads,
                           0,
                           0,
                           nrow,
                           dst->ELL.max_row,
                           src.row_offset,
                           coo_row_nnz);
        CHECK_HIP_ERROR(__FILE__, __LINE__);

        // Inclusive sum on coo_row_nnz
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;

        // Obtain hipcub buffer size
        hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, coo_row_nnz, coo_row_nnz + 1, nrow);

        // Allocate hipcub buffer
        hipMalloc(&d_temp_storage, temp_storage_bytes);

        // Do inclusive sum
        hipcub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, coo_row_nnz, coo_row_nnz + 1, nrow);

        // Clear hipcub buffer
        hipFree(d_temp_storage);

        set_to_zero_hip(blocksize, 1, coo_row_nnz);

        // Copy result to host
        hipMemcpy(nnz_coo, coo_row_nnz + nrow, sizeof(IndexType), hipMemcpyDeviceToHost);
    }

    *nnz_hyb = *nnz_coo + *nnz_ell;

    if(*nnz_hyb <= 0)
    {
        return false;
    }

    // Allocate COO part
    allocate_hip(*nnz_coo, &dst->COO.row);
    allocate_hip(*nnz_coo, &dst->COO.col);
    allocate_hip(*nnz_coo, &dst->COO.val);

    // Fill HYB structures
    dim3 blocks((nrow - 1) / blocksize + 1);
    dim3 threads(blocksize);

    hipLaunchKernelGGL((kernel_hyb_csr2hyb<ValueType>),
                       blocks,
                       threads,
                       0,
                       0,
                       nrow,
                       src.val,
                       src.row_offset,
                       src.col,
                       dst->ELL.max_row,
                       dst->ELL.col,
                       dst->ELL.val,
                       dst->COO.row,
                       dst->COO.col,
                       dst->COO.val,
                       coo_row_nnz);
    CHECK_HIP_ERROR(__FILE__, __LINE__);

    free_hip(&coo_row_nnz);

    return true;
}

// csr_to_coo
template bool csr_to_coo_hip(const hipsparseHandle_t handle,
                             int nnz, int nrow, int ncol,
                             const MatrixCSR<float, int> &src,
                             MatrixCOO<float, int> *dst);

template bool csr_to_coo_hip(const hipsparseHandle_t handle,
                             int nnz, int nrow, int ncol,
                             const MatrixCSR<double, int> &src,
                             MatrixCOO<double, int> *dst);

// coo_to_csr
template bool coo_to_csr_hip(const hipsparseHandle_t handle,
                             int nnz, int nrow, int ncol,
                             const MatrixCOO<float, int> &src,
                             MatrixCSR<float, int> *dst);

template bool coo_to_csr_hip(const hipsparseHandle_t handle,
                             int nnz, int nrow, int ncol,
                             const MatrixCOO<double, int> &src,
                             MatrixCSR<double, int> *dst);

// csr_to_ell
template bool csr_to_ell_hip(const hipsparseHandle_t handle,
                             int nnz, int nrow, int ncol,
                             const MatrixCSR<float, int> &src,
                             const hipsparseMatDescr_t src_descr,
                             MatrixELL<float, int> *dst,
                             const hipsparseMatDescr_t dst_descr,
                             int *nnz_ell);

template bool csr_to_ell_hip(const hipsparseHandle_t handle,
                             int nnz, int nrow, int ncol,
                             const MatrixCSR<double, int>& src,
                             const hipsparseMatDescr_t src_descr,
                             MatrixELL<double, int>* dst,
                             const hipsparseMatDescr_t dst_descr,
                             int* nnz_ell);

// csr_to_dia
template bool csr_to_dia_hip(int blocksize,
                             int nnz, int nrow, int ncol,
                             const MatrixCSR<float, int>& src,
                             MatrixDIA<float, int>* dst,
                             int* nnz_dia,
                             int* num_diag);

template bool csr_to_dia_hip(int blocksize,
                             int nnz, int nrow, int ncol,
                             const MatrixCSR<double, int>& src,
                             MatrixDIA<double, int>* dst,
                             int* nnz_dia,
                             int* num_diag);

// csr_to_hyb
template bool csr_to_hyb_hip(int blocksize,
                             int nnz, int nrow, int ncol,
                             const MatrixCSR<float, int>& src,
                             MatrixHYB<float, int>* dst,
                             int* nnz_hyb, int* nnz_ell, int* nnz_coo);

template bool csr_to_hyb_hip(int blocksize,
                             int nnz, int nrow, int ncol,
                             const MatrixCSR<double, int>& src,
                             MatrixHYB<double, int>* dst,
                             int* nnz_hyb, int* nnz_ell, int* nnz_coo);

} // namespace rocalution
