/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HOST_CONVERSION_HPP_
#define ROCALUTION_HOST_CONVERSION_HPP_

#include "../matrix_formats.hpp"

namespace rocalution
{

    template <typename ValueType, typename IndexType>
    bool csr_to_coo(int                                    omp_threads,
                    IndexType                              nnz,
                    IndexType                              nrow,
                    IndexType                              ncol,
                    const MatrixCSR<ValueType, IndexType>& src,
                    MatrixCOO<ValueType, IndexType>*       dst);

    template <typename ValueType, typename IndexType>
    bool csr_to_mcsr(int                                    omp_threads,
                     IndexType                              nnz,
                     IndexType                              nrow,
                     IndexType                              ncol,
                     const MatrixCSR<ValueType, IndexType>& src,
                     MatrixMCSR<ValueType, IndexType>*      dst);

    template <typename ValueType, typename IndexType>
    bool csr_to_dia(int                                    omp_threads,
                    IndexType                              nnz,
                    IndexType                              nrow,
                    IndexType                              ncol,
                    const MatrixCSR<ValueType, IndexType>& src,
                    MatrixDIA<ValueType, IndexType>*       dst,
                    IndexType*                             nnz_dia);

    template <typename ValueType, typename IndexType>
    bool csr_to_dense(int                                    omp_threads,
                      IndexType                              nnz,
                      IndexType                              nrow,
                      IndexType                              ncol,
                      const MatrixCSR<ValueType, IndexType>& src,
                      MatrixDENSE<ValueType>*                dst);

    template <typename ValueType, typename IndexType>
    bool csr_to_ell(int                                    omp_threads,
                    IndexType                              nnz,
                    IndexType                              nrow,
                    IndexType                              ncol,
                    const MatrixCSR<ValueType, IndexType>& src,
                    MatrixELL<ValueType, IndexType>*       dst,
                    IndexType*                             nnz_ell);

    template <typename ValueType, typename IndexType>
    bool csr_to_hyb(int                                    omp_threads,
                    IndexType                              nnz,
                    IndexType                              nrow,
                    IndexType                              ncol,
                    const MatrixCSR<ValueType, IndexType>& src,
                    MatrixHYB<ValueType, IndexType>*       dst,
                    IndexType*                             nnz_hyb,
                    IndexType*                             nnz_ell,
                    IndexType*                             nnz_coo);

    template <typename ValueType, typename IndexType>
    bool dense_to_csr(int                              omp_threads,
                      IndexType                        nrow,
                      IndexType                        ncol,
                      const MatrixDENSE<ValueType>&    src,
                      MatrixCSR<ValueType, IndexType>* dst,
                      IndexType*                       nnz);

    template <typename ValueType, typename IndexType>
    bool dia_to_csr(int                                    omp_threads,
                    IndexType                              nnz,
                    IndexType                              nrow,
                    IndexType                              ncol,
                    const MatrixDIA<ValueType, IndexType>& src,
                    MatrixCSR<ValueType, IndexType>*       dst,
                    IndexType*                             nnz_csr);

    template <typename ValueType, typename IndexType>
    bool ell_to_csr(int                                    omp_threads,
                    IndexType                              nnz,
                    IndexType                              nrow,
                    IndexType                              ncol,
                    const MatrixELL<ValueType, IndexType>& src,
                    MatrixCSR<ValueType, IndexType>*       dst,
                    IndexType*                             nnz_csr);

    template <typename ValueType, typename IndexType>
    bool coo_to_csr(int                                    omp_threads,
                    IndexType                              nnz,
                    IndexType                              nrow,
                    IndexType                              ncol,
                    const MatrixCOO<ValueType, IndexType>& src,
                    MatrixCSR<ValueType, IndexType>*       dst);

    template <typename ValueType, typename IndexType>
    bool mcsr_to_csr(int                                     omp_threads,
                     IndexType                               nnz,
                     IndexType                               nrow,
                     IndexType                               ncol,
                     const MatrixMCSR<ValueType, IndexType>& src,
                     MatrixCSR<ValueType, IndexType>*        dst);

    template <typename ValueType, typename IndexType>
    bool hyb_to_csr(int                                    omp_threads,
                    IndexType                              nnz,
                    IndexType                              nrow,
                    IndexType                              ncol,
                    IndexType                              nnz_ell,
                    IndexType                              nnz_coo,
                    const MatrixHYB<ValueType, IndexType>& src,
                    MatrixCSR<ValueType, IndexType>*       dst,
                    IndexType*                             nnz_csr);

} // namespace rocalution

#endif // ROCALUTION_HOST_CONVERSION_HPP_
