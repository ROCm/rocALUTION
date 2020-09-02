/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
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

#include "hip_sparse.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"

#include <complex>
#include <rocsparse.h>

namespace rocalution
{

    // rocsparse csrmv analysis
    template <>
    rocsparse_status rocsparseTcsrmv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              int                       m,
                                              int                       n,
                                              int                       nnz,
                                              const rocsparse_mat_descr descr,
                                              const float*              csr_val,
                                              const int*                csr_row_ptr,
                                              const int*                csr_col_ind,
                                              rocsparse_mat_info        info)
    {
        return rocsparse_scsrmv_analysis(
            handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
    }

    template <>
    rocsparse_status rocsparseTcsrmv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              int                       m,
                                              int                       n,
                                              int                       nnz,
                                              const rocsparse_mat_descr descr,
                                              const double*             csr_val,
                                              const int*                csr_row_ptr,
                                              const int*                csr_col_ind,
                                              rocsparse_mat_info        info)
    {
        return rocsparse_dcsrmv_analysis(
            handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
    }

    template <>
    rocsparse_status rocsparseTcsrmv_analysis(rocsparse_handle           handle,
                                              rocsparse_operation        trans,
                                              int                        m,
                                              int                        n,
                                              int                        nnz,
                                              const rocsparse_mat_descr  descr,
                                              const std::complex<float>* csr_val,
                                              const int*                 csr_row_ptr,
                                              const int*                 csr_col_ind,
                                              rocsparse_mat_info         info)
    {
        return rocsparse_ccsrmv_analysis(handle,
                                         trans,
                                         m,
                                         n,
                                         nnz,
                                         descr,
                                         (const rocsparse_float_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info);
    }

    template <>
    rocsparse_status rocsparseTcsrmv_analysis(rocsparse_handle            handle,
                                              rocsparse_operation         trans,
                                              int                         m,
                                              int                         n,
                                              int                         nnz,
                                              const rocsparse_mat_descr   descr,
                                              const std::complex<double>* csr_val,
                                              const int*                  csr_row_ptr,
                                              const int*                  csr_col_ind,
                                              rocsparse_mat_info          info)
    {
        return rocsparse_zcsrmv_analysis(handle,
                                         trans,
                                         m,
                                         n,
                                         nnz,
                                         descr,
                                         (const rocsparse_double_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info);
    }

    // rocsparse csrmv
    template <>
    rocsparse_status rocsparseTcsrmv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              csr_val,
                                     const int*                csr_row_ptr,
                                     const int*                csr_col_ind,
                                     rocsparse_mat_info        info,
                                     const float*              x,
                                     const float*              beta,
                                     float*                    y)
    {
        return rocsparse_scsrmv(handle,
                                trans,
                                m,
                                n,
                                nnz,
                                alpha,
                                descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                info,
                                x,
                                beta,
                                y);
    }

    template <>
    rocsparse_status rocsparseTcsrmv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             csr_val,
                                     const int*                csr_row_ptr,
                                     const int*                csr_col_ind,
                                     rocsparse_mat_info        info,
                                     const double*             x,
                                     const double*             beta,
                                     double*                   y)
    {
        return rocsparse_dcsrmv(handle,
                                trans,
                                m,
                                n,
                                nnz,
                                alpha,
                                descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                info,
                                x,
                                beta,
                                y);
    }

    template <>
    rocsparse_status rocsparseTcsrmv(rocsparse_handle           handle,
                                     rocsparse_operation        trans,
                                     int                        m,
                                     int                        n,
                                     int                        nnz,
                                     const std::complex<float>* alpha,
                                     const rocsparse_mat_descr  descr,
                                     const std::complex<float>* csr_val,
                                     const int*                 csr_row_ptr,
                                     const int*                 csr_col_ind,
                                     rocsparse_mat_info         info,
                                     const std::complex<float>* x,
                                     const std::complex<float>* beta,
                                     std::complex<float>*       y)
    {
        return rocsparse_ccsrmv(handle,
                                trans,
                                m,
                                n,
                                nnz,
                                (const rocsparse_float_complex*)alpha,
                                descr,
                                (const rocsparse_float_complex*)csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                info,
                                (const rocsparse_float_complex*)x,
                                (const rocsparse_float_complex*)beta,
                                (rocsparse_float_complex*)y);
    }

    template <>
    rocsparse_status rocsparseTcsrmv(rocsparse_handle            handle,
                                     rocsparse_operation         trans,
                                     int                         m,
                                     int                         n,
                                     int                         nnz,
                                     const std::complex<double>* alpha,
                                     const rocsparse_mat_descr   descr,
                                     const std::complex<double>* csr_val,
                                     const int*                  csr_row_ptr,
                                     const int*                  csr_col_ind,
                                     rocsparse_mat_info          info,
                                     const std::complex<double>* x,
                                     const std::complex<double>* beta,
                                     std::complex<double>*       y)
    {
        return rocsparse_zcsrmv(handle,
                                trans,
                                m,
                                n,
                                nnz,
                                (const rocsparse_double_complex*)alpha,
                                descr,
                                (const rocsparse_double_complex*)csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                info,
                                (const rocsparse_double_complex*)x,
                                (const rocsparse_double_complex*)beta,
                                (rocsparse_double_complex*)y);
    }

    // rocsparse csrsv buffer size
    template <>
    rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 int                       m,
                                                 int                       nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const float*              csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        return rocsparse_scsrsv_buffer_size(
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 int                       m,
                                                 int                       nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const double*             csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        return rocsparse_dcsrsv_buffer_size(
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle           handle,
                                                 rocsparse_operation        trans,
                                                 int                        m,
                                                 int                        nnz,
                                                 const rocsparse_mat_descr  descr,
                                                 const std::complex<float>* csr_val,
                                                 const int*                 csr_row_ptr,
                                                 const int*                 csr_col_ind,
                                                 rocsparse_mat_info         info,
                                                 size_t*                    buffer_size)
    {
        return rocsparse_ccsrsv_buffer_size(handle,
                                            trans,
                                            m,
                                            nnz,
                                            descr,
                                            (const rocsparse_float_complex*)csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            info,
                                            buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle            handle,
                                                 rocsparse_operation         trans,
                                                 int                         m,
                                                 int                         nnz,
                                                 const rocsparse_mat_descr   descr,
                                                 const std::complex<double>* csr_val,
                                                 const int*                  csr_row_ptr,
                                                 const int*                  csr_col_ind,
                                                 rocsparse_mat_info          info,
                                                 size_t*                     buffer_size)
    {
        return rocsparse_zcsrsv_buffer_size(handle,
                                            trans,
                                            m,
                                            nnz,
                                            descr,
                                            (const rocsparse_double_complex*)csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            info,
                                            buffer_size);
    }

    // rocsparse csrsv analysis
    template <>
    rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              int                       m,
                                              int                       nnz,
                                              const rocsparse_mat_descr descr,
                                              const float*              csr_val,
                                              const int*                csr_row_ptr,
                                              const int*                csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer)
    {
        return rocsparse_scsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              int                       m,
                                              int                       nnz,
                                              const rocsparse_mat_descr descr,
                                              const double*             csr_val,
                                              const int*                csr_row_ptr,
                                              const int*                csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer)
    {
        return rocsparse_dcsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle           handle,
                                              rocsparse_operation        trans,
                                              int                        m,
                                              int                        nnz,
                                              const rocsparse_mat_descr  descr,
                                              const std::complex<float>* csr_val,
                                              const int*                 csr_row_ptr,
                                              const int*                 csr_col_ind,
                                              rocsparse_mat_info         info,
                                              rocsparse_analysis_policy  analysis,
                                              rocsparse_solve_policy     solve,
                                              void*                      temp_buffer)
    {
        return rocsparse_ccsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         (const rocsparse_float_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle            handle,
                                              rocsparse_operation         trans,
                                              int                         m,
                                              int                         nnz,
                                              const rocsparse_mat_descr   descr,
                                              const std::complex<double>* csr_val,
                                              const int*                  csr_row_ptr,
                                              const int*                  csr_col_ind,
                                              rocsparse_mat_info          info,
                                              rocsparse_analysis_policy   analysis,
                                              rocsparse_solve_policy      solve,
                                              void*                       temp_buffer)
    {
        return rocsparse_zcsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         (const rocsparse_double_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    // rocsparse csrsv
    template <>
    rocsparse_status rocsparseTcsrsv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       nnz,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              csr_val,
                                     const int*                csr_row_ptr,
                                     const int*                csr_col_ind,
                                     rocsparse_mat_info        info,
                                     const float*              x,
                                     float*                    y,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer)
    {
        return rocsparse_scsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      alpha,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      x,
                                      y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       nnz,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             csr_val,
                                     const int*                csr_row_ptr,
                                     const int*                csr_col_ind,
                                     rocsparse_mat_info        info,
                                     const double*             x,
                                     double*                   y,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer)
    {
        return rocsparse_dcsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      alpha,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      x,
                                      y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv(rocsparse_handle           handle,
                                     rocsparse_operation        trans,
                                     int                        m,
                                     int                        nnz,
                                     const std::complex<float>* alpha,
                                     const rocsparse_mat_descr  descr,
                                     const std::complex<float>* csr_val,
                                     const int*                 csr_row_ptr,
                                     const int*                 csr_col_ind,
                                     rocsparse_mat_info         info,
                                     const std::complex<float>* x,
                                     std::complex<float>*       y,
                                     rocsparse_solve_policy     policy,
                                     void*                      temp_buffer)
    {
        return rocsparse_ccsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      (const rocsparse_float_complex*)alpha,
                                      descr,
                                      (const rocsparse_float_complex*)csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      (const rocsparse_float_complex*)x,
                                      (rocsparse_float_complex*)y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv(rocsparse_handle            handle,
                                     rocsparse_operation         trans,
                                     int                         m,
                                     int                         nnz,
                                     const std::complex<double>* alpha,
                                     const rocsparse_mat_descr   descr,
                                     const std::complex<double>* csr_val,
                                     const int*                  csr_row_ptr,
                                     const int*                  csr_col_ind,
                                     rocsparse_mat_info          info,
                                     const std::complex<double>* x,
                                     std::complex<double>*       y,
                                     rocsparse_solve_policy      policy,
                                     void*                       temp_buffer)
    {
        return rocsparse_zcsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      (const rocsparse_double_complex*)alpha,
                                      descr,
                                      (const rocsparse_double_complex*)csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      (const rocsparse_double_complex*)x,
                                      (rocsparse_double_complex*)y,
                                      policy,
                                      temp_buffer);
    }

    // rocsparse coomv
    template <>
    rocsparse_status rocsparseTcoomv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              coo_val,
                                     const int*                coo_row_ind,
                                     const int*                coo_col_ind,
                                     const float*              x,
                                     const float*              beta,
                                     float*                    y)
    {
        return rocsparse_scoomv(
            handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
    }

    template <>
    rocsparse_status rocsparseTcoomv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             coo_val,
                                     const int*                coo_row_ind,
                                     const int*                coo_col_ind,
                                     const double*             x,
                                     const double*             beta,
                                     double*                   y)
    {
        return rocsparse_dcoomv(
            handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
    }

    template <>
    rocsparse_status rocsparseTcoomv(rocsparse_handle           handle,
                                     rocsparse_operation        trans,
                                     int                        m,
                                     int                        n,
                                     int                        nnz,
                                     const std::complex<float>* alpha,
                                     const rocsparse_mat_descr  descr,
                                     const std::complex<float>* coo_val,
                                     const int*                 coo_row_ind,
                                     const int*                 coo_col_ind,
                                     const std::complex<float>* x,
                                     const std::complex<float>* beta,
                                     std::complex<float>*       y)
    {
        return rocsparse_ccoomv(handle,
                                trans,
                                m,
                                n,
                                nnz,
                                (const rocsparse_float_complex*)alpha,
                                descr,
                                (const rocsparse_float_complex*)coo_val,
                                coo_row_ind,
                                coo_col_ind,
                                (const rocsparse_float_complex*)x,
                                (const rocsparse_float_complex*)beta,
                                (rocsparse_float_complex*)y);
    }

    template <>
    rocsparse_status rocsparseTcoomv(rocsparse_handle            handle,
                                     rocsparse_operation         trans,
                                     int                         m,
                                     int                         n,
                                     int                         nnz,
                                     const std::complex<double>* alpha,
                                     const rocsparse_mat_descr   descr,
                                     const std::complex<double>* coo_val,
                                     const int*                  coo_row_ind,
                                     const int*                  coo_col_ind,
                                     const std::complex<double>* x,
                                     const std::complex<double>* beta,
                                     std::complex<double>*       y)
    {
        return rocsparse_zcoomv(handle,
                                trans,
                                m,
                                n,
                                nnz,
                                (const rocsparse_double_complex*)alpha,
                                descr,
                                (const rocsparse_double_complex*)coo_val,
                                coo_row_ind,
                                coo_col_ind,
                                (const rocsparse_double_complex*)x,
                                (const rocsparse_double_complex*)beta,
                                (rocsparse_double_complex*)y);
    }

    // rocsparse ellmv
    template <>
    rocsparse_status rocsparseTellmv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              ell_val,
                                     const int*                ell_col_ind,
                                     int                       ell_width,
                                     const float*              x,
                                     const float*              beta,
                                     float*                    y)
    {
        return rocsparse_sellmv(
            handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
    }

    template <>
    rocsparse_status rocsparseTellmv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             ell_val,
                                     const int*                ell_col_ind,
                                     int                       ell_width,
                                     const double*             x,
                                     const double*             beta,
                                     double*                   y)
    {
        return rocsparse_dellmv(
            handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
    }

    template <>
    rocsparse_status rocsparseTellmv(rocsparse_handle           handle,
                                     rocsparse_operation        trans,
                                     int                        m,
                                     int                        n,
                                     const std::complex<float>* alpha,
                                     const rocsparse_mat_descr  descr,
                                     const std::complex<float>* ell_val,
                                     const int*                 ell_col_ind,
                                     int                        ell_width,
                                     const std::complex<float>* x,
                                     const std::complex<float>* beta,
                                     std::complex<float>*       y)
    {
        return rocsparse_cellmv(handle,
                                trans,
                                m,
                                n,
                                (const rocsparse_float_complex*)alpha,
                                descr,
                                (const rocsparse_float_complex*)ell_val,
                                ell_col_ind,
                                ell_width,
                                (const rocsparse_float_complex*)x,
                                (const rocsparse_float_complex*)beta,
                                (rocsparse_float_complex*)y);
    }

    template <>
    rocsparse_status rocsparseTellmv(rocsparse_handle            handle,
                                     rocsparse_operation         trans,
                                     int                         m,
                                     int                         n,
                                     const std::complex<double>* alpha,
                                     const rocsparse_mat_descr   descr,
                                     const std::complex<double>* ell_val,
                                     const int*                  ell_col_ind,
                                     int                         ell_width,
                                     const std::complex<double>* x,
                                     const std::complex<double>* beta,
                                     std::complex<double>*       y)
    {
        return rocsparse_zellmv(handle,
                                trans,
                                m,
                                n,
                                (const rocsparse_double_complex*)alpha,
                                descr,
                                (const rocsparse_double_complex*)ell_val,
                                ell_col_ind,
                                ell_width,
                                (const rocsparse_double_complex*)x,
                                (const rocsparse_double_complex*)beta,
                                (rocsparse_double_complex*)y);
    }

    // rocsparse bsrmv
    template <>
    rocsparse_status rocsparseTbsrmv(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_operation       trans,
                                     int                       mb,
                                     int                       nb,
                                     int                       nnzb,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              bsr_val,
                                     const int*                bsr_row_ptr,
                                     const int*                bsr_col_ind,
                                     int                       bsr_dim,
                                     const float*              x,
                                     const float*              beta,
                                     float*                    y)
    {
        return rocsparse_sbsrmv(handle,
                                dir,
                                trans,
                                mb,
                                nb,
                                nnzb,
                                alpha,
                                descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                bsr_dim,
                                x,
                                beta,
                                y);
    }

    template <>
    rocsparse_status rocsparseTbsrmv(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_operation       trans,
                                     int                       mb,
                                     int                       nb,
                                     int                       nnzb,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             bsr_val,
                                     const int*                bsr_row_ptr,
                                     const int*                bsr_col_ind,
                                     int                       bsr_dim,
                                     const double*             x,
                                     const double*             beta,
                                     double*                   y)
    {
        return rocsparse_dbsrmv(handle,
                                dir,
                                trans,
                                mb,
                                nb,
                                nnzb,
                                alpha,
                                descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                bsr_dim,
                                x,
                                beta,
                                y);
    }

    template <>
    rocsparse_status rocsparseTbsrmv(rocsparse_handle           handle,
                                     rocsparse_direction        dir,
                                     rocsparse_operation        trans,
                                     int                        mb,
                                     int                        nb,
                                     int                        nnzb,
                                     const std::complex<float>* alpha,
                                     const rocsparse_mat_descr  descr,
                                     const std::complex<float>* bsr_val,
                                     const int*                 bsr_row_ptr,
                                     const int*                 bsr_col_ind,
                                     int                        bsr_dim,
                                     const std::complex<float>* x,
                                     const std::complex<float>* beta,
                                     std::complex<float>*       y)
    {
        return rocsparse_cbsrmv(handle,
                                dir,
                                trans,
                                mb,
                                nb,
                                nnzb,
                                (const rocsparse_float_complex*)alpha,
                                descr,
                                (const rocsparse_float_complex*)bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                bsr_dim,
                                (const rocsparse_float_complex*)x,
                                (const rocsparse_float_complex*)beta,
                                (rocsparse_float_complex*)y);
    }

    template <>
    rocsparse_status rocsparseTbsrmv(rocsparse_handle            handle,
                                     rocsparse_direction         dir,
                                     rocsparse_operation         trans,
                                     int                         mb,
                                     int                         nb,
                                     int                         nnzb,
                                     const std::complex<double>* alpha,
                                     const rocsparse_mat_descr   descr,
                                     const std::complex<double>* bsr_val,
                                     const int*                  bsr_row_ptr,
                                     const int*                  bsr_col_ind,
                                     int                         bsr_dim,
                                     const std::complex<double>* x,
                                     const std::complex<double>* beta,
                                     std::complex<double>*       y)
    {
        return rocsparse_zbsrmv(handle,
                                dir,
                                trans,
                                mb,
                                nb,
                                nnzb,
                                (const rocsparse_double_complex*)alpha,
                                descr,
                                (const rocsparse_double_complex*)bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                bsr_dim,
                                (const rocsparse_double_complex*)x,
                                (const rocsparse_double_complex*)beta,
                                (rocsparse_double_complex*)y);
    }

    // rocsparse csrgemm
    template <>
    rocsparse_status rocsparseTcsrgemm_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const float*              alpha,
                                                   const rocsparse_mat_descr descr_A,
                                                   int                       nnz_A,
                                                   const int*                csr_row_ptr_A,
                                                   const int*                csr_col_ind_A,
                                                   const rocsparse_mat_descr descr_B,
                                                   int                       nnz_B,
                                                   const int*                csr_row_ptr_B,
                                                   const int*                csr_col_ind_B,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_scsrgemm_buffer_size(handle,
                                              trans_A,
                                              trans_B,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              descr_A,
                                              nnz_A,
                                              csr_row_ptr_A,
                                              csr_col_ind_A,
                                              descr_B,
                                              nnz_B,
                                              csr_row_ptr_B,
                                              csr_col_ind_B,
                                              NULL,
                                              NULL,
                                              0,
                                              NULL,
                                              NULL,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const double*             alpha,
                                                   const rocsparse_mat_descr descr_A,
                                                   int                       nnz_A,
                                                   const int*                csr_row_ptr_A,
                                                   const int*                csr_col_ind_A,
                                                   const rocsparse_mat_descr descr_B,
                                                   int                       nnz_B,
                                                   const int*                csr_row_ptr_B,
                                                   const int*                csr_col_ind_B,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_dcsrgemm_buffer_size(handle,
                                              trans_A,
                                              trans_B,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              descr_A,
                                              nnz_A,
                                              csr_row_ptr_A,
                                              csr_col_ind_A,
                                              descr_B,
                                              nnz_B,
                                              csr_row_ptr_B,
                                              csr_col_ind_B,
                                              NULL,
                                              NULL,
                                              0,
                                              NULL,
                                              NULL,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm_buffer_size(rocsparse_handle           handle,
                                                   rocsparse_operation        trans_A,
                                                   rocsparse_operation        trans_B,
                                                   int                        m,
                                                   int                        n,
                                                   int                        k,
                                                   const std::complex<float>* alpha,
                                                   const rocsparse_mat_descr  descr_A,
                                                   int                        nnz_A,
                                                   const int*                 csr_row_ptr_A,
                                                   const int*                 csr_col_ind_A,
                                                   const rocsparse_mat_descr  descr_B,
                                                   int                        nnz_B,
                                                   const int*                 csr_row_ptr_B,
                                                   const int*                 csr_col_ind_B,
                                                   rocsparse_mat_info         info,
                                                   size_t*                    buffer_size)
    {
        return rocsparse_ccsrgemm_buffer_size(handle,
                                              trans_A,
                                              trans_B,
                                              m,
                                              n,
                                              k,
                                              (const rocsparse_float_complex*)alpha,
                                              descr_A,
                                              nnz_A,
                                              csr_row_ptr_A,
                                              csr_col_ind_A,
                                              descr_B,
                                              nnz_B,
                                              csr_row_ptr_B,
                                              csr_col_ind_B,
                                              NULL,
                                              NULL,
                                              0,
                                              NULL,
                                              NULL,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm_buffer_size(rocsparse_handle            handle,
                                                   rocsparse_operation         trans_A,
                                                   rocsparse_operation         trans_B,
                                                   int                         m,
                                                   int                         n,
                                                   int                         k,
                                                   const std::complex<double>* alpha,
                                                   const rocsparse_mat_descr   descr_A,
                                                   int                         nnz_A,
                                                   const int*                  csr_row_ptr_A,
                                                   const int*                  csr_col_ind_A,
                                                   const rocsparse_mat_descr   descr_B,
                                                   int                         nnz_B,
                                                   const int*                  csr_row_ptr_B,
                                                   const int*                  csr_col_ind_B,
                                                   rocsparse_mat_info          info,
                                                   size_t*                     buffer_size)
    {
        return rocsparse_zcsrgemm_buffer_size(handle,
                                              trans_A,
                                              trans_B,
                                              m,
                                              n,
                                              k,
                                              (const rocsparse_double_complex*)alpha,
                                              descr_A,
                                              nnz_A,
                                              csr_row_ptr_A,
                                              csr_col_ind_A,
                                              descr_B,
                                              nnz_B,
                                              csr_row_ptr_B,
                                              csr_col_ind_B,
                                              NULL,
                                              NULL,
                                              0,
                                              NULL,
                                              NULL,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       const float*              alpha,
                                       const rocsparse_mat_descr descr_A,
                                       int                       nnz_A,
                                       const float*              csr_val_A,
                                       const int*                csr_row_ptr_A,
                                       const int*                csr_col_ind_A,
                                       const rocsparse_mat_descr descr_B,
                                       int                       nnz_B,
                                       const float*              csr_val_B,
                                       const int*                csr_row_ptr_B,
                                       const int*                csr_col_ind_B,
                                       const rocsparse_mat_descr descr_C,
                                       float*                    csr_val_C,
                                       const int*                csr_row_ptr_C,
                                       int*                      csr_col_ind_C,
                                       rocsparse_mat_info        info,
                                       void*                     temp_buffer)
    {
        return rocsparse_scsrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  descr_A,
                                  nnz_A,
                                  csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  descr_B,
                                  nnz_B,
                                  csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  NULL,
                                  NULL,
                                  0,
                                  NULL,
                                  NULL,
                                  NULL,
                                  descr_C,
                                  csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C,
                                  info,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       const double*             alpha,
                                       const rocsparse_mat_descr descr_A,
                                       int                       nnz_A,
                                       const double*             csr_val_A,
                                       const int*                csr_row_ptr_A,
                                       const int*                csr_col_ind_A,
                                       const rocsparse_mat_descr descr_B,
                                       int                       nnz_B,
                                       const double*             csr_val_B,
                                       const int*                csr_row_ptr_B,
                                       const int*                csr_col_ind_B,
                                       const rocsparse_mat_descr descr_C,
                                       double*                   csr_val_C,
                                       const int*                csr_row_ptr_C,
                                       int*                      csr_col_ind_C,
                                       rocsparse_mat_info        info,
                                       void*                     temp_buffer)
    {
        return rocsparse_dcsrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  descr_A,
                                  nnz_A,
                                  csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  descr_B,
                                  nnz_B,
                                  csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  NULL,
                                  NULL,
                                  0,
                                  NULL,
                                  NULL,
                                  NULL,
                                  descr_C,
                                  csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C,
                                  info,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm(rocsparse_handle           handle,
                                       rocsparse_operation        trans_A,
                                       rocsparse_operation        trans_B,
                                       int                        m,
                                       int                        n,
                                       int                        k,
                                       const std::complex<float>* alpha,
                                       const rocsparse_mat_descr  descr_A,
                                       int                        nnz_A,
                                       const std::complex<float>* csr_val_A,
                                       const int*                 csr_row_ptr_A,
                                       const int*                 csr_col_ind_A,
                                       const rocsparse_mat_descr  descr_B,
                                       int                        nnz_B,
                                       const std::complex<float>* csr_val_B,
                                       const int*                 csr_row_ptr_B,
                                       const int*                 csr_col_ind_B,
                                       const rocsparse_mat_descr  descr_C,
                                       std::complex<float>*       csr_val_C,
                                       const int*                 csr_row_ptr_C,
                                       int*                       csr_col_ind_C,
                                       rocsparse_mat_info         info,
                                       void*                      temp_buffer)
    {
        return rocsparse_ccsrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  n,
                                  k,
                                  (const rocsparse_float_complex*)alpha,
                                  descr_A,
                                  nnz_A,
                                  (const rocsparse_float_complex*)csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  descr_B,
                                  nnz_B,
                                  (const rocsparse_float_complex*)csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  NULL,
                                  NULL,
                                  0,
                                  NULL,
                                  NULL,
                                  NULL,
                                  descr_C,
                                  (rocsparse_float_complex*)csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C,
                                  info,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm(rocsparse_handle            handle,
                                       rocsparse_operation         trans_A,
                                       rocsparse_operation         trans_B,
                                       int                         m,
                                       int                         n,
                                       int                         k,
                                       const std::complex<double>* alpha,
                                       const rocsparse_mat_descr   descr_A,
                                       int                         nnz_A,
                                       const std::complex<double>* csr_val_A,
                                       const int*                  csr_row_ptr_A,
                                       const int*                  csr_col_ind_A,
                                       const rocsparse_mat_descr   descr_B,
                                       int                         nnz_B,
                                       const std::complex<double>* csr_val_B,
                                       const int*                  csr_row_ptr_B,
                                       const int*                  csr_col_ind_B,
                                       const rocsparse_mat_descr   descr_C,
                                       std::complex<double>*       csr_val_C,
                                       const int*                  csr_row_ptr_C,
                                       int*                        csr_col_ind_C,
                                       rocsparse_mat_info          info,
                                       void*                       temp_buffer)
    {
        return rocsparse_zcsrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  n,
                                  k,
                                  (const rocsparse_double_complex*)alpha,
                                  descr_A,
                                  nnz_A,
                                  (const rocsparse_double_complex*)csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  descr_B,
                                  nnz_B,
                                  (const rocsparse_double_complex*)csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  NULL,
                                  NULL,
                                  0,
                                  NULL,
                                  NULL,
                                  NULL,
                                  descr_C,
                                  (rocsparse_double_complex*)csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C,
                                  info,
                                  temp_buffer);
    }

    // rocsparse csric0 buffer size
    template <>
    rocsparse_status rocsparseTcsric0_buffer_size(rocsparse_handle          handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const rocsparse_mat_descr descr,
                                                  float*                    csr_val,
                                                  const int*                csr_row_ptr,
                                                  const int*                csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_scsric0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsric0_buffer_size(rocsparse_handle          handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const rocsparse_mat_descr descr,
                                                  double*                   csr_val,
                                                  const int*                csr_row_ptr,
                                                  const int*                csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_dcsric0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsric0_buffer_size(rocsparse_handle          handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const rocsparse_mat_descr descr,
                                                  std::complex<float>*      csr_val,
                                                  const int*                csr_row_ptr,
                                                  const int*                csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_ccsric0_buffer_size(handle,
                                             m,
                                             nnz,
                                             descr,
                                             (rocsparse_float_complex*)csr_val,
                                             csr_row_ptr,
                                             csr_col_ind,
                                             info,
                                             buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsric0_buffer_size(rocsparse_handle          handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const rocsparse_mat_descr descr,
                                                  std::complex<double>*     csr_val,
                                                  const int*                csr_row_ptr,
                                                  const int*                csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_zcsric0_buffer_size(handle,
                                             m,
                                             nnz,
                                             descr,
                                             (rocsparse_double_complex*)csr_val,
                                             csr_row_ptr,
                                             csr_col_ind,
                                             info,
                                             buffer_size);
    }

    // rocsparse csric0 analysis
    template <>
    rocsparse_status rocsparseTcsric0_analysis(rocsparse_handle          handle,
                                               int                       m,
                                               int                       nnz,
                                               const rocsparse_mat_descr descr,
                                               float*                    csr_val,
                                               const int*                csr_row_ptr,
                                               const int*                csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_scsric0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0_analysis(rocsparse_handle          handle,
                                               int                       m,
                                               int                       nnz,
                                               const rocsparse_mat_descr descr,
                                               double*                   csr_val,
                                               const int*                csr_row_ptr,
                                               const int*                csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_dcsric0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0_analysis(rocsparse_handle          handle,
                                               int                       m,
                                               int                       nnz,
                                               const rocsparse_mat_descr descr,
                                               std::complex<float>*      csr_val,
                                               const int*                csr_row_ptr,
                                               const int*                csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_ccsric0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          (rocsparse_float_complex*)csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0_analysis(rocsparse_handle          handle,
                                               int                       m,
                                               int                       nnz,
                                               const rocsparse_mat_descr descr,
                                               std::complex<double>*     csr_val,
                                               const int*                csr_row_ptr,
                                               const int*                csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_zcsric0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          (rocsparse_double_complex*)csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    // rocsparse csric0
    template <>
    rocsparse_status rocsparseTcsric0(rocsparse_handle          handle,
                                      int                       m,
                                      int                       nnz,
                                      const rocsparse_mat_descr descr,
                                      float*                    csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_scsric0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0(rocsparse_handle          handle,
                                      int                       m,
                                      int                       nnz,
                                      const rocsparse_mat_descr descr,
                                      double*                   csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_dcsric0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0(rocsparse_handle          handle,
                                      int                       m,
                                      int                       nnz,
                                      const rocsparse_mat_descr descr,
                                      std::complex<float>*      csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_ccsric0(handle,
                                 m,
                                 nnz,
                                 descr,
                                 (rocsparse_float_complex*)csr_val,
                                 csr_row_ptr,
                                 csr_col_ind,
                                 info,
                                 policy,
                                 temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0(rocsparse_handle          handle,
                                      int                       m,
                                      int                       nnz,
                                      const rocsparse_mat_descr descr,
                                      std::complex<double>*     csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_zcsric0(handle,
                                 m,
                                 nnz,
                                 descr,
                                 (rocsparse_double_complex*)csr_val,
                                 csr_row_ptr,
                                 csr_col_ind,
                                 info,
                                 policy,
                                 temp_buffer);
    }

    // rocsparse csrilu0 buffer size
    template <>
    rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle          handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const rocsparse_mat_descr descr,
                                                   float*                    csr_val,
                                                   const int*                csr_row_ptr,
                                                   const int*                csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_scsrilu0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle          handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const rocsparse_mat_descr descr,
                                                   double*                   csr_val,
                                                   const int*                csr_row_ptr,
                                                   const int*                csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_dcsrilu0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle          handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const rocsparse_mat_descr descr,
                                                   std::complex<float>*      csr_val,
                                                   const int*                csr_row_ptr,
                                                   const int*                csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_ccsrilu0_buffer_size(handle,
                                              m,
                                              nnz,
                                              descr,
                                              (rocsparse_float_complex*)csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle          handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const rocsparse_mat_descr descr,
                                                   std::complex<double>*     csr_val,
                                                   const int*                csr_row_ptr,
                                                   const int*                csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_zcsrilu0_buffer_size(handle,
                                              m,
                                              nnz,
                                              descr,
                                              (rocsparse_double_complex*)csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              buffer_size);
    }

    // rocsparse csrilu0 analysis
    template <>
    rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle          handle,
                                                int                       m,
                                                int                       nnz,
                                                const rocsparse_mat_descr descr,
                                                float*                    csr_val,
                                                const int*                csr_row_ptr,
                                                const int*                csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_scsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle          handle,
                                                int                       m,
                                                int                       nnz,
                                                const rocsparse_mat_descr descr,
                                                double*                   csr_val,
                                                const int*                csr_row_ptr,
                                                const int*                csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_dcsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle          handle,
                                                int                       m,
                                                int                       nnz,
                                                const rocsparse_mat_descr descr,
                                                std::complex<float>*      csr_val,
                                                const int*                csr_row_ptr,
                                                const int*                csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_ccsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           (rocsparse_float_complex*)csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle          handle,
                                                int                       m,
                                                int                       nnz,
                                                const rocsparse_mat_descr descr,
                                                std::complex<double>*     csr_val,
                                                const int*                csr_row_ptr,
                                                const int*                csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_zcsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           (rocsparse_double_complex*)csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    // rocsparse csrilu0
    template <>
    rocsparse_status rocsparseTcsrilu0(rocsparse_handle          handle,
                                       int                       m,
                                       int                       nnz,
                                       const rocsparse_mat_descr descr,
                                       float*                    csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_scsrilu0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0(rocsparse_handle          handle,
                                       int                       m,
                                       int                       nnz,
                                       const rocsparse_mat_descr descr,
                                       double*                   csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_dcsrilu0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0(rocsparse_handle          handle,
                                       int                       m,
                                       int                       nnz,
                                       const rocsparse_mat_descr descr,
                                       std::complex<float>*      csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_ccsrilu0(handle,
                                  m,
                                  nnz,
                                  descr,
                                  (rocsparse_float_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0(rocsparse_handle          handle,
                                       int                       m,
                                       int                       nnz,
                                       const rocsparse_mat_descr descr,
                                       std::complex<double>*     csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_zcsrilu0(handle,
                                  m,
                                  nnz,
                                  descr,
                                  (rocsparse_double_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    // rocsparse_csr2bsr
    template <>
    rocsparse_status rocsparseTcsr2bsr(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       m,
                                       int                       n,
                                       const rocsparse_mat_descr csr_descr,
                                       const float*              csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       int                       block_dim,
                                       const rocsparse_mat_descr bsr_descr,
                                       float*                    bsr_val,
                                       int*                      bsr_row_ptr,
                                       int*                      bsr_col_ind)
    {
        return rocsparse_scsr2bsr(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTcsr2bsr(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       m,
                                       int                       n,
                                       const rocsparse_mat_descr csr_descr,
                                       const double*             csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       int                       block_dim,
                                       const rocsparse_mat_descr bsr_descr,
                                       double*                   bsr_val,
                                       int*                      bsr_row_ptr,
                                       int*                      bsr_col_ind)
    {
        return rocsparse_dcsr2bsr(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTcsr2bsr(rocsparse_handle           handle,
                                       rocsparse_direction        dir,
                                       int                        m,
                                       int                        n,
                                       const rocsparse_mat_descr  csr_descr,
                                       const std::complex<float>* csr_val,
                                       const int*                 csr_row_ptr,
                                       const int*                 csr_col_ind,
                                       int                        block_dim,
                                       const rocsparse_mat_descr  bsr_descr,
                                       std::complex<float>*       bsr_val,
                                       int*                       bsr_row_ptr,
                                       int*                       bsr_col_ind)
    {
        return rocsparse_ccsr2bsr(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  (const rocsparse_float_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  (rocsparse_float_complex*)bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTcsr2bsr(rocsparse_handle            handle,
                                       rocsparse_direction         dir,
                                       int                         m,
                                       int                         n,
                                       const rocsparse_mat_descr   csr_descr,
                                       const std::complex<double>* csr_val,
                                       const int*                  csr_row_ptr,
                                       const int*                  csr_col_ind,
                                       int                         block_dim,
                                       const rocsparse_mat_descr   bsr_descr,
                                       std::complex<double>*       bsr_val,
                                       int*                        bsr_row_ptr,
                                       int*                        bsr_col_ind)
    {
        return rocsparse_zcsr2bsr(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  (const rocsparse_double_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  (rocsparse_double_complex*)bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind);
    }

    // rocsparse csr2csc
    template <>
    rocsparse_status rocsparseTcsr2csc(rocsparse_handle     handle,
                                       int                  m,
                                       int                  n,
                                       int                  nnz,
                                       const float*         csr_val,
                                       const int*           csr_row_ptr,
                                       const int*           csr_col_ind,
                                       float*               csc_val,
                                       int*                 csc_row_ind,
                                       int*                 csc_col_ptr,
                                       rocsparse_action     copy_values,
                                       rocsparse_index_base idx_base,
                                       void*                temp_buffer)
    {
        return rocsparse_scsr2csc(handle,
                                  m,
                                  n,
                                  nnz,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsr2csc(rocsparse_handle     handle,
                                       int                  m,
                                       int                  n,
                                       int                  nnz,
                                       const double*        csr_val,
                                       const int*           csr_row_ptr,
                                       const int*           csr_col_ind,
                                       double*              csc_val,
                                       int*                 csc_row_ind,
                                       int*                 csc_col_ptr,
                                       rocsparse_action     copy_values,
                                       rocsparse_index_base idx_base,
                                       void*                temp_buffer)
    {
        return rocsparse_dcsr2csc(handle,
                                  m,
                                  n,
                                  nnz,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsr2csc(rocsparse_handle           handle,
                                       int                        m,
                                       int                        n,
                                       int                        nnz,
                                       const std::complex<float>* csr_val,
                                       const int*                 csr_row_ptr,
                                       const int*                 csr_col_ind,
                                       std::complex<float>*       csc_val,
                                       int*                       csc_row_ind,
                                       int*                       csc_col_ptr,
                                       rocsparse_action           copy_values,
                                       rocsparse_index_base       idx_base,
                                       void*                      temp_buffer)
    {
        return rocsparse_ccsr2csc(handle,
                                  m,
                                  n,
                                  nnz,
                                  (const rocsparse_float_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  (rocsparse_float_complex*)csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsr2csc(rocsparse_handle            handle,
                                       int                         m,
                                       int                         n,
                                       int                         nnz,
                                       const std::complex<double>* csr_val,
                                       const int*                  csr_row_ptr,
                                       const int*                  csr_col_ind,
                                       std::complex<double>*       csc_val,
                                       int*                        csc_row_ind,
                                       int*                        csc_col_ptr,
                                       rocsparse_action            copy_values,
                                       rocsparse_index_base        idx_base,
                                       void*                       temp_buffer)
    {
        return rocsparse_zcsr2csc(handle,
                                  m,
                                  n,
                                  nnz,
                                  (const rocsparse_double_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  (rocsparse_double_complex*)csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
    }

    // rocsparse csr2ell
    template <>
    rocsparse_status rocsparseTcsr2ell(rocsparse_handle          handle,
                                       int                       m,
                                       const rocsparse_mat_descr csr_descr,
                                       const float*              csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       const rocsparse_mat_descr ell_descr,
                                       int                       ell_width,
                                       float*                    ell_val,
                                       int*                      ell_col_ind)
    {
        return rocsparse_scsr2ell(handle,
                                  m,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  ell_descr,
                                  ell_width,
                                  ell_val,
                                  ell_col_ind);
    }

    template <>
    rocsparse_status rocsparseTcsr2ell(rocsparse_handle          handle,
                                       int                       m,
                                       const rocsparse_mat_descr csr_descr,
                                       const double*             csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       const rocsparse_mat_descr ell_descr,
                                       int                       ell_width,
                                       double*                   ell_val,
                                       int*                      ell_col_ind)
    {
        return rocsparse_dcsr2ell(handle,
                                  m,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  ell_descr,
                                  ell_width,
                                  ell_val,
                                  ell_col_ind);
    }

    template <>
    rocsparse_status rocsparseTcsr2ell(rocsparse_handle           handle,
                                       int                        m,
                                       const rocsparse_mat_descr  csr_descr,
                                       const std::complex<float>* csr_val,
                                       const int*                 csr_row_ptr,
                                       const int*                 csr_col_ind,
                                       const rocsparse_mat_descr  ell_descr,
                                       int                        ell_width,
                                       std::complex<float>*       ell_val,
                                       int*                       ell_col_ind)
    {
        return rocsparse_ccsr2ell(handle,
                                  m,
                                  csr_descr,
                                  (const rocsparse_float_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  ell_descr,
                                  ell_width,
                                  (rocsparse_float_complex*)ell_val,
                                  ell_col_ind);
    }

    template <>
    rocsparse_status rocsparseTcsr2ell(rocsparse_handle            handle,
                                       int                         m,
                                       const rocsparse_mat_descr   csr_descr,
                                       const std::complex<double>* csr_val,
                                       const int*                  csr_row_ptr,
                                       const int*                  csr_col_ind,
                                       const rocsparse_mat_descr   ell_descr,
                                       int                         ell_width,
                                       std::complex<double>*       ell_val,
                                       int*                        ell_col_ind)
    {
        return rocsparse_zcsr2ell(handle,
                                  m,
                                  csr_descr,
                                  (const rocsparse_double_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  ell_descr,
                                  ell_width,
                                  (rocsparse_double_complex*)ell_val,
                                  ell_col_ind);
    }

    // rocsparse ell2csr
    template <>
    rocsparse_status rocsparseTell2csr(rocsparse_handle          handle,
                                       int                       m,
                                       int                       n,
                                       const rocsparse_mat_descr ell_descr,
                                       int                       ell_width,
                                       const float*              ell_val,
                                       const int*                ell_col_ind,
                                       const rocsparse_mat_descr csr_descr,
                                       float*                    csr_val,
                                       const int*                csr_row_ptr,
                                       int*                      csr_col_ind)
    {
        return rocsparse_sell2csr(handle,
                                  m,
                                  n,
                                  ell_descr,
                                  ell_width,
                                  ell_val,
                                  ell_col_ind,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTell2csr(rocsparse_handle          handle,
                                       int                       m,
                                       int                       n,
                                       const rocsparse_mat_descr ell_descr,
                                       int                       ell_width,
                                       const double*             ell_val,
                                       const int*                ell_col_ind,
                                       const rocsparse_mat_descr csr_descr,
                                       double*                   csr_val,
                                       const int*                csr_row_ptr,
                                       int*                      csr_col_ind)
    {
        return rocsparse_dell2csr(handle,
                                  m,
                                  n,
                                  ell_descr,
                                  ell_width,
                                  ell_val,
                                  ell_col_ind,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTell2csr(rocsparse_handle           handle,
                                       int                        m,
                                       int                        n,
                                       const rocsparse_mat_descr  ell_descr,
                                       int                        ell_width,
                                       const std::complex<float>* ell_val,
                                       const int*                 ell_col_ind,
                                       const rocsparse_mat_descr  csr_descr,
                                       std::complex<float>*       csr_val,
                                       const int*                 csr_row_ptr,
                                       int*                       csr_col_ind)
    {
        return rocsparse_cell2csr(handle,
                                  m,
                                  n,
                                  ell_descr,
                                  ell_width,
                                  (const rocsparse_float_complex*)ell_val,
                                  ell_col_ind,
                                  csr_descr,
                                  (rocsparse_float_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTell2csr(rocsparse_handle            handle,
                                       int                         m,
                                       int                         n,
                                       const rocsparse_mat_descr   ell_descr,
                                       int                         ell_width,
                                       const std::complex<double>* ell_val,
                                       const int*                  ell_col_ind,
                                       const rocsparse_mat_descr   csr_descr,
                                       std::complex<double>*       csr_val,
                                       const int*                  csr_row_ptr,
                                       int*                        csr_col_ind)
    {
        return rocsparse_zell2csr(handle,
                                  m,
                                  n,
                                  ell_descr,
                                  ell_width,
                                  (const rocsparse_double_complex*)ell_val,
                                  ell_col_ind,
                                  csr_descr,
                                  (rocsparse_double_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTcsr2dense(rocsparse_handle          handle,
                                         int                       m,
                                         int                       n,
                                         const rocsparse_mat_descr csr_descr,
                                         const float*              csr_val,
                                         const int*                csr_row_ptr,
                                         const int*                csr_col_ind,
                                         float*                    A,
                                         int                       ld)
    {
        return rocsparse_scsr2dense(
            handle, m, n, csr_descr, csr_val, csr_row_ptr, csr_col_ind, A, ld);
    }

    template <>
    rocsparse_status rocsparseTcsr2dense(rocsparse_handle          handle,
                                         int                       m,
                                         int                       n,
                                         const rocsparse_mat_descr csr_descr,
                                         const double*             csr_val,
                                         const int*                csr_row_ptr,
                                         const int*                csr_col_ind,
                                         double*                   A,
                                         int                       ld)
    {
        return rocsparse_dcsr2dense(
            handle, m, n, csr_descr, csr_val, csr_row_ptr, csr_col_ind, A, ld);
    }

    template <>
    rocsparse_status rocsparseTcsr2dense(rocsparse_handle           handle,
                                         int                        m,
                                         int                        n,
                                         const rocsparse_mat_descr  csr_descr,
                                         const std::complex<float>* csr_val,
                                         const int*                 csr_row_ptr,
                                         const int*                 csr_col_ind,
                                         std::complex<float>*       A,
                                         int                        ld)
    {
        return rocsparse_ccsr2dense(handle,
                                    m,
                                    n,
                                    csr_descr,
                                    (const rocsparse_float_complex*)csr_val,
                                    csr_row_ptr,
                                    csr_col_ind,
                                    (rocsparse_float_complex*)A,
                                    ld);
    }

    template <>
    rocsparse_status rocsparseTcsr2dense(rocsparse_handle            handle,
                                         int                         m,
                                         int                         n,
                                         const rocsparse_mat_descr   csr_descr,
                                         const std::complex<double>* csr_val,
                                         const int*                  csr_row_ptr,
                                         const int*                  csr_col_ind,
                                         std::complex<double>*       A,
                                         int                         ld)
    {
        return rocsparse_zcsr2dense(handle,
                                    m,
                                    n,
                                    csr_descr,
                                    (const rocsparse_double_complex*)csr_val,
                                    csr_row_ptr,
                                    csr_col_ind,
                                    (rocsparse_double_complex*)A,
                                    ld);
    }

    template <>
    rocsparse_status rocsparseTdense2csr(rocsparse_handle          handle,
                                         int                       m,
                                         int                       n,
                                         const rocsparse_mat_descr descr_A,
                                         const float*              A,
                                         int                       lda,
                                         const int*                nnz_per_row,
                                         float*                    csr_val,
                                         int*                      csr_row_ptr,
                                         int*                      csr_col_ind)
    {
        return rocsparse_sdense2csr(
            handle, m, n, descr_A, A, lda, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTdense2csr(rocsparse_handle          handle,
                                         int                       m,
                                         int                       n,
                                         const rocsparse_mat_descr descr_A,
                                         const double*             A,
                                         int                       lda,
                                         const int*                nnz_per_row,
                                         double*                   csr_val,
                                         int*                      csr_row_ptr,
                                         int*                      csr_col_ind)
    {
        return rocsparse_ddense2csr(
            handle, m, n, descr_A, A, lda, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTdense2csr(rocsparse_handle           handle,
                                         int                        m,
                                         int                        n,
                                         const rocsparse_mat_descr  descr_A,
                                         const std::complex<float>* A,
                                         int                        lda,
                                         const int*                 nnz_per_row,
                                         std::complex<float>*       csr_val,
                                         int*                       csr_row_ptr,
                                         int*                       csr_col_ind)
    {
        return rocsparse_cdense2csr(handle,
                                    m,
                                    n,
                                    descr_A,
                                    (const rocsparse_float_complex*)A,
                                    lda,
                                    nnz_per_row,
                                    (rocsparse_float_complex*)csr_val,
                                    csr_row_ptr,
                                    csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTdense2csr(rocsparse_handle            handle,
                                         int                         m,
                                         int                         n,
                                         const rocsparse_mat_descr   descr_A,
                                         const std::complex<double>* A,
                                         int                         lda,
                                         const int*                  nnz_per_row,
                                         std::complex<double>*       csr_val,
                                         int*                        csr_row_ptr,
                                         int*                        csr_col_ind)
    {
        return rocsparse_zdense2csr(handle,
                                    m,
                                    n,
                                    descr_A,
                                    (const rocsparse_double_complex*)A,
                                    lda,
                                    nnz_per_row,
                                    (rocsparse_double_complex*)csr_val,
                                    csr_row_ptr,
                                    csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTnnz(rocsparse_handle          handle,
                                   rocsparse_direction       dir_A,
                                   int                       m,
                                   int                       n,
                                   const rocsparse_mat_descr descr_A,
                                   const float*              A,
                                   int                       lda,
                                   int*                      nnz_per_row_column,
                                   int*                      nnz_total)
    {
        return rocsparse_snnz(handle, dir_A, m, n, descr_A, A, lda, nnz_per_row_column, nnz_total);
    }

    template <>
    rocsparse_status rocsparseTnnz(rocsparse_handle          handle,
                                   rocsparse_direction       dir_A,
                                   int                       m,
                                   int                       n,
                                   const rocsparse_mat_descr descr_A,
                                   const double*             A,
                                   int                       lda,
                                   int*                      nnz_per_row_column,
                                   int*                      nnz_total)
    {
        return rocsparse_dnnz(handle, dir_A, m, n, descr_A, A, lda, nnz_per_row_column, nnz_total);
    }

    template <>
    rocsparse_status rocsparseTnnz(rocsparse_handle           handle,
                                   rocsparse_direction        dir_A,
                                   int                        m,
                                   int                        n,
                                   const rocsparse_mat_descr  descr_A,
                                   const std::complex<float>* A,
                                   int                        lda,
                                   int*                       nnz_per_row_column,
                                   int*                       nnz_total)
    {
        return rocsparse_cnnz(handle,
                              dir_A,
                              m,
                              n,
                              descr_A,
                              (const rocsparse_float_complex*)A,
                              lda,
                              nnz_per_row_column,
                              nnz_total);
    }

    template <>
    rocsparse_status rocsparseTnnz(rocsparse_handle            handle,
                                   rocsparse_direction         dir_A,
                                   int                         m,
                                   int                         n,
                                   const rocsparse_mat_descr   descr_A,
                                   const std::complex<double>* A,
                                   int                         lda,
                                   int*                        nnz_per_row_column,
                                   int*                        nnz_total)
    {
        return rocsparse_znnz(handle,
                              dir_A,
                              m,
                              n,
                              descr_A,
                              (const rocsparse_double_complex*)A,
                              lda,
                              nnz_per_row_column,
                              nnz_total);
    }

} // namespace rocalution
