/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HOST_MATRIX_CSR_HPP_
#define ROCALUTION_HOST_MATRIX_CSR_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace rocalution
{

    template <typename ValueType>
    class HostMatrixCSR : public HostMatrix<ValueType>
    {
    public:
        HostMatrixCSR();
        explicit HostMatrixCSR(const Rocalution_Backend_Descriptor& local_backend);
        virtual ~HostMatrixCSR();

        virtual void         Info(void) const;
        virtual unsigned int GetMatFormat(void) const
        {
            return CSR;
        }

        virtual bool Check(void) const;
        virtual void AllocateCSR(int64_t nnz, int nrow, int ncol);
        virtual void SetDataPtrCSR(
            PtrType** row_offset, int** col, ValueType** val, int64_t nnz, int nrow, int ncol);
        virtual void LeaveDataPtrCSR(PtrType** row_offset, int** col, ValueType** val);

        virtual void Clear(void);
        virtual bool Zeros(void);

        virtual bool Scale(ValueType alpha);
        virtual bool ScaleDiagonal(ValueType alpha);
        virtual bool ScaleOffDiagonal(ValueType alpha);
        virtual bool AddScalar(ValueType alpha);
        virtual bool AddScalarDiagonal(ValueType alpha);
        virtual bool AddScalarOffDiagonal(ValueType alpha);

        virtual bool ExtractSubMatrix(int                    row_offset,
                                      int                    col_offset,
                                      int                    row_size,
                                      int                    col_size,
                                      BaseMatrix<ValueType>* mat) const;

        virtual bool ExtractDiagonal(BaseVector<ValueType>* vec_diag) const;
        virtual bool ExtractInverseDiagonal(BaseVector<ValueType>* vec_inv_diag) const;
        virtual bool ExtractU(BaseMatrix<ValueType>* U) const;
        virtual bool ExtractUDiagonal(BaseMatrix<ValueType>* U) const;
        virtual bool ExtractL(BaseMatrix<ValueType>* L) const;
        virtual bool ExtractLDiagonal(BaseMatrix<ValueType>* L) const;

        virtual bool
            MultiColoring(int& num_colors, int** size_colors, BaseVector<int>* permutation) const;

        virtual bool MaximalIndependentSet(int& size, BaseVector<int>* permutation) const;

        virtual bool ZeroBlockPermutation(int& size, BaseVector<int>* permutation) const;

        virtual bool SymbolicPower(int p);

        virtual bool SymbolicMatMatMult(const BaseMatrix<ValueType>& src);
        virtual bool MatMatMult(const BaseMatrix<ValueType>& A, const BaseMatrix<ValueType>& B);
        virtual bool SymbolicMatMatMult(const BaseMatrix<ValueType>& A,
                                        const BaseMatrix<ValueType>& B);
        virtual bool NumericMatMatMult(const BaseMatrix<ValueType>& A,
                                       const BaseMatrix<ValueType>& B);

        virtual bool DiagonalMatrixMultR(const BaseVector<ValueType>& diag);
        virtual bool DiagonalMatrixMultL(const BaseVector<ValueType>& diag);

        virtual bool MatrixAdd(const BaseMatrix<ValueType>& mat,
                               ValueType                    alpha,
                               ValueType                    beta,
                               bool                         structure);

        virtual bool Permute(const BaseVector<int>& permutation);

        virtual bool CMK(BaseVector<int>* permutation) const;
        virtual bool RCMK(BaseVector<int>* permutation) const;
        virtual bool ConnectivityOrder(BaseVector<int>* permutation) const;

        virtual bool ConvertFrom(const BaseMatrix<ValueType>& mat);

        virtual void CopyFrom(const BaseMatrix<ValueType>& mat);

        virtual void CopyFromCSR(const PtrType* row_offsets, const int* col, const ValueType* val);
        virtual void CopyToCSR(PtrType* row_offsets, int* col, ValueType* val) const;

        virtual void CopyTo(BaseMatrix<ValueType>* mat) const;

        virtual void CopyFromHostCSR(const PtrType*   row_offset,
                                     const int*       col,
                                     const ValueType* val,
                                     int64_t          nnz,
                                     int              nrow,
                                     int              ncol);

        virtual bool ReadFileCSR(const std::string& filename);
        virtual bool WriteFileCSR(const std::string& filename) const;

        virtual bool CreateFromMap(const BaseVector<int>& map, int n, int m);
        virtual bool
            CreateFromMap(const BaseVector<int>& map, int n, int m, BaseMatrix<ValueType>* pro);

        virtual bool ICFactorize(BaseVector<ValueType>* inv_diag);

        virtual bool ILU0Factorize(void);
        virtual bool ILUpFactorizeNumeric(int p, const BaseMatrix<ValueType>& mat);
        virtual bool ILUTFactorize(double t, int maxrow);
        virtual bool ItILU0Factorize(ItILU0Algorithm alg,
                                     int             option,
                                     int             max_iter,
                                     double          tolerance,
                                     int*            niter,
                                     double*         history);

        virtual void LUAnalyse(void);
        virtual void LUAnalyseClear(void);
        virtual bool LUSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

        virtual void LLAnalyse(void);
        virtual void LLAnalyseClear(void);
        virtual bool LLSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
        virtual bool LLSolve(const BaseVector<ValueType>& in,
                             const BaseVector<ValueType>& inv_diag,
                             BaseVector<ValueType>*       out) const;

        virtual void LAnalyse(bool diag_unit = false);
        virtual void LAnalyseClear(void);
        virtual bool LSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

        virtual void UAnalyse(bool diag_unit = false);
        virtual void UAnalyseClear(void);
        virtual bool USolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

        virtual void ItLUAnalyse(void);
        virtual void ItLUAnalyseClear(void);
        virtual bool ItLUSolve(int                          max_iter,
                               double                       tolerance,
                               bool                         use_tol,
                               const BaseVector<ValueType>& in,
                               BaseVector<ValueType>*       out) const;

        virtual void ItLLAnalyse(void);
        virtual void ItLLAnalyseClear(void);
        virtual bool ItLLSolve(int                          max_iter,
                               double                       tolerance,
                               bool                         use_tol,
                               const BaseVector<ValueType>& in,
                               BaseVector<ValueType>*       out) const;
        virtual bool ItLLSolve(int                          max_iter,
                               double                       tolerance,
                               bool                         use_tol,
                               const BaseVector<ValueType>& in,
                               const BaseVector<ValueType>& inv_diag,
                               BaseVector<ValueType>*       out) const;

        virtual void ItLAnalyse(bool diag_unit = false);
        virtual void ItLAnalyseClear(void);
        virtual bool ItLSolve(int                          max_iter,
                              double                       tolerance,
                              bool                         use_tol,
                              const BaseVector<ValueType>& in,
                              BaseVector<ValueType>*       out) const;

        virtual void ItUAnalyse(bool diag_unit = false);
        virtual void ItUAnalyseClear(void);
        virtual bool ItUSolve(int                          max_iter,
                              double                       tolerance,
                              bool                         use_tol,
                              const BaseVector<ValueType>& in,
                              BaseVector<ValueType>*       out) const;

        virtual bool Gershgorin(ValueType& lambda_min, ValueType& lambda_max) const;

        virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
        virtual void ApplyAdd(const BaseVector<ValueType>& in,
                              ValueType                    scalar,
                              BaseVector<ValueType>*       out) const;

        virtual bool Compress(double drop_off);
        virtual bool Transpose(void);
        virtual bool Transpose(BaseMatrix<ValueType>* T) const;
        virtual bool Sort(void);
        virtual bool Key(long int& row_key, long int& col_key, long int& val_key) const;

        virtual bool ReplaceColumnVector(int idx, const BaseVector<ValueType>& vec);
        virtual bool ExtractColumnVector(int idx, BaseVector<ValueType>* vec) const;

        virtual bool ReplaceRowVector(int idx, const BaseVector<ValueType>& vec);
        virtual bool ExtractRowVector(int idx, BaseVector<ValueType>* vec) const;

        virtual bool ExtractGlobalColumnIndices(int                        ncol,
                                                int64_t                    global_offset,
                                                const BaseVector<int64_t>& l2g,
                                                BaseVector<int64_t>*       global_col) const;
        virtual bool ExtractExtRowNnz(int offset, BaseVector<PtrType>* row_nnz) const;
        virtual bool ExtractBoundaryRowNnz(BaseVector<PtrType>*         row_nnz,
                                           const BaseVector<int>&       boundary_index,
                                           const BaseMatrix<ValueType>& gst) const;
        virtual bool ExtractBoundaryRows(const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                         BaseVector<int64_t>*         bnd_csr_col_ind,
                                         BaseVector<ValueType>*       bnd_csr_val,
                                         int64_t                      global_column_offset,
                                         const BaseVector<int>&       boundary_index,
                                         const BaseVector<int64_t>&   ghost_mapping,
                                         const BaseMatrix<ValueType>& gst) const;
        virtual bool MergeToLocal(const BaseMatrix<ValueType>& mat_int,
                                  const BaseMatrix<ValueType>& mat_gst,
                                  const BaseMatrix<ValueType>& mat_ext,
                                  const BaseVector<int>&       vec_ext);

        virtual bool AMGConnect(ValueType eps, BaseVector<int>* connections) const;
        virtual bool AMGAggregate(const BaseVector<int>& connections,
                                  BaseVector<int>*       aggregates) const;
        virtual bool AMGPMISAggregate(const BaseVector<int>& connections,
                                      BaseVector<int>*       aggregates) const;
        virtual bool AMGSmoothedAggregation(ValueType              relax,
                                            const BaseVector<int>& aggregates,
                                            const BaseVector<int>& connections,
                                            BaseMatrix<ValueType>* prolong,
                                            int                    lumping_strat = 0) const;
        virtual bool AMGAggregation(const BaseVector<int>& aggregates,
                                    BaseMatrix<ValueType>* prolong) const;

        virtual bool AMGGreedyAggregate(const BaseVector<bool>& connections,
                                        BaseVector<int64_t>*    aggregates,
                                        BaseVector<int64_t>*    aggregate_root_nodes) const;
        virtual bool AMGBoundaryNnz(const BaseVector<int>&       boundary,
                                    const BaseVector<bool>&      connections,
                                    const BaseMatrix<ValueType>& ghost,
                                    BaseVector<PtrType>*         row_nnz) const;
        virtual bool AMGExtractBoundary(int64_t                      global_column_begin,
                                        const BaseVector<int>&       boundary,
                                        const BaseVector<int64_t>&   l2g,
                                        const BaseVector<bool>&      connections,
                                        const BaseMatrix<ValueType>& ghost,
                                        const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                        BaseVector<int64_t>*         bnd_csr_col_ind) const;
        virtual bool AMGComputeStrongConnections(ValueType                    eps,
                                                 const BaseVector<ValueType>& diag,
                                                 const BaseVector<int64_t>&   l2g,
                                                 BaseVector<bool>*            connections,
                                                 const BaseMatrix<ValueType>& ghost) const;
        virtual bool AMGPMISInitializeState(int64_t                      global_column_begin,
                                            const BaseVector<bool>&      connections,
                                            BaseVector<int>*             state,
                                            BaseVector<int>*             hash,
                                            const BaseMatrix<ValueType>& ghost) const;
        virtual bool AMGExtractBoundaryState(const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                             const BaseVector<bool>&      connections,
                                             const BaseVector<int>&       max_state,
                                             const BaseVector<int>&       hash,
                                             BaseVector<int>*             bnd_max_state,
                                             BaseVector<int>*             bnd_hash,
                                             int64_t                      global_column_offset,
                                             const BaseVector<int>&       boundary_index,
                                             const BaseMatrix<ValueType>& gst) const;
        virtual bool AMGPMISFindMaxNeighbourNode(int64_t                      global_column_begin,
                                                 int64_t                      global_column_end,
                                                 bool&                        undecided,
                                                 const BaseVector<bool>&      connections,
                                                 const BaseVector<int>&       state,
                                                 const BaseVector<int>&       hash,
                                                 const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                                 const BaseVector<int64_t>&   bnd_csr_col_ind,
                                                 const BaseVector<int>&       bnd_state,
                                                 const BaseVector<int>&       bnd_hash,
                                                 BaseVector<int>*             max_state,
                                                 BaseVector<int64_t>*         aggregates,
                                                 const BaseMatrix<ValueType>& ghost) const;
        virtual bool
                     AMGPMISAddUnassignedNodesToAggregations(int64_t                    global_column_begin,
                                                             const BaseVector<bool>&    connections,
                                                             const BaseVector<int>&     state,
                                                             const BaseVector<int64_t>& l2g,
                                                             BaseVector<int>*           max_state,
                                                             BaseVector<int64_t>*       aggregates,
                                                             BaseVector<int64_t>*       aggregate_root_nodes,
                                                             const BaseMatrix<ValueType>& ghost) const;
        virtual bool AMGPMISInitializeAggregateGlobalIndices(
            int64_t                    global_column_begin,
            const BaseVector<int64_t>& aggregates,
            BaseVector<int64_t>*       aggregate_root_nodes) const;

        //virtual bool AMGSmoothedAggregation(ValueType                  relax,
        //                                    const BaseVector<bool>&    connections,
        //                                    const BaseVector<int64_t>& aggregates,
        //                                    BaseMatrix<ValueType>*     prolong,
        //                                    int                        lumping_strat = 0) const;
        virtual bool AMGUnsmoothedAggregation(const BaseVector<int64_t>& aggregates,
                                              BaseMatrix<ValueType>*     prolong) const;

        virtual bool
            AMGSmoothedAggregationProlongNnz(int64_t                      global_column_begin,
                                             int64_t                      global_column_end,
                                             const BaseVector<bool>&      connections,
                                             const BaseVector<int64_t>&   aggregates,
                                             const BaseVector<int64_t>&   aggregate_root_nodes,
                                             const BaseMatrix<ValueType>& ghost,
                                             BaseVector<int>*             f2c,
                                             BaseMatrix<ValueType>*       prolong_int,
                                             BaseMatrix<ValueType>*       prolong_gst) const;

        virtual bool
            AMGSmoothedAggregationProlongFill(int64_t                      global_column_begin,
                                              int64_t                      global_column_end,
                                              int                          lumping_strat,
                                              ValueType                    relax,
                                              const BaseVector<bool>&      connections,
                                              const BaseVector<int64_t>&   aggregates,
                                              const BaseVector<int64_t>&   aggregate_root_nodes,
                                              const BaseVector<int64_t>&   l2g,
                                              const BaseVector<int>&       f2c,
                                              const BaseMatrix<ValueType>& ghost,
                                              BaseMatrix<ValueType>*       prolong_int,
                                              BaseMatrix<ValueType>*       prolong_gst,
                                              BaseVector<int64_t>*         global_ghost_col) const;

        virtual bool
            AMGUnsmoothedAggregationProlongNnz(int64_t                      global_column_begin,
                                               int64_t                      global_column_end,
                                               const BaseVector<int64_t>&   aggregates,
                                               const BaseVector<int64_t>&   aggregate_root_nodes,
                                               const BaseMatrix<ValueType>& ghost,
                                               BaseVector<int>*             f2c,
                                               BaseMatrix<ValueType>*       prolong_int,
                                               BaseMatrix<ValueType>*       prolong_gst) const;

        virtual bool
            AMGUnsmoothedAggregationProlongFill(int64_t                      global_column_begin,
                                                int64_t                      global_column_end,
                                                const BaseVector<int64_t>&   aggregates,
                                                const BaseVector<int64_t>&   aggregate_root_nodes,
                                                const BaseVector<int>&       f2c,
                                                const BaseMatrix<ValueType>& ghost,
                                                BaseMatrix<ValueType>*       prolong_int,
                                                BaseMatrix<ValueType>*       prolong_gst,
                                                BaseVector<int64_t>* global_ghost_col) const;

        virtual bool RSCoarsening(float eps, BaseVector<int>* CFmap, BaseVector<bool>* S) const;

        virtual bool RSPMISStrongInfluences(float                        eps,
                                            BaseVector<bool>*            S,
                                            BaseVector<float>*           omega,
                                            int64_t                      global_row_offset,
                                            const BaseMatrix<ValueType>& ghost) const;
        virtual bool RSPMISUnassignedToCoarse(BaseVector<int>*         CFmap,
                                              BaseVector<bool>*        marked,
                                              const BaseVector<float>& omega) const;
        virtual bool RSPMISCorrectCoarse(BaseVector<int>*             CFmap,
                                         const BaseVector<bool>&      S,
                                         const BaseVector<bool>&      marked,
                                         const BaseVector<float>&     omega,
                                         const BaseMatrix<ValueType>& ghost) const;
        virtual bool RSPMISCoarseEdgesToFine(BaseVector<int>*             CFmap,
                                             const BaseVector<bool>&      S,
                                             const BaseMatrix<ValueType>& ghost) const;
        virtual bool RSPMISCheckUndecided(bool& undecided, const BaseVector<int>& CFmap) const;

        virtual bool RSDirectProlongNnz(const BaseVector<int>&       CFmap,
                                        const BaseVector<bool>&      S,
                                        const BaseMatrix<ValueType>& ghost,
                                        BaseVector<ValueType>*       Amin,
                                        BaseVector<ValueType>*       Amax,
                                        BaseVector<int>*             f2c,
                                        BaseMatrix<ValueType>*       prolong_int,
                                        BaseMatrix<ValueType>*       prolong_gst) const;
        virtual bool RSDirectProlongFill(const BaseVector<int64_t>&   l2g,
                                         const BaseVector<int>&       f2c,
                                         const BaseVector<int>&       CFmap,
                                         const BaseVector<bool>&      S,
                                         const BaseMatrix<ValueType>& ghost,
                                         const BaseVector<ValueType>& Amin,
                                         const BaseVector<ValueType>& Amax,
                                         BaseMatrix<ValueType>*       prolong_int,
                                         BaseMatrix<ValueType>*       prolong_gst,
                                         BaseVector<int64_t>*         global_ghost_col) const;
        virtual bool RSExtPIBoundaryNnz(const BaseVector<int>&       boundary,
                                        const BaseVector<int>&       CFmap,
                                        const BaseVector<bool>&      S,
                                        const BaseMatrix<ValueType>& ghost,
                                        BaseVector<PtrType>*         row_nnz) const;
        virtual bool RSExtPIExtractBoundary(int64_t                      global_column_begin,
                                            const BaseVector<int>&       boundary,
                                            const BaseVector<int64_t>&   l2g,
                                            const BaseVector<int>&       CFmap,
                                            const BaseVector<bool>&      S,
                                            const BaseMatrix<ValueType>& ghost,
                                            const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                            BaseVector<int64_t>*         bnd_csr_col_ind) const;
        virtual bool RSExtPIProlongNnz(int64_t                      global_column_begin,
                                       int64_t                      global_column_end,
                                       bool                         FF1,
                                       const BaseVector<int64_t>&   l2g,
                                       const BaseVector<int>&       CFmap,
                                       const BaseVector<bool>&      S,
                                       const BaseMatrix<ValueType>& ghost,
                                       const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                       const BaseVector<int64_t>&   bnd_csr_col_ind,
                                       BaseVector<int>*             f2c,
                                       BaseMatrix<ValueType>*       prolong_int,
                                       BaseMatrix<ValueType>*       prolong_gst) const;
        virtual bool RSExtPIProlongFill(int64_t                      global_column_begin,
                                        int64_t                      global_column_end,
                                        bool                         FF1,
                                        const BaseVector<int64_t>&   l2g,
                                        const BaseVector<int>&       f2c,
                                        const BaseVector<int>&       CFmap,
                                        const BaseVector<bool>&      S,
                                        const BaseMatrix<ValueType>& ghost,
                                        const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                        const BaseVector<int64_t>&   bnd_csr_col_ind,
                                        const BaseVector<PtrType>&   ext_csr_row_ptr,
                                        const BaseVector<int64_t>&   ext_csr_col_ind,
                                        const BaseVector<ValueType>& ext_csr_val,
                                        BaseMatrix<ValueType>*       prolong_int,
                                        BaseMatrix<ValueType>*       prolong_gst,
                                        BaseVector<int64_t>*         global_ghost_col) const;

        virtual bool FSAI(int power, const BaseMatrix<ValueType>* pattern);
        virtual bool SPAI(void);

        virtual bool InitialPairwiseAggregation(ValueType        beta,
                                                int&             nc,
                                                BaseVector<int>* G,
                                                int&             Gsize,
                                                int**            rG,
                                                int&             rGsize,
                                                int              ordering) const;
        virtual bool InitialPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                                ValueType                    beta,
                                                int&                         nc,
                                                BaseVector<int>*             G,
                                                int&                         Gsize,
                                                int**                        rG,
                                                int&                         rGsize,
                                                int                          ordering) const;
        virtual bool FurtherPairwiseAggregation(ValueType        beta,
                                                int&             nc,
                                                BaseVector<int>* G,
                                                int&             Gsize,
                                                int**            rG,
                                                int&             rGsize,
                                                int              ordering) const;
        virtual bool FurtherPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                                ValueType                    beta,
                                                int&                         nc,
                                                BaseVector<int>*             G,
                                                int&                         Gsize,
                                                int**                        rG,
                                                int&                         rGsize,
                                                int                          ordering) const;
        virtual bool CoarsenOperator(BaseMatrix<ValueType>* Ac,
                                     int                    nrow,
                                     int                    ncol,
                                     const BaseVector<int>& G,
                                     int                    Gsize,
                                     const int*             rG,
                                     int                    rGsize) const;

        virtual bool CombineAndRenumber(int                        ncol,
                                        int64_t                    ext_nnz,
                                        int64_t                    col_begin,
                                        int64_t                    col_end,
                                        const BaseVector<int64_t>& l2g,
                                        const BaseVector<int64_t>& ext,
                                        BaseVector<int>*           merged,
                                        BaseVector<int64_t>*       mapping,
                                        BaseVector<int>*           local_col) const;

        virtual bool SplitInteriorGhost(BaseMatrix<ValueType>* interior,
                                        BaseMatrix<ValueType>* ghost) const;
        virtual bool CopyGhostFromGlobalReceive(const BaseVector<int>&       boundary,
                                                const BaseVector<PtrType>&   recv_csr_row_ptr,
                                                const BaseVector<int64_t>&   recv_csr_col_ind,
                                                const BaseVector<ValueType>& recv_csr_val,
                                                BaseVector<int64_t>*         global_col);
        virtual bool CopyFromGlobalReceive(int                          nrow,
                                           int64_t                      global_column_begin,
                                           int64_t                      global_column_end,
                                           const BaseVector<int>&       boundary,
                                           const BaseVector<PtrType>&   recv_csr_row_ptr,
                                           const BaseVector<int64_t>&   recv_csr_col_ind,
                                           const BaseVector<ValueType>& recv_csr_val,
                                           BaseMatrix<ValueType>*       ghost,
                                           BaseVector<int64_t>*         global_col);
        virtual bool RenumberGlobalToLocal(const BaseVector<int64_t>& column_indices);
        virtual bool CompressAdd(const BaseVector<int64_t>&   l2g,
                                 const BaseVector<int64_t>&   global_ghost_col,
                                 const BaseMatrix<ValueType>& ext,
                                 BaseVector<int64_t>*         global_col);

    private:
        MatrixCSR<ValueType, int, PtrType> mat_;

        bool L_diag_unit_;
        bool U_diag_unit_;

        // Matrix buffer (itcsrsv)
        size_t mat_buffer_size_;
        char*  mat_buffer_;

        HostVector<ValueType>* tmp_vec_;

        friend class BaseVector<ValueType>;
        friend class HostVector<ValueType>;
        friend class HostMatrixCOO<ValueType>;
        friend class HostMatrixDIA<ValueType>;
        friend class HostMatrixELL<ValueType>;
        friend class HostMatrixHYB<ValueType>;
        friend class HostMatrixDENSE<ValueType>;
        friend class HostMatrixMCSR<ValueType>;
        friend class HostMatrixBCSR<ValueType>;

        friend class HIPAcceleratorMatrixCSR<ValueType>;
    };

} // namespace rocalution

#endif // ROCALUTION_HOST_MATRIX_CSR_HPP_
