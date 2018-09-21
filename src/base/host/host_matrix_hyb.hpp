/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HOST_MATRIX_HYB_HPP_
#define ROCALUTION_HOST_MATRIX_HYB_HPP_

#include "../base_vector.hpp"
#include "../base_matrix.hpp"
#include "../matrix_formats.hpp"

namespace rocalution {

template <typename ValueType>
class HostMatrixHYB : public HostMatrix<ValueType>
{
    public:
    HostMatrixHYB();
    HostMatrixHYB(const Rocalution_Backend_Descriptor local_backend);
    virtual ~HostMatrixHYB();

    inline int GetELLMaxRow(void) const { return this->mat_.ELL.max_row; }
    inline int GetELLNnz(void) const { return this->ell_nnz_; }
    inline int GetCOONnz(void) const { return this->coo_nnz_; }

    virtual void Info(void) const;
    virtual unsigned int GetMatFormat(void) const { return HYB; }

    virtual void Clear(void);
    virtual void AllocateHYB(int ell_nnz, int coo_nnz, int ell_max_row, int nrow, int ncol);

    virtual bool ConvertFrom(const BaseMatrix<ValueType>& mat);

    virtual void CopyFrom(const BaseMatrix<ValueType>& mat);
    virtual void CopyTo(BaseMatrix<ValueType>* mat) const;

    virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
    virtual void
    ApplyAdd(const BaseVector<ValueType>& in, ValueType scalar, BaseVector<ValueType>* out) const;

    private:
    MatrixHYB<ValueType, int> mat_;
    int ell_nnz_;
    int coo_nnz_;

    friend class BaseVector<ValueType>;
    friend class HostVector<ValueType>;
    friend class HostMatrixCSR<ValueType>;
    friend class HostMatrixCOO<ValueType>;
    friend class HostMatrixELL<ValueType>;
    friend class HostMatrixDENSE<ValueType>;

    friend class HIPAcceleratorMatrixHYB<ValueType>;
};

} // namespace rocalution

#endif // ROCALUTION_HOST_MATRIX_HYB_HPP_
