/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HOST_MATRIX_DENSE_HPP_
#define ROCALUTION_HOST_MATRIX_DENSE_HPP_

#include "../base_vector.hpp"
#include "../base_matrix.hpp"
#include "../matrix_formats.hpp"

namespace rocalution {

template <typename ValueType>
class HostMatrixDENSE : public HostMatrix<ValueType>
{
    public:
    HostMatrixDENSE();
    HostMatrixDENSE(const Rocalution_Backend_Descriptor local_backend);
    virtual ~HostMatrixDENSE();

    virtual void Info(void) const;
    virtual unsigned int GetMatFormat(void) const { return DENSE; }

    virtual void Clear(void);
    virtual void AllocateDENSE(int nrow, int ncol);
    virtual void SetDataPtrDENSE(ValueType** val, int nrow, int ncol);
    virtual void LeaveDataPtrDENSE(ValueType** val);

    virtual bool ConvertFrom(const BaseMatrix<ValueType>& mat);

    virtual void CopyFrom(const BaseMatrix<ValueType>& mat);
    virtual void CopyTo(BaseMatrix<ValueType>* mat) const;

    virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
    virtual void
    ApplyAdd(const BaseVector<ValueType>& in, ValueType scalar, BaseVector<ValueType>* out) const;

    virtual bool MatMatMult(const BaseMatrix<ValueType>& A, const BaseMatrix<ValueType>& B);

    virtual bool Householder(int idx, ValueType& beta, BaseVector<ValueType>* vec) const;
    virtual bool QRDecompose(void);
    virtual bool QRSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

    virtual bool LUFactorize(void);
    virtual bool LUSolve(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;

    virtual bool Invert(void);

    virtual bool ReplaceColumnVector(int idx, const BaseVector<ValueType>& vec);
    virtual bool ReplaceRowVector(int idx, const BaseVector<ValueType>& vec);
    virtual bool ExtractColumnVector(int idx, BaseVector<ValueType>* vec) const;
    virtual bool ExtractRowVector(int idx, BaseVector<ValueType>* vec) const;

    private:
    MatrixDENSE<ValueType> mat_;

    friend class BaseVector<ValueType>;
    friend class HostVector<ValueType>;
    friend class HostMatrixCSR<ValueType>;
    friend class HostMatrixCOO<ValueType>;
    friend class HostMatrixELL<ValueType>;
    friend class HostMatrixHYB<ValueType>;
    friend class HostMatrixDIA<ValueType>;

    friend class HIPAcceleratorMatrixDENSE<ValueType>;
};

} // namespace rocalution

#endif // ROCALUTION_HOST_MATRIX_DENSE_HPP_
