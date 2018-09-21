/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HIP_MATRIX_MCSR_HPP_
#define ROCALUTION_HIP_MATRIX_MCSR_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace rocalution {

template <typename ValueType>
class HIPAcceleratorMatrixMCSR : public HIPAcceleratorMatrix<ValueType>
{
    public:
    HIPAcceleratorMatrixMCSR();
    HIPAcceleratorMatrixMCSR(const Rocalution_Backend_Descriptor local_backend);
    virtual ~HIPAcceleratorMatrixMCSR();

    virtual void Info(void) const;
    virtual unsigned int GetMatFormat(void) const { return MCSR; }

    virtual void Clear(void);
    virtual void AllocateMCSR(int nnz, int nrow, int ncol);
    virtual void
    SetDataPtrMCSR(int** row_offset, int** col, ValueType** val, int nnz, int nrow, int ncol);
    virtual void LeaveDataPtrMCSR(int** row_offset, int** col, ValueType** val);

    virtual bool ConvertFrom(const BaseMatrix<ValueType>& mat);

    virtual void CopyFrom(const BaseMatrix<ValueType>& mat);
    virtual void CopyFromAsync(const BaseMatrix<ValueType>& mat);
    virtual void CopyTo(BaseMatrix<ValueType>* mat) const;
    virtual void CopyToAsync(BaseMatrix<ValueType>* mat) const;

    virtual void CopyFromHost(const HostMatrix<ValueType>& src);
    virtual void CopyFromHostAsync(const HostMatrix<ValueType>& src);
    virtual void CopyToHost(HostMatrix<ValueType>* dst) const;
    virtual void CopyToHostAsync(HostMatrix<ValueType>* dst) const;

    virtual void Apply(const BaseVector<ValueType>& in, BaseVector<ValueType>* out) const;
    virtual void
    ApplyAdd(const BaseVector<ValueType>& in, ValueType scalar, BaseVector<ValueType>* out) const;

    private:
    MatrixMCSR<ValueType, int> mat_;

    friend class BaseVector<ValueType>;
    friend class AcceleratorVector<ValueType>;
    friend class HIPAcceleratorVector<ValueType>;
};

} // namespace rocalution

#endif // ROCALUTION_HIP_MATRIX_MCSR_HPP_
