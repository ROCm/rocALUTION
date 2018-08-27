#ifndef ROCALUTION_HIP_MATRIX_ELL_HPP_
#define ROCALUTION_HIP_MATRIX_ELL_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

#include <rocsparse.h>

namespace rocalution {

template <typename ValueType>
class HIPAcceleratorMatrixELL : public HIPAcceleratorMatrix<ValueType>
{
    public:
    HIPAcceleratorMatrixELL();
    HIPAcceleratorMatrixELL(const Rocalution_Backend_Descriptor local_backend);
    virtual ~HIPAcceleratorMatrixELL();

    inline int GetMaxRow(void) const { return mat_.max_row; }

    virtual void Info(void) const;
    virtual unsigned int GetMatFormat(void) const { return ELL; }

    virtual void Clear(void);
    virtual void AllocateELL(const int nnz, const int nrow, const int ncol, const int max_row);
    virtual void SetDataPtrELL(int** col,
                               ValueType** val,
                               const int nnz,
                               const int nrow,
                               const int ncol,
                               const int max_row);
    virtual void LeaveDataPtrELL(int** col, ValueType** val, int& max_row);

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
    virtual void ApplyAdd(const BaseVector<ValueType>& in,
                          const ValueType scalar,
                          BaseVector<ValueType>* out) const;

    private:
    MatrixELL<ValueType, int> mat_;

    rocsparse_mat_descr mat_descr_;

    friend class HIPAcceleratorMatrixCSR<ValueType>;

    friend class BaseVector<ValueType>;
    friend class AcceleratorVector<ValueType>;
    friend class HIPAcceleratorVector<ValueType>;
};

} // namespace rocalution

#endif // ROCALUTION_HIP_MATRIX_ELL_HPP_
