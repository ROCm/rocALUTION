/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_LOCAL_VECTOR_HPP_
#define ROCALUTION_LOCAL_VECTOR_HPP_

#include "../utils/types.hpp"
#include "vector.hpp"

namespace rocalution {

template <typename ValueType>
class BaseVector;
template <typename ValueType>
class HostVector;

template <typename ValueType>
class LocalMatrix;
template <typename ValueType>
class GlobalMatrix;

template <typename ValueType>
class LocalStencil;

/** \ingroup op_vec_module
  * \class LocalVector
  * \brief LocalVector class
  * \details
  * A LocalVector is called local, because it will always stay on a single system. The
  * system can contain several CPUs via UMA or NUMA memory system or it can contain an
  * accelerator.
  *
  * \tparam ValueType - can be int, float, double, std::complex<float> and
  *                     std::complex<double>
  */
template <typename ValueType>
class LocalVector : public Vector<ValueType>
{
    public:
    LocalVector();
    virtual ~LocalVector();

    virtual void MoveToAccelerator(void);
    virtual void MoveToAcceleratorAsync(void);
    virtual void MoveToHost(void);
    virtual void MoveToHostAsync(void);
    virtual void Sync(void);

    virtual void Info(void) const;
    virtual IndexType2 GetSize(void) const;

    /** \private */
    const LocalVector<ValueType>& GetInterior() const;
    /** \private */
    LocalVector<ValueType>& GetInterior();

    virtual bool Check(void) const;

    /** \brief Allocate a local vector with name and size */
    void Allocate(std::string name, IndexType2 size);

    /** \brief Initialize a vector with externally allocated data */
    void SetDataPtr(ValueType** ptr, std::string name, int size);
    /** \brief Get a pointer from the vector data and free the vector object */
    void LeaveDataPtr(ValueType** ptr);

    virtual void Clear();
    virtual void Zeros();
    virtual void Ones();
    virtual void SetValues(ValueType val);
    virtual void SetRandomUniform(unsigned long long seed,
                                  ValueType a = static_cast<ValueType>(-1),
                                  ValueType b = static_cast<ValueType>(1));
    virtual void SetRandomNormal(unsigned long long seed,
                                 ValueType mean = static_cast<ValueType>(0),
                                 ValueType var  = static_cast<ValueType>(1));

    /** \brief Access operator (only for host data) */
    ValueType& operator[](int i);
    /** \brief Access operator (only for host data) */
    const ValueType& operator[](int i) const;

    virtual void ReadFileASCII(const std::string filename);
    virtual void WriteFileASCII(const std::string filename) const;
    virtual void ReadFileBinary(const std::string filename);
    virtual void WriteFileBinary(const std::string filename) const;

    virtual void CopyFrom(const LocalVector<ValueType>& src);
    virtual void CopyFromAsync(const LocalVector<ValueType>& src);
    virtual void CopyFromFloat(const LocalVector<float>& src);
    virtual void CopyFromDouble(const LocalVector<double>& src);

    virtual void CopyFrom(const LocalVector<ValueType>& src, int src_offset, int dst_offset, int size);

    /** \brief Copy a vector under permutation (forward permutation) */
    void CopyFromPermute(const LocalVector<ValueType>& src, const LocalVector<int>& permutation);

    /** \brief Copy a vector under permutation (backward permutation) */
    void CopyFromPermuteBackward(const LocalVector<ValueType>& src,
                                 const LocalVector<int>& permutation);

    virtual void CloneFrom(const LocalVector<ValueType>& src);

    /** \brief Copy (import) vector described in one array (values).
      * The object data has to be allocated (call Allocate first)
      */
    void CopyFromData(const ValueType* data);

    /** \brief Copy (export) vector described in one array (values).
      * The output array has to be allocated
      */
    void CopyToData(ValueType* data) const;

    /** \brief Perform in-place permutation (forward) of the vector */
    void Permute(const LocalVector<int>& permutation);

    /** \brief Perform in-place permutation (backward) of the vector */
    void PermuteBackward(const LocalVector<int>& permutation);

    /** \brief Restriction operator based on restriction mapping vector */
    void Restriction(const LocalVector<ValueType>& vec_fine, const LocalVector<int>& map);

    /** \brief Prolongation operator based on restriction mapping vector */
    void Prolongation(const LocalVector<ValueType>& vec_coarse, const LocalVector<int>& map);

    virtual void AddScale(const LocalVector<ValueType>& x, ValueType alpha);
    virtual void ScaleAdd(ValueType alpha, const LocalVector<ValueType>& x);
    virtual void ScaleAddScale(ValueType alpha, const LocalVector<ValueType>& x, ValueType beta);
    virtual void ScaleAddScale(ValueType alpha,
                               const LocalVector<ValueType>& x,
                               ValueType beta,
                               int src_offset,
                               int dst_offset,
                               int size);
    virtual void ScaleAdd2(ValueType alpha,
                           const LocalVector<ValueType>& x,
                           ValueType beta,
                           const LocalVector<ValueType>& y,
                           ValueType gamma);
    virtual void Scale(ValueType alpha);
    virtual ValueType Dot(const LocalVector<ValueType>& x) const;
    virtual ValueType DotNonConj(const LocalVector<ValueType>& x) const;
    virtual ValueType Norm(void) const;
    virtual ValueType Reduce(void) const;
    virtual ValueType Asum(void) const;
    virtual int Amax(ValueType& value) const;
    virtual void PointWiseMult(const LocalVector<ValueType>& x);
    virtual void PointWiseMult(const LocalVector<ValueType>& x, const LocalVector<ValueType>& y);
    virtual void Power(double power);

    /** \brief Set index array */
    void SetIndexArray(int size, const int* index);
    /** \brief Get indexed values */
    void GetIndexValues(ValueType* values) const;
    /** \brief Set indexed values */
    void SetIndexValues(const ValueType* values);
    /** \brief Get continuous indexed values */
    void GetContinuousValues(int start, int end, ValueType* values) const;
    /** \brief Set continuous indexed values */
    void SetContinuousValues(int start, int end, const ValueType* values);
    /** \brief Extract coarse boundary mapping */
    void
    ExtractCoarseMapping(int start, int end, const int* index, int nc, int* size, int* map) const;
    /** \brief Extract coarse boundary index */
    void ExtractCoarseBoundary(
        int start, int end, const int* index, int nc, int* size, int* boundary) const;

    protected:
    virtual bool is_host(void) const;
    virtual bool is_accel(void) const;

    private:
    // Pointer from the base vector class to the current allocated vector (host_ or accel_)
    BaseVector<ValueType>* vector_;

    // Host Vector
    HostVector<ValueType>* vector_host_;

    // Accelerator Vector
    AcceleratorVector<ValueType>* vector_accel_;

    friend class LocalVector<double>;
    friend class LocalVector<float>;
    friend class LocalVector<std::complex<double>>;
    friend class LocalVector<std::complex<float>>;
    friend class LocalVector<int>;

    friend class LocalMatrix<double>;
    friend class LocalMatrix<float>;
    friend class LocalMatrix<std::complex<double>>;
    friend class LocalMatrix<std::complex<float>>;

    friend class LocalStencil<double>;
    friend class LocalStencil<float>;
    friend class LocalStencil<std::complex<double>>;
    friend class LocalStencil<std::complex<float>>;

    friend class GlobalVector<ValueType>;
    friend class LocalMatrix<ValueType>;
    friend class GlobalMatrix<ValueType>;
};

} // namespace rocalution

#endif // ROCALUTION_LOCAL_VECTOR_HPP_
