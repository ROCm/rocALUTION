/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_VECTOR_HPP_
#define ROCALUTION_VECTOR_HPP_

#include "../utils/types.hpp"
#include "base_rocalution.hpp"

#include <iostream>
#include <string>
#include <cstdlib>

namespace rocalution {

template <typename ValueType>
class GlobalVector;
template <typename ValueType>
class LocalVector;

// Vector
template <typename ValueType>
class Vector : public BaseRocalution<ValueType>
{
    public:
    Vector();
    virtual ~Vector();

    /// Return the size of the vector
    virtual IndexType2 GetSize(void) const = 0;
    /// Return the size of the local vector
    virtual int GetLocalSize(void) const;
    /// Return the size of the ghost vector
    virtual int GetGhostSize(void) const;

    /// Return true if the vector is ok (empty vector is also ok)
    /// and false if some of values are NaN
    virtual bool Check(void) const = 0;

    /// Clear (free) the vector
    virtual void Clear(void) = 0;

    /// Set the values of the vector to zero
    virtual void Zeros(void) = 0;

    /// Set the values of the vector to one
    virtual void Ones(void) = 0;

    /// Set the values of the vector to given argument
    virtual void SetValues(ValueType val) = 0;

    /// Set random values from interval [a,b]
    virtual void SetRandomUniform(unsigned long long seed,
                                  ValueType a = static_cast<ValueType>(-1),
                                  ValueType b = static_cast<ValueType>(1)) = 0;

    /// Set random values from normal distribution
    virtual void SetRandomNormal(unsigned long long seed,
                                 ValueType mean = static_cast<ValueType>(0),
                                 ValueType var = static_cast<ValueType>(1)) = 0;

    /// Read vector from ASCII file
    virtual void ReadFileASCII(const std::string filename) = 0;

    /// Write vector to ASCII file
    virtual void WriteFileASCII(const std::string filename) const = 0;

    /// Read vector from binary file
    virtual void ReadFileBinary(const std::string filename) = 0;

    /// Write vector to binary file
    virtual void WriteFileBinary(const std::string filename) const = 0;

    /// Copy values from another local vector
    virtual void CopyFrom(const LocalVector<ValueType>& src);

    /// Copy values from another global vector
    virtual void CopyFrom(const GlobalVector<ValueType>& src);

    /// Async copy
    virtual void CopyFromAsync(const LocalVector<ValueType>& src);

    /// Copy values from float vector
    virtual void CopyFromFloat(const LocalVector<float>& src);

    /// Copy values from double vector
    virtual void CopyFromDouble(const LocalVector<double>& src);

    /// Copy data (not entire vector) from another local vector with specified src/dst offsets and
    /// size
    virtual void
    CopyFrom(const LocalVector<ValueType>& src, int src_offset, int dst_offset, int size);
    /// Clone the entire vector (data+backend descr) from another local vector
    virtual void CloneFrom(const LocalVector<ValueType>& src);

    /// Clone the entire vector (data+backend descr) from another global vector
    virtual void CloneFrom(const GlobalVector<ValueType>& src);

    /// Perform vector update of type this = this + alpha*x
    virtual void AddScale(const LocalVector<ValueType>& x, ValueType alpha);
    /// Perform vector update of type this = this + alpha*x
    virtual void AddScale(const GlobalVector<ValueType>& x, ValueType alpha);

    /// Perform vector update of type this = alpha*this + x
    virtual void ScaleAdd(ValueType alpha, const LocalVector<ValueType>& x);
    /// Perform vector update of type this = alpha*this + x
    virtual void ScaleAdd(ValueType alpha, const GlobalVector<ValueType>& x);

    /// Perform vector update of type this = alpha*this + x*beta
    virtual void ScaleAddScale(ValueType alpha, const LocalVector<ValueType>& x, ValueType beta);
    /// Perform vector update of type this = alpha*this + x*beta
    virtual void ScaleAddScale(ValueType alpha, const GlobalVector<ValueType>& x, ValueType beta);
    /// Perform vector update of type this = alpha*this + x*beta with offsets for a specified part
    /// of a vector
    virtual void ScaleAddScale(ValueType alpha,
                               const LocalVector<ValueType>& x,
                               ValueType beta,
                               int src_offset,
                               int dst_offset,
                               int size);
    /// Perform vector update of type this = alpha*this + x*beta with offsets for a specified part
    /// of a vector
    virtual void ScaleAddScale(ValueType alpha,
                               const GlobalVector<ValueType>& x,
                               ValueType beta,
                               int src_offset,
                               int dst_offset,
                               int size);

    /// Perform vector update of type this = alpha*this + x*beta + y*gamma
    virtual void ScaleAdd2(ValueType alpha,
                           const LocalVector<ValueType>& x,
                           ValueType beta,
                           const LocalVector<ValueType>& y,
                           ValueType gamma);
    /// Perform vector update of type this = alpha*this + x*beta + y*gamma
    virtual void ScaleAdd2(ValueType alpha,
                           const GlobalVector<ValueType>& x,
                           ValueType beta,
                           const GlobalVector<ValueType>& y,
                           ValueType gamma);

    /// Perform vector scaling this = alpha*this
    virtual void Scale(ValueType alpha) = 0;

    /// Performs exlusive scan
    virtual void ExclusiveScan(const LocalVector<ValueType>& x);
    /// Performs exlusive scan
    virtual void ExclusiveScan(const GlobalVector<ValueType>& x);

    /// Compute dot (scalar) product, return this^T y
    virtual ValueType Dot(const LocalVector<ValueType>& x) const;
    /// Compute dot (scalar) product, return this^T y
    virtual ValueType Dot(const GlobalVector<ValueType>& x) const;

    /// Compute non-conjugate dot (scalar) product, return this^T y
    virtual ValueType DotNonConj(const LocalVector<ValueType>& x) const;
    /// Compute non-conjugate dot (scalar) product, return this^T y
    virtual ValueType DotNonConj(const GlobalVector<ValueType>& x) const;

    /// Compute L2 norm of the vector, return =  srqt(this^T this)
    virtual ValueType Norm(void) const = 0;

    /// Reduce the vector
    virtual ValueType Reduce(void) const = 0;

    /// Compute the sum of the absolute values of the vector (L1 norm), return =  sum(|this|)
    virtual ValueType Asum(void) const = 0;

    /// Compute the absolute max value of the vector, return = index(max(|this|))
    virtual int Amax(ValueType& value) const = 0;

    /// Perform point-wise multiplication (element-wise) of type this = this * x
    virtual void PointWiseMult(const LocalVector<ValueType>& x);
    /// Perform point-wise multiplication (element-wise) of type this = this * x
    virtual void PointWiseMult(const GlobalVector<ValueType>& x);

    /// Perform point-wise multiplication (element-wise) of type this = x*y
    virtual void PointWiseMult(const LocalVector<ValueType>& x, const LocalVector<ValueType>& y);
    /// Perform point-wise multiplication (element-wise) of type this = x*y
    virtual void PointWiseMult(const GlobalVector<ValueType>& x, const GlobalVector<ValueType>& y);

    /// Perform power operation to a vector
    virtual void Power(double power) = 0;
};

} // namespace rocalution

#endif // ROCALUTION_VECTOR_HPP_
