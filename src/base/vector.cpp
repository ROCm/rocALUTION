/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "../utils/def.hpp"
#include "vector.hpp"
#include "global_vector.hpp"
#include "local_vector.hpp"
#include "../utils/log.hpp"

#include <complex>

namespace rocalution {

template <typename ValueType>
Vector<ValueType>::Vector()
{
    log_debug(this, "Vector::Vector()");

    this->object_name_ = "";
}

template <typename ValueType>
Vector<ValueType>::~Vector()
{
    log_debug(this, "Vector::~Vector()");
}

template <typename ValueType>
int Vector<ValueType>::GetLocalSize(void) const
{
    return IndexTypeToInt(this->GetSize());
}

template <typename ValueType>
int Vector<ValueType>::GetGhostSize(void) const
{
    return 0;
}

template <typename ValueType>
void Vector<ValueType>::CopyFrom(const LocalVector<ValueType>& src)
{
    LOG_INFO("Vector<ValueType>::CopyFrom(const LocalVector<ValueType>& src)");
    LOG_INFO("Mismatched types:");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::CopyFrom(const GlobalVector<ValueType>& src)
{
    LOG_INFO("Vector<ValueType>::CopyFrom(const GlobalVector<ValueType>& src)");
    LOG_INFO("Mismatched types:");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::CopyFromAsync(const LocalVector<ValueType>& src)
{
    LOG_INFO("Vector<ValueType>::CopyFromAsync(const LocalVector<ValueType>& src)");
    LOG_INFO("Mismatched types:");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::CopyFromFloat(const LocalVector<float>& src)
{
    LOG_INFO("Vector<ValueType>::CopyFromFloat(const LocalVector<float>& src)");
    LOG_INFO("Mismatched types:");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::CopyFromDouble(const LocalVector<double>& src)
{
    LOG_INFO("Vector<ValueType>::CopyFromDouble(const LocalVector<double>& src)");
    LOG_INFO("Mismatched types:");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::CopyFrom(const LocalVector<ValueType>& src,
                                 int src_offset,
                                 int dst_offset,
                                 int size)
{
    LOG_INFO("Vector<ValueType>::CopyFrom(const LocalVector<ValueType>& src,"
             "int src_offset,"
             "int dst_offset,"
             "int size");
    LOG_INFO("Mismatched types:");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::CloneFrom(const LocalVector<ValueType>& src)
{
    LOG_INFO("Vector<ValueType>::CloneFrom(const LocalVector<ValueType>& src)");
    LOG_INFO("Mismatched types:");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::CloneFrom(const GlobalVector<ValueType>& src)
{
    LOG_INFO("Vector<ValueType>::CloneFrom(const GlobalVector<ValueType>& src)");
    LOG_INFO("Mismatched types:");
    this->Info();
    src.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::AddScale(const LocalVector<ValueType>& x, ValueType alpha)
{
    LOG_INFO("Vector<ValueType>::AddScale(const LocalVector<ValueType>& x, ValueType alpha)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::AddScale(const GlobalVector<ValueType>& x, ValueType alpha)
{
    LOG_INFO("Vector<ValueType>::AddScale(const GlobalVector<ValueType>& x, ValueType alpha)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::ScaleAdd(ValueType alpha, const LocalVector<ValueType>& x)
{
    LOG_INFO("Vector<ValueType>::ScaleAdd(ValueType alpha, const LocalVector<ValueType>& x)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::ScaleAdd(ValueType alpha, const GlobalVector<ValueType>& x)
{
    LOG_INFO("Vector<ValueType>::ScaleAdd(ValueType alpha, const GlobalVector<ValueType>& x)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::ScaleAddScale(ValueType alpha,
                                      const LocalVector<ValueType>& x,
                                      ValueType beta,
                                      int src_offset,
                                      int dst_offset,
                                      int size)
{
    LOG_INFO("Vector<ValueType>::ScaleAddScale(ValueType alpha,"
             "const LocalVector<ValueType>& x,"
             "ValueType beta,"
             "int src_offset,"
             "int dst_offset,"
             "int size)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::ScaleAddScale(ValueType alpha,
                                      const GlobalVector<ValueType>& x,
                                      ValueType beta,
                                      int src_offset,
                                      int dst_offset,
                                      int size)
{
    LOG_INFO("Vector<ValueType>::ScaleAddScale(ValueType alpha,"
             "const GlobalVector<ValueType>& x,"
             "ValueType beta,"
             "int src_offset,"
             "int dst_offset,"
             "int size)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
ValueType Vector<ValueType>::Dot(const LocalVector<ValueType>& x) const
{
    LOG_INFO("Vector<ValueType>::Dot(const LocalVector<ValueType>& x) const");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
ValueType Vector<ValueType>::Dot(const GlobalVector<ValueType>& x) const
{
    LOG_INFO("Vector<ValueType>::Dot(const GlobalVector<ValueType>& x) const");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
ValueType Vector<ValueType>::DotNonConj(const LocalVector<ValueType>& x) const
{
    LOG_INFO("Vector<ValueType>::DotNonConj(const LocalVector<ValueType>& x) const");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
ValueType Vector<ValueType>::DotNonConj(const GlobalVector<ValueType>& x) const
{
    LOG_INFO("Vector<ValueType>::DotNonConj(const GlobalVector<ValueType>& x) const");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::PointWiseMult(const LocalVector<ValueType>& x)
{
    LOG_INFO("Vector<ValueType>::PointWiseMult(const LocalVector<ValueType>& x)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::PointWiseMult(const GlobalVector<ValueType>& x)
{
    LOG_INFO("Vector<ValueType>::PointWiseMult(const GlobalVector<ValueType>& x)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::PointWiseMult(const LocalVector<ValueType>& x,
                                      const LocalVector<ValueType>& y)
{
    LOG_INFO("Vector<ValueType>::PointWiseMult(const LocalVector<ValueType>& x, const "
             "LocalVector<ValueType>& y)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    y.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::PointWiseMult(const GlobalVector<ValueType>& x,
                                      const GlobalVector<ValueType>& y)
{
    LOG_INFO("Vector<ValueType>::PointWiseMult(const GlobalVector<ValueType>& x, const "
             "GlobalVector<ValueType>& y)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    y.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::ScaleAddScale(ValueType alpha,
                                      const LocalVector<ValueType>& x,
                                      ValueType beta)
{
    LOG_INFO("ScaleAddScale(ValueType alpha, const LocalVector<ValueType>& x, "
             "ValueType beta)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::ScaleAddScale(ValueType alpha,
                                      const GlobalVector<ValueType>& x,
                                      ValueType beta)
{
    LOG_INFO("ScaleAddScale(ValueType alpha, const GlobalVector<ValueType>& x, "
             "ValueType beta)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::ScaleAdd2(ValueType alpha,
                                  const LocalVector<ValueType>& x,
                                  ValueType beta,
                                  const LocalVector<ValueType>& y,
                                  ValueType gamma)
{
    LOG_INFO("ScaleAdd2(ValueType alpha, const LocalVector<ValueType>& x, ValueType "
             "beta, const LocalVector<ValueType>& y, ValueType gamma)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    y.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::ScaleAdd2(ValueType alpha,
                                  const GlobalVector<ValueType>& x,
                                  ValueType beta,
                                  const GlobalVector<ValueType>& y,
                                  ValueType gamma)
{
    LOG_INFO("ScaleAdd2(ValueType alpha, const GlobalVector<ValueType>& x, ValueType "
             "beta, const GlobalVector<ValueType>& y, ValueType gamma)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    y.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::ExclusiveScan(const LocalVector<ValueType>& x)
{
    LOG_INFO("ExclusiveScan(const LocalVector<ValueType>& x)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template <typename ValueType>
void Vector<ValueType>::ExclusiveScan(const GlobalVector<ValueType>& x)
{
    LOG_INFO("ExclusiveScan(const GlobalVector<ValueType>& x)");
    LOG_INFO("Mismatched types:");
    this->Info();
    x.Info();
    FATAL_ERROR(__FILE__, __LINE__);
}

template class Vector<double>;
template class Vector<float>;
#ifdef SUPPORT_COMPLEX
template class Vector<std::complex<double>>;
template class Vector<std::complex<float>>;
#endif

template class Vector<int>;

} // namespace rocalution
