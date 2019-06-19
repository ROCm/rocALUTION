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

#include "base_vector.hpp"
#include "../utils/def.hpp"
#include "../utils/log.hpp"
#include "backend_manager.hpp"

#include <complex>
#include <fstream>
#include <stdlib.h>

namespace rocalution
{

    template <typename ValueType>
    BaseVector<ValueType>::BaseVector()
    {
        log_debug(this, "BaseVector::BaseVector()");

        this->size_       = 0;
        this->index_size_ = 0;
    }

    template <typename ValueType>
    BaseVector<ValueType>::~BaseVector()
    {
        log_debug(this, "BaseVector::~BaseVector()");
    }

    template <typename ValueType>
    inline int BaseVector<ValueType>::GetSize(void) const
    {
        return this->size_;
    }

    template <typename ValueType>
    void BaseVector<ValueType>::set_backend(const Rocalution_Backend_Descriptor local_backend)
    {
        this->local_backend_ = local_backend;
    }

    template <typename ValueType>
    bool BaseVector<ValueType>::Check(void) const
    {
        LOG_INFO("BaseVector::Check()");
        this->Info();
        LOG_INFO("Only host version!");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void BaseVector<ValueType>::CopyFromData(const ValueType* data)
    {
        LOG_INFO("CopyFromData(const ValueType* data)");
        this->Info();
        LOG_INFO("This function is not available for this backend");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void BaseVector<ValueType>::CopyToData(ValueType* data) const
    {
        LOG_INFO("CopyToData(ValueType *val) const");
        this->Info();
        LOG_INFO("This function is not available for this backend");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void BaseVector<ValueType>::CopyFromFloat(const BaseVector<float>& vec)
    {
        LOG_INFO("BaseVector::CopyFromFloat(const BaseVector<float>& vec)");
        this->Info();
        vec.Info();
        LOG_INFO("Float casting is not available for this backend");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void BaseVector<ValueType>::CopyFromDouble(const BaseVector<double>& vec)
    {
        LOG_INFO("BaseVector::CopyFromDouble(const BaseVector<double>& vec)");
        this->Info();
        vec.Info();
        LOG_INFO("Float casting is not available for this backend");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    bool BaseVector<ValueType>::Restriction(const BaseVector<ValueType>& vec_fine,
                                            const BaseVector<int>&       map)
    {
        return false;
    }

    template <typename ValueType>
    bool BaseVector<ValueType>::Prolongation(const BaseVector<ValueType>& vec_coarse,
                                             const BaseVector<int>&       map)
    {
        return false;
    }

    template <typename ValueType>
    void BaseVector<ValueType>::CopyFromAsync(const BaseVector<ValueType>& vec)
    {
        // default is no async
        LOG_VERBOSE_INFO(4, "*** info: BaseVector::CopyFromAsync() no async available)");

        this->CopyFrom(vec);
    }

    template <typename ValueType>
    void BaseVector<ValueType>::CopyToAsync(BaseVector<ValueType>* vec) const
    {
        // default is no async
        LOG_VERBOSE_INFO(4, "*** info: BaseVector::CopyToAsync() no async available)");

        this->CopyTo(vec);
    }

    template <typename ValueType>
    AcceleratorVector<ValueType>::AcceleratorVector()
    {
    }

    template <typename ValueType>
    AcceleratorVector<ValueType>::~AcceleratorVector()
    {
    }

    template <typename ValueType>
    void AcceleratorVector<ValueType>::CopyFromHostAsync(const HostVector<ValueType>& src)
    {
        // default is no async
        this->CopyFromHost(src);
    }

    template <typename ValueType>
    void AcceleratorVector<ValueType>::CopyToHostAsync(HostVector<ValueType>* dst) const
    {
        // default is no async
        this->CopyToHost(dst);
    }

    template class BaseVector<double>;
    template class BaseVector<float>;
#ifdef SUPPORT_COMPLEX
    template class BaseVector<std::complex<double>>;
    template class BaseVector<std::complex<float>>;
#endif
    template class BaseVector<int>;

    template class AcceleratorVector<double>;
    template class AcceleratorVector<float>;
#ifdef SUPPORT_COMPLEX
    template class AcceleratorVector<std::complex<double>>;
    template class AcceleratorVector<std::complex<float>>;
#endif
    template class AcceleratorVector<int>;

} // namespace rocalution
