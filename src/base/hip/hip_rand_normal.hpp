/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

#ifndef ROCALUTION_HIP_RAND_NORMAL_HPP_
#define ROCALUTION_HIP_RAND_NORMAL_HPP_

#include "hip_rand.hpp"
#include <rocrand/rocrand.hpp>

namespace rocalution
{
    //
    // Implementation of a normal distribution using rocRAND.
    //
    template <typename T>
    class HIPRandNormal_rocRAND;

    template <typename T>
    struct CRTP_HIPRand_Traits<HIPRandNormal_rocRAND<T>>
    {
        using data_t = T;
    };

    template <typename T>
    class HIPRandNormal_rocRAND : public CRTP_HIPRand<HIPRandNormal_rocRAND<T>>
    {
    protected:
        using value_type = typename numeric_traits<T>::value_type;
        rocrand_cpp::mtgp32_engine<0UL>              m_engine;
        rocrand_cpp::normal_distribution<value_type> m_distribution;

    public:
        inline HIPRandNormal_rocRAND(unsigned long long seed, value_type mean, value_type var)
            : m_engine(seed)
            , m_distribution(mean, var){};

        inline void Generate(T* data, size_t size)
        {
            if(size > 0)
            {
                assert(0 == sizeof(T) % sizeof(value_type));
                this->m_distribution(
                    this->m_engine, ((value_type*)data), size * (sizeof(T) / sizeof(value_type)));
            }
        };
    };

} // namespace rocalution

#endif // ROCALUTION_BASE_VECTOR_HPP_
