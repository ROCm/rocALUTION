/* ************************************************************************
 * Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_LOCAL_STENCIL_HPP_
#define ROCALUTION_LOCAL_STENCIL_HPP_

#include "../utils/types.hpp"
#include "local_vector.hpp"
#include "operator.hpp"
#include "rocalution/export.hpp"
#include "stencil_types.hpp"

namespace rocalution
{

    template <typename ValueType>
    class BaseStencil;
    template <typename ValueType>
    class HostStencil;
    template <typename ValueType>
    class AcceleratorStencil;

    template <typename ValueType>
    class LocalVector;
    template <typename ValueType>
    class GlobalVector;

    /** \ingroup op_vec_module
  * \class LocalStencil
  * \brief LocalStencil class
  * \details
  * A LocalStencil is called local, because it will always stay on a single system. The
  * system can contain several CPUs via UMA or NUMA memory system or it can contain an
  * accelerator.
  *
  * \tparam ValueType - can be int, float, double, std::complex<float> and
  *                     std::complex<double>
  */
    template <typename ValueType>
    class LocalStencil : public Operator<ValueType>
    {
    public:
        ROCALUTION_EXPORT
        LocalStencil();
        /** \brief Initialize a local stencil with a type */
        ROCALUTION_EXPORT
        LocalStencil(unsigned int type);
        ROCALUTION_EXPORT
        virtual ~LocalStencil();

        ROCALUTION_EXPORT
        virtual void Info() const;

        /** \brief Return the dimension of the stencil */
        ROCALUTION_EXPORT
        int GetNDim(void) const;
        ROCALUTION_EXPORT
        virtual IndexType2 GetM(void) const;
        ROCALUTION_EXPORT
        virtual IndexType2 GetN(void) const;
        ROCALUTION_EXPORT
        virtual IndexType2 GetNnz(void) const;

        /** \brief Set the stencil grid size */
        ROCALUTION_EXPORT
        void SetGrid(int size);

        ROCALUTION_EXPORT
        virtual void Clear();

        ROCALUTION_EXPORT
        virtual void Apply(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;
        ROCALUTION_EXPORT
        virtual void ApplyAdd(const LocalVector<ValueType>& in,
                              ValueType                     scalar,
                              LocalVector<ValueType>*       out) const;

        ROCALUTION_EXPORT
        virtual void MoveToAccelerator(void);
        ROCALUTION_EXPORT
        virtual void MoveToHost(void);

    protected:
        virtual bool is_host_(void) const
        {
            return true;
        };
        virtual bool is_accel_(void) const
        {
            return false;
        };

    private:
        std::string object_name_;

        BaseStencil<ValueType>* stencil_;

        HostStencil<ValueType>*        stencil_host_;
        AcceleratorStencil<ValueType>* stencil_accel_;

        friend class LocalVector<ValueType>;
        friend class GlobalVector<ValueType>;
    };

} // namespace rocalution

#endif // ROCALUTION_LOCAL_STENCIL_HPP_
