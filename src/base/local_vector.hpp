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

#ifndef ROCALUTION_LOCAL_VECTOR_HPP_
#define ROCALUTION_LOCAL_VECTOR_HPP_

#include "../utils/types.hpp"
#include "rocalution/export.hpp"
#include "vector.hpp"

namespace rocalution
{

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
        ROCALUTION_EXPORT
        LocalVector();
        ROCALUTION_EXPORT
        virtual ~LocalVector();

        ROCALUTION_EXPORT
        virtual void MoveToAccelerator(void);
        ROCALUTION_EXPORT
        virtual void MoveToAcceleratorAsync(void);
        ROCALUTION_EXPORT
        virtual void MoveToHost(void);
        ROCALUTION_EXPORT
        virtual void MoveToHostAsync(void);
        ROCALUTION_EXPORT
        virtual void Sync(void);

        ROCALUTION_EXPORT
        virtual void Info(void) const;
        ROCALUTION_EXPORT
        virtual IndexType2 GetSize(void) const;

        /** \private */
        ROCALUTION_EXPORT
        const LocalVector<ValueType>& GetInterior() const;
        /** \private */
        ROCALUTION_EXPORT
        LocalVector<ValueType>& GetInterior();

        ROCALUTION_EXPORT
        virtual bool Check(void) const;

        /** \brief Allocate a local vector with name and size
      * \details
      * The local vector allocation function requires a name of the object (this is only
      * for information purposes) and corresponding size description for vector objects.
      *
      * @param[in]
      * name    object name
      * @param[in]
      * size    number of elements in the vector
      *
      * \par Example
      * \code{.cpp}
      *   LocalVector<ValueType> vec;
      *
      *   vec.Allocate("my vector", 100);
      *   vec.Clear();
      * \endcode
      */
        ROCALUTION_EXPORT
        void Allocate(std::string name, IndexType2 size);

        /** \brief Initialize a LocalVector on the host with externally allocated data
      * \details
      * \p SetDataPtr has direct access to the raw data via pointers. Already allocated
      * data can be set by passing the pointer.
      *
      * \note
      * Setting data pointer will leave the original pointer empty (set to \p NULL).
      *
      * \par Example
      * \code{.cpp}
      *   // Allocate vector
      *   ValueType* ptr_vec = new ValueType[200];
      *
      *   // Fill vector
      *   // ...
      *
      *   // rocALUTION local vector object
      *   LocalVector<ValueType> vec;
      *
      *   // Set the vector data, ptr_vec will become invalid
      *   vec.SetDataPtr(&ptr_vec, "my_vector", 200);
      * \endcode
      */
        ROCALUTION_EXPORT
        void SetDataPtr(ValueType** ptr, std::string name, int size);

        /** \brief Leave a LocalVector to host pointers
      * \details
      * \p LeaveDataPtr has direct access to the raw data via pointers. A LocalVector
      * object can leave its raw data to a host pointer. This will leave the LocalVector
      * empty.
      *
      * \par Example
      * \code{.cpp}
      *   // rocALUTION local vector object
      *   LocalVector<ValueType> vec;
      *
      *   // Allocate the vector
      *   vec.Allocate("my_vector", 100);
      *
      *   // Fill vector
      *   // ...
      *
      *   ValueType* ptr_vec = NULL;
      *
      *   // Get (steal) the data from the vector, this will leave the local vector object empty
      *   vec.LeaveDataPtr(&ptr_vec);
      * \endcode
      */
        ROCALUTION_EXPORT
        void LeaveDataPtr(ValueType** ptr);

        ROCALUTION_EXPORT
        virtual void Clear();
        ROCALUTION_EXPORT
        virtual void Zeros();
        ROCALUTION_EXPORT
        virtual void Ones();
        ROCALUTION_EXPORT
        virtual void SetValues(ValueType val);
        ROCALUTION_EXPORT
        virtual void SetRandomUniform(unsigned long long seed,
                                      ValueType          a = static_cast<ValueType>(-1),
                                      ValueType          b = static_cast<ValueType>(1));
        ROCALUTION_EXPORT
        virtual void SetRandomNormal(unsigned long long seed,
                                     ValueType          mean = static_cast<ValueType>(0),
                                     ValueType          var  = static_cast<ValueType>(1));

        /** \brief Access operator (only for host data)
      * \details
      * The elements in the vector can be accessed via [] operators, when the vector is
      * allocated on the host.
      *
      * @param[in]
      * i   access data at index \p i
      *
      * \returns    value at index \p i
      *
      * \par Example
      * \code{.cpp}
      *   // rocALUTION local vector object
      *   LocalVector<ValueType> vec;
      *
      *   // Allocate vector
      *   vec.Allocate("my_vector", 100);
      *
      *   // Initialize vector with 1
      *   vec.Ones();
      *
      *   // Set even elements to -1
      *   for(int i = 0; i < vec.GetSize(); i += 2)
      *   {
      *     vec[i] = -1;
      *   }
      * \endcode
      */
        /**@{*/
        ROCALUTION_EXPORT
        ValueType& operator[](int i);
        ROCALUTION_EXPORT
        const ValueType& operator[](int i) const;
        /**@}*/

        ROCALUTION_EXPORT
        virtual void ReadFileASCII(const std::string& filename);
        ROCALUTION_EXPORT
        virtual void WriteFileASCII(const std::string& filename) const;
        ROCALUTION_EXPORT
        virtual void ReadFileBinary(const std::string& filename);
        ROCALUTION_EXPORT
        virtual void WriteFileBinary(const std::string& filename) const;

        ROCALUTION_EXPORT
        virtual void CopyFrom(const LocalVector<ValueType>& src);
        ROCALUTION_EXPORT
        virtual void CopyFromAsync(const LocalVector<ValueType>& src);
        ROCALUTION_EXPORT
        virtual void CopyFromFloat(const LocalVector<float>& src);
        ROCALUTION_EXPORT
        virtual void CopyFromDouble(const LocalVector<double>& src);

        ROCALUTION_EXPORT
        virtual void
            CopyFrom(const LocalVector<ValueType>& src, int src_offset, int dst_offset, int size);

        /** \brief Copy a vector under permutation (forward permutation) */
        ROCALUTION_EXPORT
        void CopyFromPermute(const LocalVector<ValueType>& src,
                             const LocalVector<int>&       permutation);

        /** \brief Copy a vector under permutation (backward permutation) */
        ROCALUTION_EXPORT
        void CopyFromPermuteBackward(const LocalVector<ValueType>& src,
                                     const LocalVector<int>&       permutation);

        ROCALUTION_EXPORT
        virtual void CloneFrom(const LocalVector<ValueType>& src);

        /** \brief Copy (import) vector
      * \details
      * Copy (import) vector data that is described in one array (values). The object
      * data has to be allocated with Allocate(), using the corresponding size of the
      * data, first.
      *
      * @param[in]
      * data    data to be imported.
      */
        ROCALUTION_EXPORT
        void CopyFromData(const ValueType* data);

        /** \brief Copy (export) vector
      * \details
      * Copy (export) vector data that is described in one array (values). The output
      * array has to be allocated, using the corresponding size of the data, first.
      * Size can be obtain by GetSize().
      *
      * @param[out]
      * data    exported data.
      */
        ROCALUTION_EXPORT
        void CopyToData(ValueType* data) const;

        /** \brief Perform in-place permutation (forward) of the vector */
        ROCALUTION_EXPORT
        void Permute(const LocalVector<int>& permutation);

        /** \brief Perform in-place permutation (backward) of the vector */
        ROCALUTION_EXPORT
        void PermuteBackward(const LocalVector<int>& permutation);

        /** \brief Restriction operator based on restriction mapping vector */
        ROCALUTION_EXPORT
        void Restriction(const LocalVector<ValueType>& vec_fine, const LocalVector<int>& map);

        /** \brief Prolongation operator based on restriction mapping vector */
        ROCALUTION_EXPORT
        void Prolongation(const LocalVector<ValueType>& vec_coarse, const LocalVector<int>& map);

        ROCALUTION_EXPORT
        virtual void AddScale(const LocalVector<ValueType>& x, ValueType alpha);
        ROCALUTION_EXPORT
        virtual void ScaleAdd(ValueType alpha, const LocalVector<ValueType>& x);
        ROCALUTION_EXPORT
        virtual void
            ScaleAddScale(ValueType alpha, const LocalVector<ValueType>& x, ValueType beta);
        ROCALUTION_EXPORT
        virtual void ScaleAddScale(ValueType                     alpha,
                                   const LocalVector<ValueType>& x,
                                   ValueType                     beta,
                                   int                           src_offset,
                                   int                           dst_offset,
                                   int                           size);
        ROCALUTION_EXPORT
        virtual void ScaleAdd2(ValueType                     alpha,
                               const LocalVector<ValueType>& x,
                               ValueType                     beta,
                               const LocalVector<ValueType>& y,
                               ValueType                     gamma);
        ROCALUTION_EXPORT
        virtual void Scale(ValueType alpha);
        ROCALUTION_EXPORT
        virtual ValueType Dot(const LocalVector<ValueType>& x) const;
        ROCALUTION_EXPORT
        virtual ValueType DotNonConj(const LocalVector<ValueType>& x) const;
        ROCALUTION_EXPORT
        virtual ValueType Norm(void) const;
        ROCALUTION_EXPORT
        virtual ValueType Reduce(void) const;
        ROCALUTION_EXPORT
        virtual ValueType Asum(void) const;
        ROCALUTION_EXPORT
        virtual int Amax(ValueType& value) const;
        ROCALUTION_EXPORT
        virtual void PointWiseMult(const LocalVector<ValueType>& x);
        ROCALUTION_EXPORT
        virtual void PointWiseMult(const LocalVector<ValueType>& x,
                                   const LocalVector<ValueType>& y);
        ROCALUTION_EXPORT
        virtual void Power(double power);

        /** \brief Set index array */
        ROCALUTION_EXPORT
        void SetIndexArray(int size, const int* index);
        /** \brief Get indexed values */
        ROCALUTION_EXPORT
        void GetIndexValues(ValueType* values) const;
        /** \brief Set indexed values */
        ROCALUTION_EXPORT
        void SetIndexValues(const ValueType* values);
        /** \brief Get continuous indexed values */
        ROCALUTION_EXPORT
        void GetContinuousValues(int start, int end, ValueType* values) const;
        /** \brief Set continuous indexed values */
        ROCALUTION_EXPORT
        void SetContinuousValues(int start, int end, const ValueType* values);
        /** \brief Extract coarse boundary mapping */
        ROCALUTION_EXPORT
        void ExtractCoarseMapping(
            int start, int end, const int* index, int nc, int* size, int* map) const;
        /** \brief Extract coarse boundary index */
        ROCALUTION_EXPORT
        void ExtractCoarseBoundary(
            int start, int end, const int* index, int nc, int* size, int* boundary) const;

    protected:
        virtual bool is_host_(void) const;
        virtual bool is_accel_(void) const;

    private:
        // Pointer from the base vector class to the current allocated vector (host_ or accel_)
        BaseVector<ValueType>* vector_;

        // Host Vector
        HostVector<ValueType>* vector_host_;

        // Accelerator Vector
        AcceleratorVector<ValueType>* vector_accel_;

        friend class LocalVector<bool>;
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
