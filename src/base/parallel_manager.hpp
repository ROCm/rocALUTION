/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_PARALLEL_MANAGER_HPP_
#define ROCALUTION_PARALLEL_MANAGER_HPP_

#include "../utils/types.hpp"
#include "base_rocalution.hpp"
#include "rocalution/export.hpp"

#include <complex>
#include <string>

namespace rocalution
{

    template <typename ValueType>
    class GlobalMatrix;
    template <typename ValueType>
    class GlobalVector;

    /** \ingroup backend_module
  * \brief Parallel Manager class
  * \details
  * The parallel manager class handles the communication and the mapping of the global
  * operators. Each global operator and vector need to be initialized with a valid
  * parallel manager in order to perform any operation. For many distributed simulations,
  * the underlying operator is already distributed. This information need to be passed to
  * the parallel manager.
  */
    class ParallelManager : public RocalutionObj
    {
    public:
        ROCALUTION_EXPORT
        ParallelManager();
        ROCALUTION_EXPORT
        ~ParallelManager();

        /** \brief Set the MPI communicator */
        ROCALUTION_EXPORT
        void SetMPICommunicator(const void* comm);
        /** \brief Clear all allocated resources */
        ROCALUTION_EXPORT
        void Clear(void);

        /** \brief Return rank */
        ROCALUTION_EXPORT
        int GetRank(void) const
        {
            return this->rank_;
        }

        /** \brief Return the global size */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#else
        [[deprecated("This function will be removed in a future release. Use "
                     "GetGlobalNrow() or GetGlobalNcol() instead")]]
#endif
        ROCALUTION_EXPORT
        IndexType2 GetGlobalSize(void) const;
        /** \brief Return the global number of rows */
        ROCALUTION_EXPORT
        IndexType2 GetGlobalNrow(void) const;
        /** \brief Return the global number of columns */
        ROCALUTION_EXPORT
        IndexType2 GetGlobalNcol(void) const;
        /** \brief Return the local size */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#else
        [[deprecated("This function will be removed in a future release. Use "
                     "GetLocalNrow() or GetLocalNcol() instead")]]
#endif
        ROCALUTION_EXPORT
        int GetLocalSize(void) const;
        /** \brief Return the local number of rows */
        ROCALUTION_EXPORT
        int GetLocalNrow(void) const;
        /** \brief Return the local number of columns */
        ROCALUTION_EXPORT
        int GetLocalNcol(void) const;

        /** \brief Return the number of receivers */
        ROCALUTION_EXPORT
        int GetNumReceivers(void) const;
        /** \brief Return the number of senders */
        ROCALUTION_EXPORT
        int GetNumSenders(void) const;
        /** \brief Return the number of involved processes */
        ROCALUTION_EXPORT
        int GetNumProcs(void) const;

        /** \brief Initialize the global size */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#else
        [[deprecated("This function will be removed in a future release. Use "
                     "SetGlobalNrow() or SetGlobalNcol() instead")]]
#endif
        ROCALUTION_EXPORT
        void SetGlobalSize(IndexType2 size);
        /** \brief Initialize the global number of rows */
        ROCALUTION_EXPORT
        void SetGlobalNrow(IndexType2 nrow);
        /** \brief Initialize the global number of columns */
        ROCALUTION_EXPORT
        void SetGlobalNcol(IndexType2 ncol);
        /** \brief Initialize the local size */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#else
        [[deprecated("This function will be removed in a future release. Use "
                     "SetLocalNrow() or SetLocalNcol() instead")]]
#endif
        ROCALUTION_EXPORT
        void SetLocalSize(int size);
        /** \brief Initialize the local number of rows */
        ROCALUTION_EXPORT
        void SetLocalNrow(int nrow);
        /** \brief Initialize the local number of columns */
        ROCALUTION_EXPORT
        void SetLocalNcol(int ncol);

        /** \brief Set all boundary indices of this ranks process */
        ROCALUTION_EXPORT
        void SetBoundaryIndex(int size, const int* index);

        /** \brief Number of processes, the current process is receiving data from, array of
      * the processes, the current process is receiving data from and offsets, where the
      * boundary for process 'receiver' starts
      */
        ROCALUTION_EXPORT
        void SetReceivers(int nrecv, const int* recvs, const int* recv_offset);

        /** \brief Number of processes, the current process is sending data to, array of the
      * processes, the current process is sending data to and offsets where the ghost
      * part for process 'sender' starts
      */
        ROCALUTION_EXPORT
        void SetSenders(int nsend, const int* sends, const int* send_offset);

        /** \brief Mapping local to global */
        ROCALUTION_EXPORT
        void LocalToGlobal(int proc, int local, int& global);
        /** \brief Mapping global to local */
        ROCALUTION_EXPORT
        void GlobalToLocal(int global, int& proc, int& local);

        /** \brief Check sanity status of parallel manager */
        ROCALUTION_EXPORT
        bool Status(void) const;

        /** \brief Read file that contains all relevant parallel manager data */
        ROCALUTION_EXPORT
        void ReadFileASCII(const std::string& filename);
        /** \brief Write file that contains all relevant parallel manager data */
        ROCALUTION_EXPORT
        void WriteFileASCII(const std::string& filename) const;

    private:
        const void* comm_;
        int         rank_;
        int         num_procs_;

        IndexType2 global_nrow_;
        IndexType2 global_ncol_;
        int        local_nrow_;
        int        local_ncol_;

        // Number of total ids, the current process is receiving
        int recv_index_size_;
        // Number of total ids, the current process is sending
        int send_index_size_;

        // Number of processes, the current process receives data from
        int nrecv_;
        // Number of processes, the current process sends data to
        int nsend_;

        // Array of process ids, the current process receives data from
        int* recvs_;
        // Array of process ids, the current process sends data to
        int* sends_;

        // Array of offsets, the current process receives data from
        int* recv_offset_index_;
        // Array of offsets, the current process sends data to
        int* send_offset_index_;

        // Boundary index ids
        int* boundary_index_;

        friend class GlobalMatrix<double>;
        friend class GlobalMatrix<float>;
        friend class GlobalMatrix<std::complex<double>>;
        friend class GlobalMatrix<std::complex<float>>;
        friend class GlobalVector<double>;
        friend class GlobalVector<float>;
        friend class GlobalVector<std::complex<double>>;
        friend class GlobalVector<std::complex<float>>;
        friend class GlobalVector<int>;
    };

} // namespace rocalution

#endif // ROCALUTION_PARALLEL_MANAGER_HPP_
