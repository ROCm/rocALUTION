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

#ifndef ROCALUTION_BACKEND_MANAGER_HPP_
#define ROCALUTION_BACKEND_MANAGER_HPP_

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>

namespace rocalution {

template <typename ValueType>
class AcceleratorVector;
template <typename ValueType>
class AcceleratorMatrix;
template <typename ValueType>
class HostMatrix;

// Backend descriptor - keeps information about the
// hardware - OpenMP (threads); HIP (blocksizes, handles, etc);
struct Rocalution_Backend_Descriptor
{
    // set by initbackend();
    bool init;

    // current backend
    int backend;
    bool accelerator;
    bool disable_accelerator;

    // OpenMP threads
    int OpenMP_threads;
    // OpenMP threads before ROCALUTION init
    int OpenMP_def_threads;
    // OpenMP nested before ROCALUTION init
    int OpenMP_def_nested;
    // Host affinity (true-yes/false-no)
    bool OpenMP_affinity;
    // Host threshold size
    int OpenMP_threshold;

    // HIP section
    // handles
    // rocblas_handle casted in void **
    void* ROC_blas_handle;
    // rocsparse_handle casted in void **
    void* ROC_sparse_handle;

    int HIP_dev;
    int HIP_warp;
    int HIP_block_size;
    int HIP_max_threads;
    int HIP_num_procs;
    int HIP_threads_per_proc;

    // MPI rank/id
    int rank;

    // Logging
    int log_mode;
    std::ofstream* log_file;
};

// Global backend descriptor
extern struct Rocalution_Backend_Descriptor _Backend_Descriptor;

// Host name
extern const std::string _rocalution_host_name[1];

// Backend names
extern const std::string _rocalution_backend_name[2];

// Backend IDs
enum _rocalution_backend_id
{
    None = 0,
    HIP  = 1
};

/** \ingroup backend_module
  * \brief Initialize rocALUTION platform
  * \details
  * \p init_rocalution initializes the rocALUTION platform.
  *
  * @param[in]
  * rank            specifies MPI rank when multi-node environment
  * @param[in]
  * dev_per_node    number of accelerator devices per node, when in multi-GPU environment
  */
int init_rocalution(int rank = -1, int dev_per_node = 1);

/** \ingroup backend_module
  * \brief Shutdown rocALUTION platform
  * \details
  * \p stop_rocalution shuts down the rocALUTION platform.
  */
int stop_rocalution(void);

/** \ingroup backend_module
  * \brief Set the accelerator device
  * \details
  * \p set_device_rocalution sets the accelerator device used for computation.
  *
  * @param[in]
  * dev     accelerator device ID for computation
  */
void set_device_rocalution(int dev);

/** \ingroup backend_module
  * \brief Set number of OpenMP threads
  * \details
  * \p set_omp_threads_rocalution sets the number of OpenMP threads for computation.
  *
  * @param[in]
  * nthreads    number of OpenMP threads
  */
void set_omp_threads_rocalution(int nthreads);

/** \ingroup backend_module
  * \brief Enable/disable OpenMP host affinity
  * \details
  * \p set_omp_affinity_rocalution enables / disables OpenMP host affinity.
  *
  * @param[in]
  * affinity    boolean to turn on/off OpenMP host affinity
  */
void set_omp_affinity_rocalution(bool affinity);

/** \ingroup backend_module
  * \brief Set OpenMP threshold size
  * \details
  * \p set_omp_threshold_rocalution sets the OpenMP threshold size.
  *
  * @param[in]
  * threshold   OpenMP threshold size
  */
void set_omp_threshold_rocalution(int threshold);

/** \ingroup backend_module
  * \brief Print info about rocALUTION
  * \details
  * \p info_rocalution prints information about the rocALUTION platform
  */
void info_rocalution(void);

/** \ingroup backend_module
  * \brief Print info about specific rocALUTION backend descriptor
  * \details
  * \p info_rocalution prints information about the rocALUTION platform of the specific
  * backend descriptor.
  *
  * @param[in]
  * backend_descriptor  rocALUTION backend descriptor
  */
void info_rocalution(const struct Rocalution_Backend_Descriptor backend_descriptor);

/** \ingroup backend_module
  * \brief Disable/Enable the accelerator
  * \details
  * \p disable_accelerator_rocalution disables / enables the accelerator.
  *
  * @param[in]
  * onoff   boolean to turn on/off the accelerator
  */
void disable_accelerator_rocalution(bool onoff = true);

// Return true if any accelerator is available
bool _rocalution_available_accelerator(void);

// Return backend descriptor
struct Rocalution_Backend_Descriptor* _get_backend_descriptor(void);

// Set backend descriptor
void _set_backend_descriptor(const struct Rocalution_Backend_Descriptor backend_descriptor);

// Set the OMP threads based on the size threshold
void _set_omp_backend_threads(const struct Rocalution_Backend_Descriptor backend_descriptor,
                              int size);

// Build (and return) a vector on the selected in the descriptor accelerator
template <typename ValueType>
AcceleratorVector<ValueType>*
_rocalution_init_base_backend_vector(const struct Rocalution_Backend_Descriptor backend_descriptor);

// Build (and return) a matrix on the host
template <typename ValueType>
HostMatrix<ValueType>*
_rocalution_init_base_host_matrix(const struct Rocalution_Backend_Descriptor backend_descriptor,
                                  unsigned int matrix_format);

// Build (and return) a matrix on the selected in the descriptor accelerator
template <typename ValueType>
AcceleratorMatrix<ValueType>*
_rocalution_init_base_backend_matrix(const struct Rocalution_Backend_Descriptor backend_descriptor,
                                     unsigned int matrix_format);

/** \ingroup backend_module
  * \brief Sync rocALUTION
  * \details
  * \p _rocalution_sync blocks the host until all active asynchronous transfers are completed.
  */
void _rocalution_sync(void);

size_t _rocalution_add_obj(class RocalutionObj* ptr);
bool _rocalution_del_obj(class RocalutionObj* ptr, size_t id);
void _rocalution_delete_all_obj(void);
bool _rocalution_check_if_any_obj(void);

} // namespace rocalution

#endif // ROCALUTION_BACKEND_MANAGER_HPP_
