/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
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

/// Backend descriptor - keeps information about the
/// hardware - OpenMP (threads); HIP (blocksizes, handles, etc);
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

/// Global backend descriptor
extern struct Rocalution_Backend_Descriptor _Backend_Descriptor;

/// Host name
extern const std::string _rocalution_host_name[1];

/// Backend names
extern const std::string _rocalution_backend_name[2];

/// Backend IDs
enum _rocalution_backend_id
{
    None = 0,
    HIP  = 1
};

/// Initialization of the rocalution platform
int init_rocalution(int rank = -1, int dev_per_node = 1);

/// Shutdown the rocalution platform
int stop_rocalution(void);

/// Select a device
void set_device_rocalution(int dev);

/// Set the number of threads in the platform
void set_omp_threads_rocalution(int nthreads);

/// Set host affinity (true-on/false-off)
void set_omp_affinity_rocalution(bool affinity);

/// Set OpenMP threshold size
void set_omp_threshold_rocalution(int threshold);

/// Print information about the platform
void info_rocalution(void);

/// Print information about the platform via specific backend descriptor
void info_rocalution(const struct Rocalution_Backend_Descriptor backend_descriptor);

/// Return true if any accelerator is available
bool _rocalution_available_accelerator(void);

/// Disable/Enable the accelerator
void disable_accelerator_rocalution(bool onoff = true);

/// Return backend descriptor
struct Rocalution_Backend_Descriptor* _get_backend_descriptor(void);

/// Set backend descriptor
void _set_backend_descriptor(const struct Rocalution_Backend_Descriptor backend_descriptor);

/// Set the OMP threads based on the size threshold
void _set_omp_backend_threads(const struct Rocalution_Backend_Descriptor backend_descriptor,
                              int size);

/// Build (and return) a vector on the selected in the descriptor accelerator
template <typename ValueType>
AcceleratorVector<ValueType>*
_rocalution_init_base_backend_vector(const struct Rocalution_Backend_Descriptor backend_descriptor);

/// Build (and return) a matrix on the host
template <typename ValueType>
HostMatrix<ValueType>*
_rocalution_init_base_host_matrix(const struct Rocalution_Backend_Descriptor backend_descriptor,
                                  unsigned int matrix_format);

/// Build (and return) a matrix on the selected in the descriptor accelerator
template <typename ValueType>
AcceleratorMatrix<ValueType>*
_rocalution_init_base_backend_matrix(const struct Rocalution_Backend_Descriptor backend_descriptor,
                                     unsigned int matrix_format);

/// Sync the active async transfers
void _rocalution_sync(void);

size_t _rocalution_add_obj(class RocalutionObj* ptr);
bool _rocalution_del_obj(class RocalutionObj* ptr, size_t id);
void _rocalution_delete_all_obj(void);
bool _rocalution_check_if_any_obj(void);

} // namespace rocalution

#endif // ROCALUTION_BACKEND_MANAGER_HPP_
