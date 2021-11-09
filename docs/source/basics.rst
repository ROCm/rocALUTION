******
Basics
******

Operators and Vectors
=====================
The main objects in rocALUTION are linear operators and vectors.
All objects can be moved to an accelerator at run time.
The linear operators are defined as local or global matrices (i.e. on a single node or distributed/multi-node) and local stencils (i.e. matrix-free linear operations).
The only template parameter of the operators and vectors is the data type (ValueType).
The operator data type could be float, double, complex float or complex double, while the vector data type can be int, float, double, complex float or complex double (int is used mainly for the permutation vectors).
In the current version, cross ValueType object operations are not supported. :numref:`operators` gives an overview of supported operators and vectors.
Further details are also given in the :ref:`design_document`.

.. _operators:
.. figure:: ../fig/operators.png
  :alt: operator and vector classes
  :align: center

  Operator and vector classes.

Each of the objects contain a local copy of the hardware descriptor created by the :cpp:func:`rocalution::init_rocalution` function. This allows the user to modify it according to his needs and to obtain two or more objects with different hardware specifications (e.g. different amount of OpenMP threads, HIP block sizes, etc.).

Local Operators and Vectors
---------------------------
By Local Operators and Vectors we refer to Local Matrices and Stencils and to Local Vectors. By Local we mean the fact that they stay on a single system. The system can contain several CPUs via UMA or NUMA memory system, it can also contain an accelerator.

.. doxygenclass:: rocalution::LocalMatrix
.. doxygenclass:: rocalution::LocalStencil
.. doxygenclass:: rocalution::LocalVector

Global Operators and Vectors
----------------------------
By Global Operators and Vectors we refer to Global Matrix and to Global Vectors. By Global we mean the fact they can stay on a single or multiple nodes in a network. For this type of computation, the communication is based on MPI.

.. doxygenclass:: rocalution::GlobalMatrix
.. doxygenclass:: rocalution::GlobalVector

Backend Descriptor and User Control
===================================
Naturally, not all routines and algorithms can be performed efficiently on many-core systems (i.e. on accelerators).
To provide full functionality, the library has internal mechanisms to check if a particular routine is implemented on the accelerator.
If not, the object is moved to the host and the routine is computed there.
This guarantees that your code will run (maybe not in the most efficient way) with any accelerator regardless of the available functionality for it.

Initialization of rocALUTION
----------------------------
The body of a rocALUTION code is very simple, it should contain the header file and the namespace of the library.
The program must contain an initialization call to :cpp:func:`init_rocalution <rocalution::init_rocalution>` which will check and allocate the hardware and a finalizing call to :cpp:func:`stop_rocalution <rocalution::stop_rocalution>` which will release the allocated hardware.

.. doxygenfunction:: rocalution::init_rocalution
.. doxygenfunction:: rocalution::stop_rocalution

Thread-core Mapping
-------------------
The number of threads which rocALUTION will use can be modified by the function :cpp:func:`set_omp_threads_rocalution <rocalution::set_omp_threads_rocalution>` or by the global OpenMP environment variable (for Unix-like OS this is `OMP_NUM_THREADS`).
During the initialization phase, the library provides affinity thread-core mapping:

- If the number of cores (including SMT cores) is greater or equal than two times the number of threads, then all the threads can occupy every second core ID (e.g. 0,2,4,...).
  This is to avoid having two threads working on the same physical core, when SMT is enabled.
- If the number of threads is less or equal to the number of cores (including SMT), and the previous clause is false, then the threads can occupy every core ID (e.g. 0,1,2,3,...).
- If non of the above criteria is matched, then the default thread-core mapping is used (typically set by the operating system).

.. note:: The thread-core mapping is available for Unix-like operating systems only.
.. note:: The user can disable the thread affinity by :cpp:func:`set_omp_affinity_rocalution <rocalution::set_omp_affinity_rocalution>`, before initializing the library.

OpenMP Threshold Size
---------------------
Whenever working on a small problem, OpenMP host backend might be slightly slower than using no OpenMP.
This is mainly attributed to the small amount of work, which every thread should perform and the large overhead of forking/joining threads.
This can be avoid by the OpenMP threshold size parameter in rocALUTION.
The default threshold is set to 10.000, which means that all matrices under (and equal to) this size will use only one thread (disregarding the number of OpenMP threads set in the system).
The threshold can be modified with :cpp:func:`set_omp_threshold_rocalution <rocalution::set_omp_threshold_rocalution>`.

Accelerator Selection
---------------------
The accelerator device id that is supposed to be used for the computation can be selected by the user by :cpp:func:`set_device_rocalution <rocalution::set_device_rocalution>`.

Disable the Accelerator
-----------------------
Furthermore, the accelerator can be disabled without having to re-compile the library by calling :cpp:func:`disable_accelerator_rocalution <rocalution::disable_accelerator_rocalution>`.

Backend Information
-------------------
Detailed information about the current backend / accelerator in use as well as the available accelerators can be printed by :cpp:func:`info_rocalution <rocalution::info_rocalution>`.

MPI and Multi-Accelerators
--------------------------
When initializing the library with MPI, the user need to pass the rank of the MPI process as well as the number of accelerators available on each node.
Basically, this way the user can specify the mapping of MPI process and accelerators - the allocated accelerator will be `rank % num_dev_per_node`.
Thus, the user can run two MPI processes on systems with two accelerators by specifying the number of devices to 2, as illustrated in the example code below.

.. code-block:: cpp

  #include <rocalution/rocalution.hpp>
  #include <mpi.h>

  using namespace rocalution;

  int main(int argc, char* argv[])
  {
      MPI_Init(&argc, &argv);
      MPI_Comm comm = MPI_COMM_WORLD;

      int num_processes;
      int rank;

      MPI_Comm_size(comm, &num_processes);
      MPI_Comm_rank(comm, &rank);

      int nacc_per_node = 2;

      init_rocalution(rank, nacc_per_node);

      // ... do some work

      stop_rocalution();

      return 0;
  }

.. _rocalution_obj_tracking:

Automatic Object Tracking
=========================
rocALUTION supports automatic object tracking.
After the initialization of the library, all objects created by the user application can be tracked.
Once :cpp:func:`stop_rocalution <rocalution::stop_rocalution>` is called, all memory from tracked objects gets deallocated.
This will avoid memory leaks when the objects are allocated but not freed.
The user can enable or disable the tracking by editing `src/utils/def.hpp`.
By default, automatic object tracking is disabled.

.. _rocalution_verbose:

Verbose Output
==============
rocALUTION provides different levels of output messages.
The `VERBOSE_LEVEL` can be modified in `src/utils/def.hpp` before the compilation of the library.
By setting a higher level, the user will obtain more detailed information about the internal calls and data transfers to and from the accelerators.
By default, `VERBOSE_LEVEL` is set to 2.

.. _rocalution_logging:

Verbose Output and MPI
======================
To prevent all MPI processes from printing information to `stdout`, the default configuration is that only `RANK 0` outputs information.
The user can change the `RANK` or allow all processes to print setting `LOG_MPI_RANK` to 1 in `src/utils/def.hpp`.
If file logging is enabled, all ranks write into the corresponding log files.

.. _rocalution_debug:

Debug Output
============
Debug output will print almost every detail in the program, including object constructor / destructor, address of the object, memory allocation, data transfers, all function calls for matrices, vectors, solvers and preconditioners.
The flag `DEBUG_MODE` can be set in `src/utils/def.hpp`.
When enabled, additional `assert()s` are being checked during the computation.
This might decrease performance of some operations significantly.

File Logging
============
rocALUTION trace file logging can be enabled by setting the environment variable `ROCALUTION_LAYER` to 1.
rocALUTION will then log each rocALUTION function call including object constructor / destructor, address of the object, memory allocation, data transfers, all function calls for matrices, vectors, solvers and preconditioners.
The log file will be placed in the working directory.
The log file naming convention is `rocalution-rank-<rank>-<time_since_epoch_in_msec>.log`.
By default, the environment variable `ROCALUTION_LAYER` is unset, and logging is disabled.

.. note:: Performance might degrade when logging is enabled.

Versions
========
For checking the rocALUTION version in an application, pre-defined macros can be used:

.. code-block:: cpp

  #define __ROCALUTION_VER_MAJOR  // version major
  #define __ROCALUTION_VER_MINOR  // version minor
  #define __ROCALUTION_VER_PATCH  // version patch
  #define __ROCALUTION_VER_TWEAK  // commit id (sha-1)

  #define __ROCALUTION_VER_PRE    // version pre-release (alpha or beta)

  #define __ROCALUTION_VER        // version

The final `__ROCALUTION_VER` holds the version number as `10000 * major + 100 * minor + patch`, as defined in `src/base/version.hpp.in`.
