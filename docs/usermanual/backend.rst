.. meta::
   :description: A sparse linear algebra library with focus on exploring fine-grained parallelism on top of the AMD ROCm runtime and toolchains
   :keywords: rocALUTION, ROCm, library, API, tool

.. _backends:

********
Backends
********
The support of accelerator devices is embedded in the structure of rocALUTION. The primary goal is to use this technology whenever possible to decrease the computational time.
.. note:: Not all functions are ported and present on the accelerator backend. This limited functionality is natural, since not all operations can be performed efficiently on the accelerators (e.g. sequential algorithms, I/O from the file system, etc.).

Currently, rocALUTION supports HIP capable GPUs starting with ROCm 1.9. Due to its design, the library can be easily extended to support future accelerator technologies. Such an extension of the library will not reflect the algorithms which are based on it.

If a particular function is not implemented for the used accelerator, the library will move the object to the host and compute the routine there. In this case a warning message of level 2 will be printed. For example, if the user wants to perform an ILUT factorization on the HIP backend which is currently not available, the library will move the object to the host, perform the routine there and print the following warning message

::

  *** warning: LocalMatrix::ILUTFactorize() is performed on the host

Moving Objects To and From the Accelerator
==========================================
All objects in rocALUTION can be moved to the accelerator and to the host.

.. doxygenfunction:: rocalution::BaseRocalution::MoveToAccelerator
.. doxygenfunction:: rocalution::BaseRocalution::MoveToHost

.. code-block:: cpp

  LocalMatrix<ValueType> mat;
  LocalVector<ValueType> vec1, vec2;

  // Perform matrix vector multiplication on the host
  mat.Apply(vec1, &vec2);

  // Move data to the accelerator
  mat.MoveToAccelerator();
  vec1.MoveToAccelerator();
  vec2.MoveToAccelerator();

  // Perform matrix vector multiplication on the accelerator
  mat.Apply(vec1, &vec2);

  // Move data to the host
  mat.MoveToHost();
  vec1.MoveToHost();
  vec2.MoveToHost();

Asynchronous Transfers
======================
The rocALUTION library also provides asynchronous transfers of data between host and HIP backend.

.. doxygenfunction:: rocalution::BaseRocalution::MoveToAcceleratorAsync
.. doxygenfunction:: rocalution::BaseRocalution::MoveToHostAsync
.. doxygenfunction:: rocalution::BaseRocalution::Sync

This can be done with :cpp:func:`rocalution::LocalVector::CopyFromAsync` and :cpp:func:`rocalution::LocalMatrix::CopyFromAsync` or with `MoveToAcceleratorAsync()` and `MoveToHostAsync()`. These functions return immediately and perform the asynchronous transfer in background mode. The synchronization is done with `Sync()`.

When using the `MoveToAcceleratorAsync()` and `MoveToHostAsync()` functions, the object will still point to its original location (i.e. host for calling `MoveToAcceleratorAsync()` and accelerator for `MoveToHostAsync()`). The object will switch to the new location after the `Sync()` function is called.

.. note:: The objects should not be modified during an active asynchronous transfer. However, if this happens, the values after the synchronization might be wrong.
.. note:: To use the asynchronous transfers, you need to enable the pinned memory allocation. Uncomment `#define ROCALUTION_HIP_PINNED_MEMORY` in `src/utils/allocate_free.hpp`.

Systems without Accelerators
============================
rocALUTION provides full code compatibility on systems without accelerators, the user can take the code from the GPU system, re-compile the same code on a machine without a GPU and it will provide the same results. Any calls to :cpp:func:`rocalution::BaseRocalution::MoveToAccelerator` and :cpp:func:`rocalution::BaseRocalution::MoveToHost` will be ignored.

Memory Allocations
==================
All data which is passed to and from rocALUTION is using the memory handling functions described in the code. By default, the library uses standard C++ *new* and *delete* functions for the host data. This can be changed by modifying `src/utils/allocate_free.cpp`.

Allocation Problems
-------------------
If the allocation fails, the library will report an error and exits. If the user requires a special treatment, it has to be placed in `src/utils/allocate_free.cpp`.

Memory Alignment
----------------
The library can also handle special memory alignment functions. This feature need to be uncommented before the compilation process in `src/utils/allocate_free.cpp`.

Pinned Memory Allocation (HIP)
------------------------------
By default, the standard host memory allocation is realized by C++ *new* and *delete*. For faster PCI-Express transfers on HIP backend, the user can also use pinned host memory. This can be activated by uncommenting the corresponding macro in `src/utils/allocate_free.hpp`.

