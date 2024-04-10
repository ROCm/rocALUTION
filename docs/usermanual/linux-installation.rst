.. meta::
   :description: A sparse linear algebra library with focus on exploring fine-grained parallelism on top of the AMD ROCm runtime and toolchains
   :keywords: rocALUTION, ROCm, library, API, tool

.. _linux-installation:

===================================
Installation on Linux
===================================

This document provides information required to install and configure rocALUTION on Linux.

-------------
Prerequisites
-------------

A ROCm enabled platform. For information on supported GPUs, Linux distributions, ROCm installation, and Windows SKUs, refer to `ROCm Documentation <https://rocm.docs.amd.com/>`_.

-----------------------------
Installing pre-built packages
-----------------------------

You can install rocALUTION from `AMD ROCm repository <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html>`_.
The repository hosts the single-node, accelerator enabled version of the library.
If a different setup is required, e.g. multi-node support, build :ref:`rocALUTION from source <rocalution_build_from_source>`.

For detailed instructions on how to set up ROCm on different platforms, see the `AMD ROCm Platform Installation Guide for Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html>`_.

rocALUTION has the following run-time dependencies:

- `AMD ROCm <https://github.com/RadeonOpenCompute/ROCm>`_ 2.9 or later (optional, for HIP support)
- `rocSPARSE <https://github.com/ROCmSoftwarePlatform/rocSPARSE>`_ (optional, for HIP support)
- `rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_ (optional, for HIP support)
- `rocPRIM <https://github.com/ROCmSoftwarePlatform/rocPRIM>`_ (optional, for HIP support)
- `OpenMP <https://www.openmp.org/>`_ (optional, for OpenMP support)
- `MPI <https://www.mcs.anl.gov/research/projects/mpi/>`_ (optional, for multi-node / multi-GPU support)

.. _rocalution_build_from_source:

-------------------------------
Building from GitHub repository
-------------------------------

Requirements
^^^^^^^^^^^^

To build rocALUTION from source, ensure that the following compile-time and run-time dependencies are met:

- `git <https://git-scm.com/>`_
- `CMake <https://cmake.org/>`_ 3.5 or later
- `AMD ROCm <https://github.com/RadeonOpenCompute/ROCm>`_ 2.9 or later (optional, for HIP support)
- `rocSPARSE <https://github.com/ROCmSoftwarePlatform/rocSPARSE>`_ (optional, for HIP support)
- `rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_ (optional, for HIP support)
- `rocPRIM <https://github.com/ROCmSoftwarePlatform/rocPRIM>`_ (optional, for HIP support)
- `OpenMP <https://www.openmp.org/>`_ (optional, for OpenMP support)
- `MPI <https://www.mcs.anl.gov/research/projects/mpi/>`_ (optional, for multi-node / multi-GPU support)
- `googletest <https://github.com/google/googletest>`_ (optional, for clients)

Download rocALUTION
^^^^^^^^^^^^^^^^^^^
The rocALUTION source code is available at the `rocALUTION GitHub page <https://github.com/ROCmSoftwarePlatform/rocALUTION>`_.
Download the master branch using:

::

  $ git clone -b master https://github.com/ROCmSoftwarePlatform/rocALUTION.git
  $ cd rocALUTION

Below are the steps to build different packages of the library, including dependencies and clients.
It is recommended to install rocALUTION using the ``install.sh`` script.

Using `install.sh` script to build rocALUTION with dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following table lists the common uses of ``install.sh`` to build dependencies and the library. Accelerator support via HIP and OpenMP are enabled by default, whereas MPI is disabled.

.. tabularcolumns::
      |\X{1}{6}|\X{5}{6}|

============================ ====================================================================================================================================================================================================
Command                      Description
============================ ===================================================================================================================================================================================================
``./install.sh -h``          Prints help information.
``./install.sh -d``          Builds dependencies and library in your local directory. The ``-d`` flag only needs to be used once. For subsequent invocations of ``install.sh`` it is not necessary to rebuild the dependencies.
``./install.sh``             Builds library in your local directory assuming the dependencies to be available.
``./install.sh -i``          Builds library, then builds and installs rocALUTION package in ``/opt/rocm/rocalution``. It prompts for sudo access which installs for all users.
``./install.sh --host``      Builds library in your local directory without HIP support assuming the dependencies to be available.
``./install.sh --mpi=<dir>`` Builds library in your local directory with HIP and MPI support assuming the dependencies to be available.
============================ ===================================================================================================================================================================================================

Using ``install.sh`` script to build rocALUTION with dependencies and clients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The client contains example code, unit tests and benchmarks. Common uses of ``install.sh`` to build them are listed in the table below:

.. tabularcolumns::
      |\X{1}{6}|\X{5}{6}|

===================== =========================================================================================================================================================================================================
Command               Description
===================== ==========================================================================================================================================================================================================
``./install.sh -h``   Prints help information.
``./install.sh -dc``  Builds dependencies, library and client in your local directory. The ``-d`` flag only needs to be used once. For subsequent invocations of ``install.sh`` it is not necessary to rebuild the dependencies.
``./install.sh -c``   Builds library and client in your local directory assuming the dependencies to be available.
``./install.sh -idc`` Builds library, dependencies and client, then builds and installs rocALUTION package in ``/opt/rocm/rocalution``. It prompts for sudo access which installs for all users.
``./install.sh -ic``  Builds library and client, then builds and installs rocALUTION package in ``opt/rocm/rocalution``. It prompts for sudo access which installs for all users.
===================== ===========================================================================================================================================================================================================

Using individual commands to build rocALUTION
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CMake 3.5 or later is required to build rocALUTION without the use of ``install.sh``.

rocALUTION can be built with ``cmake`` using the following commands:

::

  # Create and change to build directory
  mkdir -p build/release ; cd build/release

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path>
  # to adjust it. In this case, rocALUTION is built with HIP and
  # OpenMP support.
  # MPI support is disabled.
  cmake ../.. -DSUPPORT_HIP=ON \
              -DSUPPORT_MPI=OFF \
              -DSUPPORT_OMP=ON

  # Compile rocALUTION library
  make -j$(nproc)

  # Install rocALUTION to /opt/rocm
  sudo make install

`GoogleTest <https://github.com/google/googletest>`_ is required to build all rocALUTION clients.

rocALUTION with dependencies and clients can be built using the following commands:

::

  # Install googletest
  mkdir -p build/release/deps ; cd build/release/deps
  cmake ../../../deps
  sudo make -j$(nproc) install

  # Change to build directory
  cd ..

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path>
  # to adjust it. By default, HIP and OpenMP support are enabled,
  # MPI support is disabled.
  cmake ../.. -DBUILD_CLIENTS_TESTS=ON \
              -DBUILD_CLIENTS_SAMPLES=ON

  # Compile rocALUTION library
  make -j$(nproc)

  # Install rocALUTION to /opt/rocm
  sudo make install

The compilation process produces a shared library file ``librocalution.so`` and ``librocalution_hip.so`` if HIP support is enabled.
Ensure that the library objects can be found in your library path.
If you don't copy the library to a specific location you can add the path under Linux in the ``LD_LIBRARY_PATH`` variable.

::

  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_rocalution>

Common build problems
^^^^^^^^^^^^^^^^^^^^^^^

#. **Issue:** Could not find any of the following package files provided by "ROCM":
            - ROCMConfig.cmake
            - rocm-config.cmake

   **Solution:** Install `ROCm cmake modules <https://github.com/ROCm/rocm-cmake>`_ either from source or from `AMD ROCm repository <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html>`_.

#. **Issue:** Could not find any of the following package files provided by "ROCSPARSE":
            - ROCSPARSE.cmake
            - rocsparse-config.cmake

   **Solution:** Install `rocSPARSE <https://github.com/ROCmSoftwarePlatform/rocSPARSE>`_ either from source or from `AMD ROCm repository <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html>`_.

#. **Issue:** Could not find any of the following package files provided by "ROCBLAS":
            - ROCBLAS.cmake
            - rocblas-config.cmake

   **Solution:** Install `rocBLAS <https://github.com/ROCm/rocBLAS>`_ either from the source or from `AMD ROCm repository <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html>`_.

Simple test
^^^^^^^^^^^

You can test the installation by running a CG solver on a sparse matrix.
After successfully compiling the library, the CG solver example can be executed.

::

  cd rocALUTION/build/release/clients/staging

  wget ftp://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/laplace/gr_30_30.mtx.gz
  gzip -d gr_30_30.mtx.gz

  ./cg gr_30_30.mtx
