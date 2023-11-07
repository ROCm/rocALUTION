===================================
Installation and Building for Linux
===================================

-------------
Prerequisites
-------------

- A ROCm enabled platform. `ROCm Documentation <https://docs.amd.com/>`_ has more information on
  supported GPUs, Linux distributions, and Windows SKUs. It also has information on how to install ROCm.

-----------------------------
Installing pre-built packages
-----------------------------

rocALUTION can be installed from `AMD ROCm repository <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`_.
The repository hosts the single-node, accelerator enabled version of the library.
If a different setup is required, e.g. multi-node support, rocALUTION need to be built from source, see :ref:`rocalution_build_from_source`.

For detailed instructions on how to set up ROCm on different platforms, see the `AMD ROCm Platform Installation Guide for Linux <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`_.

rocALUTION has the following run-time dependencies

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

To build rocALUTION from source, the following compile-time and run-time dependencies must be met

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

Below are steps to build different packages of the library, including dependencies and clients.
It is recommended to install rocALUTION using the `install.sh` script.

Using `install.sh` script to build rocALUTION with dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following table lists common uses of `install.sh` to build dependencies + library. Accelerator support via HIP and OpenMP will be enabled by default, whereas MPI is disabled.

.. tabularcolumns::
      |\X{1}{6}|\X{5}{6}|

========================== ====
Command                    Description
========================== ====
`./install.sh -h`          Print help information.
`./install.sh -d`          Build dependencies and library in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh` it is not necessary to rebuild the dependencies.
`./install.sh`             Build library in your local directory. It is assumed dependencies are available.
`./install.sh -i`          Build library, then build and install rocALUTION package in `/opt/rocm/rocalution`. You will be prompted for sudo access. This will install for all users.
`./install.sh --host`      Build library in your local directory without HIP support. It is assumed dependencies are available.
`./install.sh --mpi=<dir>` Build library in your local directory with HIP and MPI support. It is assumed dependencies are available.
========================== ====

Using `install.sh` script to build rocALUTION with dependencies and clients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The client contains example code, unit tests and benchmarks. Common uses of `install.sh` to build them are listed in the table below.

.. tabularcolumns::
      |\X{1}{6}|\X{5}{6}|

=================== ====
Command             Description
=================== ====
`./install.sh -h`   Print help information.
`./install.sh -dc`  Build dependencies, library and client in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh` it is not necessary to rebuild the dependencies.
`./install.sh -c`   Build library and client in your local directory. It is assumed dependencies are available.
`./install.sh -idc` Build library, dependencies and client, then build and install rocALUTION package in `/opt/rocm/rocalution`. You will be prompted for sudo access. This will install for all users.
`./install.sh -ic`  Build library and client, then build and install rocALUTION package in `opt/rocm/rocalution`. You will be prompted for sudo access. This will install for all users.
=================== ====

Using individual commands to build rocALUTION
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
CMake 3.5 or later is required in order to build rocALUTION without the use of `install.sh`.

rocALUTION can be built with cmake using the following commands:

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

`GoogleTest <https://github.com/google/googletest>`_ is required in order to build all rocALUTION clients.

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

The compilation process produces a shared library file `librocalution.so` and `librocalution_hip.so` if HIP support is enabled.
Ensure that the library objects can be found in your library path.
If you do not copy the library to a specific location you can add the path under Linux in the `LD_LIBRARY_PATH` variable.

::

  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_rocalution>

Common build problems
^^^^^^^^^^^^^^^^^^^^^
#. **Issue:** Could not find a package file provided by "ROCM" with any of the following names:
              ROCMConfig.cmake
              rocm-config.cmake

   **Solution:** Install `ROCm cmake modules <https://github.com/RadeonOpenCompute/rocm-cmake>`_ either from source or from `AMD ROCm repository <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`_.

#. **Issue:** Could not find a package file provided by "ROCSPARSE" with any of the following names:
              ROCSPARSE.cmake
              rocsparse-config.cmake

   **Solution:** Install `rocSPARSE <https://github.com/ROCmSoftwarePlatform/rocSPARSE>`_ either from source or from `AMD ROCm repository <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`_.

#. **Issue:** Could not find a package file provided by "ROCBLAS" with any of the following names:
              ROCBLAS.cmake
              rocblas-config.cmake

   **Solution:** Install `rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_ either from source or from `AMD ROCm repository <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`_.

Simple Test
^^^^^^^^^^^
You can test the installation by running a CG solver on a sparse matrix.
After successfully compiling the library, the CG solver example can be executed.

::

  cd rocALUTION/build/release/clients/staging

  wget ftp://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/laplace/gr_30_30.mtx.gz
  gzip -d gr_30_30.mtx.gz

  ./cg gr_30_30.mtx
