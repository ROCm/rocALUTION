.. meta::
   :description: A sparse linear algebra library with focus on exploring fine-grained parallelism on top of the AMD ROCm runtime and toolchains
   :keywords: rocALUTION, ROCm, library, API, tool

.. _windows-installation:

=====================================
Installation on Windows
=====================================

This document provides information required to install and configure rocALUTION on Windows.

-------------
Prerequisites
-------------

- An AMD HIP SDK-enabled platform. For more information, refer to the `ROCm documentation <https://rocm.docs.amd.com/>`_.
- rocALUTION is supported on the same Windows versions and toolchains that are supported by the HIP SDK.

.. note::
   
   As the AMD HIP SDK is under continuous development, the information updated regarding the SDK's internal contents may overrule the statements in this document on installing and building on Windows.

----------------------------
Installing prebuilt packages
----------------------------

rocALUTION can be installed on Windows 11 or Windows 10 using the AMD HIP SDK installer.

The simplest way to use rocALUTION in your code is to use ``CMake`` that requires you to add the SDK installation location to your
`DCMAKE_PREFIX_PATH`. Note that you need to use quotes as the path contains a space, e.g.,

::

    -DCMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\5.5"


After CMake configuration, in your ``CMakeLists.txt`` use:

::

    find_package(rocalution)

    target_link_libraries( your_exe PRIVATE roc::rocalution )

Once rocALUTION is installed, you can find ``rocalution.hpp`` in the HIP SDK ``\\include\\rocalution``
directory. Use only the installed file in the user application if needed.
You must include ``rocalution.hpp`` header file in the user code to make calls
into rocALUTION, so that the rocALUTION import library and dynamic link library become the respective link-time and run-time
dependencies for the user application.

----------------------------------
Building and installing rocALUTION
----------------------------------

Building from source is not necessary, as rocALUTION can be used after installing the pre-built packages as described above.
If desired, you can follow the instructions below to build rocALUTION from source.

Requirements
^^^^^^^^^^^^
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

The rocALUTION source code, which is the same as for the ROCm linux distributions, is available at the `rocALUTION github page <https://github.com/ROCmSoftwarePlatform/rocSPARSE>`_.
The version of the ROCm HIP SDK may be shown in the path of default installation, but
you can run the HIP SDK compiler to report the version from the ``bin/`` folder using:

::

    hipcc --version

The HIP version has major, minor, and patch fields, possibly followed by a build-specific identifier. For example, a HIP version 5.4.22880-135e1ab4 corresponds to major = 5, minor = 4, patch = 22880, and build identifier 135e1ab4.
There are GitHub branches at the rocALUTION site with names ``release/rocm-rel-major.minor`` where major and minor are the same as in the HIP version.
To download rocALUTION, use:

::

   git clone -b release/rocm-rel-x.y https://github.com/ROCmSoftwarePlatform/rocALUTION.git
   cd rocALUTION

Replace ``x.y`` in the above command with the version of HIP SDK installed on your machine. For example, if you have HIP 5.5 installed, then use ``-b release/rocm-rel-5.5``.
You can add the SDK tools to your path using: 

::

   %HIP_PATH%\bin

Build
^^^^^^^^

Below are the steps required to build using the `rmake.py` script. The user can build either of the following:

* library

* library and client

You only need (library) if you call rocALUTION from your code and want to build the library alone.
The client contains testing and benchmarking tools. ``rmake.py`` prints the full ``cmake`` command being used to configure rocALUTION based on your ``rmake`` command-line options.
This full ``cmake`` command can be used in your own build scripts if you want to bypass the Python helper script for a fixed set of build options.

Build library
^^^^^^^^^^^^^^

Common uses of ``rmake.py`` to build (library) are listed below:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+--------------------+-----------------------------+
| Command            | Description                 |
+====================+=============================+
| ``./rmake.py -h``  | Help information.           |
+--------------------+-----------------------------+
| ``./rmake.py``     | Builds library.             |
+--------------------+-----------------------------+
| ``./rmake.py -i``  | Builds library, then        |
|                    | builds and installs         |
|                    | rocALUTION package.         |
|                    | If you want to keep         |
|                    | rocALUTION in your local    |
|                    | tree, don't use ``-i`` flag.|
+--------------------+-----------------------------+

Build library and client
^^^^^^^^^^^^^^^^^^^^^^^^^^

Some client executables (.exe) are listed below:

====================== ==================================================
Executable name        Description
====================== ==================================================
``rocalution-test``     Runs Google Tests to test the library
``rocalution-bench``    Executable to benchmark or test functions
``./cg lap_25.mtx``     Executes conjugate gradient example 
                        (must download ``mtx`` matrix file you wish to use)
====================== ==================================================

Common uses of ``rmake.py`` to build (library and client) are listed below:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+------------------------+----------------------------------+
| Command                | Description                      |
+========================+==================================+
| ``./rmake.py -h``      | Help information.                |
+------------------------+----------------------------------+
| ``./rmake.py -c``      | Builds library and client        |
|                        | in your local directory.         |
+------------------------+----------------------------------+
| ``./rmake.py -ic``     | Builds and installs              |
|                        | rocALUTION package, and          |
|                        | builds the client.               |
|                        | If you want to keep              |
|                        | rocALUTION in your local         |
|                        | directory, don't use ``-i`` flag.|
+------------------------+----------------------------------+
