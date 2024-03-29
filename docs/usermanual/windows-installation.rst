=====================================
Installation on Windows
=====================================

This document provides information required to install and configure rocALUTION on Linux.

-------------
Prerequisites
-------------

- An AMD HIP SDK-enabled platform. You can find more information in the `ROCm documentation <https://rocm.docs.amd.com/>`_.
- rocALUTION is supported on the same Windows versions and toolchains that are supported by the HIP SDK.
- As the AMD HIP SDK is new and quickly evolving it will have more up to date information regarding the SDK's internal contents. Thus it may overrule statements found in this section on installing and building for Windows.


----------------------------
Installing prebuilt packages
----------------------------

rocALUTION can be installed on Windows 11 or Windows 10 using the AMD HIP SDK installer.

The simplest way to use rocALUTION in your code would be using CMake for which you would add the SDK installation location to your
`CMAKE_PREFIX_PATH`. Note you need to use quotes as the path contains a space, e.g.,

::

    -DCMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\5.5"


in your CMake configure step and then in your CMakeLists.txt use

::

    find_package(rocalution)

    target_link_libraries( your_exe PRIVATE roc::rocalution )

The rocalution.hpp header file must be included in the user code to make calls
into rocALUTION, and the rocALUTION import library and dynamic link library will become respective link-time and run-time
dependencies for the user application.

Once installed, find rocalution.hpp in the HIP SDK ``\\include\\rocalution``
directory. Only use these two installed files when needed in user code.

----------------------------------
Building and installing rocALUTION
----------------------------------

Building from source is not necessary, as rocALUTION can be used after installing the pre-built packages as described above.
If desired, the following instructions can be used to build rocALUTION from source.

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
you can run the HIP SDK compiler to report the verison from the bin/ folder with:

::

    hipcc --version

The HIP version has major, minor, and patch fields, possibly followed by a build specific identifier. For example, HIP version could be 5.4.22880-135e1ab4;
this corresponds to major = 5, minor = 4, patch = 22880, build identifier 135e1ab4.
There are GitHub branches at the rocALUTION site with names release/rocm-rel-major.minor where major and minor are the same as in the HIP version.
For example for you can use the following to download rocALUTION:

::

   git clone -b release/rocm-rel-x.y https://github.com/ROCmSoftwarePlatform/rocALUTION.git
   cd rocALUTION

Replace x.y in the above command with the version of HIP SDK installed on your machine. For example, if you have HIP 5.5 installed, then use -b release/rocm-rel-5.5
You can can add the SDK tools to your path with an entry like:

::

   %HIP_PATH%\bin

Build
^^^^^^^^

Below are steps to build using the `rmake.py` script. The user can build either:

* library

* library and client

You only need (library) if you call rocALUTION from your code and only want the library built.
The client contains testing and benchmark tools.  rmake.py will print to the screen the full cmake command being used to configure rocALUTION based on your rmake command line options.
This full cmake command can be used in your own build scripts if you want to bypass the python helper script for a fixed set of build options.


Build library
^^^^^^^^^^^^^

Common uses of rmake.py to build (library) are
in the table below:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+--------------------+--------------------------+
| Command            | Description              |
+====================+==========================+
| ``./rmake.py -h``  | Help information.        |
+--------------------+--------------------------+
| ``./rmake.py``     | Build library.           |
+--------------------+--------------------------+
| ``./rmake.py -i``  | Build library, then      |
|                    | build and install        |
|                    | rocALUTION package.      |
|                    | If you want to keep      |
|                    | rocALUTION in your local |
|                    | tree, you do not         |
|                    | need the -i flag.        |
+--------------------+--------------------------+

Build library and client
^^^^^^^^^^^^^^^^^^^^^^^^^^

Some client executables (.exe) are listed in the table below:

====================== ==================================================
executable name        description
====================== ==================================================
rocalution-test           runs Google Tests to test the library
rocalution-bench          executable to benchmark or test functions
./cg lap_25.mtx           execute conjugate gradient example
                          (must download mtx matrix file you wish to use)
====================== ==================================================

Common uses of rmake.py to build (library + client) are
in the table below:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+------------------------+--------------------------+
| Command                | Description              |
+========================+==========================+
| ``./rmake.py -h``      | Help information.        |
+------------------------+--------------------------+
| ``./rmake.py -c``      | Build library and client |
|                        | in your local directory. |
+------------------------+--------------------------+
| ``./rmake.py -ic``     | Build and install        |
|                        | rocALUTION package, and  |
|                        | build the client.        |
|                        | If you want to keep      |
|                        | rocALUTION in your local |
|                        | directory, you do not    |
|                        | need the -i flag.        |
+------------------------+--------------------------+
