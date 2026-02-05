.. meta::
   :description: Install rocALUTION
   :keywords: rocALUTION, ROCm, library, API, install, windows, linux, HIP SDK, building, installing


********************************
Install rocALUTION
********************************

You can install rocALUTION as part of the AMD ROCm software stack or `HIP SDK <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/>`__ for Windows, or you can build it directly from source.
The installation method you choose depends on your operating system and whether you need a custom configuration, such as multi-node execution.

Install on Linux
--------------------------------

On Linux systems, rocALUTION is typically installed as part of ROCm. 
You must install ROCm before building or using rocALUTION. 

When installed through ROCm, rocALUTION is provided as a single-node, accelerator-enabled library.
If you require a different configuration, such as multi-node or distributed execution, you can build rocALUTION from source.


Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building rocALUTION from source on Linux requires the following prerequisites:

- `CMake <https://cmake.org/>`__
- `rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html>`__
- `rocSPARSE <https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/index.html>`__
- `rocRAND <https://rocm.docs.amd.com/projects/rocRAND/en/latest/index.html>`__
- `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`__

Ensure that these components are installed before building rocALUTION. Refer to their respective documentation for installation instructions.

For multi-node configurations, you must also install:

- `OpenMP <https://www.openmp.org/>`__
- `MPI <https://www.mcs.anl.gov/research/projects/mpi/>`__

Build from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Obtain the source code from the `rocALUTION GitHub repository <https://github.com/ROCm/rocALUTION>`__.
   Use the branch that matches the version of ROCm installed on your system.

2. Create a ``build`` directory in the rocALUTION root directory and change into it:

   .. code:: shell

      mkdir build
      cd build

3. Use CMake to generate the build files. You must set the ``ROCM_PATH`` directive to point to the ROCm installation directory.
   The following optional directives can also be configured:

   - ``SUPPORT_HIP``: Enable HIP support. This option is ``ON`` by default.
   - ``SUPPORT_OMP``: Enable OpenMP support. This option is ``ON`` by default.
   - ``SUPPORT_MPI``: Enable MPI support for multi-node execution. This option is ``OFF`` by default.
   - ``BUILD_SHARED_LIBS``: Build rocALUTION as a shared library. This option is ``ON`` by default and is recommended.
   - ``BUILD_EXAMPLES``: Build the example programs. This option is ``ON`` by default.

   For example, to build rocALUTION with MPI support enabled:

   .. code:: shell

      cmake .. -DSUPPORT_MPI=ON -DROCM_PATH=/opt/rocm/

4. Build and install rocALUTION:

   .. code:: shell

      make
      make install

   The library is installed under the ROCm installation directory.

Test your rocALUTION installation on Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To test your installation on Linux, run a CG solver on a Laplacian matrix:

.. code:: shell

   cd rocALUTION; cd build
   wget ftp://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/laplace/gr_30_30.mtx.gz
   gzip -d gr_30_30.mtx.gz
   ./clients/staging/cg gr_30_30.mtx

Install on Windows
--------------------------------

On Microsoft Windows, rocALUTION is built and used with the HIP SDK for Windows.
You must install the HIP SDK for Windows before building or using rocALUTION.

When installed through the HIP SDK, rocALUTION is provided as a single-node, accelerator-enabled library.
If you require a different configuration, such as multi-node or distributed execution, you can build rocALUTION from source.


Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building rocALUTION from source on Windows requires the following prerequisites:

- `CMake <https://cmake.org/>`_
- `rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html>`_
- `rocSPARSE <https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/index.html>`_
- `rocRAND <https://rocm.docs.amd.com/projects/rocRAND/en/latest/index.html>`_
- `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_
- `Python 3 <https://www.python.org/downloads/>`_
- `Ninja <https://ninja-build.org/>`_
- `Strawberry Perl <https://strawberryperl.com/>`_


Build from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Obtain the source code from the `rocALUTION GitHub repository <https://github.com/ROCm/rocALUTION>`__.
   Use the branch that matches the installed version of the HIP SDK for Windows.

2. Verify the installed HIP SDK version by running:

   .. code:: shell

      hipcc --version

   .. note::

      If ``hipcc`` is not found, add ``%HIP_PATH%\\bin`` to your ``PATH`` environment variable.

3. Use the ``rmake.py`` script to build rocALUTION without installing it:

   .. code:: shell

      python3 rmake.py

   The built library files are placed in ``build\\release\\include\\rocalution``.

4. To build and install the library, use the ``-i`` option:

   .. code:: shell

      python3 rmake.py -i

   The library files are installed under ``%HIP_PATH%\\include\\rocalution``.

5. To build the library and its clients and install the library files, use:

   .. code:: shell

      python3 rmake.py -ci

   You can also omit the ``i`` option to build the library and clients without installing the library.


Test your rocALUTION installation on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To test your installation on Windows, run a CG solver on a Laplacian matrix. On Windows, use Windows Subsystem for Linux (WSL).

1. Install WSL (run in PowerShell as Administrator):

   .. code-block:: powershell

      wsl --install

2. Reboot and open the **Ubuntu** application.

3. Run the test:

   .. code-block:: shell

      sudo apt update
      sudo apt install -y wget gzip
      cd rocALUTION/build
      wget ftp://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/laplace/gr_30_30.mtx.gz
      gzip -d gr_30_30.mtx.gz
      ./clients/staging/cg gr_30_30.mtx
