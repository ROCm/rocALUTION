# ########################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Find OpenMP package
find_package(OpenMP)
if (NOT OPENMP_FOUND)
  message("-- OpenMP not found. Compiling WITHOUT OpenMP support.")
else()
  option(SUPPORT_OMP "Compile WITH OpenMP support." ON)
endif()

# MPI
find_package(MPI)
if (NOT MPI_FOUND)
  message("-- MPI not found. Compiling WITHOUT MPI support.")
else()
  option(SUPPORT_MPI "Compile WITH MPI support." OFF)
  if (SUPPORT_MPI)
    set(CMAKE_C_COMPILER ${MPI_COMPILER})
    set(CMAKE_CXX_COMPILER ${MPI_COMPILER})
  endif()
endif()

# Find HIP package
find_package(HIP 1.5.19055) # ROCm 2.2
if (NOT HIP_FOUND)
  message("-- HIP not found. Compiling WITHOUT HIP support.")
else()
  option(SUPPORT_HIP "Compile WITH HIP support." ON)
  if (SUPPORT_HIP)
    find_package(ROCBLAS 2.0.1 REQUIRED) # ROCm 2.2
    find_package(ROCSPARSE 1.0.2 REQUIRED) # ROCm 2.2

    # Find HCC executable
    find_program(
        HIP_HCC_EXECUTABLE
        NAMES hcc
        PATHS
        "${HIP_ROOT_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        /opt/rocm/hip
        ${CMAKE_PREFIX_PATH}
        PATH_SUFFIXES bin
        NO_DEFAULT_PATH
        )
    if(NOT HIP_HCC_EXECUTABLE)
        # Now search in default paths
        find_program(HIP_HCC_EXECUTABLE hcc)
    endif()
    mark_as_advanced(HIP_HCC_EXECUTABLE)
  endif()
endif()
