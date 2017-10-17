cmake_minimum_required(VERSION 3.1)
project(PARALUTION)

#######################################################################################################################

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

if (${CMAKE_SOURCE_DIR} MATCHES ${CMAKE_BINARY_DIR})
  message(SEND_ERROR)
  message("-- ERROR: in-tree-build not allowed.")
  message("-- TRY: rm -f CMakeCache.txt; mkdir -p build; cd build; cmake ..; make")
  return()
endif()

#######################################################################################################################

set(PARALUTION_MAJOR_VERSION 1)
set(PARALUTION_MINOR_VERSION 2)
set(PARALUTION_BUILD_VERSION 0)
set(PARALUTION_VERSION "${PARALUTION_MAJOR_VERSION}.${PARALUTION_MINOR_VERSION}.${PARALUTION_BUILD_VERSION}")

#######################################################################################################################

set(PARALUTION_LIBRARY paralution)

set(BIN_BUILD_DIR ${CMAKE_BINARY_DIR}/bin)
set(INC_BUILD_DIR ${CMAKE_BINARY_DIR}/inc)
set(LIB_BUILD_DIR ${CMAKE_BINARY_DIR}/lib)
set(PLUGINS_BUILD_DIR ${CMAKE_BINARY_DIR}/inc/plug-ins)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_BUILD_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIB_BUILD_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIB_BUILD_DIR})

set(UTILS_DIR ${PROJECT_SOURCE_DIR}/src/utils)

#######################################################################################################################
# Generate ptk file
configure_file(${UTILS_DIR}/paralution.ptk.in ${CMAKE_BINARY_DIR}/paralution.ptk)

#######################################################################################################################
# Determine if library should be build shared or static
option(BUILD_SHARED "Build the PARALUTION library shared." ON)
if(BUILD_SHARED)
  SET(LIBRARY_TYPE SHARED)
else()
  SET(LIBRARY_TYPE STATIC)
endif()

#######################################################################################################################
# Find packages

# OpenMP
find_package(OpenMP)
if (NOT OPENMP_FOUND)
  message("-- OpenMP not found. Compiling WITHOUT OpenMP support.")
else()
  option(SUPPORT_OMP "Compile WITH OpenMP support." ON)
  if (SUPPORT_OMP)
    set(OMP_CXX_FLAGS ${OpenMP_CXX_FLAGS})
  endif()
endif()

#######################################################################################################################
# CUDA
find_package(CUDA)
if (NOT CUDA_FOUND)
  message("-- CUDA not found. Compiling WITHOUT CUDA support.")
else()
  option(SUPPORT_CUDA "Compile WITH CUDA support." ON)
  if (SUPPORT_CUDA)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch sm_20 --compiler-options -Wall -D_FORCE_INLINES")
    set(CUDA_CXX_FLAGS -DSUPPORT_CUDA)
  endif()
endif()

#######################################################################################################################
# OpenCL - only compile OpenCL when CUDA is not picked
find_package(OpenCL)
if (NOT OPENCL_FOUND)
  message("-- OpenCL not found. Compiling WITHOUT OpenCL support.")
else()
  if (NOT SUPPORT_CUDA)
    option(SUPPORT_OCL "Compile WITH OpenCL support." ON)
    if (SUPPORT_OCL)
      set(OCL_CXX_FLAGS -DSUPPORT_OCL)
      include_directories(${OPENCL_INCLUDE_DIRS})
    endif()
  else()
    message("-- OpenCL not supported while CUDA is enabled.")
  endif()
endif()

#######################################################################################################################
# MIC - only compile MIC when CUDA and OpenCL is not picked
if (NOT SUPPORT_CUDA)
  if (NOT SUPPORT_OCL)
    # Check for Intel Compiler
    find_program(CMAKE_C_COMPILER NAMES icc)
    find_program(CMAKE_CXX_COMPILER NAMES icpc)
    if (CMAKE_C_COMPILER MATCHES CMAKE_C_COMPILER-NOTFOUND OR
        CMAKE_CXX_COMPILER MATCHES CMAKE_CXX_COMPILER-NOTFOUND)
      message("-- MIC only supported by Intel Compiler. Compiling WITHOUT MIC support.")
    else()
      option(SUPPORT_MIC "Compile WITH MIC support." OFF)
      if (SUPPORT_MIC)
        include(CMakeForceCompiler)
        CMAKE_FORCE_C_COMPILER(icc "Intel C Compiler")
        CMAKE_FORCE_CXX_COMPILER(icpc "Intel C++ Compiler")
        set(CMAKE_AR xiar)
        set(CMAKE_LINKER xild)
        set(MIC_CXX_FLAGS "-DSUPPORT_MIC -offload-option,mic,compiler,\"-z defs\"")
      endif()
    endif()
  else()
    message("-- MIC not supported while OpenCL is enabled.")
  endif()
else()
  message("-- MIC not supported while CUDA is enabled.")
endif()

#######################################################################################################################
# MKL
set(MKL_MULTI_THREADED ON)
find_package(MKL)
if (NOT MKL_FOUND)
  message("-- MKL not found. Compiling WITHOUT MKL support.")
else()
  if (NOT SUPPORT_OCL)
    option(SUPPORT_MKL "Compile WITH MKL support." OFF)
    if (SUPPORT_MKL)
      set(MKL_CXX_FLAGS -DSUPPORT_MKL)
    endif()
  else()
    message("-- MKL not supported while OpenCL is enabled.")
  endif()
endif()

#######################################################################################################################
# MPI
find_package(MPI)
if (NOT MPI_FOUND)
  message("-- MPI not found. Compiling WITHOUT MPI support.")
else()
  option(SUPPORT_MPI "Compile WITH MPI support." OFF)
  if (SUPPORT_MPI)
    set(CMAKE_C_COMPILER mpicc)
    set(CMAKE_CXX_COMPILER mpicxx)
    set(MPI_CXX_FLAGS "-DMPICH_IGNORE_CXX_SEEK -DSUPPORT_MULTINODE")
  endif()
endif()

#######################################################################################################################
# Debug or RelWithDebInfo
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "debug")
  set(CXX_FLAGS "-g -O0 -Wall -fPIC")
elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo" OR "${CMAKE_BUILD_TYPE}" STREQUAL "relwithdebinfo")
  set(CXX_FLAGS "-O3 -Wall -g -fPIC")
else()
  set(CXX_FLAGS "-O3 -Wall -fPIC")
endif()

# Build examples
option(BUILD_EXAMPLES "Build examples." ON)

#######################################################################################################################

set(CMAKE_CXX_FLAGS "${CXX_FLAGS} ${OMP_CXX_FLAGS} ${CUDA_CXX_FLAGS} ${OCL_CXX_FLAGS} ${MKL_CXX_FLAGS} ${MIC_CXX_FLAGS} ${MPI_CXX_FLAGS}")

add_subdirectory(src)
if (BUILD_EXAMPLES)
  add_subdirectory(src/examples)
  if (SUPPORT_MPI)
    add_subdirectory(src/examples/mpi)
  endif()
endif()

#######################################################################################################################
# Testing

if (EXISTS "${CMAKE_SOURCE_DIR}/not-for-release/ctest")
  option(BUILD_TESTING "Build tests." OFF)
  include(CTest)
  enable_testing()

  set(BIN_TESTS_DIR ${CMAKE_BINARY_DIR}/not-for-release/ctest)

  if (BUILD_TESTING)
    add_subdirectory(not-for-release/ctest)
  endif()
endif()

