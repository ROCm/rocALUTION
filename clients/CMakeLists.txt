# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required(VERSION 3.5.1)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# CMake modules
list(APPEND CMAKE_MODULE_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT TARGET rocalution)
  # This project may compile dependencies for clients
  project(rocalution-clients LANGUAGES CXX)

  find_package(rocalution REQUIRED CONFIG PATHS /opt/rocm/rocalution)

  option(BUILD_CLIENTS_SAMPLES "Build examples." ON)
  option(SUPPORT_MPI "Build MPI examples." OFF)
  option(BUILD_CLIENTS_BENCHMARKS "Build benchmarks." OFF)
  option(BUILD_CLIENTS_TESTS "Build tests." OFF)

  if(SUPPORT_MPI)
    find_package(MPI REQUIRED)
  endif()
endif()

if(BUILD_CLIENTS_SAMPLES)
  add_subdirectory(samples)
endif()

if(BUILD_CLIENTS_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(BUILD_CLIENTS_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
