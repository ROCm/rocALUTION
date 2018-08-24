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
     /opt/rocm/hip/cmake
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# rocALUTION project
project(rocalution VERSION 1.3.1.0 LANGUAGES CXX)
set(rocalution_SOVERSION 1)

# Set toolchain
include(cmake/SetToolchain.cmake)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOLEAN "Add paths to linker search and installed rpath")

# Build flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_SHARED_LIBS "Build rocALUTION as a shared library" ON)
option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
option(BUILD_CLIENTS_BENCHMARKS "Build benchmarks (requires boost)" OFF)
option(BUILD_CLIENTS_SAMPLES "Build examples" ON)

# Dependencies
include(cmake/Dependencies.cmake)

# AMD targets
set(AMDGPU_TARGETS gfx803;gfx900 CACHE STRING "List of specific machine types for library to target")

# rocALUTION library
add_subdirectory(src)

# Clients
if(BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  enable_testing()
  add_subdirectory(clients)
endif()
