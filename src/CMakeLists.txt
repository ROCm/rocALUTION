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

# Get full git commit SHA-1
set(GIT_COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD)
execute_process(COMMAND ${GIT_COMMAND}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE rocalution_GIT_REV
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)

# Configure a header file to pass the rocALUTION version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/base/version.hpp.in"
               "${PROJECT_BINARY_DIR}/include/version.hpp"
               @ONLY
)

# Include sub-directories
include(base/CMakeLists.txt)
include(base/host/CMakeLists.txt)
include(solvers/CMakeLists.txt)
include(utils/CMakeLists.txt)

if(SUPPORT_HIP)
  include(base/hip/CMakeLists.txt)
endif()

# Public rocALUTION headers
set(PUBLIC_HEADERS
    rocalution.hpp
    ${BASE_PUBLIC_HEADERS}
    ${SOLVERS_PUBLIC_HEADERS}
    ${UTILS_PUBLIC_HEADERS}
)

# Copy public headers to include directory
foreach(i ${PUBLIC_HEADERS})
  configure_file("${i}" "${PROJECT_BINARY_DIR}/include/${i}" COPYONLY)
endforeach()

# rocALUTION source
set(SOURCE
    ${BASE_SOURCES}
    ${HOST_SOURCES}
    ${SOLVERS_SOURCES}
    ${UTILS_SOURCES}
    ${PUBLIC_HEADERS}
    ${PROJECT_BINARY_DIR}/include
)

if(SUPPORT_MPI)
  list(APPEND SOURCE ${UTILS_MPI_SOURCES})
endif()

# TODO plug-ins



# Create rocALUTION host library
add_library(rocalution ${SOURCE})
add_library(roc::rocalution ALIAS rocalution)

# Target link libraries
if(SUPPORT_OMP)
target_link_libraries(rocalution PRIVATE ${OpenMP_CXX_FLAGS})
endif()
if(SUPPORT_MPI)
  target_link_libraries(rocalution PUBLIC ${MPI_CXX_LIBRARIES})
endif()

# Target include directories
target_include_directories(rocalution PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)
if(SUPPORT_MPI)
  target_include_directories(rocalution PUBLIC ${MPI_CXX_INCLUDE_DIRS})
endif()

# Target compile definitions
if(SUPPORT_MPI)
  target_compile_definitions(rocalution PRIVATE SUPPORT_MULTINODE)
  target_compile_definitions(rocalution PUBLIC ${MPI_COMPILE_DEFINITIONS})
endif()
if(SUPPORT_HIP)
  target_compile_definitions(rocalution PRIVATE SUPPORT_HIP)
endif()

# Target compile options
if(SUPPORT_OMP)
  target_compile_options(rocalution PUBLIC ${OpenMP_CXX_FLAGS})
endif()
if(SUPPORT_MPI)
  target_compile_options(rocalution PUBLIC ${MPI_COMPILE_OPTIONS})
endif()

# Target properties
set_target_properties(rocalution PROPERTIES VERSION ${rocalution_VERSION} SOVERSION ${rocalution_SOVERSION})
set_target_properties(rocalution PROPERTIES DEBUG_POSTFIX "-d")
#set_target_properties(rocalution PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)

# Generate export header
#include(GenerateExportHeader)
#generate_export_header(rocalution EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/export.hpp)

# Create rocALUTION hip library
if(SUPPORT_HIP)

  # Flag source file as a hip source file
  foreach(i ${HIP_SOURCES})
    set_source_files_properties(${i} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
  endforeach()

  # For some reason, hip does not recognize build type
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(HIP_BUILD_OPTIONS "-O3")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(HIP_BUILD_OPTIONS "-g")
  endif()

  HIP_PREPARE_TARGET_COMMANDS(rocalution_hip OBJ _generated_files _source_files ${HIP_SOURCES} SHARED HIPCC_OPTIONS ${HIP_BUILD_OPTIONS})
  list(REMOVE_ITEM HIP_SOURCES ${_source_files})
  list(APPEND HIP_SOURCES ${_generated_files})

  set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)

  add_library(rocalution_hip SHARED ${HIP_SOURCES})

  set_target_properties(rocalution_hip PROPERTIES VERSION ${rocalution_VERSION} SOVERSION ${rocalution_SOVERSION})
  set_target_properties(rocalution_hip PROPERTIES DEBUG_POSTFIX "-d")

  target_compile_options(rocalution_hip PRIVATE -fno-gpu-rdc)
  foreach(target ${AMDGPU_TARGETS})
    target_compile_options(rocalution_hip PRIVATE --amdgpu-target=${target})
  endforeach()

  target_link_libraries(rocalution_hip PRIVATE ${ROCBLAS_LIBRARIES} ${ROCSPARSE_LIBRARIES})

  target_include_directories(rocalution_hip
                             PRIVATE
                             $<BUILD_INTERFACE:${HIPCUB_INCLUDE_DIRS}>
                             $<BUILD_INTERFACE:${ROCPRIM_INCLUDE_DIRS}>)

  target_link_libraries(rocalution PRIVATE rocalution_hip)
endif()







# Install targets
if(SUPPORT_HIP)
  rocm_install_targets(TARGETS rocalution rocalution_hip
                       INCLUDE
                       ${CMAKE_BINARY_DIR}/include
                       PREFIX rocalution
  )
else()
  rocm_install_targets(TARGETS rocalution
                       INCLUDE
                       ${CMAKE_BINARY_DIR}/include
                       PREFIX rocalution
  )
endif()

# Export targets
rocm_export_targets(TARGETS roc::rocalution
                    PREFIX rocalution
                    DEPENDS PACKAGE HIP
                    NAMESPACE roc::
)

# Symbolic links
rocm_install_symlink_subdir(rocalution)

# Package specific CPACK vars
if(SUPPORT_HIP)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc (>= 1.5.19055), rocsparse (>= 1.0.5), rocblas (>= 2.2.2)")
  set(CPACK_RPM_PACKAGE_REQUIRES "hip_hcc >= 1.5.19055, rocsparse >= 1.0.5, rocblas >= 2.2.2")
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

set(package_name rocalution)

set(ROCALUTION_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "Radeon Open Compute library for sparse linear systems"
  MAINTAINER "Nico Trost"
  LDCONFIG
  LDCONFIG_DIR ${ROCALUTION_CONFIG_DIR}
)
