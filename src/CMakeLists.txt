# ########################################################################
# Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print verbose compiler flags
if(BUILD_VERBOSE)
  include(../cmake/Verbose.cmake)
endif()

# Configure a header file to pass the rocALUTION version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/base/version.hpp.in"
               "${PROJECT_BINARY_DIR}/include/rocalution/version.hpp"
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
  configure_file("${i}" "${PROJECT_BINARY_DIR}/include/rocalution/${i}" COPYONLY)
endforeach()

source_group("Header Files\\Public" FILES ${PUBLIC_HEADERS})

include(GNUInstallDirs)

set(BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

# rocALUTION source
set(SOURCE
    ${BASE_SOURCES}
    ${HOST_SOURCES}
    ${SOLVERS_SOURCES}
    ${UTILS_SOURCES}
)

if(SUPPORT_MPI)
  list(APPEND SOURCE ${UTILS_MPI_SOURCES})
endif()

# Create rocALUTION host library
add_library(rocalution ${SOURCE} ${PUBLIC_HEADERS})
add_library(roc::rocalution ALIAS rocalution)

# Target link libraries
if(SUPPORT_OMP)
  if(WIN32)
    target_link_libraries(rocalution PRIVATE OpenMP::OpenMP_CXX libomp)
  else()
    target_link_libraries(rocalution PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

if(SUPPORT_MPI)
  target_link_libraries(rocalution PUBLIC MPI::MPI_CXX)
endif()

# Target compile definitions
if(SUPPORT_MPI)
  target_compile_definitions(rocalution PRIVATE SUPPORT_MULTINODE)
endif()

if(SUPPORT_HIP)
  target_compile_definitions(rocalution PRIVATE SUPPORT_HIP)
endif()

# Target properties
rocm_set_soversion(rocalution ${rocalution_SOVERSION})
set_target_properties(rocalution PROPERTIES DEBUG_POSTFIX "-d")
if(WIN32)
  set_target_properties(rocalution PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
endif()

# Generate export header
include(GenerateExportHeader)
generate_export_header(rocalution EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/export.hpp)
# Private Header include
target_include_directories(rocalution PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

list(APPEND CMAKE_HOST_FLAGS "-O3;-march=native")
target_compile_options(rocalution PRIVATE ${CMAKE_HOST_FLAGS})

# Create rocALUTION hip library
if(SUPPORT_HIP)
  # Flag source file as a hip source file
  foreach(i ${HIP_SOURCES})
    set_source_files_properties(${i} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
  endforeach()

  # HIP flags workaround while target_compile_options do not work
  list(APPEND HIP_HIPCC_FLAGS "-O3 -march=native -Wno-unused-command-line-argument -fPIC -std=c++14")
  foreach(target ${AMDGPU_TARGETS})
    list(APPEND HIP_HIPCC_FLAGS "--amdgpu-target=${target}")
  endforeach()

  # Create rocALUTION HIP library
  if(WIN32)
    add_library(rocalution_hip OBJECT ${HIP_SOURCES})
  else()
    hip_add_library(rocalution_hip ${HIP_SOURCES})
  endif()

  rocm_set_soversion(rocalution_hip ${rocalution_SOVERSION})
  set_target_properties(rocalution_hip PROPERTIES DEBUG_POSTFIX "-d")

  if(WIN32)
    target_link_libraries(rocalution_hip PRIVATE roc::rocblas roc::rocsparse roc::rocprim roc::rocrand hip::device)
  else()
    target_link_libraries(rocalution_hip PRIVATE roc::rocblas roc::rocsparse roc::rocprim roc::rocrand)
  endif()

  target_link_libraries(rocalution PRIVATE rocalution_hip hip::host)
endif()

set_target_properties(rocalution PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

if(WIN32 AND BUILD_CLIENTS AND BUILD_SHARED_LIBS)
  add_custom_command(TARGET rocalution POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:rocalution> ${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:rocalution>)
  if( ${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_custom_command(TARGET rocalution POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/rocalution.pdb ${PROJECT_BINARY_DIR}/clients/staging/rocalution.pdb )
  endif()
endif()

# Install targets
if(SUPPORT_HIP)
  rocm_install_targets(TARGETS rocalution rocalution_hip)
else()
  rocm_install_targets(TARGETS rocalution)
endif()
install(
  DIRECTORY
  "${PROJECT_BINARY_DIR}/include/rocalution/"
  DESTINATION "include/rocalution/"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)

# Export targets
if(SUPPORT_OMP)
  rocm_export_targets(TARGETS roc::rocalution
                      DEPENDS PACKAGE HIP
                      DEPENDS PACKAGE OpenMP
                      NAMESPACE roc::)
else()
  rocm_export_targets(TARGETS roc::rocalution
                      DEPENDS PACKAGE HIP
                      NAMESPACE roc::)
endif()


# Package specific CPACK vars
if(SUPPORT_HIP)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip-rocclr (>= 4.0.0), rocsparse (>= 1.12.10), rocblas (>= 2.22.0), rocrand (>= 0.0.1)")
  set(CPACK_RPM_PACKAGE_REQUIRES "hip-rocclr >= 4.0.0, rocsparse >= 1.12.10, rocblas >= 2.22.0, rocrand >= 0.0.1")
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

if(WIN32)
  set(CPACK_SOURCE_GENERATOR "ZIP")
  set(CPACK_GENERATOR "ZIP")
  set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE)
  set(INSTALL_PREFIX "C:/hipSDK")
  set(CPACK_SET_DESTDIR OFF)
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
  set(CPACK_PACKAGING_INSTALL_PREFIX "")
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
  if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
  endif()
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

# Package name
set(package_name rocalution)

set(ROCALUTION_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "Radeon Open Compute library for sparse linear systems"
  MAINTAINER "rocALUTION Maintainer <rocalution-maintainer@amd.com>"
  LDCONFIG
  LDCONFIG_DIR ${ROCALUTION_CONFIG_DIR}
)
