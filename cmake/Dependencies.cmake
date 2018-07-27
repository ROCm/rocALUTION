# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

# Dependencies

# Git
find_package(Git REQUIRED)

# DownloadProject package
include(cmake/DownloadProject.cmake)

if(SUPPORT_HIP)
  # HIP configuration
  if(HIP_PLATFORM STREQUAL "hcc")
    # Workaround until hcc & hip cmake modules fixes symlink logic in their config files.
    # (Thanks to rocBLAS devs for finding workaround for this problem!)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip)
    # Ignore hcc warning: argument unused during compilation: '-isystem /opt/rocm/hip/include'
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-command-line-argument")
    find_package(hip REQUIRED CONFIG PATHS /opt/rocm)
    set(CMAKE_SHARED_LIBRARY_CREATE_HIP_FLAGS ${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS})
    set(CMAKE_HIP_COMPILE_OPTIONS_PIC ${CMAKE_C_COMPILE_OPTIONS_PIC})
    set(CMAKE_HIP_COMPILE_OPTIONS_PIE ${CMAKE_C_COMPILE_OPTIONS_PIE})
    set(CMAKE_HIP_COMPILE_OPTIONS_DLL ${CMAKE_C_COMPILE_OPTIONS_DLL})
    set(CMAKE_SHARED_LIBRARY_HIP_FLAGS ${CMAKE_SHARED_LIBRARY_C_FLAGS})
    set(CMAKE_SHARED_LIBRARY_LINK_HIP_FLAGS ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS})
    set(CMAKE_SHARED_LIBRARY_RUNTIME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG})
    set(CMAKE_SHARED_LIBRARY_RUNTIME_HIP_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG_SEP})
    set(CMAKE_SHARED_LIBRARY_RPATH_LINK_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_C_FLAG})
    set(CMAKE_EXE_EXPORTS_HIP_FLAG ${CMAKE_EXE_EXPORTS_C_FLAG})
    set(CMAKE_SHARED_LIBRARY_SONAME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_SONAME_C_FLAG})
    set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG})
    set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG_SEP})
    set(CMAKE_EXECUTABLE_RPATH_LINK_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG})
    set(CMAKE_SHARED_LIBRARY_LINK_HIP_WITH_RUNTIME_PATH ${CMAKE_SHARED_LIBRARY_LINK_C_WITH_RUNTIME_PATH})
    set(CMAKE_INCLUDE_FLAG_HIP ${CMAKE_INCLUDE_FLAG_C})
    set(CMAKE_INCLUDE_FLAG_SEP_HIP ${CMAKE_INCLUDE_FLAG_SEP_C})
    set(CMAKE_HIP_CREATE_SHARED_LIBRARY
        "${HIP_HIPCC_EXECUTABLE} ${HIP_LIBRARY_FLAGS} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
  else()
    find_package(CUDA REQUIRED)
  endif()
endif()

# rocPRIM package
if(SUPPORT_HIP)
  if(HIP_PLATFORM STREQUAL "hcc")
    find_package(ROCPRIM QUIET CONFIG PATHS /opt/rocm)
    find_package(HIPCUB QUIET CONFIG PATHS /opt/rocm)
    if(NOT ROCPRIM_FOUND)
      set(ROCPRIM_ROOT ${CMAKE_CURRENT_BINARY_DIR}/rocPRIM CACHE PATH "")
      message(STATUS "Downloading rocPRIM.")
      download_project(PROJ    rocPRIM
           GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocPRIM.git
           GIT_TAG             master
           INSTALL_DIR         ${ROCPRIM_ROOT}
           CMAKE_ARGS          -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hcc
           LOG_DOWNLOAD        TRUE
           LOG_CONFIGURE       TRUE
           LOG_INSTALL         TRUE
           BUILD_PROJECT       TRUE
           UPDATE_DISCONNECT   TRUE
      )
    find_package(ROCPRIM REQUIRED CONFIG PATHS ${ROCPRIM_ROOT})
    find_package(HIPCUB REQUIRED CONFIG PATHS ${ROCPRIM_ROOT})
    endif()
  elseif(HIP_PLATFORM STREQUAL "nvcc")
    find_package(HIPCUB QUIET CONFIG PATHS /opt/rocm)
    if(NOT HIPCUB_FOUND)
      set(ROCPRIM_ROOT ${CMAKE_CURRENT_BINARY_DIR}/rocPRIM CACHE PATH "")
      message(STATUS "Downloading rocPRIM.")
      download_project(PROJ    rocPRIM
           GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocPRIM.git
           GIT_TAG             master
           INSTALL_DIR         ${ROCPRIM_ROOT}
           CMAKE_ARGS          -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_CXX_COMPILER=${HIP_HIPCC_EXECUTABLE}
           LOG_DOWNLOAD        TRUE
           LOG_CONFIGURE       TRUE
           LOG_INSTALL         TRUE
           BUILD_PROJECT       TRUE
           UPDATE_DISCONNECT   TRUE
      )
    find_package(HIPCUB REQUIRED CONFIG PATHS ${ROCPRIM_ROOT})
    endif()
  endif()
endif()

# Test dependencies
if(BUILD_TEST)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(GTest QUIET)
  endif()
  if(NOT GTEST_FOUND)
    message(STATUS "GTest not found. Downloading and building GTest.")
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")
    download_project(PROJ        googletest
             GIT_REPOSITORY      https://github.com/google/googletest.git
             GIT_TAG             master
             INSTALL_DIR         ${GTEST_ROOT}
             CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             LOG_DOWNLOAD        TRUE
             LOG_CONFIGURE       TRUE
             LOG_BUILD           TRUE
             LOG_INSTALL         TRUE
             BUILD_PROJECT       TRUE
             UPDATE_DISCONNECTED TRUE
    )
  endif()
  find_package(GTest REQUIRED)
endif()

# ROCm package
find_package(ROCM QUIET CONFIG PATHS /opt/rocm)
if(NOT ROCM_FOUND)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  file(DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
       ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
  )
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  find_package(ROCM REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag})
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
