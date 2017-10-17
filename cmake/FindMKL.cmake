find_package(PackageHandleStandardArgs)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(MKL_INTERFACE_LAYER "_lp64")
  set(MKL_ARCH "intel64")
else()
  set(MKL_INTERFACE_LAYER "")
  set(MKL_ARCH "ia32")
endif()

if(NOT MKL_ROOT)
  set(MKL_ROOT $ENV{MKLROOT})
endif()

find_path(MKL_INCLUDE_DIR
  mkl.h
  PATHS
    ${MKL_ROOT}/include
)

set(_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(WIN32)
  if(MKL_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib)
  elseif(MKL_SDL)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib)
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES _dll.lib)
  endif()
elseif(APPLE)
  if(MKL_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .dylib)
  endif()
else()
  if(MKL_STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .so)
  endif()
endif()

if(MKL_SDL)
  find_library(MKL_LIBRARY mkl_rt
    PATHS ${MKL_ROOT}/lib/${MKL_ARCH}/)

    set(MKL_MINIMAL_LIBRARY ${MKL_LIBRARY})
else()
    set(MKL_INTERFACE_LIBNAME "mkl_intel${MKL_INTERFACE_LAYER}")

    find_library(MKL_INTERFACE_LIBRARY ${MKL_INTERFACE_LIBNAME}
        PATHS ${MKL_ROOT}/lib/${MKL_ARCH}/)

    ######################## Threading layer ########################
    if(MKL_MULTI_THREADED)
        if(WIN32)
            set(MKL_THREADING_LIBNAME mkl_intel_thread)
        else(WIN32)
            set(MKL_THREADING_LIBNAME mkl_gnu_thread)
        endif(WIN32)
    else()
        set(MKL_THREADING_LIBNAME mkl_sequential)
    endif()

    find_library(MKL_THREADING_LIBRARY ${MKL_THREADING_LIBNAME}
        PATHS ${MKL_ROOT}/lib/${MKL_ARCH}/)

    ####################### Computational layer #####################
    find_library(MKL_CORE_LIBRARY mkl_core
        PATHS ${MKL_ROOT}/lib/${MKL_ARCH}/)

    if(WIN32)
        set(MKL_RTL_LIBNAME libiomp5md)
        find_library(MKL_RTL_LIBRARY ${MKL_RTL_LIBNAME}
            PATHS ${MKL_ROOT}/../compiler/lib/${MKL_ARCH})
    endif()

    if(WIN32)
        set(MKL_MINIMAL_LIBRARY ${MKL_INTERFACE_LIBRARY} ${MKL_THREADING_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_RTL_LIBRARY})
        set(MKL_LIBRARY ${MKL_INTERFACE_LIBRARY} ${MKL_THREADING_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_RTL_LIBRARY})
    else(WIN32)
        set(MKL_LIBRARY "-Wl,--start-group ${MKL_INTERFACE_LIBRARY} ${MKL_THREADING_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_RTL_LIBRARY} -Wl,--end-group -ldl -lpthread -lm -fopenmp")
        set(MKL_MINIMAL_LIBRARY "-Wl,--start-group ${MKL_INTERFACE_LIBRARY} ${MKL_THREADING_LIBRARY} ${MKL_CORE_LIBRARY} -Wl,--end-group ${MKL_ROOT}/lib/${MKL_ARCH}/libmkl_blacs_openmpi_lp64.a -ldl -lpthread -lm -fopenmp")
    endif(WIN32)
endif()

set(CMAKE_FIND_LIBRARY_SUFFIXES ${_MKL_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})

find_package_handle_standard_args(MKL DEFAULT_MSG
    MKL_INCLUDE_DIR MKL_LIBRARY MKL_MINIMAL_LIBRARY)

if(MKL_FOUND)
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
    set(MKL_LIBRARIES ${MKL_LIBRARY})
    set(MKL_MINIMAL_LIBRARIES ${MKL_MINIMAL_LIBRARY})
endif()
