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
find_package(HIP 1.5.18353) # ROCm 1.9
if (NOT HIP_FOUND)
  message("-- HIP not found. Compiling WITHOUT HIP support.")
else()
  option(SUPPORT_HIP "Compile WITH HIP support." ON)
  if (SUPPORT_HIP)
    find_package(ROCBLAS 0.14.2.4 REQUIRED) # ROCm 1.9
    find_package(ROCSPARSE 0.1.3.0 REQUIRED) # ROCm 1.9
  endif()
endif()
