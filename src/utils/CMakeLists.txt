# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

set(UTILS_SOURCES
  utils/log.cpp
  utils/allocate_free.cpp
  utils/math_functions.cpp
  utils/time_functions.cpp
)

set(UTILS_PUBLIC_HEADERS
  utils/def.hpp
  utils/allocate_free.hpp
  utils/time_functions.hpp
  utils/types.hpp
)

set(UTILS_MPI_SOURCES
  utils/communicator.cpp
)
