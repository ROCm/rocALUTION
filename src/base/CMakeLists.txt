# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

set(BASE_SOURCES
  base/base_rocalution.cpp
  base/operator.cpp
  base/vector.cpp
  base/local_matrix.cpp
  base/global_matrix.cpp
  base/local_vector.cpp
  base/global_vector.cpp
  base/base_matrix.cpp
  base/base_vector.cpp
  base/backend_manager.cpp
  base/parallel_manager.cpp
  base/local_stencil.cpp
  base/base_stencil.cpp
)

set(BASE_PUBLIC_HEADERS
  base/matrix_formats.hpp
  base/matrix_formats_ind.hpp
  base/base_rocalution.hpp
  base/operator.hpp
  base/vector.hpp
  base/local_matrix.hpp
  base/global_matrix.hpp
  base/local_vector.hpp
  base/global_vector.hpp
  base/backend_manager.hpp
  base/parallel_manager.hpp
  base/local_stencil.hpp
  base/stencil_types.hpp
)
