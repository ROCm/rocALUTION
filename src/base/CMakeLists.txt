set(BASE_SOURCES
  base_paralution.cpp
  operator.cpp
  vector.cpp
  local_matrix.cpp
  global_matrix.cpp
  local_vector.cpp
  global_vector.cpp
  base_matrix.cpp
  base_vector.cpp
  backend_manager.cpp
  parallel_manager.cpp
  local_stencil.cpp
  base_stencil.cpp
)

set(BASE_PUBLIC_HEADERS
  version.hpp
  matrix_formats.hpp
  matrix_formats_ind.hpp
  base_paralution.hpp
  operator.hpp
  vector.hpp
  local_matrix.hpp
  global_matrix.hpp
  local_vector.hpp
  global_vector.hpp
  backend_manager.hpp
  parallel_manager.hpp
  local_stencil.hpp
  stencil_types.hpp
)

include(base/host/CMakeLists.txt)
foreach(i ${HOST_SOURCES})
  list(APPEND BASE_SOURCES host/${i})
endforeach()

foreach(i ${HOST_PUBLIC_HEADERS})
  list(APPEND BASE_PUBLIC_HEADERS host/${i})
endforeach()

if (SUPPORT_CUDA)
  include(base/gpu/CMakeLists.txt)
  foreach(i ${GPU_SOURCES})
    list(APPEND BASE_SOURCES gpu/${i})
  endforeach()

  foreach(i ${GPU_PUBLIC_HEADERS})
    list(APPEND BASE_PUBLIC_HEADERS gpu/${i})
  endforeach()
endif()
