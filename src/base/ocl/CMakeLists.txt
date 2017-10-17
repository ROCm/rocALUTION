include_directories(base/ocl)

set(OCL_SOURCES
  ocl_allocate_free.cpp
  backend_ocl.cpp
  ocl_vector.cpp
  ocl_matrix_csr.cpp
  ocl_matrix_coo.cpp
  ocl_matrix_dense.cpp
  ocl_matrix_mcsr.cpp
  ocl_matrix_bcsr.cpp
  ocl_matrix_ell.cpp
  ocl_matrix_dia.cpp
  ocl_matrix_hyb.cpp
)

set(OCL_PUBLIC_HEADERS
)
