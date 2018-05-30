include_directories(base/hip)

set(HIP_SOURCES
  base/hip/hip_blas.cpp
  base/hip/hip_sparse.cpp
  base/hip/backend_hip.cpp
  base/hip/hip_allocate_free.cpp
  base/hip/hip_vector.cpp
  base/hip/hip_matrix_csr.cpp
  base/hip/hip_matrix_coo.cpp
  base/hip/hip_matrix_dense.cpp
  base/hip/hip_matrix_mcsr.cpp
  base/hip/hip_matrix_bcsr.cpp
  base/hip/hip_matrix_ell.cpp
  base/hip/hip_matrix_dia.cpp
  base/hip/hip_matrix_hyb.cpp
)
