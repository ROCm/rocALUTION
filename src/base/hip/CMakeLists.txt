include_directories(base/hip)

set(HIP_SOURCES
  hip_allocate_free.cu
  backend_hip.cu
  hip_vector.cu
  hip_matrix_csr.cu
  hip_matrix_coo.cu
  hip_matrix_dense.cu
  hip_matrix_mcsr.cu
  hip_matrix_bcsr.cu
  hip_matrix_ell.cu
  hip_matrix_dia.cu
  hip_matrix_hyb.cu
)

set(HIP_PUBLIC_HEADERS
)
