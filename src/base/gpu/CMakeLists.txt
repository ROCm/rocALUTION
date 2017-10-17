include_directories(base/gpu)

set(GPU_SOURCES
  gpu_allocate_free.cu
  backend_gpu.cu
  gpu_vector.cu
  gpu_matrix_csr.cu
  gpu_matrix_coo.cu
  gpu_matrix_dense.cu
  gpu_matrix_mcsr.cu
  gpu_matrix_bcsr.cu
  gpu_matrix_ell.cu
  gpu_matrix_dia.cu
  gpu_matrix_hyb.cu
)

set(GPU_PUBLIC_HEADERS
)
