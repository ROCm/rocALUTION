# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

set(HOST_SOURCES
  base/host/host_matrix_csr.cpp
  base/host/host_matrix_mcsr.cpp
  base/host/host_matrix_bcsr.cpp
  base/host/host_matrix_coo.cpp
  base/host/host_matrix_dia.cpp
  base/host/host_matrix_ell.cpp
  base/host/host_matrix_hyb.cpp
  base/host/host_matrix_dense.cpp
  base/host/host_vector.cpp
  base/host/host_conversion.cpp  
  base/host/host_affinity.cpp
  base/host/host_io.cpp
  base/host/host_stencil_laplace2d.cpp
)
