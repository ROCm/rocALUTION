include_directories(${INC_BUILD_DIR})

set(MPI_EXAMPLES
  cg_mpi
  bicgstab_mpi
  fgmres_mpi
  cg-amg_mpi
  global-io_mpi
  idr_mpi
  qmrcgstab_mpi
  fcg_mpi
  benchmark_mpi
)

foreach(example ${MPI_EXAMPLES})
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} ${rocALUTION_LIBRARY})
  install(PROGRAMS ${example} DESTINATION bin/mpi)
endforeach()
