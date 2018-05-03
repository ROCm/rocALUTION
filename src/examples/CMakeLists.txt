include_directories(${INC_BUILD_DIR})

set(EXAMPLES
  amg
  benchmark
  bicgstab
  cg
#  complex
  cg-amg
  cmk
  idr
  gmres
  sp-precond
  block-precond
  as-precond
  fixed-point
  mixed-precision
  me-preconditioner
  power-method
  simple-spmv
  async
  direct
  key
  var-precond
  stencil
  tns
)

foreach(example ${EXAMPLES})
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} ${rocALUTION_LIBRARY})
  install(PROGRAMS ${example} DESTINATION bin)
endforeach()
