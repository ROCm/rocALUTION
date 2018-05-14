include_directories(${INC_BUILD_DIR})

###################################
# CHANGE THIS TO ADD NEW MATRICES #
###################################

set(SYM_MATRICES
  gr_30_30
  nos3
)

set(ASYM_MATRICES
  bfw398a
)

set(CPLX_MATRICES
  qc324
)

set(AMG_MATRICES
  gr_30_30
)

###################################
# CHANGE THIS TO ADD NEW SOLVERS  #
###################################

set(SYM_SOLVER
  CG
  FCG
  CR
)

set(ASYM_SOLVER
  BiCGStab
  BiCGStabl
  QMRCGStab
  IDR
  GMRES
  FGMRES
)

set(CPLX_SOLVER
  CG
  CR
  BiCGStab
  GMRES
  FGMRES
)

###################################
# CHANGE THIS TO ADD NEW TESTS    #
###################################

set(TESTS
  sym_solver
  asym_solver
  fsai_solver
  ilu_solver
  mcilu_solver
#  complex_solver
  amg_solver
)

###################################
# CHANGE THIS TO ADD NEW PRECONDS #
###################################

set(SYM_PRECOND
  None
  Jacobi
  SGS
  IC
  FSAI
  TNS
  MCSGS
)

set(ASYM_PRECOND
  None
  Jacobi
  GS
  ILU
  ILUT
  SPAI
  TNS
  MCILU
  MCGS
)

set(CPLX_PRECOND
  ILU
  MCILU
  FSAI
)

set(FORMAT
  CSR
  MCSR
  COO
  DIA
  ELL
  HYB
)

set(POWER
  0
  1
  2
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_TESTS_DIR})

################################################
# CHANGE THIS TO ADD NEW MATRICES FOR DOWNLOAD #
################################################

set(MATRICES)
foreach(i ${SYM_MATRICES})
  list(APPEND MATRICES "${i}")
endforeach()
foreach(i ${ASYM_MATRICES})
  list(APPEND MATRICES "${i}")
endforeach()

foreach(mat ${MATRICES})
  if(NOT EXISTS ${BIN_TESTS_DIR}/${mat}.mtx)
    if(NOT EXISTS ${CMAKE_BINARY_DIR}/${mat}.mtx.gz)
      set(DL_MAT TRUE)
    else()
      set(DL_MAT FALSE)
      execute_process(COMMAND gzip -d ${mat}.mtx.gz)
      execute_process(COMMAND mv ${mat}.mtx ${BIN_TESTS_DIR})
    endif()
  endif()
endforeach()

foreach(mat ${CPLX_MATRICES})
  if(NOT EXISTS ${BIN_TESTS_DIR}/${mat}.mtx)
    if(NOT EXISTS ${CMAKE_BINARY_DIR}/${mat}.mtx.gz)
      set(DL_MAT TRUE)
    else()
      set(DL_MAT FALSE)
      execute_process(COMMAND gzip -d ${mat}.mtx.gz)
      execute_process(COMMAND mv ${mat}.mtx ${BIN_TESTS_DIR})
    endif()
  endif()
endforeach()

if(DL_MAT)
  execute_process(COMMAND wget ftp://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/laplace/gr_30_30.mtx.gz)
  execute_process(COMMAND wget ftp://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/lanpro/nos3.mtx.gz)
  execute_process(COMMAND wget ftp://math.nist.gov/pub/MatrixMarket2/NEP/bfwave/bfw398a.mtx.gz)
  execute_process(COMMAND wget ftp://math.nist.gov/pub/MatrixMarket2/NEP/h2plus/qc324.mtx.gz)

  foreach(mat ${MATRICES})
    execute_process(COMMAND gzip -d ${mat}.mtx.gz)
    execute_process(COMMAND mv ${mat}.mtx ${BIN_TESTS_DIR})
  endforeach()

  foreach(mat ${CPLX_MATRICES})
    execute_process(COMMAND gzip -d ${mat}.mtx.gz)
    execute_process(COMMAND mv ${mat}.mtx ${BIN_TESTS_DIR})
  endforeach()

endif()

#######################################################################################################################

### TESTS FOR SOLVER
foreach (test ${TESTS})
  add_executable(${test} ${test}.cpp)
  target_link_libraries(${test} ${rocALUTION_LIBRARY})
endforeach()








### TESTS for symmetric matrices
foreach (mat ${SYM_MATRICES})
  foreach (ls ${SYM_SOLVER})
    foreach (p ${SYM_PRECOND})
      foreach (f ${FORMAT})
        add_test(SYM_${ls}_${p}_${mat}_${f} sym_solver ${mat}.mtx ${ls} ${p} ${f})
      endforeach()
    endforeach()
  endforeach()
endforeach()

### TESTS for asymmetric matrices
foreach (mat ${MATRICES})
  foreach (ls ${ASYM_SOLVER})
    foreach (p ${ASYM_PRECOND})
      foreach (f ${FORMAT})
        add_test(ASYM_${ls}_${p}_${mat}_${f} asym_solver ${mat}.mtx ${ls} ${p} ${f})
      endforeach()
    endforeach()
  endforeach()
endforeach()

### TESTS for ILU
foreach (mat ${MATRICES})
  foreach (f ${FORMAT})
    foreach (p ${POWER})
      add_test(ILU${p}_GMRES_${mat}_${f} ilu_solver ${mat}.mtx ${f} ${p})
      add_test(MCILU${p}_GMRES_${mat}_${f} mcilu_solver ${mat}.mtx ${f} ${p})
    endforeach()
  endforeach()
endforeach()

### TESTS for FSAI
foreach (mat ${SYM_MATRICES})
  foreach (f ${FORMAT})
    foreach (p ${POWER})
      add_test(FSAI${p}_CG_${mat}_${f} fsai_solver ${mat}.mtx ${f} ${p})
    endforeach()
  endforeach()
endforeach()

### TESTS for COMPLEX
foreach (mat ${CPLX_MATRICES})
  foreach (ls ${CPLX_SOLVER})
    foreach (p ${CPLX_PRECOND})
      foreach (f ${FORMAT})
        add_test(CPLX_${ls}_${p}_${mat}_${f} complex_solver ${mat}.mtx ${ls} ${p} ${f})
      endforeach()
    endforeach()
  endforeach()
endforeach()



set(AGGREGATION
  SmoothedAggregation
  Aggregation
  RugeStueben
  Pairwise
)

set(SMOOTHER
  Jacobi
  SGS
  ILU
  SPAI
  TNS
  MCILU
  MCSGS
  FSAI
)

set(MGCYCLE
  Vcycle
  Wcycle
  Kcycle
)

### TESTS for AMG
foreach (mat ${AMG_MATRICES})
  foreach (agg ${AGGREGATION})
    foreach (sm ${SMOOTHER})
      foreach (c ${MGCYCLE})
        add_test(AMG_${agg}_${sm}_${mat}_${c} amg_solver ${mat}.mtx ${agg} ${sm} ${c})
      endforeach()
    endforeach()
  endforeach()
endforeach()



### TESTS FOR ME-PRECONDITIONER
#add_executable(mp_test mp.cpp)
#target_link_libraries(mp_test ${PARALUTION_LIBRARY})
#foreach(mat ${SYM_MATRICES})
#  add_test(mp_${mat} mp_test ${mat}.mtx ${mat}.sol)
#endforeach()
