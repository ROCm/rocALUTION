set(rocALUTION_SOURCES)
set(rocALUTION_PUBLIC_HEADERS
  rocalution.hpp
)

# Copy headers to binary include dir
foreach(i ${rocALUTION_PUBLIC_HEADERS})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${i}" "${INC_BUILD_DIR}/${i}" COPYONLY)
endforeach()

# Fill rocALUTION lists with sources and headers for each subdirectory
# base
include(base/CMakeLists.txt)
foreach(i ${BASE_SOURCES})
  list(APPEND rocALUTION_SOURCES base/${i})
endforeach()
foreach(i ${BASE_PUBLIC_HEADERS})
  list(APPEND rocALUTION_PUBLIC_HEADERS base/${i})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/base/${i}" "${INC_BUILD_DIR}/base/${i}" COPYONLY)
endforeach()

# solvers
include(solvers/CMakeLists.txt)
foreach(i ${SOLVERS_SOURCES})
  list(APPEND rocALUTION_SOURCES solvers/${i})
endforeach()
foreach(i ${SOLVERS_PUBLIC_HEADERS})
  list(APPEND rocALUTION_PUBLIC_HEADERS solvers/${i})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/solvers/${i}" "${INC_BUILD_DIR}/solvers/${i}" COPYONLY)
endforeach()

# utils
include(utils/CMakeLists.txt)
foreach(i ${UTILS_SOURCES})
  list(APPEND rocALUTION_SOURCES utils/${i})
endforeach()
foreach(i ${UTILS_PUBLIC_HEADERS})
  list(APPEND rocALUTION_PUBLIC_HEADERS utils/${i})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/utils/${i}" "${INC_BUILD_DIR}/utils/${i}" COPYONLY)
endforeach()
if (SUPPORT_MPI)
  foreach(i ${UTILS_MPI_SOURCES})
    list(APPEND rocALUTION_SOURCES utils/${i})
  endforeach()
endif()

# plug-ins
include(plug-ins/CMakeLists.txt)
foreach(i ${PLUG-INS_SOURCES})
  list(APPEND rocALUTION_SOURCES plug-ins/${i})
endforeach()
foreach(i ${PLUG-INS_PUBLIC_HEADERS})
  list(APPEND rocALUTION_PUBLIC_HEADERS plug-ins/${i})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/plug-ins/${i}" "${PLUGINS_BUILD_DIR}/${i}" COPYONLY)
endforeach()

#######################################################################################################################

if(SUPPORT_HIP)
  set(CMAKE_SHARED_LIBRARY_CREATE_HIP_FLAGS ${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS})
  set(CMAKE_HIP_COMPILE_OPTIONS_PIC ${CMAKE_C_COMPILE_OPTIONS_PIC})
  set(CMAKE_HIP_COMPILE_OPTIONS_PIE ${CMAKE_C_COMPILE_OPTIONS_PIE})
  set(CMAKE_HIP_COMPILE_OPTIONS_DLL ${CMAKE_C_COMPILE_OPTIONS_DLL})
  set(CMAKE_SHARED_LIBRARY_HIP_FLAGS ${CMAKE_SHARED_LIBRARY_C_FLAGS})
  set(CMAKE_SHARED_LIBRARY_LINK_HIP_FLAGS ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS})
  set(CMAKE_SHARED_LIBRARY_RUNTIME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG})
  set(CMAKE_SHARED_LIBRARY_RUNTIME_HIP_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG_SEP})
  set(CMAKE_SHARED_LIBRARY_RPATH_LINK_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_C_FLAG})
  set(CMAKE_EXE_EXPORTS_HIP_FLAG ${CMAKE_EXE_EXPORTS_C_FLAG})
  set(CMAKE_SHARED_LIBRARY_SONAME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_SONAME_C_FLAG})
  set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG})
  set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG_SEP})
  set(CMAKE_EXECUTABLE_RPATH_LINK_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG})
  set(CMAKE_SHARED_LIBRARY_LINK_HIP_WITH_RUNTIME_PATH ${CMAKE_SHARED_LIBRARY_LINK_C_WITH_RUNTIME_PATH})
  set(CMAKE_INCLUDE_FLAG_HIP ${CMAKE_INCLUDE_FLAG_C})
  set(CMAKE_INCLUDE_FLAG_SEP_HIP ${CMAKE_INCLUDE_FLAG_SEP_C})
  set(CMAKE_HIP_CREATE_SHARED_LIBRARY
      "${HIP_HIPCC_EXECUTABLE} ${HIP_LIBRARY_FLAGS} <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

  foreach(i ${HIP_SOURCES})
    set_source_files_properties(${i} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
  endforeach()

  HIP_PREPARE_TARGET_COMMANDS(${rocALUTION_BACKEND_HIP} OBJ _generated_files _source_files ${HIP_SOURCES} SHARED)
  list(REMOVE_ITEM HIP_SOURCES ${_source_files})
  list(APPEND HIP_SOURCES ${_generated_files})

  if (HIP_PLATFORM STREQUAL "hcc")
    set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)
  else()
    set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE CXX)
  endif()

  add_library(${rocALUTION_BACKEND_HIP} SHARED ${HIP_SOURCES})

  if (HIP_PLATFORM STREQUAL "hcc")
    target_link_libraries(${rocALUTION_BACKEND_HIP} hip_hcc hipblas hipsparse rocsparse)
  else()
    target_link_libraries(${rocALUTION_BACKEND_HIP} cudart hipblas hipsparse cusparse)
  endif()
endif()

# Create rocalution library
add_library(${rocALUTION_LIBRARY} ${LIBRARY_TYPE} ${rocALUTION_SOURCES})

if(SUPPORT_HIP)
  target_link_libraries(${rocALUTION_LIBRARY} LINK_INTERFACE_LIBRARIES ${rocALUTION_BACKEND_HIP})
endif()
