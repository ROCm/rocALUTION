# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

find_package(GTest REQUIRED)

set(ROCALUTION_TEST_SOURCES
  rocalution_host_gtest_main.cpp
)

set(ROCALUTION_CLIENTS_COMMON
)

add_executable(rocalution_host-test ${ROCALUTION_TEST_SOURCES} ${ROCALUTION_CLIENTS_COMMON})

target_compile_definitions(rocalution_host-test PRIVATE GOOGLE_TEST)

target_include_directories(rocalution_host-test PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
target_include_directories(rocalution_host-test SYSTEM PRIVATE $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>)

target_link_libraries(rocalution_host-test PRIVATE roc::rocalution ${GTEST_BOTH_LIBRARIES} Threads::Threads)

# Add tests
add_test(rocalution_host-test rocalution_host-test)
