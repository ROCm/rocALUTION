#ifndef PARALUTION_UTILS_LOG_MPI_HPP_
#define PARALUTION_UTILS_LOG_MPI_HPP_

#include "log.hpp"

#include <iostream>
#include <stdlib.h>
#include <mpi.h>

#define CHECK_MPI_ERROR(err_t, file, line) {                     \
  if (err_t != MPI_SUCCESS) {                                    \
    LOG_INFO("MPI ERROR: " << err_t);                            \
    LOG_INFO("File: " << file << "; line: " << line);            \
    exit(1);                                                     \
  }                                                              \
}


#endif // PARALUTION_UTILS_LOG_MPI_HPP_

