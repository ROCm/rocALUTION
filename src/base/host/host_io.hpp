#ifndef ROCALUTION_HOST_IO_HPP_
#define ROCALUTION_HOST_IO_HPP_

#include <string>

namespace rocalution {

template <typename ValueType>
bool read_matrix_mtx(int &nrow, int &ncol, int &nnz, int **row, int **col, ValueType **val, const char *filename);

template <typename ValueType>
bool write_matrix_mtx(const int nrow, const int ncol, const int nnz,
                      const int *row, const int *col, const ValueType *val, const char *filename);


}

#endif // ROCALUTION_HOST_IO_HPP_
