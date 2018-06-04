#include "../../utils/def.hpp"
#include "host_io.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/log.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex>

namespace rocalution {

struct mm_banner
{
    char array_type[64];
    char matrix_type[64];
    char storage_type[64];
};

bool mm_read_banner(FILE *fin, mm_banner &b)
{
    char line[1025];

    // Read banner line
    if(!fgets(line, 1025, fin))
    {
        return false;
    }

    char banner[64];
    char mtx[64];

    // Read 5 tokens from banner
    if(sscanf(line, "%s %s %s %s %s", banner, mtx, b.array_type, b.matrix_type, b.storage_type) != 5)
    {
        return false;
    }

    // clang-format off
    // Convert to lower case
    for(char *p=mtx; *p != '\0'; *p = tolower(*p), ++p);
    for(char *p=b.array_type; *p != '\0'; *p = tolower(*p), ++p);
    for(char *p=b.matrix_type; *p != '\0'; *p = tolower(*p), ++p);
    for(char *p=b.storage_type; *p != '\0'; *p = tolower(*p), ++p);

    // clang-format on

    // Check banner
    if(strncmp(banner, "%%MatrixMarket", 14))
    {
        return false;
    }

    // Check array type
    if(strncmp(mtx, "matrix", 6))
    {
        return false;
    }

    // Check array type
    if(strncmp(b.array_type, "coordinate", 10))
    {
        return false;
    }

    // Check matrix type
    if(strncmp(b.matrix_type, "real", 4) && strncmp(b.matrix_type, "complex", 7) && strncmp(b.matrix_type, "integer", 7))
    {
        return false;
    }

    // Check storage type
    if(strncmp(b.storage_type, "general", 7) && strncmp(b.storage_type, "symmetric", 9) && strncmp(b.storage_type, "hermitian", 9))
    {
        return false;
    }

    return true;
}


template <typename ValueType>
bool mm_read_coordinate(FILE *fin, mm_banner &b, int &nrow, int &ncol, int &nnz, int **row, int **col, ValueType **val) {

    char line[1025];

    // Skip banner and comments
    do
    {
        // Check for EOF
        if(!fgets(line, 1025, fin))
        {
            return false;
        }
    }
    while(line[0] == '%');

    // Read m, n, nnz
    while(sscanf(line, "%d %d %d", &nrow, &ncol, &nnz) != 3)
    {
        // Check for EOF and loop until line with 3 integer entries found
        if(!fgets(line, 1025, fin))
        {
            return false;
        }
    }

    // Allocate arrays
    allocate_host(nnz, row);
    allocate_host(nnz, col);
    allocate_host(nnz, val);

    // Read data
    if(!strncmp(b.matrix_type, "complex", 7))
    {
        double tmp1, tmp2;
        for(int i=0; i<nnz; ++i)
        {
            if(fscanf(fin, "%d %d %lg %lg", (*row)+i, (*col)+i, &tmp1, &tmp2) != 4)
            {
                return false;
            }
            --(*row)[i];
            --(*col)[i];
//            (*val)[i] = TODO
        }
    }
    else if(!strncmp(b.matrix_type, "real", 4) || !strncmp(b.matrix_type, "integer", 7))
    {
        double tmp;
        for(int i=0; i<nnz; ++i)
        {
            if(fscanf(fin, "%d %d %lg\n", (*row)+i, (*col)+i, &tmp) != 3)
            {
                return false;
            }
            --(*row)[i];
            --(*col)[i];
            (*val)[i] = static_cast<ValueType>(tmp);
        }
    }
    else
    {
        return false;
    }

    // Expand symmetric matrix
    if(strncmp(b.storage_type, "general", 7))
    {
        // Count diagonal entries
        int ndiag = 0;
        for(int i=0; i<nnz; ++i)
        {
            if((*row)[i] == (*col)[i])
            {
                ++ndiag;
            }
        }

        int tot_nnz = (nnz - ndiag) * 2 + ndiag;

        // Allocate memory
        int *sym_row = *row;
        int *sym_col = *col;
        ValueType *sym_val = *val;

        *row = NULL;
        *col = NULL;
        *val = NULL;

        allocate_host(tot_nnz, row);
        allocate_host(tot_nnz, col);
        allocate_host(tot_nnz, val);

        int idx = 0;
        for(int i=0; i<nnz; ++i)
        {
            (*row)[idx] = sym_row[i];
            (*col)[idx] = sym_col[i];
            (*val)[idx] = sym_val[i];
            ++idx;

            // Do not write diagonal again
            if(sym_row[i] != sym_col[i])
            {
                (*row)[idx] = sym_col[i];
                (*col)[idx] = sym_row[i];
                (*val)[idx] = sym_val[i];
                ++idx;
            }
        }

        if(idx != tot_nnz)
        {
            return false;
        }

        nnz = tot_nnz;

        free_host(&sym_row);
        free_host(&sym_col);
        free_host(&sym_val);
    }
    return true;
}

template <typename ValueType>
bool read_matrix_mtx(int &nrow, int &ncol, int &nnz, int **row, int **col, ValueType **val,
                     const char *filename) {  

  FILE *file = fopen(filename, "r");

  if (!file) {
    LOG_INFO("ReadFileMTX: cannot open file " << filename);
    return false;
  }

  // read banner
  mm_banner banner;
  if (mm_read_banner(file, banner) != true) {
    LOG_INFO("ReadFileMTX: invalid matrix market banner");
    return false;
  }

  if (strncmp(banner.array_type, "coordinate", 10))
  {
      return false;
  }
  else
  {
      if (mm_read_coordinate(file, banner, nrow, ncol, nnz, row, col, val) != true)
      {
        LOG_INFO("ReadFileMTX: invalid matrix data");
        return false;
      }
  }

  fclose(file);

  return true;

}

template <typename ValueType>
bool write_matrix_mtx(const int nrow, const int ncol, const int nnz,
                      const int *row, const int *col, const ValueType *val, const char *filename)
{
    FILE *file = fopen(filename, "w");

    if (!file) {
        LOG_INFO("WriteFileMTX: cannot open file " << filename);
        return false;
    }

    char sign[3];
    strcpy(sign, "%%");

    fprintf(file, "%sMatrixMarket matrix coordinate real general\n", sign);
    fprintf(file, "%d %d %d\n", nrow, ncol, nnz);

    for (int i=0; i<nnz; ++i)
    {
        fprintf(file, "%d %d %0.12lg\n", row[i] + 1, col[i] + 1, val[i]);
    }

    fclose(file);

    return true;
}


template bool read_matrix_mtx(int &nrow, int &ncol, int &nnz, int **row, int **col, float **val,
                              const char *filename);
template bool read_matrix_mtx(int &nrow, int &ncol, int &nnz, int **row, int **col, double **val,
                              const char *filename);
#ifdef SUPPORT_COMPLEX
template bool read_matrix_mtx(int &nrow, int &ncol, int &nnz, int **row, int **col, std::complex<float> **val,
                              const char *filename);
template bool read_matrix_mtx(int &nrow, int &ncol, int &nnz, int **row, int **col, std::complex<double> **val,
                              const char *filename);
#endif

template bool write_matrix_mtx(const int nrow, const int ncol, const int nnz,
                               const int *row, const int *col, const float *val,
                               const char *filename);
template bool write_matrix_mtx(const int nrow, const int ncol, const int nnz,
                               const int *row, const int *col, const double *val,
                               const char *filename);
#ifdef SUPPORT_COMPLEX
template bool write_matrix_mtx(const int nrow, const int ncol, const int nnz,
                               const int *row, const int *col, const std::complex<float> *val,
                               const char *filename);
template bool write_matrix_mtx(const int nrow, const int ncol, const int nnz,
                               const int *row, const int *col, const std::complex<double> *val,
                               const char *filename);
#endif
}
