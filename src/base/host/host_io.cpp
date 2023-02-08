/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "host_io.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "rocalution/version.hpp"

#include <cinttypes>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace rocalution
{

    struct mm_banner
    {
        char array_type[64];
        char matrix_type[64];
        char storage_type[64];
    };

    bool mm_read_banner(FILE* fin, mm_banner& b)
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
        if(sscanf(line, "%s %s %s %s %s", banner, mtx, b.array_type, b.matrix_type, b.storage_type)
           != 5)
        {
            return false;
        }

        // clang-format off
        // Convert to lower case
        for(char *p = mtx;            *p != '\0'; *p = tolower(*p), ++p);
        for(char *p = b.array_type;   *p != '\0'; *p = tolower(*p), ++p);
        for(char *p = b.matrix_type;  *p != '\0'; *p = tolower(*p), ++p);
        for(char *p = b.storage_type; *p != '\0'; *p = tolower(*p), ++p);
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
        if(strncmp(b.matrix_type, "real", 4) && strncmp(b.matrix_type, "complex", 7)
           && strncmp(b.matrix_type, "integer", 7) && strncmp(b.matrix_type, "pattern", 7))
        {
            return false;
        }

        // Check storage type
        if(strncmp(b.storage_type, "general", 7) && strncmp(b.storage_type, "symmetric", 9)
           && strncmp(b.storage_type, "hermitian", 9))
        {
            return false;
        }

        return true;
    }

    template <typename ValueType,
              typename std::enable_if<std::is_same<ValueType, float>::value
                                          || std::is_same<ValueType, double>::value,
                                      int>::type
              = 0>
    static ValueType read_complex(double real, double imag)
    {
        return static_cast<ValueType>(real);
    }

    template <typename ValueType,
              typename std::enable_if<std::is_same<ValueType, std::complex<float>>::value
                                          || std::is_same<ValueType, std::complex<double>>::value,
                                      int>::type
              = 0>
    static ValueType read_complex(double real, double imag)
    {
        return ValueType(real, imag);
    }

    template <typename ValueType>
    bool mm_read_coordinate(FILE*       fin,
                            mm_banner&  b,
                            int&        nrow,
                            int&        ncol,
                            int64_t&    nnz,
                            int**       row,
                            int**       col,
                            ValueType** val)
    {
        char line[1025];

        // Skip banner and comments
        do
        {
            // Check for EOF
            if(!fgets(line, 1025, fin))
            {
                return false;
            }
        } while(line[0] == '%');

        // Read m, n, nnz
        while(sscanf(line, "%d %d %" SCNd64, &nrow, &ncol, &nnz) != 3)
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
            double real, imag;
            for(int64_t i = 0; i < nnz; ++i)
            {
                if(fscanf(fin, "%d %d %lg %lg", (*row) + i, (*col) + i, &real, &imag) != 4)
                {
                    return false;
                }
                --(*row)[i];
                --(*col)[i];
                (*val)[i] = read_complex<ValueType>(real, imag);
            }
        }
        else if(!strncmp(b.matrix_type, "real", 4) || !strncmp(b.matrix_type, "integer", 7))
        {
            double tmp;
            for(int64_t i = 0; i < nnz; ++i)
            {
                if(fscanf(fin, "%d %d %lg\n", (*row) + i, (*col) + i, &tmp) != 3)
                {
                    return false;
                }
                --(*row)[i];
                --(*col)[i];
                (*val)[i] = read_complex<ValueType>(tmp, tmp);
            }
        }
        else if(!strncmp(b.matrix_type, "pattern", 7))
        {
            for(int64_t i = 0; i < nnz; ++i)
            {
                if(fscanf(fin, "%d %d\n", (*row) + i, (*col) + i) != 2)
                {
                    return false;
                }
                --(*row)[i];
                --(*col)[i];
                (*val)[i] = static_cast<ValueType>(1);
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
            for(int64_t i = 0; i < nnz; ++i)
            {
                if((*row)[i] == (*col)[i])
                {
                    ++ndiag;
                }
            }

            int64_t tot_nnz = (nnz - ndiag) * 2 + ndiag;

            // Allocate memory
            int*       sym_row = *row;
            int*       sym_col = *col;
            ValueType* sym_val = *val;

            *row = NULL;
            *col = NULL;
            *val = NULL;

            allocate_host(tot_nnz, row);
            allocate_host(tot_nnz, col);
            allocate_host(tot_nnz, val);

            int64_t idx = 0;
            for(int64_t i = 0; i < nnz; ++i)
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
    bool read_matrix_mtx(int&        nrow,
                         int&        ncol,
                         int64_t&    nnz,
                         int**       row,
                         int**       col,
                         ValueType** val,
                         const char* filename)
    {
        FILE* file = fopen(filename, "r");

        if(!file)
        {
            LOG_INFO("ReadFileMTX: cannot open file " << filename);
            return false;
        }

        // read banner
        mm_banner banner;
        if(mm_read_banner(file, banner) != true)
        {
            LOG_INFO("ReadFileMTX: invalid matrix market banner");
            return false;
        }

        if(strncmp(banner.array_type, "coordinate", 10))
        {
            return false;
        }
        else
        {
            if(mm_read_coordinate(file, banner, nrow, ncol, nnz, row, col, val) != true)
            {
                LOG_INFO("ReadFileMTX: invalid matrix data");
                return false;
            }
        }

        fclose(file);

        return true;
    }

    template <typename ValueType,
              typename std::enable_if<std::is_same<ValueType, float>::value
                                          || std::is_same<ValueType, double>::value,
                                      int>::type
              = 0>
    void write_banner(FILE* file)
    {
        char sign[3];
        strcpy(sign, "%%");

        fprintf(file, "%sMatrixMarket matrix coordinate real general\n", sign);
    }

    template <typename ValueType,
              typename std::enable_if<std::is_same<ValueType, std::complex<float>>::value
                                          || std::is_same<ValueType, std::complex<double>>::value,
                                      int>::type
              = 0>
    void write_banner(FILE* file)
    {
        char sign[3];
        strcpy(sign, "%%");

        fprintf(file, "%sMatrixMarket matrix coordinate complex general\n", sign);
    }

    template <typename ValueType,
              typename std::enable_if<std::is_same<ValueType, float>::value, int>::type = 0>
    void write_value(FILE* file, ValueType val)
    {
        fprintf(file, "%0.12g\n", val);
    }

    template <typename ValueType,
              typename std::enable_if<std::is_same<ValueType, double>::value, int>::type = 0>
    void write_value(FILE* file, ValueType val)
    {
        fprintf(file, "%0.12lg\n", val);
    }

    template <
        typename ValueType,
        typename std::enable_if<std::is_same<ValueType, std::complex<float>>::value, int>::type = 0>
    void write_value(FILE* file, ValueType val)
    {
        fprintf(file, "%0.12g %0.12g\n", val.real(), val.imag());
    }

    template <
        typename ValueType,
        typename std::enable_if<std::is_same<ValueType, std::complex<double>>::value, int>::type
        = 0>
    void write_value(FILE* file, ValueType val)
    {
        fprintf(file, "%0.12lg %0.12lg\n", val.real(), val.imag());
    }

    template <typename ValueType>
    bool write_matrix_mtx(int              nrow,
                          int              ncol,
                          int64_t          nnz,
                          const int*       row,
                          const int*       col,
                          const ValueType* val,
                          const char*      filename)
    {
        FILE* file = fopen(filename, "w");

        if(!file)
        {
            LOG_INFO("WriteFileMTX: cannot open file " << filename);
            return false;
        }

        // Write MTX banner
        write_banner<ValueType>(file);

        // Write matrix sizes
        fprintf(file, "%d %d %" PRId64 "\n", nrow, ncol, nnz);

        for(int64_t i = 0; i < nnz; ++i)
        {
            fprintf(file, "%d %d ", row[i] + 1, col[i] + 1);
            write_value(file, val[i]);
        }

        fclose(file);

        return true;
    }

    static inline void read_csr_values(std::ifstream& in, int64_t nnz, float* val)
    {
        // Temporary array to convert from double to float
        std::vector<double> tmp(nnz);

        // Read double values
        in.read((char*)tmp.data(), sizeof(double) * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < nnz; ++i)
        {
            val[i] = static_cast<float>(tmp[i]);
        }
    }

    static inline void read_csr_values(std::ifstream& in, int64_t nnz, double* val)
    {
        // Read double values
        in.read((char*)val, sizeof(double) * nnz);
    }

    static inline void read_csr_values(std::ifstream& in, int64_t nnz, std::complex<float>* val)
    {
        // Temporary array to convert from complex double to complex float
        std::vector<std::complex<double>> tmp(nnz);

        // Read in double complex values
        in.read((char*)tmp.data(), sizeof(std::complex<double>) * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < nnz; ++i)
        {
            val[i] = std::complex<float>(static_cast<float>(tmp[i].real()),
                                         static_cast<float>(tmp[i].imag()));
        }
    }

    static inline void read_csr_values(std::ifstream& in, int64_t nnz, std::complex<double>* val)
    {
        // Read in double complex values
        in.read((char*)val, sizeof(std::complex<double>) * nnz);
    }

    static inline void read_csr_row_ptr_32(std::ifstream& in, int64_t nrow, int* ptr)
    {
        // We can directly read into array
        in.read((char*)ptr, sizeof(int) * (nrow + 1));
    }

    static inline void read_csr_row_ptr_32(std::ifstream& in, int64_t nrow, int64_t* ptr)
    {
        // Temporary array to convert from 32 to 64 bit
        std::vector<int> tmp(nrow + 1);

        // Read 32 bit integers
        in.read((char*)tmp.data(), sizeof(int) * (nrow + 1));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < nrow + 1; ++i)
        {
            ptr[i] = static_cast<int64_t>(tmp[i]);
        }
    }

    static inline void read_csr_row_ptr_64(std::ifstream& in, int64_t nrow, int* ptr)
    {
        // We cannot read without overflow, skip and throw warning
        LOG_INFO("ReadFileCSR: cannot read 64 bit sparsity pattern into 32 bit structure");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    static inline void read_csr_row_ptr_64(std::ifstream& in, int64_t nrow, int64_t* ptr)
    {
        // We can directly read into array
        in.read((char*)ptr, sizeof(int64_t) * (nrow + 1));
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool read_matrix_csr(int64_t&      nrow,
                         int64_t&      ncol,
                         int64_t&      nnz,
                         PointerType** ptr,
                         IndexType**   col,
                         ValueType**   val,
                         const char*   filename)
    {
        std::ifstream in(filename, std::ios::in | std::ios::binary);

        if(!in.is_open())
        {
            LOG_INFO("ReadFileCSR: cannot open file " << filename);
            return false;
        }

        // Header
        std::string header;
        std::getline(in, header);

        if(header != "#rocALUTION binary csr file")
        {
            LOG_INFO("ReadFileCSR: invalid rocALUTION matrix header");
            return false;
        }

        // rocALUTION version
        int version;
        in.read((char*)&version, sizeof(int));

        // Read sizes

        // We need backward compatibility
        if(version < 30000)
        {
            // 32 bit backward compatibility sizes
            int nrow32;
            int ncol32;
            int nnz32;

            // Read sizes
            in.read((char*)&nrow32, sizeof(int));
            in.read((char*)&ncol32, sizeof(int));
            in.read((char*)&nnz32, sizeof(int));

            // Convert to 64 bit
            nrow = static_cast<int64_t>(nrow32);
            ncol = static_cast<int64_t>(ncol32);
            nnz  = static_cast<int64_t>(nnz32);

            // Always read 32 bit row pointers
            int* ptr32 = NULL;

            // Allocate for row pointers
            allocate_host(nrow32 + 1, &ptr32);
            allocate_host(nrow + 1, ptr);

            // Read row pointers in 32bit
            in.read((char*)ptr32, (nrow32 + 1) * sizeof(int));

            // Convert to actual precision
            for(int i = 0; i < nrow32 + 1; ++i)
            {
                (*ptr)[i] = ptr32[i];
            }

            free_host(&ptr32);
        }
        else
        {
            // Read sizes
            in.read((char*)&nrow, sizeof(int64_t));
            in.read((char*)&ncol, sizeof(int64_t));
            in.read((char*)&nnz, sizeof(int64_t));

            // Allocate row pointer array
            allocate_host(nrow + 1, ptr);

            // Read data - precision is determined by nnz
            if(nnz < std::numeric_limits<int>::max())
            {
                read_csr_row_ptr_32(in, nrow, *ptr);
            }
            else
            {
                read_csr_row_ptr_64(in, nrow, *ptr);
            }
        }

        //        if(version != __ROCALUTION_VER)
        //        {
        //            LOG_INFO("ReadFileCSR: file version mismatch");
        //            return false;
        //        }

        // Allocate arrays
        allocate_host(nnz, col);
        allocate_host(nnz, val);

        // Read in columns and values
        in.read((char*)*col, nnz * sizeof(int));
        read_csr_values(in, nnz, *val);

        // Check ifstream status
        if(!in)
        {
            LOG_INFO("ReadFileCSR: invalid matrix data");
            return false;
        }

        in.close();

        return true;
    }

    static inline void write_csr_values(std::ofstream& out, int64_t nnz, const float* val)
    {
        // Temporary array to convert from float to double
        std::vector<double> tmp(nnz);

        // Convert values
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < nnz; ++i)
        {
            tmp[i] = static_cast<double>(val[i]);
        }

        // Write double values
        out.write((char*)tmp.data(), sizeof(double) * nnz);
    }

    static inline void write_csr_values(std::ofstream& out, int64_t nnz, const double* val)
    {
        // Write double values
        out.write((char*)val, sizeof(double) * nnz);
    }

    static inline void
        write_csr_values(std::ofstream& out, int64_t nnz, const std::complex<float>* val)
    {
        // Temporary array to convert from complex float to complex double
        std::vector<std::complex<double>> tmp(nnz);

        // Convert values
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < nnz; ++i)
        {
            tmp[i] = std::complex<double>(static_cast<double>(val[i].real()),
                                          static_cast<double>(val[i].imag()));
        }

        // Write double complex values
        out.write((char*)tmp.data(), sizeof(std::complex<double>) * nnz);
    }

    static inline void
        write_csr_values(std::ofstream& out, int64_t nnz, const std::complex<double>* val)
    {
        // Write double complex values
        out.write((char*)val, sizeof(std::complex<double>) * nnz);
    }

    static inline void write_csr_row_ptr_32(std::ofstream& out, int64_t nrow, const int* ptr)
    {
        // We can directly write into array
        out.write((char*)ptr, (nrow + 1) * sizeof(int));
    }

    static inline void write_csr_row_ptr_32(std::ofstream& out, int64_t nrow, const int64_t* ptr)
    {
        // We have previously checked, that no entry of ptr exceeds 32 bits
        // Thus, its safe to convert to 32 bits
        std::vector<int> tmp(nrow + 1);

        // Convert integers
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < nrow + 1; ++i)
        {
            tmp[i] = static_cast<int>(ptr[i]);
        }

        // Write 32 bit integers
        out.write((char*)tmp.data(), sizeof(int) * (nrow + 1));
    }

    static inline void write_csr_row_ptr_64(std::ofstream& out, int64_t nrow, const int* ptr)
    {
        // Writing 32 bit data into 64 bit file makes not much sense, this should never be called
        LOG_INFO("This function should never be called");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    static inline void write_csr_row_ptr_64(std::ofstream& out, int64_t nrow, const int64_t* ptr)
    {
        // We can directly write into array
        out.write((char*)ptr, (nrow + 1) * sizeof(int64_t));
    }

    template <typename ValueType, typename IndexType, typename PointerType>
    bool write_matrix_csr(int64_t            nrow,
                          int64_t            ncol,
                          int64_t            nnz,
                          const PointerType* ptr,
                          const IndexType*   col,
                          const ValueType*   val,
                          const char*        filename)
    {
        std::ofstream out(filename, std::ios::out | std::ios::binary);

        if(!out.is_open())
        {
            LOG_INFO("WriteFileCSR: cannot open file " << filename);
            return false;
        }

        // Write header
        out << "#rocALUTION binary csr file" << std::endl;

        // rocALUTION version
        int version = __ROCALUTION_VER;
        out.write((char*)&version, sizeof(int));

        // Write matrix sizes
        out.write((char*)&nrow, sizeof(int64_t));
        out.write((char*)&ncol, sizeof(int64_t));
        out.write((char*)&nnz, sizeof(int64_t));

        // Write row pointer data, depending on nnz
        if(nnz <= std::numeric_limits<int>::max())
        {
            write_csr_row_ptr_32(out, nrow, ptr);
        }
        else
        {
            write_csr_row_ptr_64(out, nrow, ptr);
        }

        // Write matrix column and value data
        out.write((char*)col, nnz * sizeof(int));

        write_csr_values(out, nnz, val);

        // Check ofstream status
        if(!out)
        {
            LOG_INFO("WriteFileCSR: filename=" << filename << "; could not write to file");
            return false;
        }

        out.close();

        return true;
    }

    template bool read_matrix_mtx(int&        nrow,
                                  int&        ncol,
                                  int64_t&    nnz,
                                  int**       row,
                                  int**       col,
                                  float**     val,
                                  const char* filename);
    template bool read_matrix_mtx(int&        nrow,
                                  int&        ncol,
                                  int64_t&    nnz,
                                  int**       row,
                                  int**       col,
                                  double**    val,
                                  const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_mtx(int&                  nrow,
                                  int&                  ncol,
                                  int64_t&              nnz,
                                  int**                 row,
                                  int**                 col,
                                  std::complex<float>** val,
                                  const char*           filename);
    template bool read_matrix_mtx(int&                   nrow,
                                  int&                   ncol,
                                  int64_t&               nnz,
                                  int**                  row,
                                  int**                  col,
                                  std::complex<double>** val,
                                  const char*            filename);
#endif

    template bool write_matrix_mtx(int          nrow,
                                   int          ncol,
                                   int64_t      nnz,
                                   const int*   row,
                                   const int*   col,
                                   const float* val,
                                   const char*  filename);
    template bool write_matrix_mtx(int           nrow,
                                   int           ncol,
                                   int64_t       nnz,
                                   const int*    row,
                                   const int*    col,
                                   const double* val,
                                   const char*   filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_mtx(int                        nrow,
                                   int                        ncol,
                                   int64_t                    nnz,
                                   const int*                 row,
                                   const int*                 col,
                                   const std::complex<float>* val,
                                   const char*                filename);
    template bool write_matrix_mtx(int                         nrow,
                                   int                         ncol,
                                   int64_t                     nnz,
                                   const int*                  row,
                                   const int*                  col,
                                   const std::complex<double>* val,
                                   const char*                 filename);
#endif

    template bool read_matrix_csr(int64_t&    nrow,
                                  int64_t&    ncol,
                                  int64_t&    nnz,
                                  int**       ptr,
                                  int**       col,
                                  float**     val,
                                  const char* filename);
    template bool read_matrix_csr(int64_t&    nrow,
                                  int64_t&    ncol,
                                  int64_t&    nnz,
                                  int**       ptr,
                                  int**       col,
                                  double**    val,
                                  const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_csr(int64_t&              nrow,
                                  int64_t&              ncol,
                                  int64_t&              nnz,
                                  int**                 ptr,
                                  int**                 col,
                                  std::complex<float>** val,
                                  const char*           filename);
    template bool read_matrix_csr(int64_t&               nrow,
                                  int64_t&               ncol,
                                  int64_t&               nnz,
                                  int**                  ptr,
                                  int**                  col,
                                  std::complex<double>** val,
                                  const char*            filename);
#endif

    template bool read_matrix_csr(int64_t&    nrow,
                                  int64_t&    ncol,
                                  int64_t&    nnz,
                                  int64_t**   ptr,
                                  int**       col,
                                  float**     val,
                                  const char* filename);
    template bool read_matrix_csr(int64_t&    nrow,
                                  int64_t&    ncol,
                                  int64_t&    nnz,
                                  int64_t**   ptr,
                                  int**       col,
                                  double**    val,
                                  const char* filename);
#ifdef SUPPORT_COMPLEX
    template bool read_matrix_csr(int64_t&              nrow,
                                  int64_t&              ncol,
                                  int64_t&              nnz,
                                  int64_t**             ptr,
                                  int**                 col,
                                  std::complex<float>** val,
                                  const char*           filename);
    template bool read_matrix_csr(int64_t&               nrow,
                                  int64_t&               ncol,
                                  int64_t&               nnz,
                                  int64_t**              ptr,
                                  int**                  col,
                                  std::complex<double>** val,
                                  const char*            filename);
#endif

    template bool write_matrix_csr(int64_t      nrow,
                                   int64_t      ncol,
                                   int64_t      nnz,
                                   const int*   ptr,
                                   const int*   col,
                                   const float* val,
                                   const char*  filename);
    template bool write_matrix_csr(int64_t       nrow,
                                   int64_t       ncol,
                                   int64_t       nnz,
                                   const int*    ptr,
                                   const int*    col,
                                   const double* val,
                                   const char*   filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_csr(int64_t                    nrow,
                                   int64_t                    ncol,
                                   int64_t                    nnz,
                                   const int*                 ptr,
                                   const int*                 col,
                                   const std::complex<float>* val,
                                   const char*                filename);
    template bool write_matrix_csr(int64_t                     nrow,
                                   int64_t                     ncol,
                                   int64_t                     nnz,
                                   const int*                  ptr,
                                   const int*                  col,
                                   const std::complex<double>* val,
                                   const char*                 filename);
#endif

    template bool write_matrix_csr(int64_t        nrow,
                                   int64_t        ncol,
                                   int64_t        nnz,
                                   const int64_t* ptr,
                                   const int*     col,
                                   const float*   val,
                                   const char*    filename);
    template bool write_matrix_csr(int64_t        nrow,
                                   int64_t        ncol,
                                   int64_t        nnz,
                                   const int64_t* ptr,
                                   const int*     col,
                                   const double*  val,
                                   const char*    filename);
#ifdef SUPPORT_COMPLEX
    template bool write_matrix_csr(int64_t                    nrow,
                                   int64_t                    ncol,
                                   int64_t                    nnz,
                                   const int64_t*             ptr,
                                   const int*                 col,
                                   const std::complex<float>* val,
                                   const char*                filename);
    template bool write_matrix_csr(int64_t                     nrow,
                                   int64_t                     ncol,
                                   int64_t                     nnz,
                                   const int64_t*              ptr,
                                   const int*                  col,
                                   const std::complex<double>* val,
                                   const char*                 filename);
#endif

} // namespace rocalution
