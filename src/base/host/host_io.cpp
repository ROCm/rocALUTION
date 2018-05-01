#include "../../utils/def.hpp"
#include "host_io.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/log.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <complex>

namespace rocalution {

// ----------------------------------------------------------
// struct matrix_market_banner
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.4.0,
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - None
// ----------------------------------------------------------
struct matrix_market_banner {

  // "array" or "coordinate"
  std::string storage;
  // "general", "symmetric", "hermitian", or "skew-symmetric"
  std::string symmetry;
  // "complex", "real", "integer", or "pattern"
  std::string type;

};

// ----------------------------------------------------------
// void tokenize
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.4.0,
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - None
// ----------------------------------------------------------
inline void tokenize(std::vector<std::string>& tokens,
                     const std::string& str,
                     const std::string& delimiters = "\n\r\t ") {

  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos) {

    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);

  }

}

// ----------------------------------------------------------
// assign_complex
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.4.0,
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - None
// ----------------------------------------------------------
template <typename ValueType>
void assign_complex(ValueType &val, double real, double imag) {

  val = ValueType(real);

}

// ----------------------------------------------------------
// void assign_complex
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.4.0,
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - None
// ----------------------------------------------------------
template <typename ValueType>
void assign_complex(std::complex<ValueType> &val, double real, double imag) {

  val = std::complex<ValueType>(ValueType(real), ValueType(imag));

}

// ----------------------------------------------------------
// void write_value
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.4.0,
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - None
// ----------------------------------------------------------
template <typename ValueType>
void write_value(std::ofstream &output, const ValueType &val) {

  output << val;

}

// ----------------------------------------------------------
// void write_value
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.4.0,
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - None
// ----------------------------------------------------------
template <typename ValueType>
void write_value(std::ofstream &output, const std::complex<ValueType> &val) {

  output << val.real() << " " << val.imag();

}

// ----------------------------------------------------------
// bool read_matrix_market_banner
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.4.0,
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - None
// ----------------------------------------------------------
bool read_matrix_market_banner(matrix_market_banner &banner, std::ifstream &input) {

  std::string line;
  std::vector<std::string> tokens;

  // read first line
  std::getline(input, line);
  tokenize(tokens, line);

  if (tokens.size() != 5 || tokens[0] != "%%MatrixMarket" || tokens[1] != "matrix")
    return false;

  banner.storage  = tokens[2];
  banner.type     = tokens[3];
  banner.symmetry = tokens[4];

  if (banner.storage != "array" && banner.storage != "coordinate")
    return false;

  if (banner.type != "complex" && banner.type != "real" &&
      banner.type != "integer" && banner.type != "pattern")
    return false;

  if (banner.symmetry != "general"   && banner.symmetry != "symmetric" &&
      banner.symmetry != "hermitian" && banner.symmetry != "skew-symmetric")
    return false;

  if (tokens.size() > 10)
    return false;

  return true;

}

// ----------------------------------------------------------
// bool read_coordinate_stream
// ----------------------------------------------------------
// Modified and adapted from CUSP 0.4.0,
// http://code.google.com/p/cusp-library/
// NVIDIA, APACHE LICENSE 2.0
// ----------------------------------------------------------
// CHANGELOG
// - None
// ----------------------------------------------------------
template <typename ValueType>
bool read_coordinate_stream(int &nrow, int &ncol, int &nnz, int **row, int **col, ValueType **val,
                            std::ifstream &input, matrix_market_banner &banner) {

  // read file contents line by line
  std::string line;

  // skip over banner and comments
  do {
    std::getline(input, line);
  } while (line[0] == '%');

  // line contains [nrow num_columns nnz]
  std::vector<std::string> tokens;
  tokenize(tokens, line);

  if (tokens.size() != 3)
    return false;

  std::istringstream(tokens[0]) >> nrow;
  std::istringstream(tokens[1]) >> ncol;
  std::istringstream(tokens[2]) >> nnz;

  allocate_host(nnz, row);
  allocate_host(nnz, col);
  allocate_host(nnz, val);

  int nnz_read = 0;

  // read file contents
  if (banner.type == "pattern") {

    while(nnz_read < nnz && !input.eof()) {

      input >> (*row)[nnz_read];
      input >> (*col)[nnz_read];
      ++nnz_read;

    }

    for (int i=0; i<nnz; ++i)
      (*val)[i] = ValueType(1.0);

  } else if (banner.type == "real" || banner.type == "integer") {

    while(nnz_read < nnz && !input.eof()) {

      input >> (*row)[nnz_read];
      input >> (*col)[nnz_read];
      input >> (*val)[nnz_read];
      ++nnz_read;

    }

  } else if (banner.type == "complex") {

    while(nnz_read < nnz && !input.eof()) {

      double real, imag;

      input >> (*row)[nnz_read];
      input >> (*col)[nnz_read];
      input >> real;
      input >> imag;

      assign_complex((*val)[nnz_read], real, imag);
      ++nnz_read;

    }

  } else
    return false;

  if(nnz_read != nnz)
    return false;

  // check validity of row and column index data
  if (nnz > 0) {

    int min_row_index = 1;
    int max_row_index = nrow;
    int min_col_index = 1;
    int max_col_index = ncol;

    for (int i=0; i<nnz; ++i) {
      min_row_index = ((*row)[i] < min_row_index) ? (*row)[i] : min_row_index;
      max_row_index = ((*row)[i] > max_row_index) ? (*row)[i] : max_row_index;
      min_col_index = ((*col)[i] < min_col_index) ? (*col)[i] : min_col_index;
      max_col_index = ((*col)[i] > max_col_index) ? (*col)[i] : max_col_index;
    }

    if (min_row_index < 1)    return false;
    if (min_col_index < 1)    return false;
    if (max_row_index > nrow) return false;
    if (max_col_index > ncol) return false;

  }

  // convert base-1 indices to base-0
  for (int i=0; i<nnz; ++i) {
    --(*row)[i];
    --(*col)[i];
  }

  // expand symmetric formats to "general" format
  if (banner.symmetry != "general") {

    int off_diagonals = 0;

    for (int i=0; i<nnz; ++i)
      if((*row)[i] != (*col)[i])
        ++off_diagonals;

    int general_nnz = nnz + off_diagonals;

    int *general_row = NULL;
    int *general_col = NULL;
    ValueType *general_val = NULL;

    allocate_host(general_nnz, &general_row);
    allocate_host(general_nnz, &general_col);
    allocate_host(general_nnz, &general_val);

    if (banner.symmetry == "symmetric") {

      int symm_nnz = 0;

      for (int i=0; i<nnz; ++i) {

        // copy entry over
        general_row[symm_nnz] = (*row)[i];
        general_col[symm_nnz] = (*col)[i];
        general_val[symm_nnz] = (*val)[i];
        ++symm_nnz;

        // duplicate off-diagonals
        if ((*row)[i] != (*col)[i]) {
          general_row[symm_nnz] = (*col)[i];
          general_col[symm_nnz] = (*row)[i];
          general_val[symm_nnz] = (*val)[i];
          ++symm_nnz;
        }

      }

      nnz = symm_nnz;

    } else if (banner.symmetry == "hermitian") {
      //TODO
      return false;
    } else if (banner.symmetry == "skew-symmetric") {
      //TODO
      return false;
    }

    free_host(row);
    free_host(col);
    free_host(val);

    (*row) = general_row;
    (*col) = general_col;
    (*val) = general_val;

    general_row = NULL;
    general_col = NULL;
    general_val = NULL;

  }

  return true;

}

template <typename ValueType>
bool read_matrix_mtx(int &nrow, int &ncol, int &nnz, int **row, int **col, ValueType **val,
                     const std::string filename) {  

  std::ifstream file(filename.c_str());

  if (!file) {
    LOG_INFO("ReadFileMTX: cannot open file " << filename);
    return false;
  }

  // read banner
  matrix_market_banner banner;
  if (read_matrix_market_banner(banner, file) != true) {
    LOG_INFO("ReadFileMTX: invalid matrix market banner");
    return false;
  }

  if (banner.storage == "coordinate") {

    if (read_coordinate_stream(nrow, ncol, nnz, row, col, val, file, banner) != true) {
      LOG_INFO("ReadFileMTX: invalid matrix data");
      return false;
    }

  } else {

    return false;

  }

  file.close();

  return true;

}

template <typename ValueType>
bool write_matrix_mtx(const int nrow, const int ncol, const int nnz,
                      const int *row, const int *col, const ValueType *val, const std::string filename) {

  std::ofstream file(filename.c_str());
  file.precision(12);

  if (!file) {
    LOG_INFO("WriteFileMTX: cannot open file " << filename);
    return false;
  }

  file << "%%MatrixMarket matrix coordinate real general\n";
  file << nrow << " " << ncol << " " << nnz << "\n";

  for (int i=0; i<nnz; ++i) {

    file << row[i] + 1 << " ";
    file << col[i] + 1 << " ";

    write_value(file, val[i]);

    file << "\n";

  }

  return true;

}


template bool read_matrix_mtx(int &nrow, int &ncol, int &nnz, int **row, int **col, float **val,
                              const std::string filename);
template bool read_matrix_mtx(int &nrow, int &ncol, int &nnz, int **row, int **col, double **val,
                              const std::string filename);
#ifdef SUPPORT_COMPLEX
template bool read_matrix_mtx(int &nrow, int &ncol, int &nnz, int **row, int **col, std::complex<float> **val,
                              const std::string filename);
template bool read_matrix_mtx(int &nrow, int &ncol, int &nnz, int **row, int **col, std::complex<double> **val,
                              const std::string filename);
#endif

template bool write_matrix_mtx(const int nrow, const int ncol, const int nnz,
                               const int *row, const int *col, const float *val,
                               const std::string filename);
template bool write_matrix_mtx(const int nrow, const int ncol, const int nnz,
                               const int *row, const int *col, const double *val,
                               const std::string filename);
#ifdef SUPPORT_COMPLEX
template bool write_matrix_mtx(const int nrow, const int ncol, const int nnz,
                               const int *row, const int *col, const std::complex<float> *val,
                               const std::string filename);
template bool write_matrix_mtx(const int nrow, const int ncol, const int nnz,
                               const int *row, const int *col, const std::complex<double> *val,
                               const std::string filename);
#endif
}
