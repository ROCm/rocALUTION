#ifndef PARALUTION_MATRIX_FORMATS_HPP_
#define PARALUTION_MATRIX_FORMATS_HPP_

#include <string>

namespace paralution {

/// Matrix Names
const std::string _matrix_format_names [8] = {"DENSE", 
                                              "CSR", 
                                              "MCSR",
                                              "BCSR",
                                              "COO", 
                                              "DIA", 
                                              "ELL", 
                                              "HYB"
};

/// Matrix Enumeration
enum _matrix_format {DENSE = 0, 
                     CSR   = 1, 
                     MCSR  = 2,
                     BCSR  = 3,
                     COO   = 4, 
                     DIA   = 5, 
                     ELL   = 6, 
                     HYB   = 7};
  

/// Sparse Matrix -
/// Sparse Compressed Row Format
template <typename ValueType, typename IndexType>
struct MatrixCSR {
  /// Row offsets (row ptr)
  IndexType *row_offset;

  /// Column index
  IndexType *col;

  /// Values
  ValueType *val;
};

/// Sparse Matrix -
/// Modified Sparse Compressed Row Format
template <typename ValueType, typename IndexType>
struct MatrixMCSR {
  /// Row offsets (row ptr)
  IndexType *row_offset;

  /// Column index
  IndexType *col;

  /// Values
  ValueType *val;
};

template <typename ValueType, typename IndexType>
struct MatrixBCSR {
};

/// Sparse Matrix -
/// Coordinate Format
template <typename ValueType, typename IndexType>
struct MatrixCOO {
  /// Row index
  IndexType *row;

  /// Column index
  IndexType *col;

  // Values
  ValueType *val;
};

/// Sparse Matrix -
/// Diagonal Format (see DIA_IND for indexing)
template <typename ValueType, typename IndexType, typename Index = IndexType>
struct MatrixDIA {
  /// Number of diagonal
  Index num_diag;

  /// Offset with respect to the main diagonal
  IndexType *offset;

  /// Values
  ValueType *val;
};

/// Sparse Matrix -
/// ELL Format (see ELL_IND for indexing)
template <typename ValueType, typename IndexType, typename Index = IndexType>
struct MatrixELL {
  /// Maximal elements per row
  Index max_row;

  /// Column index
  IndexType *col;

  /// Values
  ValueType *val;
};

/// Sparse Matrix -
/// Contains ELL and COO Matrices
template <typename ValueType, typename IndexType, typename Index = IndexType>
struct MatrixHYB {
  MatrixELL<ValueType, IndexType, Index> ELL;
  MatrixCOO<ValueType, IndexType> COO;
};

/// Dense Matrix (see DENSE_IND for indexing)
template <typename ValueType>
struct MatrixDENSE {
  /// Values
  ValueType *val;
};


}

#endif // PARALUTION_MATRIX_FORMATS_HPP_
