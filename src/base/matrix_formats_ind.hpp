#ifndef ROCALUTION_MATRIX_FORMATS_IND_HPP_
#define ROCALUTION_MATRIX_FORMATS_IND_HPP_

// Matrix indexing

// DENSE indexing
#define DENSE_IND(ai, aj, nrow, ncol) (ai) + (aj) * (nrow)
//#define DENSE_IND(ai, aj, nrow, ncol) (aj) + (ai) * (ncol)

// DENSE_IND_BASE == 0 - column-major
// DENSE_IND_BASE == 1 - row-major
#define DENSE_IND_BASE (DENSE_IND(2, 2, 10, 0) == 22 ? 0 : 1)

// ELL indexing
#define ELL_IND_ROW(row, el, nrow, max_row) (el) * (nrow) + (row)
#define ELL_IND_EL(row, el, nrow, max_row) (el) + (max_row) * (row)
#define ELL_IND(row, el, nrow, max_row) ELL_IND_ROW(row, el, nrow, max_row)

// DIA indexing
#define DIA_IND_ROW(row, el, nrow, ndiag) (el) * (nrow) + (row)
#define DIA_IND_EL(row, el, nrow, ndiag) (el) + (ndiag) * (row)
#define DIA_IND(row, el, nrow, ndiag) DIA_IND_ROW(row, el, nrow, ndiag)

#endif // ROCALUTION_MATRIX_FORMATS_IND_HPP_
