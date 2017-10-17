// Allocate the CSR matrix
int *row_offsets = new int[100+1];
int *col = new int[345];
ValueType *val = new ValueType[345];

// fill the CSR matrix
...

// Create a PARALUTION matrix
LocalMatrix<ValueType> mat;

// Set the CSR matrix in PARALUTION
mat.SetDataPtrCSR(&row_offsets, &col, &val,
                  "my matrix",
                  345, 100, 100);
