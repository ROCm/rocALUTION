// Allocate the CSR matrix
int *row_offsets = new int[100+1];
int *col = new int[345];
ValueType *val = new ValueType[345];

// fill the CSR matrix
...

// Create a PARALUTION matrix
LocalMatrix<ValueType> mat;

// Import matrix to PARALUTION
mat.AllocateCSR("my matrix", 345, 100, 100);
mat.CopyFromCSR(row_offsets, col, val);

// Export matrix from PARALUTION
// the row_offsets, col, val have to be allocated
mat.CopyToCSR(row_offsets, col, val);
