// Create a PARALUTION matrix
LocalMatrix<ValueType> mat;

// Allocate and fill the PARALUTION matrix mat
...


// Define external CSR structure
int *row_offsets = NULL;
int *col = NULL;
ValueType *val = NULL;

mat.LeaveDataPtrCSR(&row_offsets, &col, &val);
