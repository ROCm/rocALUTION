LocalMatrix<ValueType> mat;

mat.AllocateCSR("my csr matrix", 456, 100, 100); // nnz, rows, columns
mat.Clear();

mat.AllocateCOO("my coo matrix", 200, 100, 100); // nnz, rows, columns
mat.Clear();
