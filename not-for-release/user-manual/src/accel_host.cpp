LocalMatrix<ValueType> mat;
LocalVector<ValueType> vec1, vec2;

// Performing the matrix-vector multiplication on host
mat.Apply(vec1, &vec2);

// Move data to the accelerator
mat. MoveToAccelerator();
vec1.MoveToAccelerator();
vec2.MoveToAccelerator();

// Performing the matrix-vector multiplication on accelerator
mat.Apply(vec1, &vec2);

// Move data to the host
mat. MoveToHost();
vec1.MoveToHost();
vec2.MoveToHost();
