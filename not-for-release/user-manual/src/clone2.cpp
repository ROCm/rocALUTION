LocalVector<ValueType> vec;
LocalMatrix<ValueType> mat;

// allocate and init vec, mat (host or accelerator)
// ...

LocalVector<ValueType> tmp;

// tmp and vec will have the same backend as mat
tmp.CloneBackend(mat);
vec.CloneBackend(mat);

// the matrix-vector multiplication will be performed 
// on the backend selected in mat
mat.Apply(vec, &tmp);

