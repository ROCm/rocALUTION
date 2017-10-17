mat.ConvertTo(CSR);
// Perform a matrix-vector multiplcation in CSR format
mat.Apply(x, &y);

mat.ConvertTo(ELL);
// Perform a matrix-vector multiplcation in ELL format
mat.Apply(x, &y);
