mat.ConvertToCSR();
// Perform a matrix-vector multiplcation in CSR format
mat.Apply(x, &y);

mat.ConvertToELL();
// Perform a matrix-vector multiplcation in ELL format
mat.Apply(x, &y);
