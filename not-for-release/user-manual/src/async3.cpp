LocalVector<ValueType> x, y;
LocalMatrix<ValueType> mat;

// mat, x and y are initially on the host
...

// Start async transfer
mat.MoveToAcceleratorAsync();
x.MoveToAcceleratorAsync();

// this will be performed on the host
mat.Apply(x, &y);

mat.Sync();
x.Sync();

// Move y
y.MoveToAccelerator();

// this will be performed on the accelerator
mat.Apply(x, &y);
