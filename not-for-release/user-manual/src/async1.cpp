LocalVector<ValueType> x, y;
LocalMatrix<ValueType> mat;

mat.MoveToAcceleratorAsync();
x.MoveToAcceleratorAsync();
y.MoveToAcceleratorAsync();

// do some computation

mat.Sync();
x.Sync();
y.Sync();
