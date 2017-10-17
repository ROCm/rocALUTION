LocalVector<ValueType> x, y;

y.MoveToHost();
x.MoveToAccelerator();

x.CopyFromAsync(y);

// do some computation

x.Sync();

