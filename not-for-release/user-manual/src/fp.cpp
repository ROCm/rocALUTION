FixedPoint<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > fp;
Jacobi<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

fp.SetOperator(mat);
fp.SetPreconditioner(p);
fp.SetRelaxation(1.3);

fp.Build();

fp.Solve(rhs, &x);
