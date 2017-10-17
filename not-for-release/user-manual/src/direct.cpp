Inversion<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;

ls.SetOperator(mat);
ls.Build();

ls.Solve(rhs, &x);
