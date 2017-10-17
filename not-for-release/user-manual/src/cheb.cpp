Chebyshev<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;

ls.SetOperator(mat);
ls.Set(lambda_min, lambda_max);
ls.Build();

ls.Solve(rhs, &x);
