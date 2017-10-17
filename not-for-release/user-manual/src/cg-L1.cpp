CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;

ls.SetOperator(mat);
ls.SetResidualNorm(1);

ls.Build();

ls.Solve(rhs, &x);
