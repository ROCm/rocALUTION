CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;
MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

ls.SetOperator(mat);
ls.SetPreconditioner(p);

ls.Build();

ls.Solve(rhs, &x);
