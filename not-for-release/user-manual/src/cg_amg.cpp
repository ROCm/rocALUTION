CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;

// AMG Preconditioner
AMG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

p.InitMaxIter(2);
p.Verbose(0);

ls.SetPreconditioner(p);
ls.SetOperator(mat);
ls.Build();

ls.Solve(rhs, &x);

