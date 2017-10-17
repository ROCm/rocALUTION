CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > cg;
MultiElimination<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;
Jacobi<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > j_p;

p.Set(j_p, 2);

cg.SetOperator(mat);
cg.SetPreconditioner(p);
cg.Build();

cg.Solve(rhs, &x);

