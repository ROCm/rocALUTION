CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;
MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

ls.SetOperator(mat);
ls.SetPreconditioner(p);

ls.Build();

mat.MoveToAccelerator();
rhs.MoveToAccelerator();
x.MoveToAccelerator();
ls.MoveToAccelerator();

ls.Solve(rhs, &x);
