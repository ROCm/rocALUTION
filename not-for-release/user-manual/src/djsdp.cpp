GMRES<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > gmres;

DiagJacobiSaddlePointPrecond<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;
MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p_k;
MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p_s;

p.Set(p_k, p_s);

gmres.SetOperator(mat);
gmres.SetBasisSize(50);
gmres.SetPreconditioner(p);
gmres.Build();

gmres.Solve(rhs, &x);

