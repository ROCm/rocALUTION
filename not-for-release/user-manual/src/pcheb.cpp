CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > cg;
AIChebyshev<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

lambda_min = lambda_max / 7;
p.Set(3, plambda_min, plambda_max);
cg.SetOperator(mat);
cg.SetPreconditioner(p);
cg.Build();

cg.Solve(rhs, &x);
