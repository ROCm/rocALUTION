CG<LocalMatrix<double>, LocalVector<double>, double > ls;

TNS<LocalMatrix<double>, LocalVector<double>, double > p;

// Explicit or implicit
//  p.Set(false);

 ls.SetOperator(mat);
 ls.SetPreconditioner(p);
 
 ls.Build();

 ls.Solve(rhs, &x);
