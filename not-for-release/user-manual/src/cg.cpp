CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;

ls.Init(1e-10,  // abs_tol
        1e-8,   // rel_tol
        1e+8,   // div_tol 
        10000); // max_iter

ls.SetOperator(mat);

ls.Build();

ls.Solve(rhs, &x);
