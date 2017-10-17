CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;

ls.SetOperator(mat);
ls.SetResidualNorm(3);

ls.Build();

ls.Solve(rhs, &x);

std::cout << "Index of the L_\infty = " << ls.GetAmaxResidualIndex() << std::end;

std::cout << "Solver status = " << ls.GetSolverStatus() << std::endl;
