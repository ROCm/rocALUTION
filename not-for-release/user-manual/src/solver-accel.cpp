CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;
MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

mat.MoveToAccelerator();
rhs.MoveToAccelerator();
x.MoveToAccelerator();


ls.SetOperator(mat);
ls.SetPreconditioner(p);

// The solver and the preconditioner will be constructed on the accelerator
ls.Build();

// The solving phase is performed on the accelerator
ls.Solve(rhs, &x);
