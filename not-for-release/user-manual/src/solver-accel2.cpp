CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;
MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

ls.SetOperator(mat);
ls.SetPreconditioner(p);

// The solver and the preconditioner will be constructed on the host
ls.Build();

// Now we move all objects (including the solver and the preconditioner
// to the accelerator

mat.MoveToAccelerator();
rhs.MoveToAccelerator();
x.MoveToAccelerator();

ls.MoveToAccelerator();

// The solving phase is performed on the accelerator
ls.Solve(rhs, &x);
