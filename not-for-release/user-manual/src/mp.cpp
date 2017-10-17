MixedPrecisionDC<LocalMatrix<double>, LocalVector<double>, double,
                 LocalMatrix<float>, LocalVector<float>, float> mp;

CG<LocalMatrix<float>, LocalVector<float>, float> cg;
MultiColoredILU<LocalMatrix<float>, LocalVector<float>, float> p;

// setup a lower tol for the inner solver
cg.SetPreconditioner(p);
cg.Init(1e-5, 1e-2, 1e+20,
        100000);

// setup the mixed-precision DC
mp.SetOperator(mat);
mp.Set(cg);

mp.Build();

mp.Solve(rhs, &x);
