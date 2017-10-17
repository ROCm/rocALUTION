LocalMatrix<ValueType> mat;

// Linear Solver
AMG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;

ls.SetOperator(mat);

// coupling strength
ls.SetCouplingStrength(0.001);
// number of unknowns on coarsest level
ls.SetCoarsestLevel(300);
// interpolation type for grid transfer operators
ls.SetInterpolation(SmoothedAggregation);
// Relaxation parameter for smoothed interpolation aggregation
ls.SetInterpRelax(2./3.);
// Manual smoothers
ls.SetManualSmoothers(true);
// Manual course grid solver
ls.SetManualSolver(true);
// grid transfer scaling
ls.SetScaling(true);

ls.BuildHierarchy();

int levels = ls.GetNumLevels();

// Smoother for each level
IterativeLinearSolver<LocalMatrix<ValueType>, LocalVector<ValueType>, 
                      ValueType > **sm = NULL;
MultiColoredGS<LocalMatrix<ValueType>, LocalVector<ValueType>, 
               ValueType > **gs = NULL;

// Coarse Grid Solver
CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > cgs;
cgs.Verbose(0);

sm = new IterativeLinearSolver<LocalMatrix<ValueType>, LocalVector<ValueType>, 
                               ValueType >*[levels-1];
gs = new MultiColoredGS<LocalMatrix<ValueType>, LocalVector<ValueType>, 
                        ValueType >*[levels-1];

// Preconditioner for the coarse grid solver
//  MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;
//  cgs->SetPreconditioner(p);

for (int i=0; i<levels-1; ++i) {
  FixedPoint<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > *fp;
  fp = new FixedPoint<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType >;
  fp->SetRelaxation(1.3);
  sm[i] = fp;
    
  gs[i] = new MultiColoredGS<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType >;
  gs[i]->SetPrecondMatrixFormat(ELL);
  
  sm[i]->SetPreconditioner(*gs[i]);
  sm[i]->Verbose(0);
 }

ls.SetOperatorFormat(CSR);
ls.SetSmoother(sm);
ls.SetSolver(cgs);
ls.SetSmootherPreIter(1);
ls.SetSmootherPostIter(2);
ls.Init(1e-10, 1e-8, 1e+8, 10000);
ls.Verbose(2);

ls.Build();

ls.Solve(rhs, &x);
