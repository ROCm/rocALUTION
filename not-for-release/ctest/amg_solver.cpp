#include <iostream>
#include <rocalution.hpp>

#define ValueType double

using namespace rocalution;

int main(int argc, char* argv[]) {

  init_rocalution();

  LocalVector<ValueType> x;
  LocalVector<ValueType> rhs;
  LocalVector<ValueType> sol;
  LocalMatrix<ValueType> mat;

  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  sol.MoveToAccelerator();

  mat.ReadFileMTX(std::string(argv[1]));

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());
  sol.Allocate("sol", mat.get_nrow());

  // b = A*1
  sol.Ones();
  x.SetRandom((ValueType) 0, (ValueType) 2, time(NULL));
  mat.Apply(sol, &rhs);

  // Solver
  CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> ls;

  BaseAMG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > *p = NULL;

  std::string sName = argv[2];

  if (sName == "SmoothedAggregation") {
    AMG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> *amg;
    amg = new AMG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
    amg->SetInterpolation(SmoothedAggregation);
    amg->SetCouplingStrength(0.01);
    amg->SetCoarsestLevel(300);
    p = amg;
  }

  if (sName == "Aggregation") {
    AMG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> *amg;
    amg = new AMG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
    amg->SetInterpolation(Aggregation);
    amg->SetCouplingStrength(0.01);
    amg->SetCoarsestLevel(300);
    p = amg;
  }

  if (sName == "RugeStueben") {
    RugeStuebenAMG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> *amg;
    amg = new RugeStuebenAMG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
    amg->SetCoarsestLevel(300);
    p = amg;
  }

  if (sName == "Pairwise") {
    PairwiseAMG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> *amg;
    amg = new PairwiseAMG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
    amg->SetOrdering(Connectivity);
    amg->SetCoarsestLevel(300);
    p = amg;
  }

  p->SetOperator(mat);
  p->SetManualSmoothers(true);
  p->SetManualSolver(true);
  p->BuildHierarchy();
  int levels = p->GetNumLevels();

  // Coarse Grid Solver
  CG<LocalMatrix<double>, LocalVector<double>, double > cgs;
  cgs.Verbose(0);

  // Smoother for each level
  IterativeLinearSolver<LocalMatrix<double>, LocalVector<double>, double > **sm = NULL;
  sm = new IterativeLinearSolver<LocalMatrix<double>, LocalVector<double>, double >*[levels-1];

  std::string pName = argv[3];

  for (int i=0; i<levels-1; ++i) {
    FixedPoint<LocalMatrix<double>, LocalVector<double>, double > *fp;
    fp = new FixedPoint<LocalMatrix<double>, LocalVector<double>, double >;
    sm[i] = fp;

    Preconditioner<LocalMatrix<double>, LocalVector<double>, double> *smoother = NULL;

    if (pName == "Jacobi") smoother = new Jacobi<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
    if (pName == "SGS")    smoother = new SGS<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
    if (pName == "ILU")    smoother = new ILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
    if (pName == "SPAI")   smoother = new SPAI<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
    if (pName == "TNS")    smoother = new TNS<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
    if (pName == "MCILU")  smoother = new MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
    if (pName == "MCSGS")  smoother = new MultiColoredSGS<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
    if (pName == "FSAI")   smoother = new FSAI<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;

    if (pName == "SGS" || pName == "MultiColoredSGS")
      fp->SetRelaxation(1.3);
    if (pName == "Jacobi")
      fp->SetRelaxation(0.67);

    sm[i]->SetPreconditioner(*smoother);
    sm[i]->Verbose(0);
  }

  p->SetSmoother(sm);
  p->SetSolver(cgs);
  p->SetSmootherPreIter(2);
  p->SetSmootherPostIter(2);

  std::string cycleName = argv[4];

  if (cycleName == "Vcycle") p->SetCycle(Vcycle);
  if (cycleName == "Wcycle") p->SetCycle(Wcycle);
  if (cycleName == "Kcycle") p->SetCycle(Kcycle);

  p->InitMaxIter(1);
  p->Verbose(0);

  ls.SetOperator(mat);
  ls.Init(1e-8, 0.0, 1e+8, 10000);
  ls.SetPreconditioner(*p);

  ls.Build();

  mat.info();

  ls.Solve(rhs, &x);

  x.ScaleAdd((ValueType) -1, sol);
  std::cout << "Error Norm = " << x.Norm() << std::endl;

  if (x.Norm() > 1e-4) {
    std::cout << "Test failed." << std::endl;
    exit(-1);
  }

  stop_rocalution();

  return 0;

}
