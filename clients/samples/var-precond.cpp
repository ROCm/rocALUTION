#include <iostream>
#include <cstdlib>

#include <rocalution.hpp>

using namespace rocalution;

int main(int argc, char* argv[]) {

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix> [Num threads]" << std::endl;
    exit(1);
  }

  init_rocalution();

  if (argc > 2) {
    set_omp_threads_rocalution(atoi(argv[2]));
  } 

  info_rocalution();

  LocalVector<double> x;
  LocalVector<double> rhs;

  LocalMatrix<double> mat;

  mat.ReadFileMTX(std::string(argv[1]));

  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());

  // Linear Solver
  FGMRES<LocalMatrix<double>, LocalVector<double>, double > ls;

  // Variable Preconditioner
  VariablePreconditioner<LocalMatrix<double>, LocalVector<double>, double > p;

  Solver<LocalMatrix<double>, LocalVector<double>, double > **vp;
  vp = new Solver<LocalMatrix<double>, LocalVector<double>, double > *[3];

  // Preconditioners
  MultiColoredSGS<LocalMatrix<double>, LocalVector<double>, double > p1;
  MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double > p2;
  ILU<LocalMatrix<double>, LocalVector<double>, double > p3;

  vp[0] = &p1;
  vp[1] = &p2;
  vp[2] = &p3;

  double tick, tack;
  
  rhs.Ones();
  x.Zeros(); 

  p.SetPreconditioner(3, vp);
  ls.SetOperator(mat);
  ls.SetPreconditioner(p);
  //  ls.Verbose(2);

  ls.Build();

  mat.Info();

  tick = rocalution_time();

  ls.Solve(rhs, &x);

  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  stop_rocalution();

  return 0;
}
