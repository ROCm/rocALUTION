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
  LocalVector<double> e;

  LocalMatrix<double> mat;

  mat.ReadFileMTX(std::string(argv[1]));
  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  e.MoveToAccelerator();

  x.Allocate("x", mat.GetN());
  rhs.Allocate("rhs", mat.GetM());
  e.Allocate("e", mat.GetN());

  // Linear Solver
  FGMRES<LocalMatrix<double>, LocalVector<double>, double > ls;

  // Preconditioner
  MultiColoredGS<LocalMatrix<double>, LocalVector<double>, double > p;

  e.Ones();
  mat.Apply(e, &rhs);
  x.Zeros(); 

  ls.SetOperator(mat);
  ls.SetBasisSize(30);
  ls.SetPreconditioner(p);

  ls.Build();

  ls.Verbose(1);

  mat.Info();

  double tick, tack;
  tick = rocalution_time();

  ls.Solve(rhs, &x);

  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  ls.Clear();

  e.ScaleAdd(-1.0, x);
  double error = e.Norm();

  std::cout << "||e - x||_2 = " << error << "\n";

  stop_rocalution();

  return 0;
}
