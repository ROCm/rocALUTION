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

  mat.ReadFileCSR(std::string(argv[1]));
  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  e.MoveToAccelerator();

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_ncol());
  e.Allocate("e", mat.get_nrow());

  // Linear Solver
  CG<LocalMatrix<double>, LocalVector<double>, double > ls;

  // Preconditioner
  Jacobi<LocalMatrix<double>, LocalVector<double>, double > p;

  mat.ConvertToDIA();

  e.Ones();
  mat.Apply(e, &rhs);
  x.Zeros(); 

  ls.SetOperator(mat);
  ls.SetPreconditioner(p);

  ls.Build();

  ls.Verbose(2);

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
