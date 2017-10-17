#include <iostream>
#include <cstdlib>

#include <paralution.hpp>

using namespace paralution;

int main(int argc, char* argv[]) {

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix> [Num threads]" << std::endl;
    exit(1);
  }

  init_paralution();

  if (argc > 2) {
    set_omp_threads_paralution(atoi(argv[2]));
  } 

  info_paralution();

  LocalVector<double> x;
  LocalVector<double> rhs;
  LocalVector<double> e;

  LocalMatrix<double> mat;

  mat.ReadFileMTX(std::string(argv[1]));
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

  e.Ones();
  mat.Apply(e, &rhs);
  x.Zeros(); 

  ls.SetOperator(mat);
  ls.SetPreconditioner(p);

  ls.Build();

  //  ls.Verbose(2);

  mat.info();

  double tick, tack;
  tick = paralution_time();

  ls.Solve(rhs, &x);

  tack = paralution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  ls.Clear();

  e.ScaleAdd(-1.0, x);
  double error = e.Norm();

  std::cout << "||e - x||_2 = " << error << "\n";

  stop_paralution();

  return 0;
}
