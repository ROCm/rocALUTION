#include <iostream>
#include <cstdlib>
#include <complex>

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

  LocalVector<std::complex<double> > x;
  LocalVector<std::complex<double> > rhs;

  LocalMatrix<std::complex<double> > mat;

  mat.ReadFileMTX(std::string(argv[1]));

  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());

  // Linear Solver
  CG<LocalMatrix<std::complex<double> >, LocalVector<std::complex<double> >, std::complex<double> > ls;

  // Preconditioner
  MultiColoredILU<LocalMatrix<std::complex<double> >, LocalVector<std::complex<double> >, std::complex<double> > p;

  rhs.Ones();
  x.Zeros(); 

  ls.SetOperator(mat);
  ls.SetPreconditioner(p);

  ls.Build();

//  ls.Verbose(2);

  mat.Info();

  double tick, tack;
  tick = rocalution_time();

  ls.Solve(rhs, &x);

  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  ls.Clear();

  stop_rocalution();

  return 0;

}
