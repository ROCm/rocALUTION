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

  // read from file 
  mat.ReadFileMTX(std::string(argv[1]));

  mat.MoveToAccelerator();
  rhs.MoveToAccelerator();
  x.MoveToAccelerator();

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());

  FixedPoint<LocalMatrix<double>, LocalVector<double>, double > fp;

  Jacobi<LocalMatrix<double>, LocalVector<double>, double > p;
  //  MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double > p;
  //  ILU<LocalMatrix<double>, LocalVector<double>, double > p;
  //  MultiColoredGS<LocalMatrix<double>, LocalVector<double>, double > p;
  //  MultiColoredSGS<LocalMatrix<double>, LocalVector<double>, double > p;

  double tick, tack;

  rhs.Ones();
  x.Zeros();

  fp.SetOperator(mat);
  fp.SetPreconditioner(p);

  fp.SetRelaxation(1.3);

  //  fp.Verbose(2);

  fp.Build();

  mat.Info();

  tick = rocalution_time();

  fp.Solve(rhs, &x);

  tack = rocalution_time();

  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  fp.Clear();

  stop_rocalution();

  return 0;
}
