#include <iostream>
#include <cstdlib>

#include <rocalution.hpp>

using namespace rocalution;

int main(int argc, char* argv[]) {

  double tick, tack, start, end;

  start = rocalution_time();

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix> [Num threads]" << std::endl;
    exit(1);
  }

  init_rocalution();

  if (argc > 2)
    set_omp_threads_rocalution(atoi(argv[2]));

  info_rocalution();

  LocalVector<double> x;
  LocalVector<double> rhs;

  LocalMatrix<double> mat;

  mat.ReadFileMTX(std::string(argv[1]));

  x.Allocate("x", mat.GetN());
  rhs.Allocate("rhs", mat.GetM());

  rhs.Ones();
  x.Zeros(); 

  tick = rocalution_time();

  CG<LocalMatrix<double>, LocalVector<double>, double > ls;
  ls.Verbose(0);

  // AMG Preconditioner
  PairwiseAMG<LocalMatrix<double>, LocalVector<double>, double > p;

  p.InitMaxIter(1);
  p.Verbose(0);

  ls.SetPreconditioner(p);
  ls.SetOperator(mat);
  ls.Build();

  p.SetHostLevels(2);

  tack = rocalution_time();
  std::cout << "Building time:" << (tack-tick)/1000000 << " sec" << std::endl;

  // move after building since AMG building is not supported on GPU yet
  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  ls.MoveToAccelerator();

  mat.Info();

  tick = rocalution_time();

  ls.Init(1e-8, 1e-8, 1e+8, 10000);
  ls.Verbose(2);

  ls.Solve(rhs, &x);

  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  ls.Clear();

  end = rocalution_time();
  std::cout << "Total runtime:" << (end-start)/1000000 << " sec" << std::endl;

  stop_rocalution();

  return 0;

}
