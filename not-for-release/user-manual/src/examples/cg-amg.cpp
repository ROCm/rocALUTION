#include <iostream>
#include <cstdlib>

#include <paralution.hpp>

using namespace paralution;

int main(int argc, char* argv[]) {

  double tick, tack, start, end;

  start = paralution_time();

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix> [Num threads]" << std::endl;
    exit(1);
  }

  init_paralution();

  if (argc > 2)
    set_omp_threads_paralution(atoi(argv[2]));

  info_paralution();

  LocalVector<double> x;
  LocalVector<double> rhs;

  LocalMatrix<double> mat;

  mat.ReadFileMTX(std::string(argv[1]));

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());

  rhs.Ones();
  x.Zeros(); 

  tick = paralution_time();

  CG<LocalMatrix<double>, LocalVector<double>, double > ls;
  ls.Verbose(0);

  // AMG Preconditioner
  AMG<LocalMatrix<double>, LocalVector<double>, double > p;

  p.InitMaxIter(1);
  p.Verbose(0);

  ls.SetPreconditioner(p);
  ls.SetOperator(mat);
  ls.Build();

  tack = paralution_time();
  std::cout << "Building time:" << (tack-tick)/1000000 << " sec" << std::endl;

  // move after building since AMG building is not supported on GPU yet
  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  ls.MoveToAccelerator();

  mat.info();

  tick = paralution_time();

  ls.Init(1e-10, 1e-8, 1e+8, 10000);
  ls.Verbose(2);

  ls.Solve(rhs, &x);

  tack = paralution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  ls.Clear();

  stop_paralution();

  end = paralution_time();
  std::cout << "Total runtime:" << (end-start)/1000000 << " sec" << std::endl;

  return 0;
}
