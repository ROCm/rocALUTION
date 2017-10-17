#if defined(SUPPORT_MIC)
#undef SUPPORT_COMPLEX
#else
#define SUPPORT_COMPLEX
#endif

#include <iostream>
#include <cstdlib>
#include <complex>

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

#ifdef SUPPORT_COMPLEX

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

  mat.info();

  double tick, tack;
  tick = paralution_time();

  ls.Solve(rhs, &x);

  tack = paralution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  ls.Clear();

#else

  std::cout << "The basic version does not support complex on CUDA/OpenCL/MIC" << std::endl;

#endif

  stop_paralution();

  return 0;

}
