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
  LocalMatrix<double> mat;

  mat.ReadFileMTX(std::string(argv[1]));

  rhs.MoveToAccelerator();
  x.MoveToAccelerator();
  mat.MoveToAccelerator();

  x.Allocate("x", mat.GetM());
  rhs.Allocate("rhs", mat.GetM());

  x.Zeros();
  rhs.Ones();

  double tick, tack;

  // Solver
  CG<LocalMatrix<double>, LocalVector<double>, double > cg;

  // Preconditioner (main)
  MultiElimination<LocalMatrix<double>, LocalVector<double>, double > p;

  // Last block-preconditioner
  MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double > mcilu_p;

  mcilu_p.Set(0);
  p.Set(mcilu_p, 2, 0.4);

  cg.SetOperator(mat);
  cg.SetPreconditioner(p);

  cg.Build();
  
  mat.Info();    
  tick = paralution_time();
  
  cg.Solve(rhs, &x);
  
  tack = paralution_time();
  
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  stop_paralution();

  return 0;
}
