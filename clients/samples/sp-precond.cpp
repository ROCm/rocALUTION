/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

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

  x.Allocate("x", mat.GetN());
  rhs.Allocate("rhs", mat.GetM());

  // Linear Solver
  BiCGStab<LocalMatrix<double>, LocalVector<double>, double > ls;

  // Preconditioner
  DiagJacobiSaddlePointPrecond<LocalMatrix<double>, LocalVector<double>, double > p;
  MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double > p1;
  MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double > p2;

  double tick, tack;
  
  rhs.Ones();
  x.Zeros(); 

  //  p1.Set(1);
  //  p2.Set(1);
  p.Set(p1, p2);

  ls.SetOperator(mat);
  ls.SetPreconditioner(p);
  ls.Verbose(2);

  ls.Build();

  mat.Info();

  tick = rocalution_time();

  ls.Solve(rhs, &x);

  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  stop_rocalution();

  return 0;
}
