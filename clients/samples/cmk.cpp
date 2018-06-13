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

  // Compute (R)CMK ordering
  LocalVector<int> cmk;
  //  mat.CMK(&cmk);
  mat.RCMK(&cmk);

  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  cmk.MoveToAccelerator();

  // Apply (R)CMK ordering
  mat.Permute(cmk);

  x.Allocate("x", mat.GetN());
  rhs.Allocate("rhs", mat.GetM());

  // Linear Solver
  CG<LocalMatrix<double>, LocalVector<double>, double > ls;

  // Preconditioner
  ILU<LocalMatrix<double>, LocalVector<double>, double > p;

  double tick, tack;

  rhs.Ones();
  x.Zeros(); 

  ls.SetOperator(mat);
  ls.SetPreconditioner(p);

  ls.Build();

  mat.Info();

  tick = rocalution_time();

  ls.Solve(rhs, &x);

  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  // Revert CMK ordering on solution vector
  x.PermuteBackward(cmk);

  ls.Clear();

  stop_rocalution();

  return 0;
}
