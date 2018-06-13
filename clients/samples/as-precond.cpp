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

  x.Allocate("x", mat.GetN());
  rhs.Allocate("rhs", mat.GetM());

  // Linear Solver
  GMRES<LocalMatrix<double>, LocalVector<double>, double > ls;

  // Preconditioner
  AS<LocalMatrix<double>, LocalVector<double>, double > p; // Additive Schwarz
  //  RAS<LocalMatrix<double>, LocalVector<double>, double > p; // Restricted Additive Schwarz

  // Second level preconditioners
  Solver<LocalMatrix<double>, LocalVector<double>, double > **p2;

  int n = 2;

  p2 = new Solver<LocalMatrix<double>, LocalVector<double>, double >*[n];

  for (int i=0; i<n; ++i) {

    MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double > *mc;
    mc = new MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double >;
    p2[i] = mc;

  }

  double tick, tack;
  
  rhs.Ones();
  x.Zeros(); 

  p.Set(n, 
        4,
        p2);
  
  ls.SetOperator(mat);
  ls.SetPreconditioner(p);
  //  ls.Verbose(2);

  ls.Build();

  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  ls.MoveToAccelerator();
  
  mat.Info();

  tick = rocalution_time();

  ls.Solve(rhs, &x);

  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  ls.Clear();

  for (int i=0; i<n; ++i) {
    delete p2[i];
    p2[i] = NULL;
  }

  delete[] p2;
  p2 = NULL;

  stop_rocalution();

  return 0;
}
