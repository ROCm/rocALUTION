#include <iostream>
#include <cstdlib>

#include <rocalution.hpp>

using namespace rocalution;

int main(int argc, char* argv[]) {

  init_rocalution();

  info_rocalution();

  LocalVector<double> x;
  LocalVector<double> rhs;

  LocalStencil<double> stencil(Laplace2D);

  stencil.SetGrid(100); // 100x100

  x.Allocate("x", stencil.GetM());
  rhs.Allocate("rhs", stencil.GetM());

  // Linear Solver
  CG<LocalStencil<double>, LocalVector<double>, double > ls;

  rhs.Ones();
  x.Zeros(); 

  ls.SetOperator(stencil);

  ls.Build();

  stencil.Info();

  double tick, tack;
  tick = rocalution_time();

  ls.Solve(rhs, &x);

  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  ls.Clear();

  stop_rocalution();

  return 0;
}
