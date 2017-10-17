#include <iostream>
#include <cstdlib>

#include <paralution.hpp>

using namespace paralution;

int main(int argc, char* argv[]) {

  init_paralution();

  info_paralution();

  LocalVector<double> x;
  LocalVector<double> rhs;

  LocalStencil<double> stencil(Laplace2D);

  stencil.SetGrid(100); // 100x100

  x.Allocate("x", stencil.get_nrow());
  rhs.Allocate("rhs", stencil.get_nrow());

  // Linear Solver
  CG<LocalStencil<double>, LocalVector<double>, double > ls;

  rhs.Ones();
  x.Zeros(); 

  ls.SetOperator(stencil);

  ls.Build();

  stencil.info();

  double tick, tack;
  tick = paralution_time();

  ls.Solve(rhs, &x);

  tack = paralution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  stop_paralution();

  return 0;
}
