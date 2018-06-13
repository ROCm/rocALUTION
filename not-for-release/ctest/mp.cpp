#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <rocalution.hpp>

#include "../../src/utils/log.hpp"

#define ValueType double

using namespace rocalution;

int main(int argc, char* argv[]) {

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix> <solution_vector>" << std::endl;
    exit(1);
  }

  init_rocalution();

  LocalVector<ValueType> x;
  LocalVector<ValueType> rhs;
  LocalVector<ValueType> compare;

  LocalMatrix<ValueType> mat;

  mat.ReadFileMTX(std::string(argv[1]));

  x.Allocate("x", mat.GetM());
  rhs.Allocate("rhs", mat.GetM());

  x.Zeros();
  rhs.Ones();


  // Solver
  CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > cg;

  // Preconditioner (main)
  MultiElimination<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

  // Last block-preconditioner
  MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > mcilu_p;

  rhs.MoveToAccelerator();
  x.MoveToAccelerator();
  mat.MoveToAccelerator();

  mcilu_p.Set(0);
  p.Set(mcilu_p, 2);

  cg.SetOperator(mat);
  cg.SetPreconditioner(p);

  cg.Build();

  cg.Solve(rhs, &x);

  compare.ReadFileASCII(std::string(argv[2]));

  x.MoveToHost();

  // |x-x_ref| < eps
  x.AddScale(compare, ValueType(-1.0));
  LOG_INFO("Error Norm = " << x.Norm())

  if (x.Norm() > 1e-4)
  {
    std::cout <<"Test failed." <<std::endl;
    exit(-1);
  }

  stop_rocalution();

  return 0;
}
