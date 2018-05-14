#include <iostream>
#include <cstdlib>

#include <rocalution.hpp>

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

  // read from file 
  mat.ReadFileMTX(std::string(argv[1]));

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());

  FixedPoint<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > fp;

  //  Jacobi<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;
  MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;
  //  ILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;
  //  MultiColoredGS<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;
  //  MultiColoredSGS<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

  rhs.Ones();
  x.Zeros();

  fp.SetOperator(mat);
  fp.SetPreconditioner(p);

  //  fp.SetRelaxation(1.3);

  fp.Build();

  fp.Solve(rhs, &x);

  compare.ReadFileASCII(std::string(argv[2]));

  x.MoveToHost();

  // |x-x_ref| < eps
  x.AddScale(compare, ValueType(-1.0));
  LOG_INFO("Error Norm = " << x.Norm())

  if (x.Norm() > 1e-3)
  {
    std::cout <<"Test failed." <<std::endl;
    exit(-1);
  }

  stop_rocalution();

  return 0;
}
