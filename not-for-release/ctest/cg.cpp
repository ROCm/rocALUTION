#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#include <rocalution.hpp>

#define ValueType double

using namespace rocalution;

int main(int argc, char* argv[]) {

  if (argc < 3) { 
    std::cerr << argv[0] << " <matrix> <solution_vector> <preconditioner>" << std::endl;
    exit(1);
  }

  init_rocalution();

  LocalVector<ValueType> x;
  LocalVector<ValueType> rhs;
  LocalVector<ValueType> compare;

  LocalMatrix<ValueType> mat;

  mat.ReadFileMTX(std::string(argv[1]));
  compare.ReadFileASCII(std::string(argv[2]));

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());

  // Linear Solver
  CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;

  // Preconditioner
  Preconditioner<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > *p = NULL;

  rhs.Ones();
  x.Zeros(); 

  ls.SetOperator(mat);

  std::string pName = argv[3];
  if ( pName == "Jacobi" ) p = new Jacobi<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType >; ls.SetPreconditioner(*p);
  if ( pName == "MultiColoredSGS" ) p = new MultiColoredSGS<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType >; ls.SetPreconditioner(*p);
  if ( pName == "ILUT" ) p = new ILUT<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>; ls.SetPreconditioner(*p);
  if ( pName == "MultiColoredILU" )
  {
    MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > *p_ilu;
    p_ilu = new MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType >;

    if ( std::string(argv[4]) == "0" ) p_ilu->Set(0,1);
    if ( std::string(argv[4]) == "1" ) p_ilu->Set(1,2);
    if ( std::string(argv[4]) == "2" ) p_ilu->Set(2,3);

    p = p_ilu;
    ls.SetPreconditioner(*p);
  }
  if ( pName == "ILU" )
  {
    ILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > *p_ilu;
    p_ilu = new ILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType >;

    if ( std::string(argv[4]) == "0" ) p_ilu->Set(0);
    if ( std::string(argv[4]) == "1" ) p_ilu->Set(1);
    if ( std::string(argv[4]) == "2" ) p_ilu->Set(2);

    p = p_ilu;
    ls.SetPreconditioner(*p);
  }

  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();

  ls.Build();

  ls.Solve(rhs, &x);

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
