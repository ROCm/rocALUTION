#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#include <paralution.hpp>

using namespace paralution;

int main(int argc, char* argv[]) {

  if (argc < 3) { 
    std::cerr << argv[0] << " <matrix> <solution_vector> <preconditioner>" << std::endl;
    exit(1);
  }

  init_paralution();

  LocalVector<double> x;
  LocalVector<double> rhs;
  LocalVector<double> compare;

  LocalMatrix<double> mat;

  mat.ReadFileMTX(std::string(argv[1]));
  compare.ReadFileASCII(std::string(argv[2]));

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());

  MixedPrecisionDC<LocalMatrix<double>, LocalVector<double>, double,
                   LocalMatrix<float>, LocalVector<float>, float> mp;

  CG<LocalMatrix<float>, LocalVector<float>, float> cg;

  Preconditioner<LocalMatrix<float>, LocalVector<float>, float> *p = NULL;

  rhs.Ones();
  x.Zeros();

  std::string pName = argv[3];
  if ( pName == "Jacobi" ) p = new Jacobi<LocalMatrix<float>, LocalVector<float>, float >; cg.SetPreconditioner(*p);
  if ( pName == "MultiColoredSGS" ) p = new MultiColoredSGS<LocalMatrix<float>, LocalVector<float>, float >; cg.SetPreconditioner(*p);
  if ( pName == "ILUT" ) p = new ILUT<LocalMatrix<float>, LocalVector<float>, float>; cg.SetPreconditioner(*p);
  if ( pName == "MultiColoredILU" )
  {
    MultiColoredILU<LocalMatrix<float>, LocalVector<float>, float > *p_ilu;
    p_ilu = new MultiColoredILU<LocalMatrix<float>, LocalVector<float>, float >;
    if ( std::string(argv[4]) == "0" ) p_ilu->Set(0,1);
    if ( std::string(argv[4]) == "1" ) p_ilu->Set(1,2);
    if ( std::string(argv[4]) == "2" ) p_ilu->Set(2,3);

    p = p_ilu;
    cg.SetPreconditioner(*p);
  }
  if ( pName == "ILU" )
  {
    ILU<LocalMatrix<float>, LocalVector<float>, float > *p_ilu;
    p_ilu = new ILU<LocalMatrix<float>, LocalVector<float>, float >;

    if ( std::string(argv[4]) == "0" ) p_ilu->Set(0,1);
    if ( std::string(argv[4]) == "1" ) p_ilu->Set(1,2);
    if ( std::string(argv[4]) == "2" ) p_ilu->Set(2,3);

    p = p_ilu;
    cg.SetPreconditioner(*p);
  }

  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();

  // setup a lower tol for the inner solver
  cg.Init(1e-5, 1e-2, 1e+20,
          100000);

  // setup the mixed-precision DC
  mp.SetOperator(mat);
  mp.Set(cg);

  mp.Build();

  mp.Solve(rhs, &x);

  x.MoveToHost();

  // |x-x_ref| < eps
  x.AddScale(compare, double(-1.0));
  LOG_INFO("Error Norm = " << x.Norm());

  if (x.Norm() > 1e-4)
  {
    std::cout <<"Test failed." <<std::endl;
    exit(-1);
  }

  stop_paralution();

  return 0;
}
