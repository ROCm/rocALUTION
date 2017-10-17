#include <iostream>
#include <cstdlib>

#include <paralution.hpp>

#define ValueType double

using namespace paralution;

int main(int argc, char* argv[]) {

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix> <solution_vector>" << std::endl;
    exit(1);
  }

  init_paralution();

  LocalVector<ValueType> b, b_old, *b_k, *b_k1, *b_tmp, compare;
  LocalMatrix<ValueType> mat;

  mat.ReadFileMTX(std::string(argv[1]));

  // Gershgorin spectrum approximation
  ValueType glambda_min, glambda_max;

  // Power method spectrum approximation
  ValueType plambda_min, plambda_max;

  // Maximum number of iteration for the power method
  int iter_max = 10000;

  // Gershgorin approximation of the eigenvalues
  mat.Gershgorin(glambda_min, glambda_max);

  mat.MoveToAccelerator();
  b.MoveToAccelerator();
  b_old.MoveToAccelerator();


  b.Allocate("b_k+1", mat.get_nrow());
  b_k1 = &b;

  b_old.Allocate("b_k", mat.get_nrow());
  b_k = &b_old;  

  b_k->Ones();

  // compute lambda max
  for (int i=0; i<=iter_max; ++i) {

    mat.Apply(*b_k, b_k1);

    //    std::cout << b_k1->Dot(*b_k) << std::endl;
    b_k1->Scale(ValueType(1.0)/b_k1->Norm());

    b_tmp = b_k1;
    b_k1 = b_k;
    b_k = b_tmp;

  }

  // get lambda max (Rayleigh quotient)
  mat.Apply(*b_k, b_k1);
  plambda_max = b_k1->Dot(*b_k) ;

  mat.AddScalarDiagonal(ValueType(-1.0)*plambda_max);

  b_k->Ones();

  // compute lambda min
  for (int i=0; i<=iter_max; ++i) {

    mat.Apply(*b_k, b_k1);

    //    std::cout << b_k1->Dot(*b_k) + plambda_max << std::endl;
    b_k1->Scale(ValueType(1.0)/b_k1->Norm());

    b_tmp = b_k1;
    b_k1 = b_k;
    b_k = b_tmp;

  }

  // get lambda min (Rayleigh quotient)
  mat.Apply(*b_k, b_k1);
  plambda_min = (b_k1->Dot(*b_k) + plambda_max);

  // back to the original matrix
  mat.AddScalarDiagonal(plambda_max);

  LocalVector<ValueType> x;
  LocalVector<ValueType> rhs;

  x.CloneBackend(mat);
  rhs.CloneBackend(mat);

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());

  // Chebyshev iteration
  Chebyshev<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;

  rhs.Ones();
  x.Zeros(); 

  ls.SetOperator(mat);

  ls.Set(plambda_min, plambda_max);

  ls.Build();

  ls.Solve(rhs, &x);

  // PCG + Chebyshev polynomial
  CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > cg;
  AIChebyshev<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

  // damping factor
  plambda_min = plambda_max / 7;
  p.Set(3, plambda_min, plambda_max);
  rhs.Ones();
  x.Zeros(); 

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

  stop_paralution();

  return 0;
}
