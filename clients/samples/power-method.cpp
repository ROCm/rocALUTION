/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
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

  LocalVector<double> b, b_old, *b_k, *b_k1, *b_tmp;
  LocalMatrix<double> mat;

  mat.ReadFileMTX(std::string(argv[1]));

  // Gershgorin spectrum approximation
  double glambda_min, glambda_max;

  // Power method spectrum approximation
  double plambda_min, plambda_max;

  // Maximum number of iteration for the power method
  int iter_max = 10000;

  double tick, tack;

  // Gershgorin approximation of the eigenvalues
  mat.Gershgorin(glambda_min, glambda_max);
  std::cout << "Gershgorin : Lambda min = " << glambda_min
            << "; Lambda max = " << glambda_max << std::endl;


  mat.MoveToAccelerator();
  b.MoveToAccelerator();
  b_old.MoveToAccelerator();


  b.Allocate("b_k+1", mat.GetM());
  b_k1 = &b;

  b_old.Allocate("b_k", mat.GetM());
  b_k = &b_old;  

  b_k->Ones();

  mat.Info();

  tick = rocalution_time();

  // compute lambda max
  for (int i=0; i<=iter_max; ++i) {

    mat.Apply(*b_k, b_k1);

    //    std::cout << b_k1->Dot(*b_k) << std::endl;
    b_k1->Scale(double(1.0)/b_k1->Norm());

    b_tmp = b_k1;
    b_k1 = b_k;
    b_k = b_tmp;

  }

  // get lambda max (Rayleigh quotient)
  mat.Apply(*b_k, b_k1);
  plambda_max = b_k1->Dot(*b_k) ;

  tack = rocalution_time();
  std::cout << "Power method (lambda max) execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  mat.AddScalarDiagonal(double(-1.0)*plambda_max);


  b_k->Ones();

  tick = rocalution_time();

  // compute lambda min
  for (int i=0; i<=iter_max; ++i) {

    mat.Apply(*b_k, b_k1);

    //    std::cout << b_k1->Dot(*b_k) + plambda_max << std::endl;
    b_k1->Scale(double(1.0)/b_k1->Norm());

    b_tmp = b_k1;
    b_k1 = b_k;
    b_k = b_tmp;

  }

  // get lambda min (Rayleigh quotient)
  mat.Apply(*b_k, b_k1);
  plambda_min = (b_k1->Dot(*b_k) + plambda_max);

  // back to the original matrix
  mat.AddScalarDiagonal(plambda_max);

  tack = rocalution_time();
  std::cout << "Power method (lambda min) execution:" << (tack-tick)/1000000 << " sec" << std::endl;


  std::cout << "Power method Lambda min = " << plambda_min
            << "; Lambda max = " << plambda_max 
            << "; iter=2x" << iter_max << std::endl;

  LocalVector<double> x;
  LocalVector<double> rhs;

  x.CloneBackend(mat);
  rhs.CloneBackend(mat);

  x.Allocate("x", mat.GetN());
  rhs.Allocate("rhs", mat.GetM());

  // Chebyshev iteration
  Chebyshev<LocalMatrix<double>, LocalVector<double>, double > ls;

  rhs.Ones();
  x.Zeros(); 

  ls.SetOperator(mat);

  ls.Set(plambda_min, plambda_max);

  ls.Build();

  tick = rocalution_time();

  ls.Solve(rhs, &x);

  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  ls.Clear();

  // PCG + Chebyshev polynomial
  CG<LocalMatrix<double>, LocalVector<double>, double > cg;
  AIChebyshev<LocalMatrix<double>, LocalVector<double>, double > p;

  // damping factor
  plambda_min = plambda_max / 7;
  p.Set(3, plambda_min, plambda_max);
  rhs.Ones();
  x.Zeros(); 

  cg.SetOperator(mat);
  cg.SetPreconditioner(p);

  cg.Build();

  tick = rocalution_time();

  cg.Solve(rhs, &x);

  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  cg.Clear();

  stop_rocalution();

  return 0;
}
