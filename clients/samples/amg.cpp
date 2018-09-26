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

  double tick, tack, start, end;

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix> [Num threads]" << std::endl;
    exit(1);
  }

  init_rocalution();

  if (argc > 2)
    set_omp_threads_rocalution(atoi(argv[2]));

  info_rocalution();

  LocalVector<double> x;
  LocalVector<double> rhs;

  LocalMatrix<double> mat;

  mat.ReadFileMTX(std::string(argv[1]));

  x.Allocate("x", mat.GetN());
  rhs.Allocate("rhs", mat.GetM());

  rhs.Ones();
  x.Zeros(); 

  tick = rocalution_time();
  start = rocalution_time();

  // Linear Solver
  SAAMG<LocalMatrix<double>, LocalVector<double>, double > ls;

  ls.SetOperator(mat);

  // coupling strength
  ls.SetCouplingStrength(0.001);
  // number of unknowns on coarsest level
  ls.SetCoarsestLevel(300);
  // Relaxation parameter for smoothed interpolation aggregation
  ls.SetInterpRelax(2./3.);
  // Manual smoothers
  ls.SetManualSmoothers(true);
  // Manual course grid solver
  ls.SetManualSolver(true);
  // grid transfer scaling
  ls.SetScaling(true);

  ls.BuildHierarchy();

  int levels = ls.GetNumLevels();

  // Smoother for each level
  IterativeLinearSolver<LocalMatrix<double>, LocalVector<double>, double > **sm = NULL;
  MultiColoredGS<LocalMatrix<double>, LocalVector<double>, double > **gs = NULL;

  // Coarse Grid Solver
  CG<LocalMatrix<double>, LocalVector<double>, double > cgs;
  cgs.Verbose(0);

  sm = new IterativeLinearSolver<LocalMatrix<double>, LocalVector<double>, double >*[levels-1];
  gs = new MultiColoredGS<LocalMatrix<double>, LocalVector<double>, double >*[levels-1];

  // Preconditioner
  //  MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double > p;
  //  cgs->SetPreconditioner(p);

  for (int i=0; i<levels-1; ++i) {
    FixedPoint<LocalMatrix<double>, LocalVector<double>, double > *fp;
    fp = new FixedPoint<LocalMatrix<double>, LocalVector<double>, double >;
    fp->SetRelaxation(1.3);
    sm[i] = fp;
    
    gs[i] = new MultiColoredGS<LocalMatrix<double>, LocalVector<double>, double >;
    
    sm[i]->SetPreconditioner(*gs[i]);
    sm[i]->Verbose(0);
  }

  ls.SetSmoother(sm);
  ls.SetSolver(cgs);
  ls.SetSmootherPreIter(1);
  ls.SetSmootherPostIter(2);
  ls.Init(1e-10, 1e-8, 1e+8, 10000);
  ls.Verbose(2);
    
  ls.Build();
  
  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  ls.MoveToAccelerator();

  mat.Info();
  
  tack = rocalution_time();
  std::cout << "Building time:" << (tack-tick)/1000000 << " sec" << std::endl;
  
  tick = rocalution_time();
  
  ls.Solve(rhs, &x);
  
  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;
  
  ls.Clear();

  // Free all allocated data
  for (int i=0; i<levels-1; ++i) {
    delete gs[i];
    delete sm[i];
  }
  delete[] gs;
  delete[] sm;

  ls.Clear();

  end = rocalution_time();
  std::cout << "Total runtime:" << (end-start)/1000000 << " sec" << std::endl;

  stop_rocalution();

  return 0;
}
