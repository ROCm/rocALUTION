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

  start = rocalution_time();

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

  CG<LocalMatrix<double>, LocalVector<double>, double > ls;
  ls.Verbose(0);

  // AMG Preconditioner
  UAAMG<LocalMatrix<double>, LocalVector<double>, double > p;

  p.InitMaxIter(1);
  p.Verbose(0);

  ls.SetPreconditioner(p);
  ls.SetOperator(mat);
  ls.Build();

  // Compute coarsest levels on the host
  if(p.GetNumLevels() > 2)
  {
    p.SetHostLevels(2);
  }

  tack = rocalution_time();
  std::cout << "Building time:" << (tack-tick)/1000000 << " sec" << std::endl;

  // move after building since AMG building is not supported on GPU yet
  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  ls.MoveToAccelerator();

  mat.Info();

  tick = rocalution_time();

  ls.Init(1e-8, 1e-8, 1e+8, 10000);
  ls.Verbose(2);

  ls.Solve(rhs, &x);

  tack = rocalution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  ls.Clear();

  end = rocalution_time();
  std::cout << "Total runtime:" << (end-start)/1000000 << " sec" << std::endl;

  stop_rocalution();

  return 0;

}
