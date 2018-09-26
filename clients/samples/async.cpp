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

  LocalVector<double> x, y;
  LocalMatrix<double> mat;

  double tick, tack;
  double tickg, tackg;

  mat.ReadFileMTX(std::string(argv[1]));
 
  x.Allocate("x", mat.GetN());
  y.Allocate("y", mat.GetM());

  x.Ones();


  // No Async
  tickg = rocalution_time();

  y.Zeros();

  mat.Info();
  x.Info();
  y.Info();

  // CPU
  tick = rocalution_time();


  for (int i=0; i<100; ++i)
    mat.ApplyAdd(x, 1.0, &y);

  tack = rocalution_time();
  std::cout << "CPU Execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  std::cout << "Dot product = " << x.Dot(y) << std::endl;


  tick = rocalution_time();

  // Memory transfer
  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  y.MoveToAccelerator();

  mat.Info();
  x.Info();
  y.Info();

  tack = rocalution_time();
  std::cout << "Sync Transfer:" << (tack-tick)/1000000 << " sec" << std::endl;

  y.Zeros();

  // Accelerator
  tick = rocalution_time();

  for (int i=0; i<100; ++i)
    mat.ApplyAdd(x, 1.0, &y);

  tack = rocalution_time();
  std::cout << "Accelerator Execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  std::cout << "Dot product = " << x.Dot(y) << std::endl;

  tackg = rocalution_time();
  std::cout << "Total execution + transfers (no async):" << (tackg-tickg)/1000000 << " sec" << std::endl;






  mat.MoveToHost();
  x.MoveToHost();
  y.MoveToHost();

  y.Zeros();

  // Async

  tickg = rocalution_time();

  tick = rocalution_time();

  // Memory transfer
  mat.MoveToAcceleratorAsync();
  x.MoveToAcceleratorAsync();

  mat.Info();
  x.Info();
  y.Info();


  tack = rocalution_time();
  std::cout << "Async Transfer:" << (tack-tick)/1000000 << " sec" << std::endl;

  // CPU
  tick = rocalution_time();

  for (int i=0; i<100; ++i)
    mat.ApplyAdd(x, 1.0, &y);

  tack = rocalution_time();
  std::cout << "CPU Execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  std::cout << "Dot product = " << x.Dot(y) << std::endl;

  mat.Sync();
  x.Sync();

  y.MoveToAccelerator();

  mat.Info();
  x.Info();
  y.Info();

  y.Zeros();

  // Accelerator
  tick = rocalution_time();

  for (int i=0; i<100; ++i)
    mat.ApplyAdd(x, 1.0, &y);

  tack = rocalution_time();
  std::cout << "Accelerator Execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  std::cout << "Dot product = " << x.Dot(y) << std::endl;

  tackg = rocalution_time();
  std::cout << "Total execution + transfers (async):" << (tackg-tickg)/1000000 << " sec" << std::endl;



  stop_rocalution();

  return 0;
}
