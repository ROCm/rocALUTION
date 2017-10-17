#include <iostream>
#include <cstdlib>

#include <paralution.hpp>

using namespace paralution;

int main(int argc, char* argv[]) {

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix> [Num threads]" << std::endl;
    exit(1);
  }

  init_paralution();

  if (argc > 2) {
    set_omp_threads_paralution(atoi(argv[2]));
  } 

  info_paralution();

  LocalVector<double> x, y;
  LocalMatrix<double> mat;

  double tick, tack;
  double tickg, tackg;

  mat.ReadFileMTX(std::string(argv[1]));
 
  x.Allocate("x", mat.get_nrow());
  y.Allocate("y", mat.get_nrow());

  x.Ones();


  // No Async
  tickg = paralution_time();

  y.Zeros();

  mat.info();
  x.info();
  y.info();

  // CPU
  tick = paralution_time();


  for (int i=0; i<100; ++i)
    mat.ApplyAdd(x, 1.0, &y);

  tack = paralution_time();
  std::cout << "CPU Execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  std::cout << "Dot product = " << x.Dot(y) << std::endl;


  tick = paralution_time();

  // Memory transfer
  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  y.MoveToAccelerator();

  mat.info();
  x.info();
  y.info();

  tack = paralution_time();
  std::cout << "Sync Transfer:" << (tack-tick)/1000000 << " sec" << std::endl;

  y.Zeros();

  // Accelerator
  tick = paralution_time();

  for (int i=0; i<100; ++i)
    mat.ApplyAdd(x, 1.0, &y);

  tack = paralution_time();
  std::cout << "Accelerator Execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  std::cout << "Dot product = " << x.Dot(y) << std::endl;

  tackg = paralution_time();
  std::cout << "Total execution + transfers (no async):" << (tackg-tickg)/1000000 << " sec" << std::endl;






  mat.MoveToHost();
  x.MoveToHost();
  y.MoveToHost();

  y.Zeros();

  // Async

  tickg = paralution_time();

  tick = paralution_time();

  // Memory transfer
  mat.MoveToAcceleratorAsync();
  x.MoveToAcceleratorAsync();

  mat.info();
  x.info();
  y.info();


  tack = paralution_time();
  std::cout << "Async Transfer:" << (tack-tick)/1000000 << " sec" << std::endl;

  // CPU
  tick = paralution_time();

  for (int i=0; i<100; ++i)
    mat.ApplyAdd(x, 1.0, &y);

  tack = paralution_time();
  std::cout << "CPU Execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  std::cout << "Dot product = " << x.Dot(y) << std::endl;

  mat.Sync();
  x.Sync();

  y.MoveToAccelerator();

  mat.info();
  x.info();
  y.info();

  y.Zeros();

  // Accelerator
  tick = paralution_time();

  for (int i=0; i<100; ++i)
    mat.ApplyAdd(x, 1.0, &y);

  tack = paralution_time();
  std::cout << "Accelerator Execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  std::cout << "Dot product = " << x.Dot(y) << std::endl;

  tackg = paralution_time();
  std::cout << "Total execution + transfers (async):" << (tackg-tickg)/1000000 << " sec" << std::endl;



  stop_paralution();

  return 0;
}
